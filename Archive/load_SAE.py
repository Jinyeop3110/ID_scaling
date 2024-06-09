#import transformer_lens

import sys
from fancy_einsum import einsum

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM, AutoConfig
from transformer_lens import ActivationCache, HookedTransformer

from huggingface_hub import notebook_login
from datasets import Dataset
import pandas as pd
import transformers
import torch
import einops  # Make sure einops is imported
from pathlib import Path

from typing import Optional, Union, Literal
from transformer_lens.loading_from_pretrained import *
from pathlib import Path
import json

def customized_get_pretrained_model_config(
    official_model_name, model_path,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    fold_ln: bool = False,
    device: Optional[str] = None,
    n_devices: int = 1,
    default_prepend_bos: bool = True,
    dtype: torch.dtype = torch.float32,
    **kwargs):

    official_model_name = get_official_model_name(official_model_name)

    cfg_json: dict = json.load(open(model_path / 'config.json', "r"))

    cfg_arch = cfg_json.get(
        "architecture", "neel" if "_old" not in official_model_name else "neel-solu-old"
    )
    cfg_dict = {
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json.get("tokenizer_name", None),
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": cfg_arch,
    }
    if "normalization" in cfg_json:
        cfg_dict["normalization_type"] = cfg_json["normalization"]
    else:
        cfg_dict["normalization_type"] = cfg_json["normalization_type"]
    if "shortformer_pos" in cfg_json:
        cfg_dict["positional_embedding_type"] = (
            "shortformer" if cfg_json["shortformer_pos"] else "standard"
        )
    else:
        cfg_dict["positional_embedding_type"] = "standard"


    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name,
            **kwargs,
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert (
                checkpoint_value in checkpoint_labels
            ), f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices
    cfg_dict["default_prepend_bos"] = default_prepend_bos

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg



def load_basic_models(model_type, set_grad_enabled=False):
    torch.set_grad_enabled(set_grad_enabled)

    model_paths = {
        'llama-7b': "../Models/Llama-2-7b-hf",
        'mistral-7b': "../Models/Mistral-7B-v0.1",
    }

    tokenizer_path = model_paths[model_type]
    model_path = model_paths.get(model_type)
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    #hf_config = AutoConfig.from_pretrained(model_path)
    #config = AutoConfig.from_pretrained(model_path+'/')
    #config['model_type']=model_type
    
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)

    for param in hf_model.parameters():
        param.requires_grad = False

    hf_model.eval()

    model = HookedTransformer.from_pretrained(model_type, hf_model=hf_model, tokenizer=tokenizer, device="cuda", dtype="float16")

    return model, tokenizer

################################################################################################
# ###################################  Attention Only models #################################### 
# ###############################################################################################


def load_attention_only_models(model_type, set_grad_enabled=False):
    torch.set_grad_enabled(set_grad_enabled)

    
    model_paths = {
        'attn-only-1l': "../Models/Attn_Only_1L512W_C4_Code",
        'attn-only-2l': "../Models/Attn_Only_2L512W_C4_Code",
        'attn-only-3l': "../Models/Attn_Only_3L512W_C4_Code",
        'attn-only-4l': "../Models/Attn_Only_4L512W_C4_Code",
    }
    model_path = Path(model_paths.get(model_type))

    tokenizer_path = "../Models/gpt-neox-tokenizer-digits"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    dtype=torch.float16

    cfg = customized_get_pretrained_model_config(
                model_type,
                model_path,
                device="cuda",
                dtype=torch.float16
            )

    model = HookedTransformer(cfg=cfg, tokenizer=tokenizer)

    state_dict = torch.load(model_path / 'model_final.pth' , map_location="cpu")

            # Convert to dtype
    state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}

    model.load_and_process_state_dict(
                state_dict,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                refactor_factored_attn_matrices=False,
            )
    
    model.move_model_modules_to_device()
    model.setup()
    
    return model,tokenizer

################################################################################################
# ###################################  GPT2 #################################### 
# ###############################################################################################


def load_gpt2_models(model_type, set_grad_enabled=False):
    
    torch.set_grad_enabled(set_grad_enabled)
    model_paths = {
            'gpt2-small': "../Models/gpt2-small",
            'gpt2-medium': "../Models/gpt2-medium",
            'gpt2-large': "../Models/gpt2-large",
            'gpt2-xl': "../Models/gpt2-xl",
        }
    model_cfg={
            'gpt2-small':{
            "d_model": 768,
            "n_layers": 12,
            "d_mlp": 3072,
            "d_head": 64,
            "n_heads": 12,
            "n_ctx": 1024,
            "d_vocab": 50257,
            "tokenizer_name": "gpt2",
            "act_fn": "gelu_new",
            "attn_only": False,
            "final_rms": False,
            "original_architecture":"GPT2LMHeadModel",
        },
            'gpt2-medium':{
            "d_model": 1024,
            "n_layers": 24,
            "d_mlp": 4096,
            "d_head": 64,
            "n_heads": 16,
            "n_ctx": 1024,
            "d_vocab": 50257,
            "tokenizer_name": "gpt2",
            "act_fn": "gelu_new",
            "attn_only": False,
            "final_rms": False,
            "original_architecture":"GPT2LMHeadModel",
        },
            'gpt2-large':{
            "d_model": 1280,
            "n_layers": 36,
            "d_mlp": 5120,
            "d_head": 64,
            "n_heads": 20,
            "n_ctx": 1024,
            "d_vocab": 50257,
            "tokenizer_name": "gpt2",
            "act_fn": "gelu_new",
            "attn_only": False,
            "final_rms": False,
            "original_architecture": "GPT2LMHeadModel",
        },
            'gpt2-xl':{
            "d_model": 1600,
            "n_layers": 48,
            "d_mlp": 6400,
            "d_head": 64,
            "n_heads": 25,
            "n_ctx": 1024,
            "d_vocab": 50257,
            "tokenizer_name": "gpt2",
            "act_fn": "gelu_new",
            "attn_only": False,
            "final_rms": False,
            "original_architecture": "GPT2LMHeadModel",
        },     
    }
    
    tokenizer_path = model_paths[model_type]
    model_path = Path(model_paths.get(model_type))
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_model.eval()
    
    for param in hf_model.parameters():
        param.requires_grad = False

    cfg = model_cfg[model_type]
    cfg= HookedTransformerConfig.from_dict(cfg)

    with open(model_path / 'config.json', 'r') as config_file:
        cfg_loaded=json.load(config_file)
        
    model = HookedTransformer(cfg=cfg, tokenizer=tokenizer, move_to_device=False, default_padding_side="right")

    #state_dict = hf_model.state_dict()
    state_dict = convert_gpt2_weights(hf_model, cfg)
    model.load_and_process_state_dict(
                state_dict,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
    )
        
    model.move_model_modules_to_device()
    model.setup()
   
    return model,tokenizer



def load_HookedTransformer(model_type, set_grad_enabled=False):
    torch.set_grad_enabled(set_grad_enabled)

    basic_model_types = ['llama-7b', 'mistral-7b']
    attention_only_model_types = ['attn-only-1l', 'attn-only-2l', 'attn-only-3l', 'attn-only-4l']
    gpt2_types=['gpt2-small', 'gpt2-medium','gpt2-large','gpt2-xl']
    
    if model_type in basic_model_types:
        model, tokenizer = load_basic_models(model_type, set_grad_enabled)
    elif model_type in attention_only_model_types:
        model, tokenizer = load_attention_only_models(model_type, set_grad_enabled)
    elif model_type in gpt2_types:
        model, tokenizer = load_gpt2_models(model_type, set_grad_enabled)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, tokenizer
