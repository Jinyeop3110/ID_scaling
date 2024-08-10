#import transformer_lens

import sys
from fancy_einsum import einsum

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM, AutoConfig
from transformer_lens import ActivationCache, HookedTransformer
from transformers import  GemmaForCausalLM  # Explicitly import Gemma classes

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



BASIC_MODEL_TYPES = ['llama2-7b', 'mistral-7b', 'gemma-2b', 'gemma-7b']
ATTENTION_ONLY_MODEL_TYPES = ['attn-only-1l', 'attn-only-2l', 'attn-only-3l', 'attn-only-4l']
GPT2_TYPES = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
OLMO_TYPES = ['olmo-1b', 'olmo-7b']
PYTHIA_TYPES = [
    'pythia-1.4b', 'pythia-12b', 'pythia-14m', 'pythia-160m-deduped', 
    'pythia-1b-deduped', 'pythia-2.8b-deduped', 'pythia-410m-deduped', 
    'pythia-6.9b-deduped', 'pythia-70m-deduped', 'pythia-1.4b-deduped', 
    'pythia-12b-deduped', 'pythia-160m', 'pythia-1b', 'pythia-2.8b', 
    'pythia-410m', 'pythia-6.9b', 'pythia-70m'
]
ALL_MODEL_TYPES=BASIC_MODEL_TYPES+ATTENTION_ONLY_MODEL_TYPES+GPT2_TYPES+OLMO_TYPES+PYTHIA_TYPES




def load_Model(model_config, set_grad_enabled=False):
    torch.set_grad_enabled(set_grad_enabled)
    
    model_type = model_config.model_name
    checkpoint = model_config.model_checkpoint

    if model_type in BASIC_MODEL_TYPES:
        model, tokenizer = load_basic_models(model_type, set_grad_enabled)
    elif model_type in ATTENTION_ONLY_MODEL_TYPES:
        model, tokenizer = load_attention_only_models(model_type, set_grad_enabled)
    elif model_type in GPT2_TYPES:
        model, tokenizer = load_gpt2_models(model_type, set_grad_enabled)
    elif model_type in OLMO_TYPES:
        if checkpoint is None:
            checkpoint = "main"
        model, tokenizer = load_olmo_models(model_type, checkpoint, set_grad_enabled)
    elif model_type in PYTHIA_TYPES:
        if checkpoint is None:
            checkpoint = "main"
        model, tokenizer = load_pythia_models(model_type, checkpoint, set_grad_enabled)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, tokenizer

# Dummy function definitions for load_olmo_models and load_pythia_models
def load_olmo_models(model_type, checkpoint, set_grad_enabled):
    torch.set_grad_enabled(set_grad_enabled)

    model_paths = {
        'olmo-1b': "../Models/OLMo-1B",
        'olmo-7b': "../Models/OLMo-7B",
        'olmo-1.7-7b': "../Models/OLMo-1.7-7B",
    }
    
    model_path= model_paths.get(model_type)
    model_path=model_path+'/'+checkpoint

    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Trying to load model : {model_type} / checkpoint : {checkpoint} -> from {model_path}")

    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path , trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, tokenizer

def load_pythia_models(model_type, checkpoint, set_grad_enabled):
    torch.set_grad_enabled(set_grad_enabled)

    model_paths = {
        'pythia-1.4b': "../Models/Pythia/pythia-1.4b",
        'pythia-12b': "../Models/Pythia/pythia-12b",
        'pythia-14m': "../Models/Pythia/pythia-14m",
        'pythia-160m-deduped': "../Models/Pythia/pythia-160m-deduped",
        'pythia-1b-deduped': "../Models/Pythia/pythia-1b-deduped",
        'pythia-2.8b-deduped': "../Models/Pythia/pythia-2.8b-deduped",
        'pythia-410m-deduped': "../Models/Pythia/pythia-410m-deduped",
        'pythia-6.9b-deduped': "../Models/Pythia/pythia-6.9b-deduped",
        'pythia-70m-deduped': "../Models/Pythia/pythia-70m-deduped",
        'pythia-1.4b-deduped': "../Models/Pythia/pythia-1.4b-deduped",
        'pythia-12b-deduped': "../Models/Pythia/pythia-12b-deduped",
        'pythia-160m': "../Models/Pythia/pythia-160m",
        'pythia-1b': "../Models/Pythia/pythia-1b",
        'pythia-2.8b': "../Models/Pythia/pythia-2.8b",
        'pythia-410m': "../Models/Pythia/pythia-410m",
        'pythia-6.9b': "../Models/Pythia/pythia-6.9b",
        'pythia-70m': "../Models/Pythia/pythia-70m"
    }
    
    model_path = model_paths.get(model_type)
    
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_path = model_path + '/' + checkpoint
    print(f"Trying to load model: {model_type} / checkpoint: {checkpoint} -> from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path , trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def load_basic_models(model_type, set_grad_enabled=False):
    torch.set_grad_enabled(set_grad_enabled)

    model_paths = {
        'llama2-7b': "../Models/Llama-2-7b-hf",
        'mistral-7b': "../Models/Mistral-7B-v0.1",
        'gemma-7b': "../Models/Gemma-7B",
    }


    tokenizer_path = model_paths[model_type]
    model_path = model_paths.get(model_type)
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Trying to load model : {model_type} from {model_path}")

    '''
    if model_type.startswith('gemma'):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            model = GemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            print("Falling back to AutoTokenizer and AutoModelForCausalLM")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
    '''
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, tokenizer

################################################################################################
# ###################################  Attention Only models #################################### 
# ###############################################################################################


# TODO: For attention only model, I skip the implementation 

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

    model_path = model_paths.get(model_type)
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Trying to load model : {model_type} from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path , trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model, tokenizer

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

    tokenizer_path = model_paths[model_type]
    model_path = Path(model_paths.get(model_type))
    if not model_path:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Trying to load model : {model_type} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path , trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True )
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
   
    return model,tokenizer



