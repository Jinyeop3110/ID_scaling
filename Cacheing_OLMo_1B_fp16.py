import gc
import os
from accelerate import Accelerator
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
import numpy as np 

from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching
import circuitsvis as cv
import matplotlib.pyplot as plt
from pathlib import Path


from utils.load_Datasets import load_Datasets
from utils.load_Model import load_Model
from utils.utils import *
import json
import argparse
import concurrent.futures


def main(session_name, model_name, model_checkpoint, dataset_name, dataset_subset, tokens_min_length):
    ##############################################################################################################################################################################################
    ################################################################################## Model ###################################################################################################
    ##############################################################################################################################################################################################
    
    session_path = f"Data/{session_name}"  # Update this to your desired path
    
    ##############################################################################################################################################################################################
    ################################################################################## Model ###################################################################################################
    ##############################################################################################################################################################################################
    
    model, tokenizer = load_Model(model_name, checkpoint=model_checkpoint)
    accelerator = Accelerator(mixed_precision='fp16') 
    model = accelerator.prepare(model)
    
    ##############################################################################################################################################################################################
    ################################################################################## DATASET ###################################################################################################
    ##############################################################################################################################################################################################
    
    dataset = load_Datasets(dataset_config_dict={
            'name': dataset_name,
            'subset': dataset_subset,
            'base_directory': None
        })
    print(f"Original dataset size: {len(dataset)}")

    def filter_item(index, item, tokenizer, tokens_min_length):
        tokens = tokenizer(item['text'], truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1] >= tokens_min_length:
            return index
        return None

    def filter_dataset_indices(dataset, tokenizer, tokens_min_length=1024, max_workers=4):
        valid_indices = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(filter_item, i, item, tokenizer, tokens_min_length): i for i, item in enumerate(dataset)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    valid_indices.append(result)
        valid_indices.sort()
        return valid_indices

    valid_indices = filter_dataset_indices(dataset, tokenizer, tokens_min_length=tokens_min_length, max_workers=8)
    dataset = dataset.select(valid_indices)
    print(f"Filtered dataset size: {len(dataset)}")

    ##############################################################################################################################################################################################
    ################################################################################## Hyperparam ################################################################################################
    ##############################################################################################################################################################################################
    ctx_len = 2048
    batch_size = 8

    pos_list = np.array([0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, -2,-3,-4,-5, -8, -16, -32, -64])
    cache = {}

    module_name_list=["attn_out", "ff_out", ""]
    layer_idx_list=np.array([0,3,7,13,14,15])
    module_inblock_sentence = "model.transformer.blocks.{layer}{module}"
    module_inblock_list = [
        module_inblock_sentence.format(layer=layer, module=f".{module}" if module else "")
        for layer in layer_idx_list for module in module_name_list
    ]

    print(module_inblock_list)
    module_outblock_list=["model.transformer.ln_f"]

    module_list = module_inblock_list + module_outblock_list


    ##############################################################################################################################################################################################
    ################################################################################## Hyperparam ################################################################################################
    ##############################################################################################################################################################################################
    # config define
    config = {
        "session_name": session_name,
        "session_path": session_path,
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "d_model":model.config.hidden_size,
        "ctx_len": ctx_len,
        "tokens_min_length": tokens_min_length,
        "batch_size": batch_size,
        "pos_list": pos_list.tolist(),  # Convert numpy array to list for better readability in logs
        "cache": cache,
        "module_name_list": module_name_list,
        "layer_idx_list": layer_idx_list.tolist(),  # Convert numpy array to list
        "module_inblock_list": module_inblock_list,
        "module_outblock_list": module_outblock_list,
        "module_list": module_list
    }






    ##############################################################################################################################################################################################
    ################################################################################## Running experiment ################################################################################################
    ##############################################################################################################################################################################################





    # Create the directory if it doesn't exist


    def hook_fn(module, input, output):
        # Retrieve the full name from the context in which the hook was registered
        full_name = hook_fn.full_names.get(module, "Unknown")  
        
        if isinstance(output, tuple):
            output = output[0]

        cache[full_name] = {}
        for pos in pos_list:
            if pos>=0:
                cache[full_name][pos]=output.detach()[:,pos,:]  # Keep tensor on CUDA
            elif pos<0:
                tensor=output.detach()
                cache[full_name][pos]=tensor[torch.arange(tensor.size(0)), np.maximum(hook_fn.len_list+pos,0),:] 


    # Register hooks to specified modules
    def register_hooks(model, module_list, pos_list):
        hooks = []
        hook_fn.full_names = {}  # Dictionary to store module names
        hook_fn.pos_list = pos_list
        for name, module in model.named_modules():
            if name in module_list:
                hooks.append(module.register_forward_hook(hook_fn))
                hook_fn.full_names[module] = name
        return hooks

    # Remove hooks after processing
    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()
        return None

    # Get the length of each input sequence
    def get_len_list(tokens):
        len_list = (tokens['input_ids'] == tokenizer.pad_token_id).int().argmax(dim=1)
        len_list[len_list == 0] = ctx_len
        return len_list.numpy()

    # Save cache to files
    def save_cache_to_files(cache, session_path):
        for full_name, pos_dict in cache.items():
            for pos, tensor in pos_dict.items():
                file_path = os.path.join(session_path, f"{full_name.replace('.', '_')}_{pos}.pt")
                with open(file_path, 'ab') as f:
                    tensor_bytes = tensor.cpu().numpy().tobytes()
                    f.write(len(tensor_bytes).to_bytes(4, byteorder='big'))
                    f.write(tensor_bytes)

    # Save entire loss tensor to file
    def save_loss_to_files(loss, session_path):
        file_path = os.path.join(session_path, "CREloss.pt")
        with open(file_path, 'ab') as f:
            tensor_bytes = loss.cpu().numpy().tobytes()
            f.write(len(tensor_bytes).to_bytes(4, byteorder='big'))
            f.write(tensor_bytes)

    # Save entropy to file
    def save_entropy_to_files(entropy, session_path):
        file_path = os.path.join(session_path, "Entropy.pt")
        with open(file_path, 'ab') as f:
            tensor_bytes = entropy.cpu().numpy().tobytes()
            f.write(len(tensor_bytes).to_bytes(4, byteorder='big'))
            f.write(tensor_bytes)

    # Save metadata to files
    def save_metadata_to_files(df_metadata, session_path):
        file_path = os.path.join(session_path, "metadata.csv")
        if not os.path.isfile(file_path):
            df_metadata.to_csv(file_path, index=False)
        else:
            df_metadata.to_csv(file_path, mode='a', header=False, index=False)

    # Save configuration to a file
    def save_config(config, session_path):
        with open(os.path.join(session_path, "config.json"), 'w') as json_file:
            json.dump(config, json_file, indent=4)

    # Load cache from file
    def load_cache_from_file(file_path, dtype=torch.float32):
        tensors = []
        with open(file_path, 'rb') as f:
            while True:
                len_bytes = f.read(4)
                if not len_bytes:
                    break
                length = int.from_bytes(len_bytes, byteorder='big')
                tensor_bytes = f.read(length)
                tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
                tensor = torch.from_numpy(tensor_array)
                tensors.append(tensor)
        return tensors

    # Clear GPU memory
    def clear_gpu_memory(cache):
        for key in list(cache.keys()):
            del cache[key]
        torch.cuda.empty_cache()

    # Function to process each batch, run the model, and calculate entropy
    def run_batch_with_cache(batch):
        prompts = batch['text']
        tokens = tokenizer(prompts, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        len_list = get_len_list(tokens)
        df_metadata = pd.DataFrame({key: value for key, value in batch.items() if key != 'text'})
        hook_fn.len_list = len_list
        df_metadata['len'] = len_list
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            del outputs
            torch.cuda.empty_cache()

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), shift_logits.size(1))

            # Calculate entropy from logits
            probs = torch.softmax(shift_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            del probs
            torch.cuda.empty_cache()
        return df_metadata, loss.detach(), entropy

    # Main function to run the experiment
    os.makedirs(session_path, exist_ok=True)
    save_config(config, session_path)

    # Register hooks
    hooks = register_hooks(model, module_list, pos_list)

    # Iterate over the dataset with batch size
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        df_metadata, loss, entropy = run_batch_with_cache(batch)
        save_cache_to_files(cache, session_path)
        save_metadata_to_files(df_metadata, session_path)
        save_loss_to_files(loss, session_path)  # Save the entire loss tensor

        # Save entropy
        save_entropy_to_files(entropy, session_path)
        
        torch.cuda.empty_cache()
        clear_gpu_memory(cache)
        print(f"{i} to {i+batch_size} done, Entropy calculated and saved.")

    # Remove hooks after processing
    remove_hooks(hooks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model with dataset.')
    
    parser.add_argument('--session_name', type=str, required=True, help='Name of the session')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Checkpoint of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--dataset_subset', type=int, nargs='*', required=True, help='Subset of the dataset')
    parser.add_argument('--tokens_min_length', type=int, required=True, help='Minimum length of tokens in the dataset')
    
    args = parser.parse_args()
    print(args)
    main(args.session_name, args.model_name, args.model_checkpoint, args.dataset_name, args.dataset_subset, args.tokens_min_length)
  
    #main("test", model_name="olmo-1b", model_checkpoint="main", dataset_name='pile_uncopyrighted_parquet', dataset_subset=[0,1], tokens_min_length=1024)