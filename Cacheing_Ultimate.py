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

import matplotlib.pyplot as plt
from pathlib import Path

from utils import *
import json
import argparse
import concurrent.futures


def main(config):
    ##############################################################################################################################################################################################
    ################################################################################## Model ###################################################################################################
    ##############################################################################################################################################################################################
    

    session_path = config.session_path  # Update this to your desired path
    
    ##############################################################################################################################################################################################
    ################################################################################## Model ###################################################################################################
    ##############################################################################################################################################################################################
    
    model, tokenizer = load_Model(config.model_config)
    if torch.cuda.is_available():
        if config.model_config.use_accelerator:
                accelerator = Accelerator(mixed_precision='fp16') 
                model = accelerator.prepare(model)
        else:
            if torch.cuda.is_available():
                model.cuda()

    ##############################################################################################################################################################################################
    ################################################################################## DATASET ###################################################################################################
    ##############################################################################################################################################################################################
    
    dataset = load_Dataset(config.dataset_config)
    #dataset=dataset.select(np.arange(1000))
    print(f"Original dataset size: {len(dataset)}")

    ##############################################################################################################################################################################################
    ############################################################################ DATASET Processing ###################################################################################################
    ##############################################################################################################################################################################################

    def process_dataset(dataset, tokenizer, ctx_len, filtering_config):

        def filter_item(index, item, tokenizer, tokens_min_length):
            tokens = tokenizer(item['text'], truncation=False, return_tensors='pt')
            if tokens['input_ids'].shape[1] >= tokens_min_length:
                return index
            return None

        def filter_dataset_indices(dataset, tokenizer, tokens_min_length, max_workers=8):
            valid_indices = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(filter_item, i, item, tokenizer, tokens_min_length): i for i, item in enumerate(dataset)}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        valid_indices.append(result)
            valid_indices.sort()
            return valid_indices
        
        tokens_min_length=ctx_len*filtering_config.min_chunks_from_a_document
        valid_indices = filter_dataset_indices(dataset, tokenizer, tokens_min_length=tokens_min_length, max_workers=8)
        dataset = dataset.select(valid_indices)
        print(f"Filtered dataset size: {len(dataset)}")

        def chunk_dataset_with_tokenizer(dataset, tokenizer, chunk_size=1024, max_chunk_idx=10, overlap=0):
            def chunk_text_with_meta(data, tokenizer, chunk_size, max_chunk_idx, overlap):
                full_text = data['text']
                chunks = []
                tokens = tokenizer.encode(full_text, truncation=False)
                start = 0
                
                while start < len(tokens):
                    end = start + chunk_size
                    chunk_tokens = tokens[start:end]
                    chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk)
                    
                    if len(chunks) >= max_chunk_idx:
                        break
                    
                    start += chunk_size - overlap
                
                new_data = []
                for j, chunk in enumerate(chunks):
                    new_entry = {'id': f"{data['id']}-{j}", 'text': chunk}
                    for key, value in data.items():
                        if key not in ['id', 'text']:
                            new_entry[key] = value
                    new_data.append(new_entry)
                
                return new_data

            new_data = []
            for i, data in enumerate(dataset):
                new_data.extend(chunk_text_with_meta(data, tokenizer, chunk_size, max_chunk_idx, overlap))

            # Converting list of dictionaries to HuggingFace Dataset
            chunked_dataset = Dataset.from_pandas(pd.DataFrame(new_data))
            return chunked_dataset
        
        chunked_dataset=chunk_dataset_with_tokenizer(dataset, tokenizer, chunk_size=ctx_len, max_chunk_idx=filtering_config.max_chunks_from_a_document)

    processed_dataset = process_dataset(dataset, tokenizer, config.ctx_len, config.dataset_config.filtering_config)



    
    ##############################################################################################################################################################################################
    ################################################################################## Module Names ################################################################################################
    ##############################################################################################################################################################################################
    batch_size = config.batch_size

    def create_module_names(module_name_mapping, layer_idx_list, module_inblock_keys, module_outblock_keys):

        module_name_mapping=dict(module_name_mapping)

        module_inblock_mapping = {
            f"{module_key}-l{layer}" : module_name_mapping[module_key].format(layer=layer) if module_key else ""
            for layer in layer_idx_list for module_key in module_inblock_keys
        }
        module_outblock_mapping = {
            f"{module_key}" : module_name_mapping[module_key] if module_key else ""
            for module_key in module_outblock_keys
        }   
        module_name_mapping = {**module_inblock_mapping, **module_outblock_mapping}
        module_name_keys = list(module_name_mapping.keys())
        return module_name_mapping, module_name_keys
    
    module_name_mapping, module_name_keys = create_module_names(config.model_config.module_name_mapping, config.layer_idx_list, config.module_inblock_keys, config.module_outblock_keys)

    print(module_name_mapping)
    print(module_name_keys)


    ##############################################################################################################################################################################################
    ################################################################################## Hyperparam ################################################################################################
    ##############################################################################################################################################################################################
    # config define
    '''
    config = {
        "session_name": session_name,
        "session_path": session_path,
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "use_accelerator" : use_accelerator,
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "d_model": model.config.hidden_size,
        "ctx_len": ctx_len,
        "tokens_min_length": tokens_min_length,
        "batch_size": batch_size,
        "pos_list": None,  # Convert numpy array to list for better readability in logs
        "cache": cache,
        "module_name_list": module_name_list,
        "layer_idx_list": layer_idx_list.tolist(),  # Convert numpy array to list
        "module_inblock_list": module_inblock_list,
        "module_outblock_list": module_outblock_list,
        "module_list": module_list
    }'''


    ##############################################################################################################################################################################################
    ################################################################################## Running experiment ################################################################################################
    ##############################################################################################################################################################################################



    cache = {}

    # Create the directory if it doesn't exist

    def hook_fn_all(module, input, output):
        # Retrieve the full name from the context in which the hook was registered
        with torch.no_grad():
            full_name = hook_fn_all.full_names.get(module, "Unknown")  
            
            if isinstance(output, tuple):
                output = output[0]

            cache[full_name]=output.detach()[:,:,:]  

    # Register hooks to specified modules
    def register_hooks(model, module_list):
        hooks = []
        hook_fn_all.full_names = {}  # Dictionary to store module names
        for name, module in model.named_modules():
            if name in module_list:
                hooks.append(module.register_forward_hook(hook_fn_all))
                hook_fn_all.full_names[module] = name
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
        for full_name, tensor in cache.items():
            file_path = os.path.join(session_path, f"{full_name.replace('.', '_')}_all.pt")
            with open(file_path, 'ab') as f:
                tensor_bytes = tensor.to(config.save_fp).cpu().numpy().tobytes()
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
        hook_fn_all.len_list = len_list
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



    ##############################################################################################################################################################################################
    ################################################################################## Running experiment ################################################################################################
    ##############################################################################################################################################################################################


    # Main function to run the experiment
    os.makedirs(session_path, exist_ok=True)
    save_config(config, session_path)

    # Register hooks
    hooks = register_hooks(model, module_list)

    # Iterate over the dataset with batch size
    for i in range(0, len(processed_dataset), batch_size):
        batch = processed_dataset[i:i + batch_size]
        df_metadata, loss, entropy = run_batch_with_cache(batch)

        save_cache_to_files(cache, session_path)
        save_metadata_to_files(df_metadata, session_path)
        save_loss_to_files(loss, session_path)  # Save the entire loss tensor
        save_entropy_to_files(entropy, session_path)
        
        torch.cuda.empty_cache()
        clear_gpu_memory(cache)
        print(f"{i} to {i+batch_size} done / vectors, loss and entropy are calculated and saved.")
        

    # Remove hooks after processing
    remove_hooks(hooks)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Load a configuration file.')

    # Add the arguments
    parser.add_argument('--config_path', type=str, help='The path to the configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Default configuration
    config = {
        "session_name": "test",
        "session_path": "/Data/test3",
        "model_config": {
            "model_name" :'llama-7b',
            "model_checkpoint": "main",
            "use_accelerator" : False,
            "module_name_mapping":{
                "mlp":"gpt_neox.layers.{layer}.mlp",
                "attn":"gpt_neox.layers.{layer}.self_attn",
                "block":"model.model.layers.{layer}",
                "emb":"model.model.embed_tokens",
                "unemb":"model.model.norm",
            }
        },
        "dataset_config": {
            "dataset_name" : "pile_uncopyrighted_parquet_test",
            "dataset_subset" : [0],
            "max_dataset_size" : 1000,
            "apply_filtering" : True,
            "filtering_config" : {
                "min_chunks_from_a_document" : 5,
                "max_chunks_from_a_document" : 5
            }

        },
        "ctx_len": 1024,
        "batch_size": 4,
        "pos_list": None,  # Convert numpy array to list for better readability in logs
        "layer_idx_list": [0,1,2,5],  # Convert numpy array to list
        "module_inblock_keys": ['mlp', 'attn', 'block'],
        "module_outblock_keys": ['emb', 'unemb'],
        "Analysis_config": {
            "save_cache" : True,
            "save_fp": torch.float16,
        }
    }

    # If a configuration file was provided, load it and overwrite the default configuration
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config_from_file = json.load(f)
        config.update(config_from_file)
    
    from bunch import Bunch

    def recursive_bunchify(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = recursive_bunchify(value)
        return Bunch(dictionary)

    config = recursive_bunchify(config)
    main(config)


    '''    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('--session_name', type=str, required=True, help='Name of the session')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Checkpoint of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--dataset_subset', type=int, nargs='*', required=True, help='Subset of the dataset')
    parser.add_argument('--tokens_min_length', type=int, required=True, help='Minimum length of tokens in the dataset')
    parser.add_argument('--use_accelerator', type=str2bool, required=True, default=False, help='Whether to use accelerator')
    '''
   #main("test3", model_name='pythia-70m-deduped', model_checkpoint='main', dataset_name='pile_uncopyrighted_parquet_test', dataset_subset=[0], tokens_min_length=1024*10, use_accelerator=False)