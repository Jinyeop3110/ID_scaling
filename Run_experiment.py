import os
from accelerate import Accelerator
from fancy_einsum import einsum
import time

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM, AutoConfig
from transformer_lens import ActivationCache, HookedTransformer

from datasets import Dataset
import pandas as pd
import transformers
import torch
import einops  # Make sure einops is imported
import numpy as np 

import matplotlib.pyplot as plt
from pathlib import Path

from utils import *
import argparse
import concurrent.futures
from bunch import Bunch
import yaml


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
    model.eval()

    ##############################################################################################################################################################################################
    ################################################################################## DATASET ###################################################################################################
    ##############################################################################################################################################################################################
    
    dataset = load_Dataset(config.dataset_config)
    #dataset=dataset.select(np.arange(1000))
    print(f"Original dataset size: {len(dataset)}")

    ##############################################################################################################################################################################################
    ############################################################################ DATASET Processing ###################################################################################################
    ##############################################################################################################################################################################################

    def process_dataset(dataset, tokenizer, ctx_len, filtering_config, return_text = False, offset=0):

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
        
        tokens_min_length=ctx_len*(filtering_config.min_chunks_from_a_document+1)
        valid_indices = filter_dataset_indices(dataset, tokenizer, tokens_min_length=tokens_min_length, max_workers=8)
        filtered_dataset = dataset.select(valid_indices)
        print(f"Filtered dataset size: {len(dataset)}")

        def chunk_dataset_with_tokenizer(dataset, tokenizer, chunk_size, max_chunk_idx, offset):
            def chunk_text_with_meta(data, tokenizer, chunk_size, max_chunk_idx, offset):
                full_text = data['text']
                chunks = []
                tokens = tokenizer.encode(full_text, truncation=False)
                start = 1 # Remove BOS
                while start < len(tokens):
                    end = start + chunk_size-1
                    chunk_tokens = tokens[start:end]
                    chunks.append(chunk_tokens)
                    if len(chunks) >= max_chunk_idx:
                        break
                    start += chunk_size -1 + offset

                new_data = []
                for j, chunk in enumerate(chunks):
                    new_entry = {'id': f"{data['id']}-{j}", 'tokens':[tokenizer.bos_token_id]+chunk}
                    for key, value in data.items():
                        if key not in ['id', 'tokens', 'text']:
                            new_entry[key] = value
                        if return_text:
                            new_entry['text']= tokenizer.decode(chunk, skip_special_tokens=False)
                    new_data.append(new_entry)
                return new_data

            new_data = []
            for i, data in enumerate(dataset):
                new_data.extend(chunk_text_with_meta(data, tokenizer, chunk_size, max_chunk_idx, offset))

            # Converting list of dictionaries to HuggingFace Dataset
            chunked_dataset = Dataset.from_pandas(pd.DataFrame(new_data))
            return chunked_dataset
        
        chunked_tokenized_dataset=chunk_dataset_with_tokenizer(filtered_dataset, tokenizer, chunk_size=ctx_len, max_chunk_idx=filtering_config.max_chunks_from_a_document, offset=50)
        return chunked_tokenized_dataset
    
    processed_dataset = process_dataset(dataset, tokenizer, config.ctx_len, config.dataset_config.filter_and_chunk_config, return_text=True, offset=50)
    
    #chunked_tokenized_datsaet=some function


    
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
    
    module_name_mapping, module_name_keys = create_module_names(config.model_config.module_name_mapping, config.cacheing_config.layer_idx_list, config.cacheing_config.module_inblock_keys, config.cacheing_config.module_outblock_keys)

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

    # Create the directory if it doesn't exist



    # Get the length of each input sequence
    def get_len_list(input_ids, pad_token_id):
        len_list = (input_ids == pad_token_id).int().argmax(dim=1)
        len_list[len_list == 0] = config.ctx_len
        return len_list.numpy()

    # Save_tensors with H5py 

    cache_manager = CacheManager(session_path, num_samples=len(processed_dataset), ctx_len=config.ctx_len, embedding_dims=model.config.hidden_size, multiprocessing=config.multiprocessing, num_cpus=config.multiprocessing_num_cpus, verbose=config.verbose)

    # Function to process each batch, run the model, and calculate entropy
    def run_batch_with_cache(batch):
        input_token_ids = torch.tensor(batch['tokens'])

        len_list = get_len_list(input_token_ids, -1)
        df_metadata = pd.DataFrame({key: value for key, value in batch.items() if key != 'text'})
        df_metadata['len'] = len_list
        
        if torch.cuda.is_available():
            input_token_ids = input_token_ids.cuda()

        with torch.no_grad():
            outputs = model(input_token_ids, use_cache=False)
            logits = outputs.logits
            del outputs

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_token_ids[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), shift_logits.size(1))

            # Calculate entropy from logits
            probs = torch.softmax(shift_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs+1e-9), dim=-1)
            # entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            del probs
            torch.cuda.empty_cache()

        return df_metadata, loss.detach(), entropy.detach()



    ##############################################################################################################################################################################################
    ################################################################################## Running experiment ################################################################################################
    ##############################################################################################################################################################################################


    # Main function to run the experiment
    os.makedirs(session_path, exist_ok=True)
    save_config(config, session_path)

    # Register hooks
    hooks = register_hooks(model, module_name_mapping)

    # Iterate over the dataset with batch size
    for i in range(0, len(processed_dataset), batch_size):

        start_time = time.time()

        batch = processed_dataset[i:i + batch_size]
        df_metadata, loss, entropy = run_batch_with_cache(batch)

        forward_pass_time = time.time()
        print(f"Forward pass time consumed: {forward_pass_time - start_time:.4f} seconds")

        if config.cacheing_config.save_cache_tensors:
            cache_tensors_start = time.time()
            cache_manager.save_cache_tensors_to_hdf5(cache)
            cache_tensors_end = time.time()
            print(f"Save cache tensors time consumed: {cache_tensors_end - cache_tensors_start:.4f} seconds")

        if config.cacheing_config.save_mean_tensors:
            mean_tensors_start = time.time()
            cache_manager.save_mean_tensors_to_hdf5(cache)
            mean_tensors_end = time.time()
            print(f"Save mean tensors time consumed: {mean_tensors_end - mean_tensors_start:.4f} seconds")

        if config.cacheing_config.save_IDs:
            ids_start = time.time()
            cache_manager.save_IDs_to_hdf5(cache, config.cacheing_config.save_IDs_list)
            ids_end = time.time()
            print(f"Save IDs time consumed: {ids_end - ids_start:.4f} seconds")

        # Save loss, entropy, and metadata without individual time checks
        cache_manager.save_loss_to_hdf5(loss)
        cache_manager.save_entropy_to_hdf5(entropy)
        cache_manager.save_metadata_to_files(df_metadata)

        total_end_time = time.time()
        print(f"Total cache save time consumed: {total_end_time - forward_pass_time:.4f} seconds")
        print(f"Total process time consumed: {total_end_time - start_time:.4f} seconds")

        # Increment the cache_manager and sanity check
        cache_manager.check_and_increment_index()
        clear_gpu_memory(cache)
        print(f"{i} to {i+batch_size} done / vectors, loss and entropy are calculated and saved.")
        


    # Remove hooks after processing
    remove_hooks(hooks)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Load a configuration file.')

    # Add the arguments
    parser.add_argument('--config_path', type=str, help='The path to the configuration file')
    # Yaml <- json

    # Parse the arguments
    args = parser.parse_args()

    # Default configuration
    default_config = {
        "session_name": "test",
        "session_path": "/home/gridsan/jsong/physics_dl_shared/ML_JY/ID_scaling/Data/test", #NFS remote folder?
        "model_config": {
            "model_name" :'llama-7b',
            "model_checkpoint": "main",
            "use_accelerator" : False,
            "module_name_mapping":{
                "mlp":"model.layers.{layer}.mlp",
                "attn":"model.layers.{layer}.self_attn",
                "block":"model.layers.{layer}",
                "emb":"model.embed_tokens",
                "unemb":"model.norm",
            }
        },
        "dataset_config": {
            "dataset_name" : "pile_uncopyrighted_parquet_test",
            "dataset_subset" : [0],
            "max_dataset_size" : 1000,
            "filter_and_chunk_config" : {
                "min_chunks_from_a_document" : 5,
                "max_chunks_from_a_document" : 5
            }
        },
        "ctx_len": 1024,
        "batch_size": 2,
        "cacheing_config" : {
            "layer_idx_list": [0,1,2,5],  # Convert numpy array to list
            "module_inblock_keys": ['mlp', 'attn', 'block'],
            "module_outblock_keys": ['unemb'],
            "save_fp": "torch.float16",
            "save_cache_tensors":True, # -> 
            "save_mean_tensors":True,
            "save_IDs":True,
            "save_IDs_list": ['mle','mind_ml', 'twoNN_f10'],
        },
        'multiprocessing' : True,
        'multiprocessing_num_cpus' : 20,
        'verbose' : True
    }

    # If a configuration file was provided, load it and overwrite the default configuration
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config_from_file = yaml.safe_load(f)
        config.update(config_from_file)

    config = recursive_bunchify(default_config)
    main(config)
