import time

import torch
import json
import os
import yaml
from bunch import Bunch
from utils import *
import multiprocessing as mp
from functools import partial
import skdim.id as id
import h5py
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mp.set_start_method('spawn', force=True)

# Global variable to store module names
MODULE_NAME_KEYS = {}

# Global cache to store outputs
cache = {}

def hook_fn_all(module, input, output):
    # Retrieve the full name from the global MODULE_NAME_KEYS
    with torch.no_grad():
        full_name = MODULE_NAME_KEYS.get(module, "Unknown")
        if isinstance(output, tuple):
            output = output[0]
        cache[full_name] = output.detach().cpu().numpy().astype(np.float16)

def register_hooks(model, module_name_mapping):
    hooks = []
    global MODULE_NAME_KEYS
    MODULE_NAME_KEYS.clear()  # Clear previous entries
    reverse_mapping = {v: k for k, v in module_name_mapping.items()}
    
    for name, module in model.named_modules():
        if name in reverse_mapping:
            hooks.append(module.register_forward_hook(hook_fn_all))
            MODULE_NAME_KEYS[module] = reverse_mapping[name]
    
    return hooks

def save_config(config, session_path):
    # Convert config to a dictionary if it's a Bunch object
    if isinstance(config, Bunch):
        config_dict = recursive_unbunchify(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError("Config must be either a dictionary or a Bunch object")

    # Debug: Print the config_dict to check its structure
    print("Config dictionary structure:")
    print(config_dict)

    # Ensure all items in config_dict are proper key-value pairs
    for key, value in config_dict.items():
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value.keys()):
                raise ValueError(f"All keys in nested dictionaries must be strings. Check the value for key '{key}'")

    # Ensure the session path exists
    os.makedirs(session_path, exist_ok=True)

    # Save the config as YAML
    yaml_path = os.path.join(session_path, "config.yaml")
    try:
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False)
    except Exception as e:
        print(f"Error while saving YAML: {e}")
        print(f"Problematic config_dict: {config_dict}")
        raise

    print(f"Config saved successfully to {yaml_path}")


# Save metadata to files


# Remove hooks after processing
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
    return None

def save_IDs_to_hdf5_worker(args):
    session_path, module_key, activation, IDs_methods_list, num_samples, next_index = args
    start_time = time.time()
    file_path = os.path.join(session_path, f"cache_{module_key}.h5")
    
    if not isinstance(activation, np.ndarray):
        activation = activation.cpu().numpy()
    
    with h5py.File(file_path, 'a') as hdf5_file:
        if 'ID' not in hdf5_file:
            id_group = hdf5_file.create_group('ID')
        else:
            id_group = hdf5_file['ID']
        
        for method in IDs_methods_list:
            if method == 'mle':
                estimator = id.MLE()
                dataset_name = 'mle'
            elif method.startswith('twoNN'):
                _, f = method.split('_')
                f = int(f[1:])
                estimator = id.TwoNN(discard_fraction=f/100)
                dataset_name = f'twoNN_f{f}'
            elif method == 'mind_ml':
                estimator = id.MiND_ML()
                dataset_name = 'mind_ml'
            else:
                raise ValueError(f"Unsupported ID method: {method}")
            
            # Process each sample individually
            id_estimates = []
            for sample in activation:
                try:
                    if np.isnan(sample).any():
                        imputer = SimpleImputer(strategy='mean')
                        sample = imputer.fit_transform(sample.reshape(1, -1)).squeeze()
                    
                    id_estimate = estimator.fit_transform(sample)
                    id_estimates.append(id_estimate)
                except Exception:
                    id_estimates.append(np.nan)
            
            id_estimates = np.array(id_estimates)
            
            if dataset_name not in id_group:
                id_group.create_dataset(dataset_name, shape=(num_samples,), dtype=float)
            
            end_index = min(next_index + len(id_estimates), num_samples)
            id_group[dataset_name][next_index:end_index] = id_estimates
    
    end_time = time.time()
    return module_key, end_time - start_time
    
# cache_manager.py

class CacheManager:
    def __init__(self, session_path, num_samples, ctx_len, embedding_dims, multiprocessing=True, num_cpus=None, verbose=False):
        self.session_path = session_path
        self.num_samples = num_samples
        self.ctx_len = ctx_len
        self.embedding_dims = embedding_dims
        self.next_index = 0
        self.current_batch_size = None
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.num_cpus = num_cpus if num_cpus is not None else mp.cpu_count()
        self.pool = None
        
        if self.multiprocessing:
            self._create_pool()

    def _create_pool(self):
        if self.pool is None:
            self.pool = mp.get_context('spawn').Pool(self.num_cpus)
            if self.verbose:
                print(f"Created multiprocessing pool with {self.num_cpus} CPUs")

    def __del__(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def save_cache_tensors_to_hdf5(self, cache):
        for module_key, activation in cache.items():
            file_path = os.path.join(self.session_path, f"cache_{module_key}.h5")
            
            if not isinstance(activation, np.ndarray):
                activation = activation.cpu().numpy().astype(np.float16)
            
            self.current_batch_size = activation.shape[0]
            
            if activation.shape != (self.current_batch_size, self.ctx_len, self.embedding_dims):
                raise ValueError(f"Activation shape {activation.shape} does not match expected shape ({self.current_batch_size}, {self.ctx_len}, {self.embedding_dims})")
            
            with h5py.File(file_path, 'a') as hdf5_file:
                if 'activations' not in hdf5_file:
                    hdf5_file.create_dataset('activations', 
                                             shape=(self.num_samples, self.ctx_len, self.embedding_dims),
                                             dtype='float16',
                                             chunks=(1, self.ctx_len, self.embedding_dims))
                
                dataset = hdf5_file['activations']
                
                end_index = min(self.next_index + self.current_batch_size, self.num_samples)
                actual_batch_size = end_index - self.next_index
                
                dataset[self.next_index:end_index] = activation[:actual_batch_size]

            if self.verbose:
                print(f"Batch of activations for {module_key} saved to {file_path}")

    def save_mean_tensors_to_hdf5(self, cache):
        return 0

    def save_IDs_to_hdf5(self, cache, IDs_methods_list):
        if self.multiprocessing:
            self._create_pool()  # Ensure pool exists
            args = [(self.session_path, module_key, activation, IDs_methods_list, self.num_samples, self.next_index) 
                    for module_key, activation in cache.items()]
            
            results = self.pool.map(save_IDs_to_hdf5_worker, args)
            
            if self.verbose:
                for module_key, processing_time in results:
                    print(f"Module {module_key} processed in {processing_time:.2f}s")
        else:
            for module_key, activation in cache.items():
                result = self._save_IDs_to_hdf5_worker((self.session_path, module_key, activation, IDs_methods_list, self.num_samples, self.next_index))
                if self.verbose:
                    print(f"Module {result[0]} processed in {result[1]:.2f}s")
    
    def save_loss_to_hdf5(self, loss):
        file_path = os.path.join(self.session_path, "creloss.h5")
        
        if not isinstance(loss, np.ndarray):
            loss = loss.cpu().numpy().astype(np.float16)
        
        if loss.shape[1] != self.ctx_len-1:
            raise ValueError(f"Loss shape {loss.shape} does not match expected shape (batch_size, {self.ctx_len})")
        
        with h5py.File(file_path, 'a') as hdf5_file:
            if 'loss' not in hdf5_file:
                hdf5_file.create_dataset('loss', shape=(self.num_samples, self.ctx_len-1),
                                         dtype=np.float16, chunks=True)
            
            dataset = hdf5_file['loss']
            end_index = min(self.next_index + self.current_batch_size, self.num_samples)
            actual_batch_size = end_index - self.next_index
            
            dataset[self.next_index:end_index] = loss[:actual_batch_size]

        if self.verbose:
            print(f"Loss saved to {file_path}")


    def save_entropy_to_hdf5(self, entropy):
        file_path = os.path.join(self.session_path, "entropy.h5")
        
        if not isinstance(entropy, np.ndarray):
            entropy = entropy.cpu().numpy().astype(np.float16)
        
        if entropy.shape[1] != self.ctx_len-1:
            raise ValueError(f"Entropy shape {entropy.shape} does not match expected shape (batch_size, {self.ctx_len})")
        
        with h5py.File(file_path, 'a') as hdf5_file:
            if 'entropy' not in hdf5_file:
                hdf5_file.create_dataset('entropy', shape=(self.num_samples, self.ctx_len-1),
                                         dtype=np.float16, chunks=True)
            
            dataset = hdf5_file['entropy']
            end_index = min(self.next_index + self.current_batch_size, self.num_samples)
            actual_batch_size = end_index - self.next_index
            
            dataset[self.next_index:end_index] = entropy[:actual_batch_size]
        
        if self.verbose:
            print(f"Entropy saved to {file_path}")

    def save_metadata_to_files(self, df_metadata):
        file_path = os.path.join(self.session_path, "metadata.csv")
        if not os.path.isfile(file_path):
            df_metadata.to_csv(file_path, index=False)
        else:
            df_metadata.to_csv(file_path, mode='a', header=False, index=False)
        
        if self.verbose:
            print(f"Metadata saved to {file_path}")

    def check_and_increment_index(self):
        if self.current_batch_size is None:
            raise ValueError("No batch size set. Make sure to call save_cache_to_hdf5 first.")
        
        self.next_index += self.current_batch_size
        if self.next_index >= self.num_samples:
            print("Reached the maximum number of samples. Resetting index to 0.")
            self.next_index = 0
        self.current_batch_size = None

    def reset(self):
        self.next_index = 0
        self.current_batch_size = None



