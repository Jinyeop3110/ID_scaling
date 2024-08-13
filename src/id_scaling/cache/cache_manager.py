import time
import os
import multiprocessing as mp
from typing import Union
from pathlib import Path
import logging

import numpy as np
import skdim.id as id
from sklearn.impute import SimpleImputer
import h5py
from rich.logging import RichHandler

from id_scaling.utils import *

# set up logging
######### LOGGING #########
logger = logging.getLogger("rich")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

mp.set_start_method('spawn', force=True)


def subprocess_save_IDs_to_hdf5_worker(args):
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
                estimator = id.MiND_ML(D=20)
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

def subprocess_save_cache_tensor_to_hdf5_worker(args):
    session_path, module_key, activation, num_samples, next_index, ctx_len, embedding_dims, verbose = args
    start_time = time.time()
    
    file_path = os.path.join(session_path, f"cache_{module_key}.h5")
    
    if not isinstance(activation, np.ndarray):
        activation = activation.cpu().numpy().astype(np.float16)
    
    current_batch_size = activation.shape[0]
    
    if activation.shape != (current_batch_size, ctx_len, embedding_dims):
        raise ValueError(f"Activation shape {activation.shape} does not match expected shape ({current_batch_size}, {ctx_len}, {embedding_dims})")
    
    with h5py.File(file_path, 'a') as hdf5_file:
        if 'activations' not in hdf5_file:
            hdf5_file.create_dataset('activations',
                                     shape=(num_samples, ctx_len, embedding_dims),
                                     dtype='float16',
                                     chunks=(1, ctx_len, embedding_dims))
        
        dataset = hdf5_file['activations']
        end_index = min(next_index + current_batch_size, num_samples)
        actual_batch_size = end_index - next_index
        dataset[next_index:end_index] = activation[:actual_batch_size]
    
    processing_time = time.time() - start_time
    return module_key, processing_time 
# cache_manager.py

def subprocess_save_mean_tensor_to_hdf5_worker(args):
    session_path, module_key, activation, num_samples, next_index, ctx_len, embedding_dims, verbose = args
    start_time = time.time()
    
    file_path = os.path.join(session_path, f"cache_{module_key}.h5")
    
    if not isinstance(activation, np.ndarray):
        activation = activation.cpu().numpy()
    
    current_batch_size = activation.shape[0]
    
    if activation.shape != (current_batch_size, ctx_len, embedding_dims):
        raise ValueError(f"Activation shape {activation.shape} does not match expected shape ({current_batch_size}, {ctx_len}, {embedding_dims})")
    
    # Calculate mean across ctx_len dimension (dim=1)
    mean_activation = np.mean(activation, axis=1).astype(np.float16)
    
    with h5py.File(file_path, 'a') as hdf5_file:
        if 'mean_activations' not in hdf5_file:
            hdf5_file.create_dataset('mean_activations',
                                     shape=(num_samples, embedding_dims),
                                     dtype='float16',
                                     chunks=(1, embedding_dims))
        
        dataset = hdf5_file['mean_activations']
        end_index = min(next_index + current_batch_size, num_samples)
        actual_batch_size = end_index - next_index
        dataset[next_index:end_index] = mean_activation[:actual_batch_size]
    
    processing_time = time.time() - start_time
    return module_key, processing_time


class CacheManager:
    def __init__(self, session_path: Union[Path, str], save_cache_tensors: bool, save_mean_tensors: bool, save_ids: bool,
                 num_samples: int, ctx_len: int, embedding_dims: int, multiprocessing: bool = True, num_cpus: int = None, 
                 verbose: bool = False):
        self.session_path = session_path
        self.to_save_cache_tensors = save_cache_tensors
        self.to_save_mean_tensors = save_mean_tensors
        self.to_save_ids = save_ids
        self.num_samples = num_samples
        self.ctx_len = ctx_len
        self.embedding_dims = embedding_dims
        self.next_index = 0
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
                

    @staticmethod
    def save_cache_tensors(self, cache):
        if not self.to_save_cache_tensors:
            logging.info('Skipping saving cache tensors')
            return
        
        start = time.time()
        if self.multiprocessing:
            self._create_pool()  # Ensure pool exists
            args = [(self.session_path, module_key, activation, self.num_samples, 
                    self.next_index, self.ctx_len, self.embedding_dims, self.verbose)
                    for module_key, activation in cache.items()]
            results = self.pool.map(subprocess_save_cache_tensor_to_hdf5_worker, args)
            if self.verbose:
                for module_key, processing_time in results:
                    print(f"Cache tensors for module {module_key} saved in {processing_time:.2f}s")
        else:
            for module_key, activation in cache.items():
                result = subprocess_save_cache_tensor_to_hdf5_worker((self.session_path, module_key, activation, 
                                                        self.num_samples, self.next_index, 
                                                        self.ctx_len, self.embedding_dims, self.verbose))
                if self.verbose:
                    print(f"Cache tensors for module {result[0]} saved in {result[1]:.2f}s")

        end = time.time()
        logging.info('Cache tensors saved in {:.2f}s'.format(end - start))

    @staticmethod
    def save_mean_tensors(self, cache):
        if self.to_save_mean_tensors:
            logging.info('Skipping saving mean tensors')
            return
        
        start = time.time()
        if self.multiprocessing:
            self._create_pool()  # Ensure pool exists
            args = [(self.session_path, module_key, activation, self.num_samples, 
                     self.next_index, self.ctx_len, self.embedding_dims, self.verbose)
                    for module_key, activation in cache.items()]
            results = self.pool.map(subprocess_save_mean_tensor_to_hdf5_worker, args)
            if self.verbose:
                for module_key, processing_time in results:
                    print(f"Mean tensors for module {module_key} saved in {processing_time:.2f}s")
        else:
            for module_key, activation in cache.items():
                result = subprocess_save_mean_tensor_to_hdf5_worker((self.session_path, module_key, activation, 
                                                          self.num_samples, self.next_index, 
                                                          self.ctx_len, self.embedding_dims, self.verbose))
                if self.verbose:
                    print(f"Mean tensors for module {result[0]} saved in {result[1]:.2f}s")
        
        end = time.time()
        logging.info('Mean tensors saved in {:.2f}s'.format(end - start))


    @staticmethod
    def save_IDs(self, cache, IDs_methods_list):
        if not self.to_save_ids:
            logging.info('Skipping saving IDs')
            return
        
        start = time.time()

        if self.multiprocessing:
            self._create_pool()  # Ensure pool exists
            args = [(self.session_path, module_key, activation, IDs_methods_list, self.num_samples, self.next_index) 
                    for module_key, activation in cache.items()]
            
            results = self.pool.map(subprocess_save_IDs_to_hdf5_worker, args)
            
            if self.verbose:
                for module_key, processing_time in results:
                    print(f"Module {module_key} processed in {processing_time:.2f}s")
        else:
            for module_key, activation in cache.items():
                result = subprocess_save_IDs_to_hdf5_worker((self.session_path, module_key, activation, IDs_methods_list, self.num_samples, self.next_index))
                if self.verbose:
                    print(f"Module {result[0]} processed in {result[1]:.2f}s")
    
        end = time.time()
        logging.info('IDs saved in {:.2f}s'.format(end - start))

    def save_loss(self, loss):
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
            end_index = min(self.next_index + loss.shape[0], self.num_samples)
            actual_batch_size = end_index - self.next_index
            
            dataset[self.next_index:end_index] = loss[:actual_batch_size]

        if self.verbose:
            print(f"Loss saved to {file_path}")


    def save_entropy(self, entropy):
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
            end_index = min(self.next_index + entropy.shape[0], self.num_samples)
            actual_batch_size = end_index - self.next_index
            
            dataset[self.next_index:end_index] = entropy[:actual_batch_size]
        
        if self.verbose:
            print(f"Entropy saved to {file_path}")


    def save_metadata(self, df_metadata):
        file_path = os.path.join(self.session_path, "metadata.csv")
        if not os.path.isfile(file_path):
            df_metadata.to_csv(file_path, index=False)
        else:
            df_metadata.to_csv(file_path, mode='a', header=False, index=False)
        
        if self.verbose:
            print(f"Metadata saved to {file_path}")


    def check_and_increment_index(self, increment):
        
        self.next_index += increment
        if self.next_index >= self.num_samples:
            print("Reached the maximum number of samples. Resetting index to 0.")
            self.next_index = 0
        self.current_batch_size = None


    def reset(self):
        self.next_index = 0
        self.current_batch_size = None



