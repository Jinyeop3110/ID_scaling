import gc
import os
import torch
import json
import os
from bunch import Bunch

def clear_gpu_memory(*args):
    """
    Clears specified PyTorch tensors or models from GPU memory, then clears the GPU memory cache.
    
    Args:
    *args: Variable length argument list. Expected to be PyTorch tensors or models.
    """
    # Attempt to delete each passed argument
    for arg in args:
        # Check if the argument is a tensor and is on GPU
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            # Delete the argument to release its GPU memory
            del arg
        # If it's a model or other structure with parameters
        elif hasattr(arg, 'parameters'):
            # Delete each parameter to release its GPU memory
            for param in arg.parameters():
                if param.is_cuda:
                    del param
    
    # Explicitly collect garbage
    gc.collect()
    
    # Empty the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache has been emptied.")


def clear_gpu_memory(cache):
    for key in list(cache.keys()):
        del cache[key]
    torch.cuda.empty_cache()

def recursive_bunchify(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = recursive_bunchify(value)
    return Bunch(dictionary)

def recursive_unbunchify(obj):
    """
    Recursively convert Bunch objects to dictionaries.
    """
    if isinstance(obj, Bunch):
        return {key: recursive_unbunchify(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: recursive_unbunchify(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [recursive_unbunchify(element) for element in obj]
    else:
        return obj