import gc
import os
import torch

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
