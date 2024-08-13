import gc
import os
import torch
import json
import os
from bunch import Bunch

# Get the length of each input sequence
def get_len_list(input_ids, pad_token_id, ctx_len):
    len_list = (input_ids == pad_token_id).int().argmax(dim=1)
    len_list[len_list == 0] = ctx_len
    return len_list.numpy()


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
