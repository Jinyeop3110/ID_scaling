import torch
import numpy as np


# Global cache to store outputs
CACHE = {}


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

def hook_fn_all(module, input, output):
    # Retrieve the full name from the global MODULE_NAME_KEYS
    with torch.no_grad():
        full_name = MODULE_NAME_KEYS.get(module, "Unknown")
        if isinstance(output, tuple):
            output = output[0]
        CACHE[full_name] = output.detach().cpu().numpy().astype(np.float16)

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