from id_scaling.configs.default_config import DefaultConfig

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
