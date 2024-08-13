from typing import Any, Dict

import yaml

from id_scaling.configs import DefaultConfig

def update_dataclass_from_dict(instance, updates: Dict[str, Any]):
    """
    Recursively update a dataclass instance with values from a dictionary.
    """
    for key, value in updates.items():
        if hasattr(instance, key):
            attr = getattr(instance, key)
            if isinstance(attr, (list, dict)):
                setattr(instance, key, value)
            elif hasattr(attr, '__dataclass_fields__'):  # If the attribute is a dataclass itself
                update_dataclass_from_dict(attr, value)
            else:
                setattr(instance, key, value)
    return instance

def update_config_from_yaml(config: DefaultConfig, yaml_path: str) -> DefaultConfig:
    """
    Load a YAML file and update the dataclass config.
    """
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    return update_dataclass_from_dict(config, yaml_data)