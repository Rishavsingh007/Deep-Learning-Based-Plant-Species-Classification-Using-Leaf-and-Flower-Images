"""
CT7160NI Computer Vision Coursework
Configuration Management Utilities
"""

import yaml
from pathlib import Path


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, config_path='config.yaml'):
    """
    Save configuration to YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class Config:
    """
    Configuration class for easy attribute access.
    
    Example:
    --------
    >>> config = Config.from_yaml('config.yaml')
    >>> print(config.data.batch_size)
    32
    """
    
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, config_path):
        config_dict = load_config(config_path)
        return cls(config_dict)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# Example usage
if __name__ == "__main__":
    # Test config loading
    try:
        config = Config.from_yaml('config.yaml')
        print("Configuration loaded successfully!")
        print(f"Project name: {config.project.name}")
        print(f"Batch size: {config.data.batch_size}")
    except FileNotFoundError:
        print("Config file not found. Creating example config...")

