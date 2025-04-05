import os
import yaml


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the config file. If None, use default path.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, 'config', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    
def get_absolute_path(relative_path):
    """
    Convert a relative path to absolute path based on project root.
    
    Args:
        relative_path (str): Relative path from project root
        
    Returns:
        str: Absolute path
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, relative_path) 