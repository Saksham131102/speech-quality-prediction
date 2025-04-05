import os
import sys
import importlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_utils import load_config


def get_model(config=None, model_type=None):
    """
    Factory function to create a model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        model_type (str): Type of model to create
        
    Returns:
        torch.nn.Module: Model instance
    """
    if config is None:
        config = load_config()
    
    # Determine model type from config if not specified
    if model_type is None:
        model_type = config['model']['type']
    
    # Map model types to their respective modules
    model_modules = {
        'cnn_lstm': 'cnn_lstm_model',
        'transformer': 'transformer_model',
        # Add more model types here as they are implemented
    }
    
    if model_type not in model_modules:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available types: {list(model_modules.keys())}")
    
    # Import the model module
    try:
        model_module = importlib.import_module(
            f"models.{model_modules[model_type]}"
        )
    except ImportError as e:
        raise ImportError(f"Failed to import model module: {e}")
    
    # Create and return the model - pass the FULL config to have access to feature info
    return model_module.create_model(config) 