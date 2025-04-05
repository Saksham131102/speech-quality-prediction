import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    """Transformer model for speech quality prediction"""
    
    def __init__(self, config):
        """
        Initialize model.
        
        Args:
            config (dict): Model configuration
        """
        super(TransformerModel, self).__init__()
        
        # Extract parameters from config
        self.hidden_size = config['hidden_size']
        self.dropout_rate = config['dropout']
        
        # Determine input feature dimension based on config and eval_feature_type
        if 'eval_feature_type' in config and config['eval_feature_type'] == 'log_mel_spectrogram':
            # For log_mel_spectrogram, use n_mels as feature dimension
            if 'features' in config and 'n_mels' in config['features']:
                self.input_dim = config['features']['n_mels']
            else:
                self.input_dim = 40  # Default n_mels value if not specified
            print(f"Initialized model for log_mel_spectrogram with {self.input_dim} features")
        elif 'features' in config and 'n_mels' in config['features'] and 'n_mfcc' in config['features']:
            # Check if we should use MFCC or Mel spectrogram features
            if 'eval_feature_type' in config and config['eval_feature_type'] == 'mfcc':
                self.input_dim = config['features']['n_mfcc']
                print(f"Initialized model for MFCC with {self.input_dim} features")
            else:
                # If feature type not explicitly set or is mel spectrogram
                self.input_dim = config['features']['n_mels']
                print(f"Initialized model with {self.input_dim} features (n_mels)")
        else:
            # Default to MFCC dimensions if nothing else is specified
            self.input_dim = 13  # Default for MFCC
            print(f"Initialized model with default {self.input_dim} features (MFCC)")
        
        # Feature dimension reduction
        self.feature_reducer = nn.Linear(self.input_dim, self.hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,  # 8 attention heads
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['num_layers'])
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output a single quality score
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features, seq_len)
            
        Returns:
            torch.Tensor: Predicted quality score
        """
        # Debug print
        print(f"Input shape: {x.shape}")
        print(f"Expected input features: {self.input_dim}")
        
        # Transpose to (batch_size, seq_len, n_features)
        x = x.transpose(1, 2)
        print(f"After transpose shape: {x.shape}")
        
        # Get actual feature dimension from the input
        actual_feature_dim = x.size(-1)
        
        # If there's a mismatch between expected and actual feature dimension,
        # recreate the feature reducer layer with the correct dimensions
        if actual_feature_dim != self.input_dim:
            print(f"Dimension mismatch! Adjusting feature reducer from {self.input_dim} to {actual_feature_dim}")
            # Create a new feature reducer with correct dimensions
            self.input_dim = actual_feature_dim
            self.feature_reducer = nn.Linear(actual_feature_dim, self.hidden_size).to(x.device)
        
        # Apply feature reduction
        x = self.feature_reducer(x)
        print(f"After feature reduction shape: {x.shape}")
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transpose to (seq_len, batch_size, hidden_size) for transformer
        x = x.transpose(0, 1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the average of all time steps for prediction
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Return predicted quality score
        return x.squeeze(-1)


def create_model(config):
    """
    Create a transformer model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        nn.Module: Model instance
    """
    # Extract just the model config part to maintain compatibility with existing code
    model_config = config['model']
    
    # Create a merged config that has both model params and feature info
    merged_config = model_config.copy()
    
    # Add feature info to model config
    if 'features' in config:
        merged_config['features'] = config['features']
    
    # Add evaluation feature type if present
    if 'eval_feature_type' in config:
        merged_config['eval_feature_type'] = config['eval_feature_type']
    
    return TransformerModel(merged_config) 