import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CNNLSTM(nn.Module):
    """CNN-LSTM model for speech quality prediction"""
    
    def __init__(self, config):
        """
        Initialize model.
        
        Args:
            config (dict): Model configuration
        """
        super(CNNLSTM, self).__init__()
        
        # Extract parameters from config
        self.input_channels = 1  # Default for MFCC/Mel spectrogram
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout']
        self.bidirectional = config['bidirectional']
        
        # CNN layers
        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate CNN output dimensions
        # Assuming input is (batch_size, 1, n_features, seq_len)
        self.cnn_output_size = 64  # Number of channels after last conv layer
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
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
        # Reshape input for CNN (batch_size, 1, n_features, seq_len)
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch_size, seq_len, n_features)
        # After CNN and pooling, we have (batch_size, channels, height, width)
        # We treat width as sequence length and collapse channels*height as features
        x = x.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        x_shape = x.size()
        x = x.reshape(batch_size, x_shape[1], -1)  # (batch_size, seq_len, features)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Use the last time step output for prediction
        x = x[:, -1, :]
        
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
    Create a model based on configuration.
    
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
    
    return CNNLSTM(merged_config) 