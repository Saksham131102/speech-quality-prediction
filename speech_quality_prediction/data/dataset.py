import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import librosa

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_audio, extract_features
from utils.config_utils import load_config, get_absolute_path


class SpeechQualityDataset(Dataset):
    """Dataset for speech quality prediction"""
    
    def __init__(self, metadata_file, config=None, transform=None, mode='train'):
        """
        Initialize dataset.
        
        Args:
            metadata_file (str): Path to metadata CSV file with file paths and scores
            config (dict): Configuration dictionary
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train', 'val', or 'test'
        """
        self.config = config if config is not None else load_config()
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.mode = mode
        self.features_config = self.config['features']
        self.sample_rate = self.config['data']['sample_rate']
        
        # Ensure the necessary columns exist
        required_cols = ['file_path', 'quality_score']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Metadata file must contain column '{col}'")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing features and target
        """
        # Get file path and quality score
        file_path = self.metadata.iloc[idx]['file_path']
        quality_score = float(self.metadata.iloc[idx]['quality_score'])
        
        # Make path absolute if it's relative
        if not os.path.isabs(file_path):
            file_path = get_absolute_path(file_path)
        
        # Load audio
        audio_data, sr = load_audio(file_path, sample_rate=self.sample_rate)
        
        # Extract features
        features = extract_features(audio_data, sr, self.features_config)
        
        # Create sample dictionary
        sample = {
            'file_path': file_path,
            'audio': audio_data,
            'features': features,
            'quality_score': quality_score
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class FeatureTransform:
    """Transform to prepare features for model input"""
    
    def __init__(self, feature_type='mfcc', max_len=None, normalize=True):
        """
        Initialize transform.
        
        Args:
            feature_type (str): Type of feature to extract ('mfcc', 'log_mel_spectrogram', etc.)
            max_len (int, optional): Maximum sequence length (will pad or truncate)
            normalize (bool): Whether to normalize features
        """
        self.feature_type = feature_type
        self.max_len = max_len
        self.normalize = normalize
    
    def __call__(self, sample):
        """
        Apply the transform to the sample.
        
        Args:
            sample (dict): Sample dictionary
            
        Returns:
            dict: Transformed sample
        """
        # Extract the specified feature
        feature = sample['features'].get(self.feature_type)
        
        if feature is None:
            raise ValueError(f"Feature type '{self.feature_type}' not found in sample")
        
        # Handle sequence length (pad or truncate)
        if self.max_len is not None:
            curr_len = feature.shape[1]
            if curr_len < self.max_len:
                # Pad
                pad_width = ((0, 0), (0, self.max_len - curr_len))
                feature = np.pad(feature, pad_width, mode='constant')
            elif curr_len > self.max_len:
                # Truncate
                feature = feature[:, :self.max_len]
        
        # Normalize if requested
        if self.normalize:
            feature_mean = np.mean(feature, axis=1, keepdims=True)
            feature_std = np.std(feature, axis=1, keepdims=True) + 1e-8
            feature = (feature - feature_mean) / feature_std
        
        # Convert to torch tensor
        feature_tensor = torch.from_numpy(feature.astype(np.float32))
        
        # Update sample with processed feature
        sample['input'] = feature_tensor
        sample['target'] = torch.tensor(sample['quality_score'], dtype=torch.float32)
        
        return sample


def create_train_val_test_split(audio_dir, output_metadata_path, quality_scores=None, 
                                split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Create train/validation/test split and save metadata.
    
    Args:
        audio_dir (str): Directory containing audio files
        output_metadata_path (str): Path to save metadata CSV
        quality_scores (dict, optional): Dictionary mapping file names to quality scores
        split_ratios (tuple): Train/val/test split ratios
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all audio files
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                rel_path = os.path.join(root, file)
                audio_files.append(rel_path)
    
    # Shuffle files
    random.shuffle(audio_files)
    
    # Calculate split indices
    n_files = len(audio_files)
    train_end = int(n_files * split_ratios[0])
    val_end = train_end + int(n_files * split_ratios[1])
    
    # Split files
    train_files = audio_files[:train_end]
    val_files = audio_files[train_end:val_end]
    test_files = audio_files[val_end:]
    
    # Create metadata dataframes
    train_df = pd.DataFrame({'file_path': train_files, 'split': 'train'})
    val_df = pd.DataFrame({'file_path': val_files, 'split': 'val'})
    test_df = pd.DataFrame({'file_path': test_files, 'split': 'test'})
    
    # Combine dataframes
    metadata_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Add quality scores if provided
    if quality_scores is not None:
        metadata_df['quality_score'] = metadata_df['file_path'].apply(
            lambda x: quality_scores.get(os.path.basename(x), np.nan)
        )
    else:
        # Add placeholder scores for demonstration
        metadata_df['quality_score'] = np.random.uniform(1.0, 5.0, size=len(metadata_df))
    
    # Save metadata
    metadata_df.to_csv(output_metadata_path, index=False)
    
    return metadata_df


def get_dataloaders(metadata_file, config=None, feature_type='mfcc', batch_size=32, 
                   num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        metadata_file (str): Path to metadata CSV file
        config (dict, optional): Configuration dictionary
        feature_type (str): Type of feature to use
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    
    # Create datasets for each split
    train_data = metadata[metadata['split'] == 'train']
    val_data = metadata[metadata['split'] == 'val']
    test_data = metadata[metadata['split'] == 'test']
    
    # Save split datasets to separate files
    train_metadata_file = os.path.join(os.path.dirname(metadata_file), 'train_metadata.csv')
    val_metadata_file = os.path.join(os.path.dirname(metadata_file), 'val_metadata.csv')
    test_metadata_file = os.path.join(os.path.dirname(metadata_file), 'test_metadata.csv')
    
    train_data.to_csv(train_metadata_file, index=False)
    val_data.to_csv(val_metadata_file, index=False)
    test_data.to_csv(test_metadata_file, index=False)
    
    # Create transform
    transform = FeatureTransform(feature_type=feature_type, normalize=True)
    
    # Create datasets
    train_dataset = SpeechQualityDataset(train_metadata_file, config, transform, mode='train')
    val_dataset = SpeechQualityDataset(val_metadata_file, config, transform, mode='val')
    test_dataset = SpeechQualityDataset(test_metadata_file, config, transform, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader 