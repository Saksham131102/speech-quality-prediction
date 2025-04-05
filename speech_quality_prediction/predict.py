import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_factory import get_model
from data.dataset import SpeechQualityDataset, FeatureTransform
from utils.audio_utils import load_audio, extract_features, calculate_pesq, calculate_stoi
from utils.config_utils import load_config, get_absolute_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Predict speech quality using trained model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_audio", type=str, help="Path to audio file or directory")
    parser.add_argument("--reference_audio", type=str, help="Path to reference audio file")
    parser.add_argument("--metadata_file", type=str, help="Path to metadata file with file paths")
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    parser.add_argument("--feature_type", type=str, default='mfcc', 
                        help="Feature type (mfcc, log_mel_spectrogram)")
    return parser.parse_args()


def load_model(model_path, config=None):
    """Load model from checkpoint"""
    if config is None:
        config = load_config()
    
    # Create model
    model = get_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def predict_single_audio(model, audio_path, config, feature_type='mfcc', device='cpu'):
    """Predict quality score for a single audio file"""
    # Load audio
    audio_data, sr = load_audio(audio_path, sample_rate=config['data']['sample_rate'])
    
    # Extract features
    features = extract_features(audio_data, sr, config['features'])
    
    # Get the requested feature
    feature = features.get(feature_type)
    
    if feature is None:
        raise ValueError(f"Feature type '{feature_type}' not available")
    
    # Normalize feature
    feature_mean = np.mean(feature, axis=1, keepdims=True)
    feature_std = np.std(feature, axis=1, keepdims=True) + 1e-8
    feature = (feature - feature_mean) / feature_std
    
    # Convert to tensor
    feature_tensor = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)  # Add batch dimension
    feature_tensor = feature_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(feature_tensor).item()
    
    return prediction


def predict_from_metadata(model, metadata_file, config, feature_type='mfcc', device='cpu'):
    """Predict quality scores for audio files in metadata"""
    # Create dataset
    transform = FeatureTransform(feature_type=feature_type, normalize=True)
    dataset = SpeechQualityDataset(metadata_file, config, transform, mode='test')
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    # Predict
    all_predictions = []
    all_targets = []
    all_file_paths = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Predicting"):
            # Extract inputs and targets
            inputs = sample['input'].to(device)
            targets = sample['target'].cpu().numpy()
            file_paths = sample['file_path']
            
            # Forward pass
            outputs = model(inputs).cpu().numpy()
            
            # Append to lists
            all_predictions.extend(outputs)
            all_targets.extend(targets)
            all_file_paths.extend(file_paths)
    
    # Create dataframe
    df = pd.DataFrame({
        'file_path': all_file_paths,
        'true_quality': all_targets,
        'predicted_quality': all_predictions
    })
    
    # Calculate metrics
    metrics = {}
    metrics['mse'] = mean_squared_error(all_targets, all_predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(all_targets, all_predictions)
    metrics['pearson'], _ = pearsonr(all_targets, all_predictions)
    metrics['spearman'], _ = spearmanr(all_targets, all_predictions)
    
    return df, metrics


def evaluate_objective_metrics(reference_path, degraded_path, sample_rate=16000):
    """Calculate objective speech quality metrics"""
    # Load audio files
    reference, sr_ref = load_audio(reference_path, sample_rate=sample_rate)
    degraded, sr_deg = load_audio(degraded_path, sample_rate=sample_rate)
    
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Calculate PESQ
    pesq_score = calculate_pesq(reference, degraded, sample_rate)
    
    # Calculate STOI
    stoi_score = calculate_stoi(reference, degraded, sample_rate)
    
    return {
        'pesq': pesq_score,
        'stoi': stoi_score
    }


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, config)
    model.to(device)
    print(f"Loaded model from {args.model_path}")
    
    # Determine feature type
    feature_type = args.feature_type
    
    # Predict based on input type
    if args.metadata_file:
        # Predict from metadata
        print(f"Predicting from metadata file: {args.metadata_file}")
        df, metrics = predict_from_metadata(
            model, args.metadata_file, config, feature_type, device
        )
        
        # Print metrics
        print("\nPrediction Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save predictions
        if args.output_file:
            df.to_csv(args.output_file, index=False)
            print(f"Predictions saved to {args.output_file}")
    
    elif args.input_audio:
        # Determine if input is file or directory
        if os.path.isfile(args.input_audio):
            # Predict for single file
            prediction = predict_single_audio(
                model, args.input_audio, config, feature_type, device
            )
            print(f"Predicted quality score: {prediction:.4f}")
            
            # Calculate objective metrics if reference is provided
            if args.reference_audio:
                metrics = evaluate_objective_metrics(
                    args.reference_audio, args.input_audio, config['data']['sample_rate']
                )
                print("\nObjective Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
            
        elif os.path.isdir(args.input_audio):
            # Predict for all audio files in directory
            audio_files = []
            for root, _, files in os.walk(args.input_audio):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(root, file))
            
            # Create dataframe to store predictions
            results = []
            
            # Predict for each file
            for audio_file in tqdm(audio_files, desc="Predicting"):
                prediction = predict_single_audio(
                    model, audio_file, config, feature_type, device
                )
                results.append({
                    'file_path': audio_file,
                    'predicted_quality': prediction
                })
            
            # Create dataframe
            df = pd.DataFrame(results)
            
            # Save predictions
            if args.output_file:
                df.to_csv(args.output_file, index=False)
                print(f"Predictions saved to {args.output_file}")
            else:
                print("\nPredictions:")
                print(df)
        
        else:
            print(f"Input audio path not found: {args.input_audio}")
    
    else:
        print("Please provide either --metadata_file or --input_audio")


if __name__ == "__main__":
    main() 