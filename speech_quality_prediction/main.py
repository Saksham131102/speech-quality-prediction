import os
import sys
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config, save_config, get_absolute_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run speech quality prediction pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument("--generate_data", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--prepare_metadata", action="store_true", help="Prepare metadata")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--model_type", type=str, default=None, help="Model type (cnn_lstm, transformer)")
    parser.add_argument("--feature_type", type=str, default=None, help="Feature type (mfcc, log_mel_spectrogram)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for synthetic dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    
    # Parse args with subcommands
    return parser.parse_args()


def setup_directories(args):
    """Set up directory structure for experiment"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(get_absolute_path("experiments"), f"experiment_{timestamp}")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    return output_dir


def generate_data(config, output_dir, num_samples):
    """Generate synthetic dataset"""
    from data.generate_test_dataset import generate_dataset
    
    # Create args object for generate_dataset
    class Args:
        pass
    
    args = Args()
    args.output_dir = os.path.join(output_dir, "data")
    args.num_samples = num_samples
    args.duration = 2.0
    args.noise_levels = "0.01,0.05,0.1,0.2,0.5"
    
    # Generate dataset
    print("\n=== Generating Synthetic Dataset ===")
    clean_dir, noisy_dir = generate_dataset(config, args)
    
    return clean_dir, noisy_dir


def prepare_metadata(config, clean_dir, noisy_dir, output_dir):
    """Prepare metadata for dataset"""
    from data.prepare_metadata import prepare_metadata
    
    # Create args object for prepare_metadata
    class Args:
        pass
    
    args = Args()
    args.clean_dir = clean_dir
    args.noisy_dir = noisy_dir
    args.output_dir = os.path.join(output_dir, "data")
    args.pair_pattern = None
    args.calculate_metrics = True
    args.artificial_mos = True
    
    # Prepare metadata
    print("\n=== Preparing Metadata ===")
    metadata_path = prepare_metadata(config, args)
    
    return metadata_path


def train_model(config, metadata_path, output_dir, model_type=None, feature_type=None,
                epochs=None, batch_size=None):
    """Train speech quality prediction model"""
    from train import train
    
    # Update config for training
    if model_type:
        config['model']['type'] = model_type
    
    if epochs:
        config['training']['num_epochs'] = epochs
    
    if batch_size:
        config['training']['batch_size'] = batch_size
    
    # Set output directory
    config['training']['checkpoint_dir'] = os.path.join(output_dir, "models")
    
    # Save updated config
    config_path = os.path.join(output_dir, "config.yaml")
    save_config(config, config_path)
    
    # Train model
    print("\n=== Training Model ===")
    print(f"Model type: {config['model']['type']}")
    print(f"Feature type: {feature_type if feature_type else 'mfcc'}")
    
    model, results = train(config, metadata_path, feature_type if feature_type else 'mfcc')
    
    return model, config_path


def evaluate_model(config, metadata_path, output_dir, feature_type=None):
    """Evaluate trained model"""
    from predict import predict_from_metadata
    import torch
    
    # Find best model
    model_dir = os.path.join(output_dir, "models")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    
    if not os.path.exists(best_model_path):
        print(f"Model not found at {best_model_path}")
        return None
    
    # Ensure config contains feature dimensions based on feature_type
    if feature_type == 'log_mel_spectrogram' and 'features' in config:
        # Explicitly tell the model to expect n_mels features (typically 40)
        print(f"Setting model to expect {config['features']['n_mels']} features for log_mel_spectrogram")
    
    # Load model
    print("\n=== Evaluating Model ===")
    from models.model_factory import get_model
    
    # Update config with feature_type information for model initialization
    config['eval_feature_type'] = feature_type
    
    model = get_model(config)
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Predict
    ft = feature_type if feature_type else 'mfcc'
    df, metrics = predict_from_metadata(model, metadata_path, config, ft, device)
    
    # Save results
    results_path = os.path.join(output_dir, "results", "evaluation_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "results", "metrics.txt")
    with open(metrics_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Evaluation results saved to {results_path}")
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def run_pipeline(args):
    """Run the complete pipeline based on arguments"""
    # Load configuration
    config = load_config(args.config)
    
    # Update config from arguments
    if args.model_type:
        config['model']['type'] = args.model_type
    
    if args.feature_type:
        feature_type = args.feature_type
    else:
        feature_type = 'mfcc'
    
    # Setup directories
    output_dir = setup_directories(args)
    print(f"Experiment output directory: {output_dir}")
    
    # Save initial config
    config_path = os.path.join(output_dir, "config.yaml")
    save_config(config, config_path)
    
    # Generate synthetic data if requested
    if args.generate_data:
        clean_dir, noisy_dir = generate_data(config, output_dir, args.num_samples)
    else:
        clean_dir = os.path.join(output_dir, "data", "clean")
        noisy_dir = os.path.join(output_dir, "data", "noisy")
    
    # Prepare metadata if requested
    if args.prepare_metadata:
        metadata_path = prepare_metadata(config, clean_dir, noisy_dir, output_dir)
    else:
        metadata_path = os.path.join(output_dir, "data", "metadata.csv")
    
    # Train model if requested
    if args.train:
        model, config_path = train_model(
            config, metadata_path, output_dir, args.model_type, 
            feature_type, args.epochs, args.batch_size
        )
    
    # Evaluate model if requested
    if args.evaluate:
        metrics = evaluate_model(config, metadata_path, output_dir, feature_type)
    
    print("\n=== Pipeline Complete ===")
    print(f"Results saved to {output_dir}")


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main() 