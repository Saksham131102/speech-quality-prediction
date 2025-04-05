import os
import sys
import argparse
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_utils import load_config, get_absolute_path
from utils.audio_utils import calculate_pesq, calculate_stoi, load_audio


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare metadata for speech quality dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--clean_dir", type=str, required=True, 
                        help="Directory containing clean audio files")
    parser.add_argument("--noisy_dir", type=str, required=True, 
                        help="Directory containing noisy/degraded audio files")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save metadata")
    parser.add_argument("--pair_pattern", type=str, default=None, 
                        help="Pattern to match clean and noisy files (e.g., '%s_noisy' where %s is clean file prefix)")
    parser.add_argument("--calculate_metrics", action="store_true", 
                        help="Calculate objective metrics (PESQ, STOI)")
    parser.add_argument("--artificial_mos", action="store_true", 
                        help="Generate artificial MOS scores based on objective metrics")
    return parser.parse_args()


def find_file_pairs(clean_dir, noisy_dir, pair_pattern=None):
    """
    Find pairs of clean and noisy files.
    
    Args:
        clean_dir (str): Directory containing clean audio files
        noisy_dir (str): Directory containing noisy audio files
        pair_pattern (str): Pattern to match clean and noisy files
        
    Returns:
        list: List of (clean_file, noisy_file) pairs
    """
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    pairs = []
    
    if pair_pattern:
        # Use the pattern to find pairs
        for clean_file in clean_files:
            clean_prefix = os.path.splitext(clean_file)[0]
            noisy_prefix = pair_pattern.replace('%s', clean_prefix)
            
            # Find matching noisy files
            for noisy_file in noisy_files:
                if noisy_file.startswith(noisy_prefix):
                    pairs.append((
                        os.path.join(clean_dir, clean_file),
                        os.path.join(noisy_dir, noisy_file)
                    ))
    else:
        # Check for synthetic dataset naming pattern (sample_XXXX.wav -> sample_XXXX_noise_YY.wav)
        for clean_file in clean_files:
            clean_prefix = os.path.splitext(clean_file)[0]
            
            # Find all noisy variants that match this clean file
            for noisy_file in noisy_files:
                # Match pattern: sample_XXXX_noise_YY.wav
                if noisy_file.startswith(clean_prefix + "_noise_"):
                    pairs.append((
                        os.path.join(clean_dir, clean_file),
                        os.path.join(noisy_dir, noisy_file)
                    ))
    
    return pairs


def calculate_objective_metrics(file_pairs, sample_rate=16000):
    """
    Calculate objective speech quality metrics for file pairs.
    
    Args:
        file_pairs (list): List of (clean_file, noisy_file) pairs
        sample_rate (int): Sample rate
        
    Returns:
        list: List of dictionaries with file paths and metrics
    """
    results = []
    
    for clean_file, noisy_file in tqdm(file_pairs, desc="Calculating metrics"):
        # Load audio files
        clean_audio, sr = load_audio(clean_file, sample_rate=sample_rate)
        noisy_audio, sr = load_audio(noisy_file, sample_rate=sample_rate)
        
        # Ensure same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # Calculate metrics
        try:
            pesq_score = calculate_pesq(clean_audio, noisy_audio, sample_rate)
        except Exception as e:
            print(f"Error calculating PESQ for {noisy_file}: {e}")
            pesq_score = None
        
        try:
            stoi_score = calculate_stoi(clean_audio, noisy_audio, sample_rate)
        except Exception as e:
            print(f"Error calculating STOI for {noisy_file}: {e}")
            stoi_score = None
        
        results.append({
            'clean_file': clean_file,
            'noisy_file': noisy_file,
            'pesq': pesq_score,
            'stoi': stoi_score
        })
    
    return results


def generate_artificial_mos(metrics_results):
    """
    Generate artificial MOS scores based on objective metrics.
    
    Args:
        metrics_results (list): List of dictionaries with metrics
        
    Returns:
        list: Updated list with artificial MOS scores
    """
    for result in metrics_results:
        # Convert PESQ to MOS-like scale (1-5)
        pesq = result.get('pesq')
        stoi = result.get('stoi')
        
        if pesq is not None and stoi is not None:
            # Simple formula: weighted combination of PESQ and STOI
            # PESQ is already roughly on a 1-4.5 scale
            # STOI is on a 0-1 scale, so we scale it to 1-5
            stoi_scaled = 1 + stoi * 4
            mos = 0.7 * pesq + 0.3 * stoi_scaled
            
            # Add some noise to simulate human variability
            noise = np.random.normal(0, 0.2)
            mos = max(1, min(5, mos + noise))
        elif pesq is not None:
            mos = pesq + np.random.normal(0, 0.2)
            mos = max(1, min(5, mos))
        elif stoi is not None:
            mos = 1 + stoi * 4 + np.random.normal(0, 0.2)
            mos = max(1, min(5, mos))
        else:
            # Random score if no metrics available
            mos = np.random.uniform(1, 5)
        
        result['quality_score'] = mos
    
    return metrics_results


def prepare_metadata(config, args):
    """
    Prepare metadata for speech quality dataset.
    
    Args:
        config (dict): Configuration dictionary
        args (Namespace): Command line arguments
        
    Returns:
        str: Path to metadata file
    """
    # Find file pairs
    print(f"Finding file pairs in {args.clean_dir} and {args.noisy_dir}...")
    file_pairs = find_file_pairs(args.clean_dir, args.noisy_dir, args.pair_pattern)
    print(f"Found {len(file_pairs)} file pairs")
    
    # Calculate metrics if requested
    if args.calculate_metrics:
        print("Calculating objective metrics...")
        results = calculate_objective_metrics(
            file_pairs, 
            sample_rate=config['data']['sample_rate']
        )
    else:
        # Create basic results without metrics
        results = [
            {'clean_file': clean_file, 'noisy_file': noisy_file}
            for clean_file, noisy_file in file_pairs
        ]
    
    # Generate artificial MOS scores if requested
    if args.artificial_mos:
        print("Generating artificial MOS scores...")
        results = generate_artificial_mos(results)
    
    # Create metadata dataframe
    df = pd.DataFrame(results)
    
    # Add file_path column (for the dataset)
    df['file_path'] = df['noisy_file']
    
    # If no quality scores, add random scores
    if 'quality_score' not in df.columns:
        print("No quality scores available, generating random scores...")
        df['quality_score'] = np.random.uniform(1, 5, size=len(df))
    
    # Add split column (train/val/test)
    random.seed(42)
    splits = ['train'] * int(0.7 * len(df)) + \
             ['val'] * int(0.15 * len(df)) + \
             ['test'] * (len(df) - int(0.7 * len(df)) - int(0.15 * len(df)))
    random.shuffle(splits)
    df['split'] = splits
    
    # Save metadata
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = get_absolute_path('data/processed')
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    # Create split files
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_path = os.path.join(output_dir, f'{split}_metadata.csv')
        split_df.to_csv(split_path, index=False)
        print(f"{split.capitalize()} metadata saved to {split_path}")
    
    return metadata_path


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare metadata
    metadata_path = prepare_metadata(config, args)
    
    print("Metadata preparation complete!")


if __name__ == "__main__":
    main() 