import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config
from utils.audio_utils import extract_features, calculate_pesq, calculate_stoi
from models.model_factory import get_model


def generate_test_audio():
    """Generate a test audio file with clean and degraded versions"""
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate a sample rate and length
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a clean signal (mixture of sine waves)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    clean_signal += 0.3 * np.sin(2 * np.pi * 880 * t)  # A5 note
    clean_signal += 0.2 * np.sin(2 * np.pi * 1320 * t)  # E5 note
    
    # Normalize
    clean_signal = clean_signal / np.max(np.abs(clean_signal))
    
    # Create degraded versions with different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    degraded_signals = []
    
    for noise_level in noise_levels:
        # Add noise
        noise = np.random.normal(0, noise_level, len(clean_signal))
        degraded = clean_signal + noise
        
        # Normalize
        degraded = degraded / np.max(np.abs(degraded))
        
        degraded_signals.append(degraded)
    
    # Save clean signal
    clean_path = 'data/raw/clean.wav'
    sf.write(clean_path, clean_signal, sample_rate)
    
    # Save degraded signals
    degraded_paths = []
    for i, degraded in enumerate(degraded_signals):
        degraded_path = f'data/raw/degraded_{i+1}.wav'
        sf.write(degraded_path, degraded, sample_rate)
        degraded_paths.append(degraded_path)
    
    return clean_path, degraded_paths, sample_rate


def test_feature_extraction(audio_path, config):
    """Test feature extraction functionality"""
    # Load audio
    audio_data, sr = librosa.load(audio_path, sr=config['data']['sample_rate'])
    
    # Extract features
    features = extract_features(audio_data, sr, config['features'])
    
    # Plot MFCCs
    if 'mfcc' in features:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            features['mfcc'], 
            x_axis='time', 
            sr=sr
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCCs')
        plt.tight_layout()
        plt.savefig('data/mfcc_example.png')
        print(f"MFCC plot saved to data/mfcc_example.png")
    
    # Plot Mel spectrogram
    if 'log_mel_spectrogram' in features:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            features['log_mel_spectrogram'], 
            x_axis='time', 
            y_axis='mel', 
            sr=sr
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig('data/mel_spec_example.png')
        print(f"Mel spectrogram plot saved to data/mel_spec_example.png")
    
    return features


def test_objective_metrics(clean_path, degraded_paths, sample_rate):
    """Test objective speech quality metrics"""
    # Load clean signal
    clean_signal, _ = librosa.load(clean_path, sr=sample_rate)
    
    results = []
    
    # Calculate metrics for each degraded signal
    for i, degraded_path in enumerate(degraded_paths):
        # Load degraded signal
        degraded_signal, _ = librosa.load(degraded_path, sr=sample_rate)
        
        # Calculate PESQ
        try:
            pesq_score = calculate_pesq(clean_signal, degraded_signal, sample_rate)
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            pesq_score = None
        
        # Calculate STOI
        try:
            stoi_score = calculate_stoi(clean_signal, degraded_signal, sample_rate)
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            stoi_score = None
        
        # Store results
        results.append({
            'file': os.path.basename(degraded_path),
            'pesq': pesq_score,
            'stoi': stoi_score
        })
    
    # Print results
    print("\nObjective Metrics:")
    print("-----------------")
    for result in results:
        print(f"File: {result['file']}")
        print(f"  PESQ: {result['pesq']:.4f}" if result['pesq'] is not None else "  PESQ: N/A")
        print(f"  STOI: {result['stoi']:.4f}" if result['stoi'] is not None else "  STOI: N/A")
    
    return results


def test_model(config):
    """Test model initialization and forward pass"""
    # Create a model
    model = get_model(config)
    print(f"\nCreated model: {config['model']['type']}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Create a random input
    batch_size = 2
    n_features = 13  # MFCCs
    seq_len = 100
    dummy_input = torch.randn(batch_size, n_features, seq_len)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.numpy()}")
    
    return model


def main():
    """Main test function"""
    print("Testing Speech Quality Prediction Project")
    print("========================================")
    
    # Load configuration
    config = load_config()
    
    # Generate test audio files
    print("\nGenerating test audio files...")
    clean_path, degraded_paths, sample_rate = generate_test_audio()
    print(f"Created clean audio: {clean_path}")
    print(f"Created degraded audio files: {len(degraded_paths)}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = test_feature_extraction(clean_path, config)
    print(f"Extracted features: {list(features.keys())}")
    
    # Test objective metrics
    print("\nTesting objective metrics...")
    metrics_results = test_objective_metrics(clean_path, degraded_paths, sample_rate)
    
    # Test model
    print("\nTesting model...")
    model = test_model(config)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 