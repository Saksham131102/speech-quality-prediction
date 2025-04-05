import os
import sys
import argparse
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_utils import load_config, get_absolute_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset for speech quality prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each sample in seconds")
    parser.add_argument("--noise_levels", type=str, default="0.01,0.05,0.1,0.2,0.5", 
                        help="Comma-separated list of noise levels")
    return parser.parse_args()


def generate_sine_wave(duration, sample_rate, freq, amplitude=1.0):
    """Generate a sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_speech_like_signal(duration, sample_rate):
    """Generate a speech-like signal (multiple sine waves with formant-like frequencies)"""
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Fundamental frequency (pitch) with slight variation
    f0 = random.uniform(100, 300)  # Hz, typical for human speech
    
    # Create a basic speech-like signal with formants
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)  # Fundamental
    
    # Add formants (simplified)
    formants = [
        (f0 * random.uniform(2.0, 2.5), random.uniform(0.2, 0.4)),  # First formant
        (f0 * random.uniform(5.0, 6.0), random.uniform(0.1, 0.3)),  # Second formant
        (f0 * random.uniform(8.0, 9.0), random.uniform(0.05, 0.2))  # Third formant
    ]
    
    for formant_freq, amplitude in formants:
        signal += amplitude * np.sin(2 * np.pi * formant_freq * t)
    
    # Add some amplitude modulation to simulate syllables
    syllable_rate = random.uniform(2, 5)  # 2-5 Hz for syllable rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t / duration)
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal


def add_noise(signal, noise_level):
    """Add Gaussian noise to signal"""
    noise = np.random.normal(0, noise_level, len(signal))
    noisy_signal = signal + noise
    
    # Normalize
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    
    return noisy_signal


def apply_random_filtering(signal, sample_rate):
    """Apply random filtering to simulate channel effects"""
    from scipy import signal as sps
    
    # Random filter parameters
    filter_type = random.choice(['lowpass', 'highpass', 'bandpass'])
    
    if filter_type == 'lowpass':
        # Lowpass filter
        cutoff = random.uniform(1000, 4000)  # Hz
        b, a = sps.butter(4, cutoff / (sample_rate / 2), btype='low')
    elif filter_type == 'highpass':
        # Highpass filter
        cutoff = random.uniform(100, 1000)  # Hz
        b, a = sps.butter(4, cutoff / (sample_rate / 2), btype='high')
    else:
        # Bandpass filter
        low_cutoff = random.uniform(300, 1000)  # Hz
        high_cutoff = random.uniform(2000, 4000)  # Hz
        b, a = sps.butter(4, [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)], btype='band')
    
    # Apply filter
    filtered_signal = sps.lfilter(b, a, signal)
    
    # Normalize
    filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
    
    return filtered_signal


def generate_dataset(config, args):
    """
    Generate synthetic dataset for speech quality prediction.
    
    Args:
        config (dict): Configuration dictionary
        args (Namespace): Command line arguments
    """
    # Parse parameters
    num_samples = args.num_samples
    duration = args.duration
    sample_rate = config['data']['sample_rate']
    noise_levels = [float(level) for level in args.noise_levels.split(',')]
    
    # Create output directories
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = get_absolute_path('data/synthetic')
    
    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples...")
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Generate clean signal
        signal = generate_speech_like_signal(duration, sample_rate)
        
        # Save clean signal
        clean_path = os.path.join(clean_dir, f'sample_{i:04d}.wav')
        sf.write(clean_path, signal, sample_rate)
        
        # Generate noisy versions
        for j, noise_level in enumerate(noise_levels):
            # Add noise
            noisy_signal = add_noise(signal, noise_level)
            
            # Optionally apply filtering (50% chance)
            if random.random() < 0.5:
                noisy_signal = apply_random_filtering(noisy_signal, sample_rate)
            
            # Save noisy signal
            noisy_path = os.path.join(noisy_dir, f'sample_{i:04d}_noise_{j:02d}.wav')
            sf.write(noisy_path, noisy_signal, sample_rate)
    
    print(f"Generated {num_samples} clean samples and {num_samples * len(noise_levels)} noisy samples")
    
    # Return directories for further processing
    return clean_dir, noisy_dir


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate dataset
    clean_dir, noisy_dir = generate_dataset(config, args)
    
    print("Dataset generation complete!")
    print(f"Clean files directory: {clean_dir}")
    print(f"Noisy files directory: {noisy_dir}")
    print("")
    print("To prepare metadata, run:")
    print(f"python -m data.prepare_metadata --clean_dir {clean_dir} --noisy_dir {noisy_dir} --calculate_metrics --artificial_mos")


if __name__ == "__main__":
    main() 