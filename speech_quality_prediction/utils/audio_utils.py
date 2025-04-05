import os
import numpy as np
import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi


def load_audio(file_path, sample_rate=16000, mono=True):
    """
    Load audio file and resample if necessary.
    
    Args:
        file_path (str): Path to audio file
        sample_rate (int): Target sampling rate
        mono (bool): Convert to mono if True
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
    return audio_data, sr


def save_audio(audio_data, file_path, sample_rate=16000):
    """
    Save audio data to file.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        file_path (str): Path to save audio file
        sample_rate (int): Sampling rate
    """
    sf.write(file_path, audio_data, sample_rate)


def extract_features(audio_data, sample_rate, config):
    """
    Extract features from audio data based on configuration.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sample_rate (int): Sampling rate
        config (dict): Feature extraction configuration
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # Convert frame lengths from ms to samples
    frame_length = int(config['frame_length_ms'] * sample_rate / 1000)
    hop_length = int(config['frame_shift_ms'] * sample_rate / 1000)
    
    # Extract MFCCs if configured
    if config['include_mfcc']:
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate,
            n_mfcc=config['n_mfcc'],
            n_fft=frame_length,
            hop_length=hop_length
        )
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features['mfcc'] = mfccs
        features['delta_mfcc'] = delta_mfccs
        features['delta2_mfcc'] = delta2_mfccs
    
    # Extract mel spectrogram if configured
    if config['include_spectrogram']:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=config['n_mels']
        )
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        features['log_mel_spectrogram'] = log_mel_spec
    
    # Extract chroma features if configured
    if config['include_chroma']:
        chroma = librosa.feature.chroma_stft(
            y=audio_data,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length
        )
        features['chroma'] = chroma
    
    # Extract spectral contrast if configured
    if config['include_contrast']:
        contrast = librosa.feature.spectral_contrast(
            y=audio_data,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length
        )
        features['spectral_contrast'] = contrast
    
    return features


def calculate_pesq(reference, degraded, sample_rate=16000):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score.
    
    Args:
        reference (numpy.ndarray): Reference audio signal
        degraded (numpy.ndarray): Degraded audio signal
        sample_rate (int): Sampling rate (must be 8000 or 16000)
        
    Returns:
        float: PESQ score
    """
    # PESQ only accepts 8000 or 16000 Hz
    if sample_rate not in [8000, 16000]:
        raise ValueError("Sample rate must be 8000 or 16000 for PESQ calculation")
    
    # Normalize audio to the range [-1, 1]
    reference = reference / np.max(np.abs(reference))
    degraded = degraded / np.max(np.abs(degraded))
    
    # Ensure signals have the same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Calculate PESQ
    mode = 'wb' if sample_rate == 16000 else 'nb'
    try:
        score = pesq(sample_rate, reference, degraded, mode)
        return score
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return None


def calculate_stoi(reference, degraded, sample_rate=16000):
    """
    Calculate STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        reference (numpy.ndarray): Reference audio signal
        degraded (numpy.ndarray): Degraded audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        float: STOI score
    """
    # Normalize audio to the range [-1, 1]
    reference = reference / np.max(np.abs(reference))
    degraded = degraded / np.max(np.abs(degraded))
    
    # Ensure signals have the same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    try:
        score = stoi(reference, degraded, sample_rate)
        return score
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return None 