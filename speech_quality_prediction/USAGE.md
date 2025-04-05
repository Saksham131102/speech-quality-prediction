# Speech Quality Prediction - Usage Instructions

This document provides instructions for using the Speech Quality Prediction system.

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd speech_quality_prediction
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Project Structure

```
├── config/        # Configuration files
├── data/          # Data handling and dataset utilities
├── models/        # Model implementations
├── notebooks/     # Jupyter notebooks for exploration
├── utils/         # Utility functions
├── train.py       # Training script
├── predict.py     # Prediction script
├── main.py        # Main pipeline script
├── requirements.txt
```

## Quick Start

The fastest way to get started is to use the `main.py` script, which runs the entire pipeline:

```
python main.py --generate_data --prepare_metadata --train --evaluate
```

This will:
1. Generate a synthetic dataset
2. Prepare metadata with quality scores
3. Train a model
4. Evaluate the model on test data

## Step-by-Step Instructions

### 1. Configuration

The system uses a YAML configuration file located at `config/config.yaml`. You can modify this file to change various parameters.

### 2. Data Preparation

#### Option A: Generate Synthetic Data

To generate synthetic test data:

```
python -m data.generate_test_dataset --num_samples 100
```

#### Option B: Use Your Own Data

If you have your own data, organize it into clean and noisy folders:

```
data/
  clean/       # Reference clean audio files
  noisy/       # Degraded audio files
```

#### Prepare Metadata

To prepare metadata with quality scores:

```
python -m data.prepare_metadata --clean_dir data/clean --noisy_dir data/noisy --calculate_metrics --artificial_mos
```

### 3. Training

To train a model:

```
python train.py --metadata_file data/processed/metadata.csv --model_type cnn_lstm --feature_type mfcc
```

Available model types:
- `cnn_lstm`: CNN-LSTM model
- `transformer`: Transformer model

Available feature types:
- `mfcc`: Mel-frequency cepstral coefficients
- `log_mel_spectrogram`: Log mel spectrogram

### 4. Prediction

To predict quality scores using a trained model:

```
python predict.py --model_path models/checkpoints/best_model.pt --metadata_file data/processed/test_metadata.csv
```

Or to predict for a single file:

```
python predict.py --model_path models/checkpoints/best_model.pt --input_audio path/to/audio.wav
```

### 5. Testing

To test the system with a simple example:

```
python test_example.py
```

## Advanced Usage

### Running the Complete Pipeline

The `main.py` script provides a way to run the complete pipeline:

```
python main.py --generate_data --prepare_metadata --train --evaluate --model_type transformer --feature_type log_mel_spectrogram --num_samples 200 --epochs 50 --batch_size 64
```

Arguments:
- `--generate_data`: Generate synthetic dataset
- `--prepare_metadata`: Prepare metadata with quality scores
- `--train`: Train the model
- `--evaluate`: Evaluate the model
- `--model_type`: Type of model (cnn_lstm, transformer)
- `--feature_type`: Type of features (mfcc, log_mel_spectrogram)
- `--num_samples`: Number of samples for synthetic dataset
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--output_dir`: Directory for outputs
- `--config`: Path to custom config file

### Using Traditional Metrics

To calculate traditional objective metrics (PESQ, STOI) for a file:

```
python predict.py --input_audio path/to/degraded.wav --reference_audio path/to/clean.wav
```

## Examples

### Example 1: Train a CNN-LSTM model with MFCCs

```
python train.py --model_type cnn_lstm --feature_type mfcc --num_epochs 50 --batch_size 32
```

### Example 2: Evaluate a trained model

```
python predict.py --model_path models/checkpoints/best_model.pt --metadata_file data/processed/test_metadata.csv --output_file results/predictions.csv
```

### Example 3: Generate a large synthetic dataset

```
python -m data.generate_test_dataset --num_samples 500 --output_dir data/large_synthetic
```

## Troubleshooting

- **Error: "No module named 'speech_quality_prediction'"**: Make sure you're running commands from the project root directory.
- **Error calculating PESQ**: PESQ only works with 8kHz or 16kHz audio. Make sure your audio is at the correct sample rate.
- **CUDA out of memory**: Reduce batch size in the configuration.
- **Unexpected prediction values**: Make sure you're using the same feature type for prediction as was used during training. 