# Speech Quality Prediction

This project implements computational models for evaluating the perceived quality of speech signals.

## Overview

Speech quality prediction models attempt to estimate human subjective quality ratings using objective computational methods. This project includes implementations of both traditional signal-processing-based metrics and modern machine learning approaches.

## Features

- Audio feature extraction from speech samples
- Implementation of traditional metrics (PESQ, STOI)
- Deep learning models for speech quality prediction
- Tools for dataset preparation and augmentation
- Evaluation framework for model comparison

## Project Structure

```
├── config/        # Configuration files
├── data/          # Data handling and dataset utilities
├── models/        # Model implementations
├── notebooks/     # Jupyter notebooks for exploration and visualization
├── utils/         # Utility functions
├── requirements.txt
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your dataset using the scripts in `data/`
4. Train models using the training scripts
5. Evaluate model performance

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

MIT 