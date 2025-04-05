import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_factory import get_model
from data.dataset import get_dataloaders
from utils.config_utils import load_config, save_config, get_absolute_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train speech quality prediction model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_type", type=str, default=None, 
                        help="Model type (cnn_lstm, transformer)")
    parser.add_argument("--feature_type", type=str, default='mfcc', 
                        help="Feature type (mfcc, log_mel_spectrogram)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--metadata_file", type=str, default=None, 
                        help="Path to metadata file")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save outputs")
    return parser.parse_args()


def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    if args.model_type is not None:
        config['model']['type'] = args.model_type
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.output_dir is not None:
        config['training']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
    
    return config


def get_optimizer(model, config):
    """Get optimizer based on configuration"""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(optimizer, config):
    """Get learning rate scheduler based on configuration"""
    scheduler_name = config['training']['lr_scheduler'].lower()
    
    if scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['num_epochs']
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch_idx, sample in enumerate(train_loader):
        # Extract inputs and targets
        inputs = sample['input'].to(device)
        targets = sample['target'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        all_predictions.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    pearson_corr, _ = pearsonr(all_targets, all_predictions)
    spearman_corr, _ = spearmanr(all_targets, all_predictions)
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, mse, rmse, mae, pearson_corr, spearman_corr


def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sample in val_loader:
            # Extract inputs and targets
            inputs = sample['input'].to(device)
            targets = sample['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Track loss and predictions
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    pearson_corr, _ = pearsonr(all_targets, all_predictions)
    spearman_corr, _ = spearmanr(all_targets, all_predictions)
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, mse, rmse, mae, pearson_corr, spearman_corr


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, path)


def train(config, metadata_file, feature_type='mfcc'):
    """Train model with given configuration"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_file=metadata_file,
        config=config,
        feature_type=feature_type,
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    model = get_model(config)
    model.to(device)
    print(f"Created model: {config['model']['type']}")
    
    # Loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config)
    
    # TensorBoard writer
    log_dir = os.path.join(get_absolute_path('logs'), time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    checkpoint_dir = get_absolute_path(config['training']['checkpoint_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping setup
    patience = config['training']['early_stopping_patience']
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_mse, train_rmse, train_mae, train_pearson, train_spearman = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_mse, val_rmse, val_mae, val_pearson, val_spearman = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('PearsonCorr/train', train_pearson, epoch)
        writer.add_scalar('PearsonCorr/val', val_pearson, epoch)
        writer.add_scalar('SpearmanCorr/train', train_spearman, epoch)
        writer.add_scalar('SpearmanCorr/val', val_spearman, epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, Pearson: {train_pearson:.4f} | "
              f"Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, Pearson: {val_pearson:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            patience_counter = 0
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Close writer
    writer.close()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_mse, test_rmse, test_mae, test_pearson, test_spearman = validate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Results | "
          f"Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, "
          f"MAE: {test_mae:.4f}, Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}")
    
    # Save test results
    test_results = {
        'loss': test_loss,
        'mse': test_mse,
        'rmse': test_rmse,
        'mae': test_mae,
        'pearson': test_pearson,
        'spearman': test_spearman
    }
    
    test_results_path = os.path.join(checkpoint_dir, "test_results.txt")
    with open(test_results_path, 'w') as f:
        for metric, value in test_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return model, test_results


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Determine metadata file
    metadata_file = args.metadata_file
    if metadata_file is None:
        metadata_file = get_absolute_path(os.path.join('data', 'processed', 'metadata.csv'))
    
    # Create output directories
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = get_absolute_path(os.path.join('models', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save updated configuration
    config_save_path = os.path.join(output_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    # Train model
    model, test_results = train(config, metadata_file, args.feature_type)
    
    print(f"Training complete! Model saved to {os.path.join(config['training']['checkpoint_dir'], 'best_model.pt')}")


if __name__ == "__main__":
    main() 