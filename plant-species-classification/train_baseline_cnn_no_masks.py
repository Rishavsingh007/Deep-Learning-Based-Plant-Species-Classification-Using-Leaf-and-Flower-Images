"""
Training Script: Baseline CNN (No Masks)
CT7160NI Computer Vision Coursework

First training run - establishes baseline performance without background removal.
"""

import sys
import os
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))

# Change to script directory to ensure relative paths work
os.chdir(script_dir)

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models import BaselineCNN
from src.data.data_loader import create_dataloaders
from src.training import Trainer

# Configuration for Baseline CNN (No Masks)
CONFIG = {
    'data_dir': 'data/raw/oxford_flowers_102',
    'batch_size': 16,  # Start with 16, adjust if OOM
    'image_size': 224,
    'num_workers': 0,  # Use 0 for Windows compatibility
    'use_masks': False,
    'apply_background_removal': False,
    'use_albumentations': True,  # Enable augmentation for better generalization
    'epochs': 100,  # Increased from 50 for better convergence
    'learning_rate': 1e-3,  # Increased to 1e-3 as recommended for better convergence
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,  # As recommended for optimal training
    'model_name': 'baseline_cnn_no_masks',
    'save_dir': 'results/models',
    'use_weighted_sampler': True,  # Handle class imbalance
    'use_class_weights': True  # Use class weights in loss function
}


def main():
    print("=" * 70)
    print("Training Baseline CNN (No Masks) - Baseline Performance")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"  Image Size: {CONFIG['image_size']}×{CONFIG['image_size']}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Early Stopping Patience: {CONFIG['early_stopping_patience']}")
    print(f"  Use Masks: {CONFIG['use_masks']}")
    print(f"  Background Removal: {CONFIG['apply_background_removal']}")
    print(f"  Data Augmentation: {CONFIG['use_albumentations']}")
    print(f"  Learning Rate Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print(f"  Model Dropout: 0.35 (reduced from 0.5 to reduce underfitting)")
    print(f"  Weighted Sampling: {CONFIG['use_weighted_sampler']} (handles class imbalance)")
    print(f"  Class Weights in Loss: {CONFIG['use_class_weights']} (handles class imbalance)")
    print("=" * 70)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    loaders = create_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        num_workers=CONFIG['num_workers'],
        use_masks=CONFIG['use_masks'],
        apply_background_removal=CONFIG['apply_background_removal'],
        use_albumentations=CONFIG['use_albumentations'],
        use_weighted_sampler=CONFIG['use_weighted_sampler']
    )
    
    print(f"  Training samples: {len(loaders['train'].dataset)}")
    print(f"  Validation samples: {len(loaders['val'].dataset)}")
    print(f"  Test samples: {len(loaders['test'].dataset)}")
    
    # Calculate class weights for loss function if enabled
    criterion = None
    if CONFIG['use_class_weights']:
        print("\nCalculating class weights for loss function...")
        train_dataset = loaders['train'].dataset
        class_counts = np.bincount(train_dataset.labels)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights calculated (min: {class_weights.min():.4f}, max: {class_weights.max():.4f})")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create model with reduced dropout (0.35 instead of 0.5 to reduce underfitting)
    print("\nCreating Baseline CNN model...")
    model = BaselineCNN(num_classes=102, dropout=0.35)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer (custom learning rate if needed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Fix 2: Add learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor validation loss (decrease LR when loss plateaus)
        factor=0.5,  # Reduce LR by half
        patience=5,  # Wait 5 epochs
        min_lr=1e-6,
        verbose=True
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=CONFIG['save_dir'],
        criterion=criterion  # Use weighted loss if enabled
    )
    
    # Train model
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    history = trainer.train(
        epochs=CONFIG['epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        save_best=True,
        model_name=CONFIG['model_name']
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"Best Validation Top-5 Accuracy: {max(history['val_top5_acc']):.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Validation Top-5 Accuracy: {history['val_top5_acc'][-1]:.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Training Top-5 Accuracy: {history['train_top5_acc'][-1]:.2f}%")
    print(f"\nModel saved to: {CONFIG['save_dir']}/{CONFIG['model_name']}_best.pth")
    print("=" * 70)
    
    return history


if __name__ == '__main__':
    try:
        history = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

