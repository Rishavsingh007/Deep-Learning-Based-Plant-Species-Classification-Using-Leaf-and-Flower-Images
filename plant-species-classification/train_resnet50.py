"""
Training Script: ResNet50 Transfer Learning
CT7160NI Computer Vision Coursework

Two-phase training approach:
1. Phase 1: Train only the classifier head with frozen backbone
2. Phase 2: Fine-tune the entire network with lower learning rate
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
from torch.amp import GradScaler, autocast
from pathlib import Path
from datetime import datetime

# Import ResNetClassifier directly to avoid timm dependency
import importlib.util
resnet_path = script_dir / 'src' / 'models' / 'resnet_model.py'
spec = importlib.util.spec_from_file_location("resnet_model", resnet_path)
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)
ResNetClassifier = resnet_module.ResNetClassifier

from src.data.data_loader import create_dataloaders
from src.training import Trainer

# Configuration for ResNet50 Transfer Learning
CONFIG = {
    'data_dir': 'data/raw/oxford_flowers_102',
    'batch_size': 16,  # Optimized for 4GB GPU memory
    'image_size': 224,  # Standard ImageNet size
    'num_workers': 0,  # Use 0 for Windows compatibility
    'use_masks': False,
    'apply_background_removal': False,
    'use_albumentations': True,  # Enable augmentation
    'use_weighted_sampler': True,  # Handle class imbalance
    'use_class_weights': True,  # Use class weights in loss function
    'model_name': 'resnet50',
    'save_dir': 'results/models',
    'metrics_dir': 'results/metrics',  # Directory for training metrics
    
    # Phase 1: Train classifier only (frozen backbone)
    'phase1': {
        'epochs': 15,
        'learning_rate': 1e-3,  # Higher LR for new layers
        'freeze_backbone': True,
        'early_stopping_patience': 5
    },
    
    # Phase 2: Fine-tune entire network
    'phase2': {
        'epochs': 35,
        'learning_rate': 1e-4,  # Lower LR for fine-tuning
        'freeze_backbone': False,
        'early_stopping_patience': 10
    },
    
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'use_amp': True,  # Enable Automatic Mixed Precision for 4GB GPU
    'gradient_accumulation_steps': 2  # Effective batch size = 16 * 2 = 32
}


def save_training_metrics(history, model_name, save_dir, phase=None):
    """
    Save all training metrics to a text file.
    
    Parameters:
    -----------
    history : dict
        Training history containing metrics
    model_name : str
        Name of the model
    save_dir : str
        Directory to save the metrics file
    phase : str, optional
        Phase name (e.g., 'phase1', 'phase2', or None for combined)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    if phase:
        filename = f"{model_name}_{phase}_training_metrics.txt"
    else:
        filename = f"{model_name}_training_metrics.txt"
    
    filepath = save_path / filename
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        if phase:
            f.write(f"Phase: {phase.upper()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Epochs: {len(history['train_loss'])}\n\n")
        
        f.write("Training Metrics:\n")
        f.write(f"  Final Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Best Loss: {min(history['train_loss']):.6f} (Epoch {history['train_loss'].index(min(history['train_loss'])) + 1})\n")
        f.write(f"  Final Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Best Accuracy: {max(history['train_acc']):.2f}% (Epoch {history['train_acc'].index(max(history['train_acc'])) + 1})\n")
        f.write(f"  Final Top-5 Accuracy: {history['train_top5_acc'][-1]:.2f}%\n")
        f.write(f"  Best Top-5 Accuracy: {max(history['train_top5_acc']):.2f}% (Epoch {history['train_top5_acc'].index(max(history['train_top5_acc'])) + 1})\n\n")
        
        f.write("Validation Metrics:\n")
        f.write(f"  Final Loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"  Best Loss: {min(history['val_loss']):.6f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})\n")
        f.write(f"  Final Accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"  Best Accuracy: {max(history['val_acc']):.2f}% (Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})\n")
        f.write(f"  Final Top-5 Accuracy: {history['val_top5_acc'][-1]:.2f}%\n")
        f.write(f"  Best Top-5 Accuracy: {max(history['val_top5_acc']):.2f}% (Epoch {history['val_top5_acc'].index(max(history['val_top5_acc'])) + 1})\n\n")
        
        # Detailed epoch-by-epoch metrics
        f.write("=" * 80 + "\n")
        f.write("DETAILED EPOCH-BY-EPOCH METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Train Top-5':<14} "
                f"{'Val Loss':<12} {'Val Acc':<12} {'Val Top-5':<14} {'LR':<12}\n")
        f.write("-" * 100 + "\n")
        
        for epoch in range(len(history['train_loss'])):
            f.write(f"{epoch+1:<8} "
                   f"{history['train_loss'][epoch]:<12.6f} "
                   f"{history['train_acc'][epoch]:<12.2f} "
                   f"{history['train_top5_acc'][epoch]:<14.2f} "
                   f"{history['val_loss'][epoch]:<12.6f} "
                   f"{history['val_acc'][epoch]:<12.2f} "
                   f"{history['val_top5_acc'][epoch]:<14.2f} "
                   f"{history['learning_rate'][epoch]:<12.2e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF METRICS\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nTraining metrics saved to: {filepath}")


def main():
    print("=" * 80)
    print("Training ResNet50 Transfer Learning Model")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"  Image Size: {CONFIG['image_size']}×{CONFIG['image_size']}")
    print(f"  Data Augmentation: {CONFIG['use_albumentations']}")
    print(f"  Weighted Sampling: {CONFIG['use_weighted_sampler']}")
    print(f"  Class Weights in Loss: {CONFIG['use_class_weights']}")
    print(f"\nPhase 1 (Classifier Training):")
    print(f"  Epochs: {CONFIG['phase1']['epochs']}")
    print(f"  Learning Rate: {CONFIG['phase1']['learning_rate']}")
    print(f"  Backbone: Frozen")
    print(f"\nPhase 2 (Fine-tuning):")
    print(f"  Epochs: {CONFIG['phase2']['epochs']}")
    print(f"  Learning Rate: {CONFIG['phase2']['learning_rate']}")
    print(f"  Backbone: Unfrozen")
    print("=" * 80)
    
    # Check CUDA and enforce GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! GPU is required for training.")
    
    device = 'cuda'
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
    
    # Enable AMP if requested
    use_amp = CONFIG.get('use_amp', False)
    if use_amp:
        print(f"AMP (Automatic Mixed Precision): ENABLED")
        scaler = GradScaler('cuda')
    else:
        scaler = None
        print(f"AMP (Automatic Mixed Precision): DISABLED")
    
    # Set memory management
    torch.cuda.empty_cache()  # Clear cache
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # Use 90% of GPU memory to leave room for system
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Create data loaders
    print("\nCreating data loaders...")
    loaders = create_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        num_workers=CONFIG['num_workers'],
        use_albumentations=CONFIG['use_albumentations'],
        use_weighted_sampler=CONFIG['use_weighted_sampler']
    )
    
    print(f"  Training samples: {len(loaders['train'].dataset):,}")
    print(f"  Validation samples: {len(loaders['val'].dataset):,}")
    print(f"  Test samples: {len(loaders['test'].dataset):,}")
    
    # Calculate class weights for loss function if enabled
    criterion = None
    if CONFIG['use_class_weights']:
        print("\nCalculating class weights for loss function...")
        train_dataset = loaders['train'].dataset
        class_counts = np.bincount(train_dataset.labels)
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights calculated (min: {class_weights.min():.4f}, max: {class_weights.max():.4f})")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ========================================================================
    # PHASE 1: Train Classifier Head Only (Backbone Frozen)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Training Classifier Head (Backbone Frozen)")
    print("=" * 80)
    
    # Create model with frozen backbone
    print("\nCreating ResNet50 model with ImageNet pre-trained weights...")
    model = ResNetClassifier(
        num_classes=102,
        pretrained=True,
        freeze_backbone=CONFIG['phase1']['freeze_backbone'],
        dropout=CONFIG['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    # Create optimizer (only for trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['phase1']['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    # Create trainer with AMP support
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=CONFIG['save_dir'],
        criterion=criterion,
        use_amp=use_amp
    )
    
    # Set gradient accumulation steps
    gradient_accumulation_steps = CONFIG.get('gradient_accumulation_steps', 1)
    if gradient_accumulation_steps > 1:
        print(f"\nGradient Accumulation: {gradient_accumulation_steps} steps")
        print(f"Effective batch size: {CONFIG['batch_size'] * gradient_accumulation_steps}")
    
    # Train Phase 1
    print("\nStarting Phase 1 training...")
    # Set gradient accumulation for trainer
    trainer.gradient_accumulation_steps = gradient_accumulation_steps
    trainer.train(
        epochs=CONFIG['phase1']['epochs'],
        early_stopping_patience=CONFIG['phase1']['early_stopping_patience'],
        save_best=True,
        model_name=f"{CONFIG['model_name']}_phase1"
    )
    
    # Extract Phase 1 history
    history_phase1 = {
        'train_loss': trainer.history['train_loss'].copy(),
        'train_acc': trainer.history['train_acc'].copy(),
        'train_top5_acc': trainer.history['train_top5_acc'].copy(),
        'val_loss': trainer.history['val_loss'].copy(),
        'val_acc': trainer.history['val_acc'].copy(),
        'val_top5_acc': trainer.history['val_top5_acc'].copy(),
        'learning_rate': trainer.history['learning_rate'].copy()
    }
    
    print(f"\nPhase 1 Complete!")
    print(f"  Best Validation Accuracy: {max(history_phase1['val_acc']):.2f}%")
    print(f"  Best Validation Top-5 Accuracy: {max(history_phase1['val_top5_acc']):.2f}%")
    
    # Save Phase 1 metrics
    save_training_metrics(history_phase1, CONFIG['model_name'], CONFIG['metrics_dir'], phase='phase1')
    
    # ========================================================================
    # PHASE 2: Fine-tune Entire Network
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Fine-tuning Entire Network")
    print("=" * 80)
    
    # Unfreeze backbone for fine-tuning
    print("\nUnfreezing backbone layers...")
    model.unfreeze_backbone(unfreeze_from_layer=0)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Create new optimizer with lower learning rate for fine-tuning
    # Use different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
    
    # Use lower LR for backbone, slightly higher for classifier
    optimizer = torch.optim.Adam(
        [
            {'params': backbone_params, 'lr': CONFIG['phase2']['learning_rate']},
            {'params': classifier_params, 'lr': CONFIG['phase2']['learning_rate'] * 2}
        ],
        weight_decay=CONFIG['weight_decay']
    )
    
    print(f"  Backbone learning rate: {CONFIG['phase2']['learning_rate']}")
    print(f"  Classifier learning rate: {CONFIG['phase2']['learning_rate'] * 2}")
    
    # New scheduler for phase 2
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=True
    )
    
    # Update trainer with new optimizer and scheduler
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.best_val_acc = max(history_phase1['val_acc'])  # Continue from phase 1 best
    trainer.gradient_accumulation_steps = gradient_accumulation_steps  # Keep gradient accumulation
    
    # Train Phase 2
    print("\nStarting Phase 2 training...")
    trainer.train(
        epochs=CONFIG['phase2']['epochs'],
        early_stopping_patience=CONFIG['phase2']['early_stopping_patience'],
        save_best=True,
        model_name=CONFIG['model_name']
    )
    
    # Extract Phase 2 history (only new entries)
    phase1_len = len(history_phase1['train_loss'])
    history_phase2 = {
        'train_loss': trainer.history['train_loss'][phase1_len:],
        'train_acc': trainer.history['train_acc'][phase1_len:],
        'train_top5_acc': trainer.history['train_top5_acc'][phase1_len:],
        'val_loss': trainer.history['val_loss'][phase1_len:],
        'val_acc': trainer.history['val_acc'][phase1_len:],
        'val_top5_acc': trainer.history['val_top5_acc'][phase1_len:],
        'learning_rate': trainer.history['learning_rate'][phase1_len:]
    }
    
    # Combined history (full training history)
    combined_history = trainer.history.copy()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nPhase 1 Results:")
    print(f"  Best Validation Accuracy: {max(history_phase1['val_acc']):.2f}%")
    print(f"  Best Validation Top-5 Accuracy: {max(history_phase1['val_top5_acc']):.2f}%")
    print(f"\nPhase 2 Results:")
    print(f"  Best Validation Accuracy: {max(history_phase2['val_acc']):.2f}%")
    print(f"  Best Validation Top-5 Accuracy: {max(history_phase2['val_top5_acc']):.2f}%")
    print(f"\nOverall Best Results:")
    print(f"  Best Validation Accuracy: {max(combined_history['val_acc']):.2f}%")
    print(f"  Best Validation Top-5 Accuracy: {max(combined_history['val_top5_acc']):.2f}%")
    print(f"  Final Validation Accuracy: {combined_history['val_acc'][-1]:.2f}%")
    print(f"  Final Validation Top-5 Accuracy: {combined_history['val_top5_acc'][-1]:.2f}%")
    print(f"  Final Training Accuracy: {combined_history['train_acc'][-1]:.2f}%")
    print(f"  Final Training Top-5 Accuracy: {combined_history['train_top5_acc'][-1]:.2f}%")
    print(f"\nModel saved to: {CONFIG['save_dir']}/{CONFIG['model_name']}_best.pth")
    print("=" * 80)
    
    # Save Phase 2 metrics
    save_training_metrics(history_phase2, CONFIG['model_name'], CONFIG['metrics_dir'], phase='phase2')
    
    # Save combined metrics
    save_training_metrics(combined_history, CONFIG['model_name'], CONFIG['metrics_dir'], phase=None)
    
    return combined_history


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

