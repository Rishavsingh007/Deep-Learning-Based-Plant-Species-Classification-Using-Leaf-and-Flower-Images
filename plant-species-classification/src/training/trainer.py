"""
CT7160NI Computer Vision Coursework
Training Loop Implementation

This module implements the training loop for CNN models including
support for learning rate scheduling, early stopping, and experiment logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """
    Trainer class for CNN models.
    
    Handles the training loop, validation, checkpointing, and logging.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function (default: CrossEntropyLoss)
    optimizer : torch.optim
        Optimizer (default: Adam with lr=1e-4)
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler (optional)
    device : str
        Device to train on ('cuda' or 'cpu')
    save_dir : str
        Directory to save checkpoints and logs
        
    Example:
    --------
    >>> trainer = Trainer(model, train_loader, val_loader)
    >>> history = trainer.train(epochs=50)
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device=None,
        save_dir='results/models',
        use_amp=False
    ):
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optimizer or optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = scheduler
        
        # Mixed precision training
        self.use_amp = use_amp and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='Validation', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        epochs,
        early_stopping_patience=None,
        save_best=True,
        model_name='model'
    ):
        """
        Train the model for specified number of epochs.
        
        Parameters:
        -----------
        epochs : int
            Number of epochs to train
        early_stopping_patience : int
            Patience for early stopping (None = disabled)
        save_best : bool
            Whether to save the best model
        model_name : str
            Name prefix for saved model files
            
        Returns:
        --------
        dict : Training history
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(f'{model_name}_best.pth')
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 60)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        
        # Save final model
        self.save_checkpoint(f'{model_name}_final.pth')
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
            
        print(f"Loaded checkpoint from {filepath}")


def create_optimizer(model, optimizer_name='adam', lr=1e-4, weight_decay=1e-4):
    """
    Create optimizer for model.
    
    Parameters:
    -----------
    model : nn.Module
        The model
    optimizer_name : str
        Name of optimizer ('adam', 'sgd', 'adamw')
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for regularization
        
    Returns:
    --------
    torch.optim.Optimizer
    """
    # Get trainable parameters
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name='plateau', **kwargs):
    """
    Create learning rate scheduler.
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer
    scheduler_name : str
        Name of scheduler ('plateau', 'step', 'cosine')
        
    Returns:
    --------
    torch.optim.lr_scheduler
    """
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    else:
        return None


# Example usage
if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Usage:")
    print("  trainer = Trainer(model, train_loader, val_loader)")
    print("  history = trainer.train(epochs=50)")

