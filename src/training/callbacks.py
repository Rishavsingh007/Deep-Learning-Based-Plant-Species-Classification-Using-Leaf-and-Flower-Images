"""
CT7160NI Computer Vision Coursework
Training Callbacks

This module implements callback functions for training including
early stopping and model checkpointing.
"""

import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    
    Parameters:
    -----------
    patience : int
        Number of epochs to wait before stopping
    min_delta : float
        Minimum change to qualify as an improvement
    mode : str
        'min' for loss, 'max' for accuracy
    verbose : bool
        Whether to print messages
        
    Example:
    --------
    >>> early_stopping = EarlyStopping(patience=10, mode='min')
    >>> for epoch in range(epochs):
    ...     train()
    ...     val_loss = validate()
    ...     if early_stopping(val_loss):
    ...         print("Early stopping triggered!")
    ...         break
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        # Set comparison function
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
            
    def __call__(self, current_value):
        """
        Check if training should stop.
        
        Parameters:
        -----------
        current_value : float
            Current metric value (loss or accuracy)
            
        Returns:
        --------
        bool : True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                print(f"  EarlyStopping: Metric improved to {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Callback to save model checkpoints during training.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save checkpoints
    filename : str
        Filename template (can include {epoch}, {val_loss}, {val_acc})
    monitor : str
        Metric to monitor ('val_loss' or 'val_acc')
    mode : str
        'min' for loss, 'max' for accuracy
    save_best_only : bool
        Whether to only save the best model
    verbose : bool
        Whether to print messages
        
    Example:
    --------
    >>> checkpoint = ModelCheckpoint('results/models', 'best_model.pth', monitor='val_acc')
    >>> for epoch in range(epochs):
    ...     train()
    ...     val_acc = validate()
    ...     checkpoint(model, val_acc, epoch)
    """
    
    def __init__(
        self,
        save_dir,
        filename='model_best.pth',
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        verbose=True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        # Determine mode
        if mode == 'auto':
            mode = 'max' if 'acc' in monitor else 'min'
        self.mode = mode
        
        # Initialize best value
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        # Set comparison function
        if mode == 'min':
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
            
    def __call__(self, model, current_value, epoch=None, optimizer=None, **kwargs):
        """
        Potentially save a checkpoint.
        
        Parameters:
        -----------
        model : nn.Module
            The model to save
        current_value : float
            Current metric value
        epoch : int
            Current epoch number
        optimizer : torch.optim.Optimizer
            The optimizer (optional)
            
        Returns:
        --------
        bool : True if checkpoint was saved
        """
        if self.save_best_only:
            if self.is_better(current_value, self.best_value):
                self.best_value = current_value
                self._save_checkpoint(model, current_value, epoch, optimizer, **kwargs)
                return True
        else:
            self._save_checkpoint(model, current_value, epoch, optimizer, **kwargs)
            return True
        
        return False
    
    def _save_checkpoint(self, model, metric_value, epoch, optimizer, **kwargs):
        """Save the checkpoint."""
        # Format filename
        filename = self.filename
        if epoch is not None:
            filename = filename.replace('{epoch}', str(epoch))
        filename = filename.replace('{val_loss}', f'{metric_value:.4f}')
        filename = filename.replace('{val_acc}', f'{metric_value:.2f}')
        
        filepath = self.save_dir / filename
        
        # Create checkpoint dict
        checkpoint = {
            'model_state_dict': model.state_dict(),
            f'{self.monitor}': metric_value,
            'epoch': epoch
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        # Add any additional kwargs
        checkpoint.update(kwargs)
        
        # Save
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"  ModelCheckpoint: Saved to {filepath}")


class LearningRateLogger:
    """
    Callback to log learning rate during training.
    """
    
    def __init__(self):
        self.lr_history = []
        
    def __call__(self, optimizer):
        """Log current learning rate."""
        lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        return lr
    
    def get_history(self):
        return self.lr_history


class GradientClipping:
    """
    Callback for gradient clipping during training.
    
    Parameters:
    -----------
    max_norm : float
        Maximum norm for gradient clipping
    norm_type : float
        Type of norm to use (default: 2.0)
    """
    
    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def __call__(self, model):
        """Clip gradients."""
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )


# Example usage
if __name__ == "__main__":
    print("Testing callbacks...")
    
    # Test EarlyStopping
    early_stopping = EarlyStopping(patience=3, mode='min', verbose=True)
    
    test_losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]  # Should stop after index 5
    
    for i, loss in enumerate(test_losses):
        print(f"\nEpoch {i+1}: Loss = {loss}")
        if early_stopping(loss):
            print(f"Training stopped at epoch {i+1}")
            break
    
    print("\nCallback tests completed!")

