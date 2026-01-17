"""
CT7160NI Computer Vision Coursework
Helper Utilities
"""

import os
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(gpu_id=0):
    """
    Get the appropriate device for training.
    
    Parameters:
    -----------
    gpu_id : int
        GPU device ID
        
    Returns:
    --------
    torch.device : The selected device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    return device


def count_parameters(model, trainable_only=True):
    """
    Count model parameters.
    
    Parameters:
    -----------
    model : nn.Module
        The model
    trainable_only : bool
        Whether to count only trainable parameters
        
    Returns:
    --------
    int : Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir='results', experiment_name=None):
    """
    Create directory for experiment results.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for results
    experiment_name : str, optional
        Name of the experiment
        
    Returns:
    --------
    str : Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = get_timestamp()
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_model_summary(model, filepath, input_size=(1, 3, 224, 224)):
    """
    Save model summary to file.
    
    Parameters:
    -----------
    model : nn.Module
        The model
    filepath : str
        Path to save summary
    input_size : tuple
        Input tensor size
    """
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Model architecture
        f.write(str(model))
        f.write("\n\n")
        
        # Parameter counts
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
    
    print(f"Model summary saved to {filepath}")


# Example usage
if __name__ == "__main__":
    print("Testing helper utilities...")
    
    # Test set_seed
    set_seed(42)
    
    # Test get_device
    device = get_device()
    
    # Test format_time
    print(f"\nFormat time test: {format_time(3723)} (should be 1h 2m 3s)")
    
    # Test timestamp
    print(f"Timestamp: {get_timestamp()}")
    
    print("\nHelper utilities tests passed!")

