"""
CT7160NI Computer Vision Coursework
Data Loader Utilities

This module provides functions for creating PyTorch DataLoaders
for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from .dataset import FlowerDataset
from .augmentation import get_train_transforms, get_val_transforms, AlbumentationsTransform


def create_dataloaders(
    data_dir,
    batch_size=32,
    image_size=224,
    num_workers=4,
    use_weighted_sampler=False,
    use_albumentations=True
):
    """
    Create train, validation, and test DataLoaders.
    
    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory
    batch_size : int
        Batch size for training and evaluation
    image_size : int
        Target image size (224 for ResNet, 300 for EfficientNet)
    num_workers : int
        Number of worker processes for data loading
    use_weighted_sampler : bool
        Whether to use weighted random sampling for class imbalance
    use_albumentations : bool
        Whether to use albumentations for augmentation
        
    Returns:
    --------
    dict : Dictionary containing 'train', 'val', 'test' DataLoaders
    """
    # Get transforms
    train_transform = get_train_transforms(image_size, use_albumentations)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    # Wrap albumentations transforms if needed
    if use_albumentations:
        train_transform = AlbumentationsTransform(train_transform)
        val_transform = AlbumentationsTransform(val_transform)
    
    # Create datasets
    train_dataset = FlowerDataset(data_dir, split='train', transform=train_transform)
    val_dataset = FlowerDataset(data_dir, split='val', transform=val_transform)
    test_dataset = FlowerDataset(data_dir, split='test', transform=val_transform)
    
    # Create weighted sampler if requested
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampler:
        # Calculate class weights
        class_counts = np.bincount(train_dataset.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_dataset.labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset statistics
    print(f"Dataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {train_dataset.num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_dataset_stats(data_loader):
    """
    Calculate mean and standard deviation of the dataset.
    
    Useful for custom normalization if not using ImageNet statistics.
    
    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader to calculate statistics from
        
    Returns:
    --------
    tuple : (mean, std) for each channel
    """
    mean = 0.0
    std = 0.0
    n_samples = 0
    
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples
    
    mean /= n_samples
    std /= n_samples
    
    return mean.tolist(), std.tolist()


# Example usage
if __name__ == "__main__":
    print("Testing DataLoader creation...")
    
    # This will fail if dataset is not downloaded yet
    try:
        loaders = create_dataloaders(
            data_dir='data/raw/oxford_flowers_102',
            batch_size=32,
            image_size=224,
            num_workers=0  # Use 0 for testing
        )
        
        # Test loading a batch
        train_loader = loaders['train']
        images, labels = next(iter(train_loader))
        
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Min value: {images.min():.4f}")
        print(f"Max value: {images.max():.4f}")
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download the Oxford 102 Flower Dataset first.")

