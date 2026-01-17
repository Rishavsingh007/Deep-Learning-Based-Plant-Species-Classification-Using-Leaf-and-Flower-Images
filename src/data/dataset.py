"""
CT7160NI Computer Vision Coursework
Custom Dataset Class for Oxford 102 Flower Dataset

This module implements a PyTorch Dataset class for loading and preprocessing
flower images from the Oxford 102 Flower Dataset.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split



class FlowerDataset(Dataset):
    """
    Custom PyTorch Dataset for Oxford 102 Flower Dataset.
    
    The Oxford 102 Flower Dataset contains 8,189 images of flowers belonging 
    to 102 different categories. The flowers are common flowers in the UK.
    
    Parameters:
    -----------
    root_dir : str
        Path to the dataset root directory containing 'jpg' folder and .mat files
    split : str
        One of 'train', 'val', or 'test'
    transform : callable, optional
        A function/transform to apply to the images
        
    Attributes:
    -----------
    image_paths : list
        List of paths to all images in the split
    labels : list
        List of corresponding labels (0-101)
    class_names : list
        List of flower category names (if available)
    num_classes : int
        Number of classes (102)
        
    Example:
    --------
    >>> from src.data import FlowerDataset
    >>> train_dataset = FlowerDataset(
    ...     root_dir='data/raw/oxford_flowers_102',
    ...     split='train',
    ...     transform=train_transforms
    ... )
    >>> image, label = train_dataset[0]
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the FlowerDataset.
        
        Parameters:
        -----------
        root_dir : str
            Path to dataset root directory
        split : str
            One of 'train', 'val', or 'test'
        transform : callable, optional
            Transform to apply to images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_dataset()
        
        # Store number of classes
        self.num_classes = 102
        
    def _load_dataset(self):
        """
        Load dataset split information and create 70/15/15 stratified split.
        
        The Oxford 102 dataset provides:
        - imagelabels.mat: Contains labels for all images (1-102, converted to 0-101)
        
        This implementation creates a stratified 70/15/15 split to ensure:
        - 70% training data for better model learning
        - 15% validation data for hyperparameter tuning
        - 15% test data for final evaluation
        - Balanced class distribution across all splits
        
        Returns:
        --------
        tuple: (image_paths, labels)
        """
        # Paths to .mat files
        labels_path = os.path.join(self.root_dir, 'imagelabels.mat')
        
        # Try both possible locations for images (standard structure vs extracted archive structure)
        jpg_dir = os.path.join(self.root_dir, 'jpg')
        if not os.path.exists(jpg_dir):
            # Try alternative location (when extracted from archive)
            alt_jpg_dir = os.path.join(self.root_dir, '102flowers', 'jpg')
            if os.path.exists(alt_jpg_dir):
                jpg_dir = alt_jpg_dir
            else:
                raise FileNotFoundError(
                    f"Images directory not found. Tried:\n"
                    f"  - {jpg_dir}\n"
                    f"  - {alt_jpg_dir}\n"
                    "Please ensure the dataset is properly extracted."
                )
        
        # Check if files exist
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                "Please download the dataset from:\n"
                "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
            )
        
        # Load labels (1-indexed in .mat file, convert to 0-indexed)
        labels_mat = sio.loadmat(labels_path)
        all_labels = labels_mat['labels'].flatten() - 1  # Convert to 0-indexed
        
        # Get all valid image indices (check which images actually exist)
        valid_indices = []
        valid_labels = []
        
        for idx in range(len(all_labels)):
            img_name = f"image_{idx + 1:05d}.jpg"
            img_path = os.path.join(jpg_dir, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
                valid_labels.append(all_labels[idx])
        
        valid_indices = np.array(valid_indices)
        valid_labels = np.array(valid_labels)
        
        # Create stratified 70/15/15 split
        # First split: 70% train, 30% temp (val + test)
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            valid_indices,
            valid_labels,
            test_size=0.30,  # 30% for val + test
            stratify=valid_labels,  # Maintain class distribution
            random_state=42  # For reproducibility
        )
        
        # Second split: Split temp 50/50 to get 15/15 of total (30% * 0.5 = 15%)
        val_indices, test_indices, val_labels, test_labels = train_test_split(
            temp_indices,
            temp_labels,
            test_size=0.50,  # Split temp 50/50
            stratify=temp_labels,  # Maintain class distribution
            random_state=42  # For reproducibility
        )
        
        # Get indices for the requested split
        if self.split == 'train':
            indices = train_indices
        elif self.split == 'val':
            indices = val_indices
        elif self.split == 'test':
            indices = test_indices
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
        
        # Build image paths and labels for this split
        image_paths = []
        labels = []
        
        for idx in indices:
            # Image files are named: image_00001.jpg, image_00002.jpg, etc.
            img_name = f"image_{idx + 1:05d}.jpg"
            img_path = os.path.join(jpg_dir, img_name)
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(all_labels[idx])
        
        return image_paths, labels
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample to retrieve
            
        Returns:
        --------
        tuple: (image, label)
            - image: Transformed image tensor
            - label: Integer class label (0-101)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
        --------
        dict: Dictionary mapping class index to count
        """
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def get_sample_images(self, n_samples=5):
        """
        Get sample images from each class for visualization.
        
        Parameters:
        -----------
        n_samples : int
            Number of sample images per class
            
        Returns:
        --------
        dict: Dictionary mapping class index to list of image paths
        """
        samples = {i: [] for i in range(self.num_classes)}
        
        for img_path, label in zip(self.image_paths, self.labels):
            if len(samples[label]) < n_samples:
                samples[label].append(img_path)
                
        return samples


# For testing the dataset class
if __name__ == "__main__":
    # Example usage
    print("Testing FlowerDataset...")
    
    # Test with dummy path (update with actual path when running)
    try:
        dataset = FlowerDataset(
            root_dir='data/raw/oxford_flowers_102',
            split='train',
            transform=None
        )
        print(f"Number of training samples: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        
        # Get class distribution
        dist = dataset.get_class_distribution()
        print(f"Class distribution (first 5): {dict(list(dist.items())[:5])}")
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download the Oxford 102 Flower Dataset first.")

