"""
CT7160NI Computer Vision Coursework
Data Augmentation Transforms

This module provides data augmentation transforms for training and validation
using torchvision and albumentations libraries.
"""

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224, use_albumentations=True):
    """
    Get training data transforms with augmentation.
    
    Augmentation techniques applied:
    - Random resized crop
    - Random horizontal flip
    - Random rotation
    - Color jittering (brightness, contrast, saturation)
    - Normalization (ImageNet statistics)
    
    Parameters:
    -----------
    image_size : int
        Target image size (default: 224 for ResNet, 300 for EfficientNet)
    use_albumentations : bool
        Whether to use albumentations (recommended) or torchvision
        
    Returns:
    --------
    transform : callable
        Transform function to apply to images
    """
    if use_albumentations:
        return A.Compose([
            # Stronger augmentation as recommended
            A.RandomResizedCrop(
                size=(image_size, image_size),  # Use size parameter instead of height/width
                scale=(0.7, 1.0),  # More aggressive crop
                ratio=(0.8, 1.2)   # Wider aspect ratio range
            ),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Translation (10%)
                scale=(0.9, 1.1),  # Scaling (±10%)
                rotate=(-30, 30),   # Rotation (±30°)
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ColorJitter(
                brightness=0.3,  # Increased from 0.2
                contrast=0.3,    # Increased from 0.2
                saturation=0.3,   # Increased from 0.2
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.OneOf([
                A.ToGray(p=1.0),  # Convert to grayscale
                A.NoOp()
            ], p=0.1),  # Convert to grayscale with 10% probability
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_val_transforms(image_size=224, use_albumentations=True):
    """
    Get validation/test data transforms (no augmentation).
    
    Only applies:
    - Resize to target size
    - Center crop
    - Normalization
    
    Parameters:
    -----------
    image_size : int
        Target image size
    use_albumentations : bool
        Whether to use albumentations or torchvision
        
    Returns:
    --------
    transform : callable
        Transform function to apply to images
    """
    if use_albumentations:
        return A.Compose([
            A.Resize(height=image_size + 32, width=image_size + 32),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class AlbumentationsTransform:
    """
    Wrapper class to use albumentations with PyTorch Dataset.
    
    Converts PIL Image to numpy array before applying transforms.
    """
    
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, image):
        """Apply albumentations transform to PIL Image."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Apply transform
        transformed = self.transform(image=image_np)
        
        return transformed['image']


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a tensor image for visualization.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Normalized image tensor of shape (C, H, W)
    mean : list
        Mean values used for normalization
    std : list
        Standard deviation values used for normalization
        
    Returns:
    --------
    tensor : torch.Tensor
        Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def tensor_to_image(tensor):
    """
    Convert a tensor to a numpy image for visualization.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Image tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
    --------
    image : numpy.ndarray
        Image array of shape (H, W, C) with values in [0, 255]
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize
    tensor = denormalize(tensor)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose to (H, W, C)
    image = tensor.permute(1, 2, 0).numpy()
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image


# Example usage and testing
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print("Testing augmentation transforms...")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    
    # Test torchvision transforms
    train_tf = get_train_transforms(224, use_albumentations=False)
    val_tf = get_val_transforms(224, use_albumentations=False)
    
    print(f"Train transform output shape: {train_tf(pil_image).shape}")
    print(f"Val transform output shape: {val_tf(pil_image).shape}")
    
    # Test albumentations transforms
    train_tf_alb = get_train_transforms(224, use_albumentations=True)
    train_tf_wrapper = AlbumentationsTransform(train_tf_alb)
    
    print(f"Albumentations train transform output shape: {train_tf_wrapper(pil_image).shape}")
    
    print("\nAugmentation tests passed!")

