"""
CT7160NI Computer Vision Coursework
Image Preprocessing Utilities

This module provides functions for preprocessing images including
resizing, normalization, and format conversion.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm


def preprocess_image(image, target_size=224, normalize=True):
    """
    Preprocess a single image for model inference.
    
    Parameters:
    -----------
    image : PIL.Image, numpy.ndarray, or str
        Input image (PIL Image, numpy array, or path to image)
    target_size : int
        Target size for resizing
    normalize : bool
        Whether to apply ImageNet normalization
        
    Returns:
    --------
    torch.Tensor : Preprocessed image tensor of shape (1, C, H, W)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy array
    image_np = np.array(image, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def resize_image(image_path, output_path, target_size=224):
    """
    Resize an image and save to output path.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    output_path : str
        Path to save resized image
    target_size : int
        Target size (image will be resized to target_size x target_size)
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)


def preprocess_dataset(input_dir, output_dir, target_size=224):
    """
    Preprocess entire dataset by resizing all images.
    
    Parameters:
    -----------
    input_dir : str
        Path to input directory containing images
    output_dir : str
        Path to output directory for resized images
    target_size : int
        Target image size
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to preprocess")
    
    for img_file in tqdm(image_files, desc="Preprocessing images"):
        # Calculate output path (maintain directory structure)
        relative_path = img_file.relative_to(input_path)
        out_file = output_path / relative_path
        
        try:
            resize_image(str(img_file), str(out_file), target_size)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")


def calculate_channel_stats(image_paths):
    """
    Calculate mean and std for each channel across a set of images.
    
    Useful for computing custom normalization statistics.
    
    Parameters:
    -----------
    image_paths : list
        List of paths to images
        
    Returns:
    --------
    tuple : (mean, std) - lists of 3 values each for RGB channels
    """
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0
    
    for img_path in tqdm(image_paths, desc="Calculating statistics"):
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        pixel_sum += image.sum(axis=(0, 1))
        pixel_sq_sum += (image ** 2).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]
    
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    
    return mean.tolist(), std.tolist()


def visualize_preprocessing(image_path, target_size=224):
    """
    Visualize the preprocessing steps applied to an image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    target_size : int
        Target size for preprocessing
        
    Returns:
    --------
    dict : Dictionary containing original, resized, and normalized images
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    original = Image.open(image_path).convert('RGB')
    
    # Resize
    resized = original.resize((target_size, target_size), Image.BILINEAR)
    
    # Normalize
    normalized = np.array(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title(f'Original ({original.size[0]}x{original.size[1]})')
    axes[0].axis('off')
    
    axes[1].imshow(resized)
    axes[1].set_title(f'Resized ({target_size}x{target_size})')
    axes[1].axis('off')
    
    # For normalized image, denormalize for visualization
    denorm = normalized * std + mean
    denorm = np.clip(denorm, 0, 1)
    axes[2].imshow(denorm)
    axes[2].set_title('Normalized (denorm for viz)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_visualization.png', dpi=150)
    plt.close()
    
    return {
        'original': original,
        'resized': resized,
        'normalized': normalized
    }


# Example usage
if __name__ == "__main__":
    print("Testing preprocessing utilities...")
    
    # Test preprocess_image function
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_image, target_size=224, normalize=True)
    
    print(f"Preprocessed tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Value range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    print("\nPreprocessing tests passed!")

