"""
CT7160NI Computer Vision Coursework
Methodology Code Examples and Visualizations

This module contains code examples for the methodology section of the mid-proposal,
demonstrating preprocessing, feature extraction, model architecture, and post-processing.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch
import torch.nn as nn
from torchvision import transforms
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# 5.1 PRE-PROCESSING
# ============================================================================

def preprocess_image_noise_reduction(image_path, method='gaussian'):
    """
    Demonstrate noise reduction techniques.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    method : str
        Noise reduction method: 'gaussian', 'median', 'bilateral'
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply noise reduction
    if method == 'gaussian':
        # Gaussian blur for noise reduction
        processed = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    elif method == 'median':
        # Median filter for salt-and-pepper noise
        processed = cv2.medianBlur(img_rgb, 5)
    elif method == 'bilateral':
        # Bilateral filter preserves edges while reducing noise
        processed = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    else:
        processed = img_rgb
    
    return img_rgb, processed


def preprocess_normalization_resize(image_path, target_size=224):
    """
    Demonstrate normalization and resizing.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    target_size : int
        Target image size (default: 224 for ResNet)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_array - mean) / std
    
    return {
        'original': np.array(img),
        'resized': img_array,
        'normalized': img_normalized,
        'original_size': original_size,
        'target_size': (target_size, target_size)
    }


def preprocess_color_space_conversion(image_path):
    """
    Demonstrate color space conversions.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to different color spaces
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return {
        'RGB': rgb,
        'HSV': hsv,
        'LAB': lab,
        'Grayscale': grayscale
    }


# ============================================================================
# 5.2 FEATURE EXTRACTION / REPRESENTATION
# ============================================================================

def extract_handcrafted_features_hog(image_path):
    """
    Extract HOG (Histogram of Oriented Gradients) features.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    """
    from skimage.feature import hog
    from skimage import exposure
    
    # Load and convert to grayscale
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize for HOG computation
    img_resized = cv2.resize(img_gray, (224, 224))
    
    # Compute HOG features
    features, hog_image = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    
    # Rescale HOG image for visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return {
        'features': features,
        'hog_image': hog_image_rescaled,
        'feature_dim': len(features)
    }


def extract_handcrafted_features_sift(image_path):
    """
    Extract SIFT (Scale-Invariant Feature Transform) features.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    """
    # Load image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=100)
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    # Draw keypoints
    img_with_keypoints = cv2.drawKeypoints(
        img_gray, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'num_keypoints': len(keypoints),
        'image_with_keypoints': img_with_keypoints
    }


def extract_handcrafted_features_harris(image_path):
    """
    Extract Harris corner features.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    """
    # Load image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32
    img_float = np.float32(img_gray)
    
    # Apply Harris corner detection
    harris_response = cv2.cornerHarris(img_float, 2, 3, 0.04)
    
    # Threshold and find corners
    threshold = 0.01 * harris_response.max()
    corners = np.argwhere(harris_response > threshold)
    
    # Mark corners on image
    img_with_corners = img.copy()
    for corner in corners:
        cv2.circle(img_with_corners, (corner[1], corner[0]), 3, (0, 255, 0), -1)
    
    return {
        'harris_response': harris_response,
        'corners': corners,
        'num_corners': len(corners),
        'image_with_corners': img_with_corners
    }


def extract_cnn_features(model, image_tensor, layer_name='backbone'):
    """
    Extract learned features from CNN model.
    
    Parameters:
    -----------
    model : nn.Module
        Trained CNN model
    image_tensor : torch.Tensor
        Preprocessed image tensor
    layer_name : str
        Name of layer to extract features from
    """
    model.eval()
    
    # Hook to capture features
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hook
    if hasattr(model, 'backbone'):
        model.backbone.avgpool.register_forward_hook(get_features('features'))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Extract feature vector
    if 'features' in features:
        feature_vector = features['features'].squeeze().cpu().numpy()
        if len(feature_vector.shape) > 1:
            feature_vector = feature_vector.flatten()
        return feature_vector
    
    return None


# ============================================================================
# 5.3 CORE ALGORITHM / MODEL
# ============================================================================

def visualize_resnet50_architecture():
    """
    Create a visual diagram of ResNet50 architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'ResNet50 Architecture for Plant Classification', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((4, 10.5), 2, 0.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.75, 'Input Image\n224×224×3', ha='center', va='center', fontsize=10)
    
    # ResNet50 Backbone
    backbone_box = FancyBboxPatch((3, 8.5), 4, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(5, 9.5, 'ResNet50 Backbone\n(Pre-trained on ImageNet)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Backbone components
    components = [
        ('Conv + BN + ReLU', 9.2),
        ('MaxPool', 9.0),
        ('Layer1 (3 blocks)', 8.8),
        ('Layer2 (4 blocks)', 8.6),
        ('Layer3 (6 blocks)', 8.4),
        ('Layer4 (3 blocks)', 8.2),
    ]
    for comp, y_pos in components:
        ax.text(5, y_pos, comp, ha='center', va='center', fontsize=9)
    
    # Adaptive Average Pooling
    pool_box = FancyBboxPatch((4.2, 7.5), 1.6, 0.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(5, 7.75, 'AdaptiveAvgPool\n2048 features', ha='center', va='center', fontsize=9)
    
    # Classifier Head
    classifier_box = FancyBboxPatch((3.5, 5.5), 3, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(5, 6.5, 'Custom Classifier Head', ha='center', va='center', fontsize=11, fontweight='bold')
    
    classifier_components = [
        ('Dense(2048 → 512)', 6.2),
        ('ReLU + Dropout(0.3)', 6.0),
        ('Dense(512 → 102)', 5.8),
    ]
    for comp, y_pos in classifier_components:
        ax.text(5, y_pos, comp, ha='center', va='center', fontsize=9)
    
    # Output
    output_box = FancyBboxPatch((4, 4), 2, 0.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 4.25, 'Softmax Output\n102 Classes', ha='center', va='center', fontsize=10)
    
    # Arrows
    arrows = [
        ((5, 10.5), (5, 10)),
        ((5, 8.5), (5, 8)),
        ((5, 7.5), (5, 7)),
        ((5, 5.5), (5, 5)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Training phases annotation
    phase1_text = 'Phase 1: Frozen Backbone\nTrain Classifier Only'
    phase2_text = 'Phase 2: Unfrozen Backbone\nFine-tune Entire Network'
    
    ax.text(1, 9.5, phase1_text, ha='left', va='center', 
           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(1, 6.5, phase2_text, ha='left', va='center', 
           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig


def visualize_baseline_cnn_architecture():
    """
    Create a visual diagram of Baseline CNN architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Baseline CNN Architecture', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((4, 10.5), 2, 0.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.75, 'Input Image\n224×224×3', ha='center', va='center', fontsize=10)
    
    # Conv Blocks
    conv_blocks = [
        ('ConvBlock(64)', 9.5, '112×112×64'),
        ('ConvBlock(128)', 8.5, '56×56×128'),
        ('ConvBlock(256)', 7.5, '28×28×256'),
        ('ConvBlock(512)', 6.5, '14×14×512'),
    ]
    
    for i, (name, y_pos, size) in enumerate(conv_blocks):
        box = FancyBboxPatch((3.5, y_pos-0.3), 3, 0.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(5, y_pos, f'{name}\n{size}', ha='center', va='center', fontsize=9)
        
        if i < len(conv_blocks) - 1:
            arrow = FancyArrowPatch((5, y_pos-0.3), (5, y_pos+0.3), 
                                   arrowstyle='->', lw=2, color='black')
            ax.add_patch(arrow)
    
    # Global Average Pooling
    gap_box = FancyBboxPatch((4.2, 5.5), 1.6, 0.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(gap_box)
    ax.text(5, 5.75, 'Global Avg Pool\n512 features', ha='center', va='center', fontsize=9)
    
    # Classifier
    classifier_box = FancyBboxPatch((3.5, 4), 3, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(5, 4.5, 'Dense(512) → ReLU\nDropout(0.5)\nDense(102)', 
            ha='center', va='center', fontsize=9)
    
    # Output
    output_box = FancyBboxPatch((4, 2.5), 2, 0.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.75, 'Softmax Output\n102 Classes', ha='center', va='center', fontsize=10)
    
    # Arrows
    arrow1 = FancyArrowPatch((5, 5.5), (5, 5), arrowstyle='->', lw=2, color='black')
    arrow2 = FancyArrowPatch((5, 4), (5, 3), arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 5.4 POST-PROCESSING
# ============================================================================

def postprocess_thresholding(probabilities, threshold=0.5):
    """
    Apply thresholding to model predictions.
    
    Parameters:
    -----------
    probabilities : np.array
        Class probabilities from model
    threshold : float
        Confidence threshold
    """
    # Get predicted class
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    # Apply threshold
    if confidence < threshold:
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_confident': False,
            'message': 'Low confidence prediction'
        }
    else:
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_confident': True,
            'message': 'Confident prediction'
        }


def postprocess_top_k_predictions(probabilities, k=5):
    """
    Get top-k predictions.
    
    Parameters:
    -----------
    probabilities : np.array
        Class probabilities from model
    k : int
        Number of top predictions to return
    """
    # Get top-k indices
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    return {
        'top_k_classes': top_k_indices,
        'top_k_probabilities': top_k_probs,
        'top_k_pairs': list(zip(top_k_indices, top_k_probs))
    }


def postprocess_smoothing(predictions_sequence, window_size=5):
    """
    Apply temporal smoothing to prediction sequence.
    
    Parameters:
    -----------
    predictions_sequence : list
        Sequence of predictions
    window_size : int
        Smoothing window size
    """
    if len(predictions_sequence) < window_size:
        return predictions_sequence
    
    smoothed = []
    for i in range(len(predictions_sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions_sequence), i + window_size // 2 + 1)
        window = predictions_sequence[start:end]
        smoothed.append(np.bincount(window).argmax())
    
    return smoothed


# ============================================================================
# COMPLETE PIPELINE VISUALIZATION
# ============================================================================

def visualize_complete_pipeline():
    """
    Create a comprehensive diagram of the complete processing pipeline.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Complete Plant Classification Pipeline', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Stage 1: Preprocessing
    stage1_box = FancyBboxPatch((0.5, 7), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(2, 8, '1. Preprocessing', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(2, 7.5, '• Resize (224×224)\n• Normalization\n• Augmentation', 
           ha='center', va='center', fontsize=9)
    
    # Stage 2: Feature Extraction
    stage2_box = FancyBboxPatch((4.5, 7), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(6, 8, '2. Feature Extraction', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(6, 7.5, '• ResNet50 Backbone\n• 2048-dim features', 
           ha='center', va='center', fontsize=9)
    
    # Stage 3: Classification
    stage3_box = FancyBboxPatch((8.5, 7), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(10, 8, '3. Classification', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(10, 7.5, '• Dense(512)\n• Dense(102)\n• Softmax', 
           ha='center', va='center', fontsize=9)
    
    # Stage 4: Post-processing
    stage4_box = FancyBboxPatch((12.5, 7), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(14, 8, '4. Post-processing', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(14, 7.5, '• Top-k selection\n• Confidence check', 
           ha='center', va='center', fontsize=9)
    
    # Arrows between stages
    for x in [3.5, 7.5, 11.5]:
        arrow = FancyArrowPatch((x, 7.75), (x+1, 7.75), 
                               arrowstyle='->', lw=3, color='black')
        ax.add_patch(arrow)
    
    # Detailed preprocessing steps
    preprocess_steps = [
        ('Noise Reduction', 0.5, 5.5, 'Gaussian/Median/Bilateral'),
        ('Resize', 4.5, 5.5, '224×224'),
        ('Normalization', 8.5, 5.5, 'ImageNet stats'),
        ('Augmentation', 12.5, 5.5, 'Rotation/Flip/Color'),
    ]
    
    for name, x, y, desc in preprocess_steps:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor='lightcyan', edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, f'{name}\n{desc}', ha='center', va='center', fontsize=8)
    
    # Feature extraction details
    feature_details = [
        ('Conv Layers', 2, 3.5, 'Hierarchical features'),
        ('Residual Blocks', 6, 3.5, 'Skip connections'),
        ('Global Pooling', 10, 3.5, 'Spatial aggregation'),
    ]
    
    for name, x, y, desc in feature_details:
        box = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor='lightgreen', edgecolor='gray', linewidth=1, alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, f'{name}\n{desc}', ha='center', va='center', fontsize=8)
    
    # Output
    output_box = FancyBboxPatch((6.5, 1.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 2.25, 'Final Prediction', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(8, 1.75, 'Class: [0-101]\nConfidence: [0-1]', 
           ha='center', va='center', fontsize=9)
    
    # Arrow to output
    arrow = FancyArrowPatch((14, 7.5), (9.5, 2.5), 
                           arrowstyle='->', lw=3, color='black')
    ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating methodology diagrams...")
    
    import os
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = 'diagrams'
    os.makedirs(diagrams_dir, exist_ok=True)
    
    # Generate architecture diagrams
    print("1. Generating ResNet50 architecture diagram...")
    fig1 = visualize_resnet50_architecture()
    fig1.savefig(os.path.join(diagrams_dir, 'resnet50_architecture.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   Saved: {os.path.join(diagrams_dir, 'resnet50_architecture.png')}")
    plt.close(fig1)
    
    print("2. Generating Baseline CNN architecture diagram...")
    fig2 = visualize_baseline_cnn_architecture()
    fig2.savefig(os.path.join(diagrams_dir, 'baseline_cnn_architecture.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   Saved: {os.path.join(diagrams_dir, 'baseline_cnn_architecture.png')}")
    plt.close(fig2)
    
    print("3. Generating complete pipeline diagram...")
    fig3 = visualize_complete_pipeline()
    fig3.savefig(os.path.join(diagrams_dir, 'complete_pipeline.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   Saved: {os.path.join(diagrams_dir, 'complete_pipeline.png')}")
    plt.close(fig3)
    
    print("\nAll diagrams generated successfully!")
    print("\nCode examples are available in this file for:")
    print("  - Preprocessing (noise reduction, normalization, resizing)")
    print("  - Feature extraction (HOG, SIFT, Harris, CNN features)")
    print("  - Model architecture visualization")
    print("  - Post-processing (thresholding, top-k, smoothing)")

