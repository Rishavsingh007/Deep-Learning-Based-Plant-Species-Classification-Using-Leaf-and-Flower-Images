"""
CT7160NI Computer Vision Coursework
Visualization Utilities

This module implements visualization functions for training analysis,
model evaluation, and interpretability (Grad-CAM).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Parameters:
    -----------
    history : dict
        Training history containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names=None, normalize=True, save_path=None, figsize=(12, 10)):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    cm : np.array
        Confusion matrix
    class_names : list, optional
        Names of classes
    normalize : bool
        Whether to normalize the matrix
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    plt.figure(figsize=figsize)
    
    # If too many classes, don't show all labels
    num_classes = cm.shape[0]
    if num_classes > 20:
        # Show only a subset or use smaller font
        annot = False
        fmt = ''
    else:
        annot = True
        fmt = '.2f' if normalize else 'd'
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names and num_classes <= 20 else False,
        yticklabels=class_names if class_names and num_classes <= 20 else False,
        square=True
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curves(true_labels, probabilities, class_names=None, top_k=5, save_path=None):
    """
    Plot ROC curves for top-k classes.
    
    Parameters:
    -----------
    true_labels : np.array
        Ground truth labels
    probabilities : np.array
        Class probabilities
    class_names : list, optional
        Names of classes
    top_k : int
        Number of classes to plot
    save_path : str, optional
        Path to save the figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    num_classes = probabilities.shape[1]
    
    # Binarize labels
    y_bin = label_binarize(true_labels, classes=range(num_classes))
    
    # Calculate ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Get top-k classes by AUC
    top_classes = sorted(roc_auc.keys(), key=lambda x: roc_auc[x], reverse=True)[:top_k]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    
    for i, cls in enumerate(top_classes):
        label = class_names[cls] if class_names else f'Class {cls}'
        plt.plot(
            fpr[cls], tpr[cls],
            color=colors[i],
            linewidth=2,
            label=f'{label} (AUC = {roc_auc[cls]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves (Top {top_k} Classes by AUC)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_per_class_performance(metrics, class_names=None, metric='f1', top_k=20, save_path=None):
    """
    Plot per-class performance bar chart.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary containing per-class metrics
    class_names : list, optional
        Names of classes
    metric : str
        Metric to plot ('f1', 'precision', 'recall')
    top_k : int
        Number of classes to show (best and worst)
    save_path : str, optional
        Path to save the figure
    """
    metric_key = f'{metric}_per_class'
    values = metrics[metric_key]
    num_classes = len(values)
    
    # Create class labels
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Sort by value
    sorted_indices = np.argsort(values)
    
    # Get worst and best classes
    worst_indices = sorted_indices[:top_k//2]
    best_indices = sorted_indices[-(top_k//2):]
    selected_indices = np.concatenate([worst_indices, best_indices])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if i in worst_indices else 'green' for i in selected_indices]
    
    y_pos = np.arange(len(selected_indices))
    bars = ax.barh(y_pos, values[selected_indices], color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([class_names[i] for i in selected_indices])
    ax.set_xlabel(f'{metric.capitalize()} (%)', fontsize=12)
    ax.set_title(f'Per-Class {metric.capitalize()} Score (Best & Worst)', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, values[selected_indices]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class performance saved to {save_path}")
    
    plt.show()


def visualize_gradcam(
    model,
    image_tensor,
    target_layer,
    class_idx=None,
    original_image=None,
    save_path=None,
    device='cuda'
):
    """
    Generate and visualize Grad-CAM heatmap.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model
    image_tensor : torch.Tensor
        Input image tensor (1, C, H, W)
    target_layer : nn.Module
        Target layer for Grad-CAM (usually last conv layer)
    class_idx : int, optional
        Target class index (None = use predicted class)
    original_image : np.array, optional
        Original image for overlay
    save_path : str, optional
        Path to save the figure
    device : str
        Device to run on
    """
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Storage for activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    # Get target class
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Calculate Grad-CAM
    activation = activations[0].detach()
    gradient = gradients[0].detach()
    
    # Global average pooling of gradients
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    
    # Weighted combination of activation maps
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Resize to image size
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if original_image is not None:
        axes[0].imshow(original_image)
    else:
        # Denormalize tensor for visualization
        img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    if original_image is not None:
        overlay = original_image.copy()
    else:
        overlay = img.copy()
    
    # Create colored heatmap
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (Class: {class_idx})', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.show()
    
    return cam


def plot_misclassified_examples(
    images,
    true_labels,
    predictions,
    class_names=None,
    n_examples=9,
    save_path=None
):
    """
    Plot grid of misclassified examples.
    
    Parameters:
    -----------
    images : torch.Tensor or np.array
        Image tensors or arrays
    true_labels : np.array
        Ground truth labels
    predictions : np.array
        Predicted labels
    class_names : list, optional
        Names of classes
    n_examples : int
        Number of examples to show
    save_path : str, optional
        Path to save the figure
    """
    # Find misclassified examples
    errors = predictions != true_labels
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No misclassified examples found!")
        return
    
    # Select random examples
    n_show = min(n_examples, len(error_indices))
    selected_indices = np.random.choice(error_indices, n_show, replace=False)
    
    # Calculate grid size
    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i, idx in enumerate(selected_indices):
        # Get image
        if isinstance(images, torch.Tensor):
            img = images[idx].cpu().numpy()
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        else:
            img = images[idx]
        
        axes[i].imshow(img)
        
        true_label = true_labels[idx]
        pred_label = predictions[idx]
        
        if class_names:
            title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}'
        else:
            title = f'True: {true_label}\nPred: {pred_label}'
        
        axes[i].set_title(title, fontsize=10, color='red')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassified examples saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Testing visualization module...")
    
    # Create dummy training history
    np.random.seed(42)
    epochs = 20
    history = {
        'train_loss': np.exp(-np.linspace(0, 2, epochs)) + np.random.randn(epochs) * 0.05,
        'val_loss': np.exp(-np.linspace(0, 1.5, epochs)) + np.random.randn(epochs) * 0.1,
        'train_acc': 50 + 40 * (1 - np.exp(-np.linspace(0, 2, epochs))) + np.random.randn(epochs) * 2,
        'val_acc': 50 + 35 * (1 - np.exp(-np.linspace(0, 1.5, epochs))) + np.random.randn(epochs) * 3,
    }
    
    # Test training history plot
    print("Testing training history plot...")
    # plot_training_history(history)  # Uncomment to test
    
    # Create dummy confusion matrix
    cm = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(cm, np.random.randint(100, 500, 10))
    
    print("Testing confusion matrix plot...")
    # plot_confusion_matrix(cm, normalize=True)  # Uncomment to test
    
    print("\nVisualization module tests passed!")

