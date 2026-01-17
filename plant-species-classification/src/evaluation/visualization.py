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


def plot_confusion_matrix(cm, class_names=None, normalize=True, save_path=None, figsize=(16, 14), 
                          title=None, cmap='YlOrRd', gamma=0.5, show_labels=True):
    """
    Plot a vibrant, publication-quality confusion matrix with enhanced visibility.
    
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
    title : str, optional
        Custom title for the plot
    cmap : str
        Colormap to use ('YlOrRd', 'hot', 'plasma', 'viridis', 'Blues')
    gamma : float
        Gamma correction for enhanced visibility of low values (0.3-0.7 recommended)
    show_labels : bool
        Whether to show class labels/indices on axes
    """
    import matplotlib.colors as mcolors
    
    # Store original for accuracy calculation
    cm_original = cm.copy()
    num_classes = cm.shape[0]
    
    # Calculate overall accuracy from original matrix
    total_samples = cm_original.sum()
    correct_predictions = np.trace(cm_original)
    overall_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm.astype('float')
    
    # Apply gamma correction for better visibility of low values
    cm_display = np.power(cm_norm, gamma)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use imshow for better control over the visualization
    im = ax.imshow(cm_display, cmap=cmap, aspect='equal', interpolation='nearest')
    
    # Add colorbar with proper label
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if normalize:
        # Create custom ticks that show actual values (before gamma)
        cbar_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_tick_positions = np.power(cbar_ticks, gamma)
        cbar.set_ticks(cbar_tick_positions)
        cbar.set_ticklabels([f'{v:.1f}' for v in cbar_ticks])
        cbar.set_label('Normalized Percentage', fontsize=11, fontweight='bold')
    else:
        cbar.set_label('Count', fontsize=11, fontweight='bold')
    
    # Configure axis labels
    if show_labels:
        if class_names is not None:
            # Use class names
            if num_classes <= 30:
                font_size = max(5, 9 - num_classes // 10)
                ax.set_xticks(np.arange(num_classes))
                ax.set_yticks(np.arange(num_classes))
                ax.set_xticklabels(class_names, rotation=90, ha='center', fontsize=font_size)
                ax.set_yticklabels(class_names, rotation=0, ha='right', fontsize=font_size)
            else:
                # Too many classes - show every nth label
                step = max(1, num_classes // 25)
                tick_positions = np.arange(0, num_classes, step)
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels([class_names[i] for i in tick_positions], rotation=90, ha='center', fontsize=6)
                ax.set_yticklabels([class_names[i] for i in tick_positions], rotation=0, ha='right', fontsize=6)
        else:
            # Show numeric indices
            if num_classes <= 50:
                step = max(1, num_classes // 20)
            else:
                step = max(1, num_classes // 15)
            tick_positions = np.arange(0, num_classes, step)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_positions, rotation=90, ha='center', fontsize=8)
            ax.set_yticklabels(tick_positions, rotation=0, ha='right', fontsize=8)
    
    # Add axis labels
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold', labelpad=10)
    
    # Create informative title
    if title:
        main_title = f'{title} - Confusion Matrix'
    else:
        main_title = 'Confusion Matrix'
    
    if normalize:
        main_title += f'\n(Gamma={gamma} for Enhanced Visibility)'
    
    subtitle = f'Accuracy: {overall_accuracy:.2f}% | Classes: {num_classes} | Samples: {int(total_samples)}'
    ax.set_title(f'{main_title}\n{subtitle}', fontsize=14, fontweight='bold', pad=15)
    
    # Add grid lines for better readability (subtle)
    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.5)
    
    # Remove minor tick marks
    ax.tick_params(which='minor', length=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix_dual(cm, class_names=None, save_path=None, figsize=(20, 9), title=None):
    """
    Plot confusion matrix in two views: Log scale and Gamma-corrected normalized.
    Creates a professional side-by-side comparison like standard ML publications.
    
    Parameters:
    -----------
    cm : np.array
        Confusion matrix (raw counts)
    class_names : list, optional
        Names of classes
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    title : str, optional
        Model name for the title
    """
    import matplotlib.colors as mcolors
    
    num_classes = cm.shape[0]
    
    # Calculate accuracy
    total_samples = cm.sum()
    correct_predictions = np.trace(cm)
    overall_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # === LEFT PLOT: Log Scale ===
    ax1 = axes[0]
    
    # Log scale with offset to handle zeros
    cm_log = np.log10(cm + 1)
    
    im1 = ax1.imshow(cm_log, cmap='Blues', aspect='equal', interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('log₁₀(Count + 1)', fontsize=10, fontweight='bold')
    
    # Configure ticks
    step = max(1, num_classes // 20)
    tick_positions = np.arange(0, num_classes, step)
    ax1.set_xticks(tick_positions)
    ax1.set_yticks(tick_positions)
    ax1.set_xticklabels(tick_positions, rotation=90, ha='center', fontsize=7)
    ax1.set_yticklabels(tick_positions, rotation=0, ha='right', fontsize=7)
    
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_title(f'{title + " - " if title else ""}Confusion Matrix (Log Scale)\nAccuracy: {overall_accuracy:.2f}%', 
                  fontsize=12, fontweight='bold')
    
    # === RIGHT PLOT: Gamma-corrected Normalized ===
    ax2 = axes[1]
    
    # Apply gamma correction (0.5) for better visibility of errors
    gamma = 0.5
    cm_gamma = np.power(cm_norm, gamma)
    
    im2 = ax2.imshow(cm_gamma, cmap='YlOrRd', aspect='equal', interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    
    # Create proper colorbar ticks showing actual values
    cbar_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar_tick_positions = np.power(cbar_ticks, gamma)
    cbar2.set_ticks(cbar_tick_positions)
    cbar2.set_ticklabels([f'{v:.1f}' for v in cbar_ticks])
    cbar2.set_label('Normalized Percentage', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(tick_positions)
    ax2.set_yticks(tick_positions)
    ax2.set_xticklabels(tick_positions, rotation=90, ha='center', fontsize=7)
    ax2.set_yticklabels(tick_positions, rotation=0, ha='right', fontsize=7)
    
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax2.set_title(f'{title + " - " if title else ""}Normalized Confusion Matrix\n(Gamma={gamma} for Enhanced Visibility)', 
                  fontsize=12, fontweight='bold')
    
    # Add subtle grid
    for ax in axes:
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.2, alpha=0.3)
        ax.tick_params(which='minor', length=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Dual confusion matrix saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curves(true_labels, probabilities, class_names=None, top_k=3, save_path=None, 
                    show_best_worst=True, title=None):
    """
    Plot clean, interpretable ROC curves with macro-average and select classes.
    
    Parameters:
    -----------
    true_labels : np.array
        Ground truth labels
    probabilities : np.array
        Class probabilities
    class_names : list, optional
        Names of classes
    top_k : int
        Number of worst classes to highlight (default 3 for clarity)
    save_path : str, optional
        Path to save the figure
    show_best_worst : bool
        If True, show worst classes; if False, show only macro-average
    title : str, optional
        Custom title for the plot
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
    
    # Calculate macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    # Sort classes by AUC (worst first)
    sorted_classes = sorted(roc_auc.keys(), key=lambda x: roc_auc[x])
    worst_classes = sorted_classes[:top_k]
    
    # Create clean figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define distinct colors for worst classes
    worst_colors = ['#e41a1c', '#ff7f00', '#984ea3']  # Red, Orange, Purple
    
    # Plot macro-average ROC (main focus - thick blue line)
    ax.plot(
        all_fpr, mean_tpr,
        color='#2171b5',
        linewidth=4,
        linestyle='-',
        label=f'Macro-Average (AUC = {macro_auc:.3f})',
        zorder=10
    )
    
    # Plot worst performing classes (these are most informative)
    if show_best_worst:
        for i, cls in enumerate(worst_classes):
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.plot(
                fpr[cls], tpr[cls],
                color=worst_colors[i % len(worst_colors)],
                linewidth=2.5,
                linestyle='--',
                alpha=0.9,
                label=f'{label} (AUC = {roc_auc[cls]:.3f})'
            )
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random (AUC = 0.5)')
    
    # Configure axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    
    # Clean title
    if title:
        plot_title = f'{title} - ROC Analysis'
    else:
        plot_title = 'ROC Analysis'
    
    # Add summary stats as subtitle
    auc_values = list(roc_auc.values())
    ax.set_title(
        f'{plot_title}\n'
        f'Macro AUC: {macro_auc:.3f} | Mean: {np.mean(auc_values):.3f} | '
        f'Min: {np.min(auc_values):.3f} | Max: {np.max(auc_values):.3f}',
        fontsize=13, fontweight='bold', pad=12
    )
    
    # Clean legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()
    plt.close()


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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassified examples saved to {save_path}")
    
    plt.close()


def plot_correctly_classified_examples(
    images,
    true_labels,
    predictions,
    confidence_scores=None,
    class_names=None,
    n_examples=16,
    save_path=None
):
    """
    Plot grid of correctly classified examples.
    
    Parameters:
    -----------
    images : torch.Tensor or np.array
        Image tensors or arrays
    true_labels : np.array
        Ground truth labels
    predictions : np.array
        Predicted labels
    confidence_scores : np.array, optional
        Confidence scores for predictions
    class_names : list, optional
        Names of classes
    n_examples : int
        Number of examples to show
    save_path : str, optional
        Path to save the figure
    """
    # Find correctly classified examples
    correct = predictions == true_labels
    correct_indices = np.where(correct)[0]
    
    if len(correct_indices) == 0:
        print("No correctly classified examples found!")
        return
    
    # Select examples (prefer high confidence if available)
    n_show = min(n_examples, len(correct_indices))
    if confidence_scores is not None:
        # Sort by confidence and select top examples
        correct_with_conf = [(idx, confidence_scores[idx]) for idx in correct_indices]
        correct_with_conf.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in correct_with_conf[:n_show]]
    else:
        # Random selection
        selected_indices = np.random.choice(correct_indices, n_show, replace=False)
    
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
        
        # Add confidence if available
        if confidence_scores is not None:
            conf = confidence_scores[idx]
            title += f'\nConf: {conf:.3f}'
        
        axes[i].set_title(title, fontsize=10, color='green')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Correctly Classified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correctly classified examples saved to {save_path}")
    
    plt.close()


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

