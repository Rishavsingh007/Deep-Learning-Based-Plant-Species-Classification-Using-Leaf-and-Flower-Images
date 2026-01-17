"""
CT7160NI Computer Vision Coursework
Evaluation Metrics

This module implements comprehensive evaluation metrics for classification models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score
)
from tqdm import tqdm


@torch.no_grad()
def get_predictions(model, data_loader, device='cuda'):
    """
    Get model predictions for a dataset.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model
    data_loader : DataLoader
        Data loader for the dataset
    device : str
        Device to run inference on
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'predictions': Predicted class labels
        - 'probabilities': Class probabilities
        - 'true_labels': Ground truth labels
        - 'features': Feature embeddings (optional)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    pbar = tqdm(data_loader, desc='Getting predictions')
    
    for images, labels in pbar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'true_labels': np.array(all_labels)
    }


def calculate_metrics(predictions, true_labels, probabilities=None, num_classes=102):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    predictions : np.array
        Predicted class labels
    true_labels : np.array
        Ground truth labels
    probabilities : np.array, optional
        Class probabilities for ROC-AUC calculation
    num_classes : int
        Number of classes
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(true_labels, predictions) * 100
    
    # Precision, Recall, F1 (macro and micro)
    metrics['precision_macro'] = precision_score(
        true_labels, predictions, average='macro', zero_division=0
    ) * 100
    metrics['precision_micro'] = precision_score(
        true_labels, predictions, average='micro', zero_division=0
    ) * 100
    
    metrics['recall_macro'] = recall_score(
        true_labels, predictions, average='macro', zero_division=0
    ) * 100
    metrics['recall_micro'] = recall_score(
        true_labels, predictions, average='micro', zero_division=0
    ) * 100
    
    metrics['f1_macro'] = f1_score(
        true_labels, predictions, average='macro', zero_division=0
    ) * 100
    metrics['f1_micro'] = f1_score(
        true_labels, predictions, average='micro', zero_division=0
    ) * 100
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(
        true_labels, predictions, average=None, zero_division=0
    ) * 100
    metrics['recall_per_class'] = recall_score(
        true_labels, predictions, average=None, zero_division=0
    ) * 100
    metrics['f1_per_class'] = f1_score(
        true_labels, predictions, average=None, zero_division=0
    ) * 100
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions)
    
    # Top-k accuracy (if probabilities provided)
    if probabilities is not None:
        try:
            metrics['top_5_accuracy'] = top_k_accuracy_score(
                true_labels, probabilities, k=5
            ) * 100
        except:
            metrics['top_5_accuracy'] = None
        
        # ROC-AUC (one-vs-rest)
        try:
            metrics['roc_auc_macro'] = roc_auc_score(
                true_labels, probabilities, multi_class='ovr', average='macro'
            )
        except:
            metrics['roc_auc_macro'] = None
    
    # Classification report
    metrics['classification_report'] = classification_report(
        true_labels, predictions, zero_division=0
    )
    
    return metrics


def print_metrics_summary(metrics):
    """
    Print a formatted summary of evaluation metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    
    if metrics.get('top_5_accuracy'):
        print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")
    
    print(f"\nPrecision:")
    print(f"  Macro: {metrics['precision_macro']:.2f}%")
    print(f"  Micro: {metrics['precision_micro']:.2f}%")
    
    print(f"\nRecall:")
    print(f"  Macro: {metrics['recall_macro']:.2f}%")
    print(f"  Micro: {metrics['recall_micro']:.2f}%")
    
    print(f"\nF1-Score:")
    print(f"  Macro: {metrics['f1_macro']:.2f}%")
    print(f"  Micro: {metrics['f1_micro']:.2f}%")
    
    if metrics.get('roc_auc_macro'):
        print(f"\nROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    print("\n" + "="*60)


def analyze_errors(predictions, true_labels, probabilities, class_names=None, top_n=10):
    """
    Analyze classification errors to identify common mistakes.
    
    Parameters:
    -----------
    predictions : np.array
        Predicted class labels
    true_labels : np.array
        Ground truth labels
    probabilities : np.array
        Class probabilities
    class_names : list, optional
        Names of classes
    top_n : int
        Number of top error pairs to analyze
        
    Returns:
    --------
    dict : Error analysis results
    """
    # Find misclassified samples
    errors = predictions != true_labels
    error_indices = np.where(errors)[0]
    
    # Count confusion pairs
    from collections import Counter
    confusion_pairs = Counter()
    
    for idx in error_indices:
        pair = (true_labels[idx], predictions[idx])
        confusion_pairs[pair] += 1
    
    # Get top confusion pairs
    top_confusions = confusion_pairs.most_common(top_n)
    
    # Calculate per-class error rates
    unique_classes = np.unique(true_labels)
    error_rates = {}
    
    for cls in unique_classes:
        cls_mask = true_labels == cls
        cls_errors = (predictions[cls_mask] != cls).sum()
        cls_total = cls_mask.sum()
        error_rates[cls] = (cls_errors / cls_total * 100) if cls_total > 0 else 0
    
    # Find most confident errors
    error_confidences = []
    for idx in error_indices:
        confidence = probabilities[idx, predictions[idx]]
        error_confidences.append((idx, confidence, true_labels[idx], predictions[idx]))
    
    # Sort by confidence (highest first - most confident errors)
    error_confidences.sort(key=lambda x: x[1], reverse=True)
    
    results = {
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(true_labels) * 100,
        'top_confusion_pairs': top_confusions,
        'per_class_error_rates': error_rates,
        'most_confident_errors': error_confidences[:top_n]
    }
    
    return results


def calculate_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """
    Calculate average inference time for the model.
    
    Parameters:
    -----------
    model : nn.Module
        The model to benchmark
    input_size : tuple
        Input tensor size
    device : str
        Device to run on
    num_runs : int
        Number of runs for averaging
        
    Returns:
    --------
    dict : Inference time statistics
    """
    import time
    
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.time() - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }


# Example usage
if __name__ == "__main__":
    print("Testing metrics module...")
    
    # Create dummy data
    np.random.seed(42)
    num_samples = 100
    num_classes = 10
    
    true_labels = np.random.randint(0, num_classes, num_samples)
    predictions = true_labels.copy()
    # Add some errors
    error_indices = np.random.choice(num_samples, 20, replace=False)
    predictions[error_indices] = np.random.randint(0, num_classes, 20)
    
    probabilities = np.random.rand(num_samples, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels, probabilities, num_classes)
    print_metrics_summary(metrics)
    
    # Analyze errors
    error_analysis = analyze_errors(predictions, true_labels, probabilities)
    print(f"\nError Analysis:")
    print(f"  Total errors: {error_analysis['total_errors']}")
    print(f"  Error rate: {error_analysis['error_rate']:.2f}%")
    
    print("\nMetrics module tests passed!")

