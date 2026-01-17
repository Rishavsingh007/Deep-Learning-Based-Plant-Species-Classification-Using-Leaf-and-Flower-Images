# Training Strategy: Model Training Implementation

**CT7160NI Computer Vision Coursework**  
**Plant Species Classification - Actual Training Implementation**

---

## Overview

This document describes the **actual training strategy** implemented for the plant species classification project. The project implemented three model architectures: a custom Baseline CNN, ResNet50 with transfer learning, and EfficientNet-B3 with transfer learning, achieving excellent classification performance on the Oxford 102 Flower Dataset.

---

## Training Approach

### Implemented Models

Three model architectures were trained and evaluated:

| Model | Architecture Type | Pre-trained | Image Size | Parameters |
|-------|------------------|-------------|------------|------------|
| **Baseline CNN** | Custom CNN from scratch | No | 224×224 | 11.9M |
| **ResNet50** | Transfer Learning | ImageNet | 224×224 | 24.6M |
| **EfficientNet-B3** | Transfer Learning | ImageNet | 300×300 | 11.1M |

### Training Strategy Summary

**Key Implementation Decisions:**
- **No background removal**: Models trained on original images without segmentation mask preprocessing
- **Two-phase training for ResNet50**: Frozen backbone followed by fine-tuning
- **Extended training for Baseline CNN**: 150 epochs to ensure convergence from scratch
- **Class imbalance handling**: Weighted sampling and class weights used for Baseline CNN

---

## 1. Baseline CNN Training

### Architecture
Custom 4-layer CNN trained from scratch:
- 4 Convolutional layers (64, 128, 256, 512 channels)
- Batch Normalization after each conv layer
- Global Average Pooling
- Fully connected layers with Dropout (0.5)
- Output layer: 102 classes

### Training Configuration

```python
Configuration:
  Epochs: 150
  Learning Rate: 5e-4 (0.0005)
  Optimizer: Adam
  Weight Decay: 1e-4
  Batch Size: 16
  Image Size: 224×224
  Loss Function: CrossEntropyLoss (with class weights)
  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
  Early Stopping: Patience=10
  Class Imbalance Handling: 
    - Weighted Random Sampling: Enabled
    - Class Weights in Loss: Enabled
  Data Augmentation: Albumentations (rotation, flip, color jitter)
```

### Training Details

**Why 150 Epochs?**
- Training from scratch requires more epochs to learn features
- Early stopping with patience=10 prevented overfitting
- Best validation accuracy achieved at epoch 149

**Class Imbalance Handling:**
- **Weighted Random Sampling**: Ensures balanced batch sampling across classes
- **Class Weights in Loss**: Inverse frequency weighting to handle imbalanced classes
- Formula: `class_weights = 1.0 / class_counts` (normalized)

**Learning Rate Strategy:**
- Initial LR: 5e-4 (higher than transfer learning models)
- Scheduler: ReduceLROnPlateau reduces LR by factor 0.5 when validation loss plateaus
- Lower learning rate appropriate for training from scratch

### Training Results

**Best Performance:**
- **Best Validation Accuracy**: 85.83% (Epoch 149)
- **Best Validation Loss**: 0.443479
- **Final Validation Accuracy**: 83.55%
- **Final Training Accuracy**: 83.41%
- **Test Accuracy**: 86.41% ⭐
- **Test Top-5 Accuracy**: 97.80%
- **Training Time**: ~2-3 hours (150 epochs on GPU)

**Key Observations:**
- Model required extended training to converge
- Validation accuracy plateaued around epoch 140-150
- Top-5 accuracy (97.80%) significantly higher than top-1, indicating good class separation

---

## 2. ResNet50 Training (Transfer Learning)

### Architecture
- Pre-trained ResNet50 backbone (ImageNet weights)
- Replaced final classification layer (2048 → 256 → 102)
- Two-phase training strategy

### Training Configuration

```python
Configuration:
  Phase 1 (Frozen Backbone):
    Epochs: 10-15 (estimated)
    Learning Rate: 1e-3
    Backbone: Frozen
    Trainable Parameters: ~2M (classification head only)
  
  Phase 2 (Fine-tuning):
    Epochs: 30-35 (estimated)
    Learning Rate: 1e-4 to 1e-5 (gradual reduction)
    Backbone: Unfrozen (all layers trainable)
    Trainable Parameters: ~24.6M (all layers)
  
  Total Epochs: 44
  Batch Size: 16-32
  Image Size: 224×224
  Optimizer: Adam
  Scheduler: ReduceLROnPlateau
  Early Stopping: Enabled
```

### Two-Phase Training Strategy

**Phase 1: Frozen Backbone**
- Freeze all ResNet50 backbone layers
- Train only the newly added classification head
- Higher learning rate (1e-3) for faster learning of new task-specific features
- Purpose: Initialize classification head with task-appropriate weights

**Phase 2: Fine-tuning**
- Unfreeze all backbone layers
- Lower learning rate (1e-4 or less) to fine-tune pre-trained features
- Gradual learning rate reduction via scheduler
- Purpose: Adapt ImageNet features to flower classification task

**Rationale:**
- Prevents destruction of useful ImageNet features during initial training
- Allows classification head to learn first, then adapts backbone features
- Standard transfer learning best practice

### Training Results

**Best Performance:**
- **Best Validation Accuracy**: 98.53% (Epoch 34)
- **Best Validation Loss**: 0.069528 (Epoch 40)
- **Final Validation Accuracy**: 98.37%
- **Final Training Accuracy**: 99.62%
- **Test Accuracy**: 97.97% 
- **Test Top-5 Accuracy**: 99.51%
- **ROC-AUC (Macro)**: 0.9994
- **Training Time**: ~1-2 hours (44 epochs on GPU)

**Key Observations:**
- Rapid convergence due to pre-trained weights
- Excellent generalization (small gap between train and validation accuracy)
- Near-perfect top-5 accuracy (99.51%) indicates strong discriminative power

---

## 3. EfficientNet-B3 Training (Transfer Learning)

### Architecture
- Pre-trained EfficientNet-B3 backbone (ImageNet weights)
- Compound scaling (width, depth, resolution)
- MBConv blocks with Squeeze-and-Excitation (SE) attention
- Global Average Pooling + classification head (256 → 102)

### Training Configuration

```python
Configuration:
  Epochs: 50
  Learning Rate: 1e-4 (initial, adaptive scheduling)
  Optimizer: Adam
  Weight Decay: 1e-4
  Batch Size: 16-32 (adjusted for 300×300 images)
  Image Size: 300×300 (larger than other models)
  Loss Function: CrossEntropyLoss
  Scheduler: ReduceLROnPlateau
  Early Stopping: Enabled (patience=10)
  Dropout: 0.3
```

### Training Details

**Higher Resolution (300×300):**
- EfficientNet-B3 benefits from larger input resolution
- Compound scaling includes resolution scaling
- Requires more GPU memory but improves performance

**Efficient Architecture:**
- Fewer parameters (11.1M) than ResNet50 (24.6M) but similar/higher accuracy
- MBConv blocks with depthwise separable convolutions
- SE attention mechanism for adaptive feature recalibration

### Training Results

**Best Performance:**
- **Best Validation Accuracy**: 99.19% (Epoch ~47-50)
- **Best Validation Loss**: 0.031144
- **Final Validation Accuracy**: 98.94%
- **Test Accuracy**: 98.94%  **Best Performance**
- **Test Top-5 Accuracy**: 99.76%
- **Precision (Macro)**: 99.03%
- **Recall (Macro)**: 98.82%
- **F1-Score (Macro)**: 98.86%
- **ROC-AUC (Macro)**: 0.9993
- **Training Time**: ~2-3 hours (50 epochs on GPU)

**Key Observations:**
- **Highest accuracy** among all models (98.94%)
- Best accuracy-to-parameter ratio (98.94% with only 11.1M parameters)
- Excellent per-class performance (high macro-averaged metrics)
- Top-5 accuracy near perfect (99.76%)

---

## Data Preprocessing & Augmentation

### Preprocessing Pipeline

**On-the-Fly Processing:**
- Images loaded directly from `data/raw/oxford_flowers_102/102flowers/jpg/`
- No manual preprocessing required before training
- All transformations applied during data loading

**Standard Preprocessing:**
- **Resize**: To target size (224×224 or 300×300)
- **Normalization**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Tensor Conversion**: PyTorch tensors (float32)

### Data Augmentation (Training Only)

**Albumentations Library:**
- **Random Horizontal Flip**: 50% probability
- **Random Rotation**: ±15 degrees
- **Color Jitter**: Brightness, contrast, saturation variations
- **Random Resized Crop**: Scale variations for robustness

**Validation/Test:**
- **No augmentation**: Only resize and normalization
- Ensures fair evaluation on original images

---

## Training Infrastructure

### Hardware Configuration
- **GPU**: CUDA-compatible GPU (recommended)
- **Batch Size**: Adjusted based on GPU memory (16-32)
- **Mixed Precision**: Not used (standard float32 training)

### Software Framework
- **Deep Learning**: PyTorch 2.0+
- **Data Processing**: torchvision, Albumentations
- **Utilities**: NumPy, Matplotlib, scikit-learn

### Training Utilities

**Trainer Class** (`src/training/trainer.py`):
- Centralized training loop
- Automatic checkpointing
- Learning rate scheduling
- Early stopping
- Metric logging

**Callbacks** (`src/training/callbacks.py`):
- Model checkpoint saving (best model based on validation accuracy)
- Early stopping (patience-based)
- Learning rate scheduling

---

## Model Comparison & Results

### Performance Summary

| Model | Test Accuracy | Top-5 Acc | Precision | Recall | F1-Score | ROC-AUC | Parameters | Inference Time |
|-------|---------------|-----------|-----------|--------|----------|---------|------------|----------------|
| **Baseline CNN** | 86.41% | 97.80% | 86.99% | 88.95% | 86.88% | 0.997 | 11.9M | 29.87 ms |
| **ResNet50** | 97.97% | 99.51% | 97.62% | 97.53% | 97.43% | 0.999 | 24.6M | 20.82 ms |
| **EfficientNet-B3** | **98.94%** ⭐ | **99.76%** | **99.03%** | **98.82%** | **98.86%** | 0.999 | **11.1M** | 24.10 ms |

### Key Findings

1. **Transfer Learning Advantage:**
   - ResNet50 and EfficientNet-B3 outperform Baseline CNN by 11-12%
   - Pre-trained ImageNet weights provide strong feature representations
   - Significant performance gain without additional data

2. **Efficiency vs. Performance:**
   - EfficientNet-B3 achieves highest accuracy with fewer parameters
   - Best accuracy-to-parameter ratio (98.94% with 11.1M params)
   - Demonstrates effectiveness of compound scaling

3. **Training Time Comparison:**
   - Baseline CNN: ~2-3 hours (150 epochs, from scratch)
   - ResNet50: ~1-2 hours (44 epochs, transfer learning)
   - EfficientNet-B3: ~2-3 hours (50 epochs, transfer learning)
   - Transfer learning models converge faster

4. **Inference Performance:**
   - ResNet50: Fastest (20.82 ms/image)
   - EfficientNet-B3: Acceptable (24.10 ms/image) for best accuracy
   - All models suitable for real-time applications

---

## Training Workflow

### 1. Data Preparation
```python
# Verify data structure
data/raw/oxford_flowers_102/
├── 102flowers/jpg/          # All images
├── imagelabels.mat          # Labels
└── setid.mat               # Train/val/test splits
```

### 2. Model Training

**Option A: Training Scripts**
```bash
# Baseline CNN
python train_baseline_cnn_no_masks.py

# ResNet50
python train_resnet50.py
```

**Option B: Jupyter Notebook**
- Run `03_model_training.ipynb` for centralized training
- All models can be trained in sequence

### 3. Model Evaluation

**Comprehensive Evaluation:**
- Run `04_model_evaluation.ipynb`
- Generates metrics, confusion matrices, ROC curves
- Per-class performance analysis

**Error Analysis:**
- Run `05_inference_error_analysis.ipynb`
- Misclassification patterns
- Grad-CAM visualizations

---

## Best Practices Implemented

### 1. Early Stopping
- Prevents overfitting
- Stops training when validation loss plateaus
- Saves best model checkpoint

### 2. Learning Rate Scheduling
- ReduceLROnPlateau adapts to training progress
- Gradual reduction prevents overshooting optima
- Different strategies for different model types

### 3. Model Checkpointing
- Save best model based on validation accuracy
- Resume training from checkpoints if interrupted
- Model versioning for reproducibility

### 4. Class Imbalance Handling (Baseline CNN)
- Weighted sampling ensures balanced batches
- Class weights in loss function
- Important for 102-class classification

### 5. Data Augmentation
- Improves generalization
- Reduces overfitting
- Increases effective dataset size

---

## Lessons Learned

### What Worked Well

1. **Transfer Learning**: Dramatic performance improvement (11-12% accuracy gain)
2. **Two-Phase Training**: Effective for ResNet50 fine-tuning
3. **Extended Training**: Baseline CNN benefited from 150 epochs
4. **Class Imbalance Handling**: Weighted sampling improved baseline performance
5. **Higher Resolution**: EfficientNet-B3 benefited from 300×300 input

### Challenges Overcome

1. **Training from Scratch**: Baseline CNN required careful hyperparameter tuning
2. **GPU Memory**: Batch size adjusted based on model and image size
3. **Convergence**: Learning rate scheduling crucial for stable training
4. **Overfitting**: Early stopping and dropout prevented overfitting

---

## Conclusion

The implemented training strategy successfully achieved excellent classification performance:

- **Baseline CNN**: 86.41% accuracy (strong baseline from scratch)
- **ResNet50**: 97.97% accuracy (excellent transfer learning performance)
- **EfficientNet-B3**: **98.94% accuracy** (best performance, most efficient)

The strategy demonstrates the effectiveness of:
- Transfer learning for multi-class classification
- Two-phase training for fine-tuning
- Appropriate hyperparameter selection
- Proper handling of class imbalance
- Effective data augmentation

All models achieved strong top-5 accuracy (>97%), indicating good class separation and practical usability for real-world applications.

---

**Document Version**: 2.0 (Based on Actual Implementation)  
**Last Updated**: January 2026
**Author**: Rishav Singh (NP01MS7A240010)
