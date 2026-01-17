# Model Comparison Analysis: Baseline CNN vs ResNet50 vs EfficientNet-B3

## Executive Summary

This document provides a comprehensive comparison of three model architectures implemented for plant species classification on the Oxford 102 Flower Dataset:

1. **Baseline CNN** - Custom CNN trained from scratch
2. **ResNet50** - Transfer learning with ImageNet pre-trained weights
3. **EfficientNet-B3** - Transfer learning with compound scaling

**Key Finding:** EfficientNet-B3 achieves the best performance with **98.94% test accuracy**, outperforming ResNet50 (97.97%) and Baseline CNN (86.41%). Transfer learning models demonstrate a **12.5% absolute improvement** over the baseline, with EfficientNet-B3 achieving the best accuracy-to-parameter ratio.

---

## 1. Performance Metrics Comparison

### 1.1 Test Set Performance (Final Evaluation)

| Metric | Baseline CNN | ResNet50 | EfficientNet-B3 | Best Model |
|--------|--------------|----------|-----------------|------------|
| **Test Accuracy** | 86.41% | 97.97% | **98.94%**  | EfficientNet-B3 |
| **Top-5 Accuracy** | 97.80% | 99.51% | **99.76%**  | EfficientNet-B3 |
| **Precision (Macro)** | 86.99% | 97.62% | **99.03%**  | EfficientNet-B3 |
| **Recall (Macro)** | 88.95% | 97.53% | **98.82%**  | EfficientNet-B3 |
| **F1-Score (Macro)** | 86.88% | 97.43% | **98.86%**  | EfficientNet-B3 |
| **ROC-AUC (Macro)** | 0.9971 | 0.9994 | 0.9993 | ResNet50 |
| **Cohen's Kappa** | 0.8624 | 0.9794 | **0.9893**  | EfficientNet-B3 |
| **Avg Precision (Macro)** | 0.9446 | 0.9945 | **0.9980**  | EfficientNet-B3 |

### 1.2 Validation Performance (Training)

| Metric | Baseline CNN | ResNet50 | EfficientNet-B3 |
|--------|--------------|----------|-----------------|
| **Best Val Accuracy** | 85.83% | 98.53% | **99.19%**  |
| **Best Val Top-5 Acc** | 97.64% | 99.92% | **100.0%**  |
| **Best Val Loss** | 0.4435 | 0.0695 | **0.0311**  |
| **Total Epochs** | 150 | 44 | 50 |

### 1.3 Performance Improvement Analysis

**Baseline CNN → ResNet50:**
- **Absolute Improvement**: +11.56 percentage points
- **Relative Improvement**: +13.4%
- **Key Factor**: Transfer learning with ImageNet pre-trained weights

**ResNet50 → EfficientNet-B3:**
- **Absolute Improvement**: +0.97 percentage points
- **Relative Improvement**: +1.0%
- **Key Factor**: Compound scaling and efficient architecture

**Baseline CNN → EfficientNet-B3:**
- **Absolute Improvement**: +12.53 percentage points
- **Relative Improvement**: +14.5%
- **Key Factor**: Transfer learning + efficient architecture design

---

## 2. Model Architecture Comparison

### 2.1 Baseline CNN

**Architecture:**
- Custom 4-layer CNN trained from scratch
- 4 Convolutional layers (64, 128, 256, 512 channels)
- Batch Normalization after each layer
- Global Average Pooling
- Fully connected layers with Dropout (0.5)
- Output: 102 classes

**Characteristics:**
- **Parameters**: 11.9M
- **Model Size**: 45.4 MB
- **FLOPs**: 23.1 G
- **Image Size**: 224×224
- **Training**: From scratch (random initialization)
- **Training Epochs**: 150

**Strengths:**
- Lightweight architecture
- No dependency on pre-trained weights
- Demonstrates baseline performance achievable from scratch

**Limitations:**
- Lower accuracy (86.41%) compared to transfer learning models
- Requires extended training (150 epochs)
- Limited feature representation capacity

### 2.2 ResNet50

**Architecture:**
- Pre-trained ResNet50 backbone (ImageNet)
- 50-layer deep residual network
- Replaced classification head (2048 → 256 → 102)
- Two-phase training (frozen → fine-tuning)

**Characteristics:**
- **Parameters**: 24.6M
- **Model Size**: 93.9 MB
- **FLOPs**: 4.1 G
- **Image Size**: 224×224
- **Training**: Transfer learning (two-phase)
- **Training Epochs**: 44 (Phase 1: ~10-15, Phase 2: ~30-35)

**Strengths:**
- Excellent accuracy (97.97%)
- Fastest inference time (20.82 ms/image)
- Well-established architecture with proven performance
- Residual connections enable deep network training

**Limitations:**
- Larger model size (93.9 MB)
- More parameters (24.6M) than EfficientNet-B3
- Slightly lower accuracy than EfficientNet-B3

### 2.3 EfficientNet-B3

**Architecture:**
- Pre-trained EfficientNet-B3 backbone (ImageNet)
- Compound scaling (width, depth, resolution)
- MBConv blocks with Squeeze-and-Excitation (SE) attention
- Classification head (1536 → 256 → 102)

**Characteristics:**
- **Parameters**: 11.1M  (fewest among transfer learning models)
- **Model Size**: 42.4 MB
- **FLOPs**: 1.8 G  (lowest computational cost)
- **Image Size**: 300×300 (higher resolution)
- **Training**: Transfer learning (single-phase fine-tuning)
- **Training Epochs**: 50

**Strengths:**
- **Highest accuracy** (98.94%)
- **Best accuracy-to-parameter ratio** (98.94% with 11.1M params)
- **Lowest FLOPs** (1.8 G) - most computationally efficient
- Efficient compound scaling architecture
- SE attention mechanism for adaptive feature recalibration

**Limitations:**
- Slightly slower inference (27.30 ms/image) than ResNet50
- Requires larger input resolution (300×300) - more GPU memory

---

## 3. Training Strategy Comparison

### 3.1 Baseline CNN Training

**Strategy:**
- Trained from scratch (random initialization)
- Extended training: 150 epochs
- Learning rate: 5e-4 (higher initial LR)
- Class imbalance handling: Weighted sampling + class weights
- Early stopping: Patience=10

**Training Dynamics:**
- Gradual convergence over 150 epochs
- Best validation accuracy at epoch 149 (85.83%)
- Required extended training to learn features from scratch
- Training accuracy (83.41%) lower than validation (85.83%) - data augmentation effects

**Key Observations:**
- Training from scratch requires more epochs
- Class imbalance handling crucial for 102-class problem
- Model capacity limitations visible in performance ceiling

### 3.2 ResNet50 Training

**Strategy:**
- Two-phase transfer learning approach
- **Phase 1**: Frozen backbone (10-15 epochs, LR=1e-3)
- **Phase 2**: Fine-tuning all layers (30-35 epochs, LR=1e-4)
- Total: 44 epochs
- Early stopping: Enabled

**Training Dynamics:**
- Rapid convergence in Phase 1 (frozen backbone)
- Continued improvement in Phase 2 (fine-tuning)
- Best validation accuracy at epoch 34 (98.53%)
- Excellent generalization (train: 99.62%, val: 98.37%)

**Key Observations:**
- Two-phase strategy prevents destruction of pre-trained features
- Faster convergence than baseline (44 vs 150 epochs)
- Minimal overfitting gap demonstrates excellent generalization

### 3.3 EfficientNet-B3 Training

**Strategy:**
- Single-phase transfer learning
- Fine-tuning all layers from start
- Learning rate: 1e-4 (adaptive scheduling)
- 50 epochs total
- Early stopping: Patience=10

**Training Dynamics:**
- Steady improvement throughout training
- Best validation accuracy at epoch ~47-50 (99.19%)
- Highest validation accuracy among all models
- Excellent convergence with minimal overfitting

**Key Observations:**
- Compound scaling architecture benefits from higher resolution (300×300)
- Efficient architecture achieves best performance with fewer parameters
- Single-phase training sufficient due to efficient design

---

## 4. Efficiency Analysis

### 4.1 Model Size & Parameters

| Model | Parameters | Model Size (MB) | Accuracy | Accuracy/Param Ratio |
|-------|------------|-----------------|----------|---------------------|
| Baseline CNN | 11.9M | 45.4 | 86.41% | 7.26% per M params |
| ResNet50 | 24.6M | 93.9 | 97.97% | 3.98% per M params |
| EfficientNet-B3 | **11.1M**  | **42.4**  | **98.94%**  | **8.91% per M params**  |

**Key Insight:** EfficientNet-B3 achieves the **best accuracy-to-parameter ratio** (8.91% per million parameters), demonstrating superior parameter efficiency.

### 4.2 Computational Efficiency

| Model | FLOPs (G) | Inference Time (ms) | Throughput (images/sec) |
|-------|-----------|---------------------|------------------------|
| EfficientNet-B3 | **1.8**  | 27.30 | ~36.6 |
| ResNet50 | 4.1 | **20.82**  | **~48.0**  |
| Baseline CNN | 23.1 | 37.42 | ~26.7 |

**Key Insights:**
- **EfficientNet-B3**: Lowest FLOPs (1.8G) - most computationally efficient
- **ResNet50**: Fastest inference (20.82 ms) - best for real-time applications
- **Baseline CNN**: Highest FLOPs (23.1G) - least efficient despite fewer parameters

### 4.3 Training Efficiency

| Model | Training Epochs | Training Time (est.) | Convergence Speed |
|-------|----------------|---------------------|-------------------|
| ResNet50 | 44 | ~1-2 hours | Fast (transfer learning) |
| EfficientNet-B3 | 50 | ~2-3 hours | Moderate (transfer learning) |
| Baseline CNN | 150 | ~2-3 hours | Slow (from scratch) |

**Key Insight:** Transfer learning models converge faster and require fewer epochs, despite similar total training time.

---

## 5. Detailed Performance Analysis

### 5.1 Accuracy Breakdown

**Top-1 Accuracy:**
- EfficientNet-B3: **98.94%** (best)
- ResNet50: 97.97% (-0.97%)
- Baseline CNN: 86.41% (-12.53%)

**Top-5 Accuracy:**
- EfficientNet-B3: **99.76%** (best)
- ResNet50: 99.51% (-0.25%)
- Baseline CNN: 97.80% (-1.96%)

**Analysis:**
- All models show strong top-5 accuracy (>97%), indicating excellent class separation
- Small gap between top-1 and top-5 suggests models are confident in predictions
- EfficientNet-B3 achieves near-perfect top-5 accuracy (99.76%)

### 5.2 Per-Class Performance

**Macro-Averaged Metrics:**

| Metric | Baseline CNN | ResNet50 | EfficientNet-B3 |
|--------|--------------|----------|-----------------|
| Precision | 86.99% | 97.62% | **99.03%**  |
| Recall | 88.95% | 97.53% | **98.82%**  |
| F1-Score | 86.88% | 97.43% | **98.86%** |

**Key Observations:**
- EfficientNet-B3 achieves highest per-class performance across all metrics
- High macro-averaged scores indicate balanced performance across all 102 classes
- Baseline CNN shows good recall (88.95%) but lower precision (86.99%)

### 5.3 Discriminative Ability (ROC-AUC)

| Model | ROC-AUC (Macro) | Interpretation |
|-------|----------------|----------------|
| ResNet50 | **0.9994**  | Near-perfect discrimination |
| EfficientNet-B3 | 0.9993 | Near-perfect discrimination |
| Baseline CNN | 0.9971 | Excellent discrimination |

**Analysis:**
- All models show excellent discriminative ability (ROC-AUC > 0.99)
- ResNet50 achieves highest ROC-AUC (0.9994)
- High ROC-AUC indicates models can effectively distinguish between classes

---

## 6. Key Findings

### 6.1 Transfer Learning Advantage

**Quantitative Impact:**
- ResNet50: +11.56% absolute improvement over Baseline CNN
- EfficientNet-B3: +12.53% absolute improvement over Baseline CNN
- Pre-trained ImageNet weights provide strong feature representations
- Transfer learning reduces training time (44-50 epochs vs 150 epochs)

**Qualitative Impact:**
- Better feature representations from ImageNet pre-training
- Improved generalization (smaller train-val gap)
- Faster convergence during training

### 6.2 Architecture Efficiency

**EfficientNet-B3 Advantages:**
- **Best accuracy-to-parameter ratio**: 8.91% per million parameters
- **Lowest FLOPs**: 1.8G (vs 4.1G for ResNet50, 23.1G for Baseline)
- **Highest accuracy**: 98.94% with only 11.1M parameters
- Demonstrates effectiveness of compound scaling

**ResNet50 Advantages:**
- **Fastest inference**: 20.82 ms/image
- **Highest throughput**: ~48 images/second
- Well-established architecture with proven reliability

### 6.3 Training Strategy Effectiveness

**Two-Phase Training (ResNet50):**
- Effective for preventing destruction of pre-trained features
- Allows gradual adaptation to new task
- Achieves excellent performance (97.97%)

**Single-Phase Training (EfficientNet-B3):**
- Sufficient due to efficient architecture design
- Simpler training procedure
- Achieves best performance (98.94%)

**Extended Training (Baseline CNN):**
- Necessary for training from scratch
- 150 epochs required for convergence
- Demonstrates baseline performance achievable without transfer learning

---

## 7. Practical Considerations

### 7.1 Model Selection Guide

**Choose EfficientNet-B3 if:**
-  Highest accuracy is priority (98.94%)
-  Parameter efficiency matters (11.1M params)
-  Computational efficiency is important (1.8G FLOPs)
-  Model size constraints (42.4 MB)
-  Best accuracy-to-parameter ratio needed

**Choose ResNet50 if:**
-  Fastest inference required (20.82 ms/image)
-  Real-time applications (48 images/sec)
-  Well-established architecture preferred
-  Slightly lower accuracy acceptable (97.97%)

**Choose Baseline CNN if:**
-  No pre-trained weights available
-  Understanding baseline performance needed
-  Lightweight model required (11.9M params)
-  Lower accuracy acceptable (86.41%)

### 7.2 Resource Requirements

**GPU Memory:**
- Baseline CNN: Low (224×224, batch size 16)
- ResNet50: Medium (224×224, batch size 16-32)
- EfficientNet-B3: Medium-High (300×300, batch size 16-32)

**Storage:**
- Baseline CNN: 45.4 MB
- ResNet50: 93.9 MB
- EfficientNet-B3: 42.4 MB  (smallest)

**Inference Speed:**
- ResNet50: Fastest (20.82 ms/image) 
- EfficientNet-B3: Moderate (27.30 ms/image)
- Baseline CNN: Slowest (37.42 ms/image)

---

## 8. Error Analysis Insights

### 8.1 Common Failure Patterns

**Baseline CNN:**
- More misclassifications across diverse classes
- Lower confidence in predictions
- Struggles with visually similar flower species

**ResNet50 & EfficientNet-B3:**
- Fewer misclassifications overall
- Higher confidence in correct predictions
- Better discrimination between similar species
- EfficientNet-B3 shows best per-class performance

### 8.2 Class-Specific Performance

**All Models:**
- Strong top-5 accuracy (>97%) indicates good class separation
- High macro-averaged F1-scores suggest balanced performance
- Some classes may have lower individual performance (see per-class analysis)

---

## 9. Conclusions

### 9.1 Performance Ranking

1. **EfficientNet-B3**: Best overall performance (98.94% accuracy)
2. **ResNet50**: Excellent performance with fastest inference (97.97%)
3. **Baseline CNN**: Strong baseline from scratch (86.41%)

### 9.2 Key Takeaways

1. **Transfer Learning is Essential:**
   - 12.5% absolute improvement over baseline
   - Pre-trained ImageNet weights provide strong features
   - Faster convergence and better generalization

2. **EfficientNet-B3 is Optimal:**
   - Highest accuracy (98.94%)
   - Best parameter efficiency (11.1M params)
   - Lowest computational cost (1.8G FLOPs)
   - Best accuracy-to-parameter ratio

3. **ResNet50 Offers Speed:**
   - Fastest inference (20.82 ms/image)
   - Excellent accuracy (97.97%)
   - Well-established architecture

4. **Baseline CNN Provides Baseline:**
   - Demonstrates performance from scratch (86.41%)
   - Valuable for comparison and understanding
   - Shows value of transfer learning

### 9.3 Recommendations

**For Production Deployment:**
- **Primary Choice**: EfficientNet-B3 (best accuracy, efficient)
- **Alternative**: ResNet50 (if speed is critical)

**For Research/Education:**
- All three models provide valuable insights
- Baseline CNN demonstrates baseline performance
- Transfer learning models show state-of-the-art results

**For Further Improvement:**
- Ensemble methods combining EfficientNet-B3 and ResNet50
- Additional data augmentation strategies
- Hyperparameter optimization
- Test-time augmentation

---

## 10. Visualizations Reference

Comprehensive visualizations are available in `results/figures/`:

**Training Curves:**
- `baseline_cnn_training_curves.png`
- `resnet50_training_curves.png`
- `efficientnet_b3_training_curves.png`

**Evaluation Metrics:**
- `baseline_cnn_confusion_matrix.png`
- `resnet50_confusion_matrix.png`
- `efficientnet_b3_confusion_matrix.png`
- `baseline_cnn_roc_curves.png`
- `resnet50_roc_curves.png`
- `efficientnet_b3_roc_curves.png`

**Per-Class Performance:**
- `baseline_cnn_per_class_performance.png`
- `resnet50_per_class_performance.png`
- `efficientnet-b3_per_class_performance.png`

**Model Comparison:**
- `model_comparison_metrics.png`
- `model_comparison_radar.png`
- `model_comparison_training.png`

---

**Document Version**: 2.0 (Based on Actual Implementation)  
**Last Updated**: January 2026  
**Author**: Rishav Singh (NP01MS7A240010)  
**Evaluation Date**: 2026-01-11
