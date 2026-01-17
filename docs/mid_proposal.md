# Mid-Term Proposal: Deep Learning-Based Plant Species Classification Using Leaf and Flower Images

**Author:** Rishav Singh  
**Student ID:** NP01MS7A240010  
**Module:** CT7160NI Computer Vision  
**Institution:** London Metropolitan University / Islington College  
**Date:** January 2025

---

## 1. Introduction & Problem Statement

### 1.1 Problem Overview

Automated plant species identification is a fundamental challenge in computer vision with significant implications for biodiversity monitoring, agriculture, and environmental conservation. This project addresses the problem of **fine-grained visual classification** of plant species using flower images, specifically focusing on the Oxford 102 Flower Dataset containing 102 distinct flower categories.

### 1.2 Problem Statement

**Task Type:** Classification (Fine-grained visual categorization)

**Technical Definition:** Given an input image I ∈ ℝ^(H×W×3) of a flower, the objective is to learn a mapping function f: I → c, where c ∈ {0, 1, 2, ..., 101} represents one of 102 flower species classes. The function f must generalize well to unseen instances while achieving high classification accuracy despite intra-class variation (different viewing angles, lighting conditions, backgrounds) and inter-class similarity (morphologically similar species).

**Key Challenges:**
- Fine-grained classification with 102 visually similar classes
- High intra-class variation (pose, scale, illumination)
- Inter-class similarity (morphologically related species)
- Limited training data per class (~80 images per class on average)
- Complex background clutter affecting feature extraction

### 1.3 Real-World Relevance

This problem is highly relevant in multiple domains:

1. **Biodiversity Monitoring:** Automated species identification aids in ecological surveys and conservation efforts, enabling rapid assessment of floral diversity in field studies.

2. **Agricultural Applications:** Precision agriculture systems can benefit from automated crop and weed identification, facilitating targeted management strategies.

3. **Citizen Science:** Mobile applications for plant identification empower non-experts to contribute to biodiversity databases and educational initiatives.

4. **Botanical Research:** Automated classification accelerates the cataloging and analysis of plant specimens in herbariums and research collections.

5. **Horticulture & Gardening:** Commercial applications for garden centers, plant nurseries, and gardening enthusiasts require reliable species identification tools.

The Oxford 102 Flower Dataset represents a well-established benchmark in computer vision literature, making this work comparable to state-of-the-art methods and contributing to reproducible research.

---

## 2. Objectives

This project aims to implement and evaluate multiple deep learning approaches for plant species classification with the following measurable objectives:

### 2.1 Primary Objectives

1. **Baseline Implementation:** Design and implement a custom Convolutional Neural Network (CNN) architecture from scratch to establish a baseline performance metric on the Oxford 102 Flower Dataset.

2. **Transfer Learning Application:** Implement and fine-tune pre-trained deep learning models (ResNet50 and EfficientNet-B3) using transfer learning to improve classification performance over the baseline.

3. **Performance Comparison:** Conduct comprehensive comparative analysis between baseline CNN, ResNet50, and EfficientNet-B3 architectures using multiple evaluation metrics (accuracy, precision, recall, F1-score, top-5 accuracy, ROC-AUC).

4. **Robustness Analysis:** Evaluate model performance under varying conditions through data augmentation techniques simulating different scales, rotations, illumination changes, and geometric transformations.

### 2.2 Success Criteria

- Baseline CNN achieves at least 60% test accuracy
- Transfer learning models (ResNet50/EfficientNet) achieve at least 85% test accuracy
- Top-5 accuracy exceeds 95% for the best-performing model
- All models demonstrate convergence with stable training dynamics
- Comprehensive evaluation metrics computed and analyzed

---

## 3. Literature Review / Background

### 3.1 Classical Computer Vision Approaches

Early plant identification systems relied on hand-crafted features and classical computer vision techniques:

**Shape Descriptors:** Methods like Hu moments, Zernike moments, and Fourier descriptors were used to capture leaf and flower shape characteristics (Söderkvist, 2001).

**Texture Features:** Local Binary Patterns (LBP), Gabor filters, and Gray-Level Co-occurrence Matrix (GLCM) features captured surface texture patterns (Wu et al., 2007).

**Color Features:** Color histograms, color moments, and color coherence vectors encoded chromatic information (Chaki & Parekh, 2011).

**Limitations:** Classical approaches suffered from limited generalization, sensitivity to background clutter, and poor performance on fine-grained classification tasks requiring discriminative feature learning.

### 3.2 Modern Deep Learning Approaches

The advent of deep learning, particularly Convolutional Neural Networks (CNNs), revolutionized plant classification:

**Pre-trained Models & Transfer Learning:** Models pre-trained on large-scale datasets like ImageNet (Deng et al., 2009) have demonstrated excellent transferability to plant classification tasks. Lee et al. (2015) showed that fine-tuning pre-trained CNNs significantly outperforms training from scratch, achieving state-of-the-art results on plant identification benchmarks.

**Residual Networks (ResNet):** He et al. (2016) introduced residual learning frameworks enabling training of very deep networks (e.g., ResNet50, ResNet101). ResNet architectures have been widely adopted for plant classification, with ResNet50 achieving strong performance on the Oxford 102 Flower Dataset (Nilsback & Zisserman, 2008).

**EfficientNet:** Tan & Le (2019) proposed EfficientNet, which uses compound scaling to optimize depth, width, and resolution simultaneously. EfficientNet models achieve comparable or superior accuracy to ResNet while using fewer parameters and computational resources, making them attractive for resource-constrained applications.

**Data Augmentation:** Modern augmentation techniques, including geometric transformations (rotation, scaling, translation), photometric transformations (color jittering, brightness/contrast adjustment), and advanced methods like Mixup (Zhang et al., 2018) and CutMix (Yun et al., 2019), improve generalization.

### 3.3 Limitations & Research Gaps

Despite significant progress, several limitations persist:

1. **Limited Data:** Fine-grained classification requires large amounts of labeled data, which is expensive and time-consuming to collect for botanical applications.

2. **Domain Shift:** Models trained on standard datasets may not generalize well to field conditions with different lighting, backgrounds, and image quality.

3. **Computational Requirements:** Deep learning models require substantial computational resources, limiting deployment on mobile devices or edge computing platforms.

4. **Interpretability:** Deep learning models are often "black boxes," making it difficult to understand why certain predictions are made, which is crucial for scientific and educational applications.

This project addresses the first limitation by employing transfer learning, which allows effective learning from limited data by leveraging pre-trained features. The comparative analysis of different architectures will provide insights into the trade-offs between accuracy and computational efficiency.

---

## 4. Dataset Description

### 4.1 Dataset Specification

**Dataset Name:** Oxford 102 Flower Dataset

**Source:** Public dataset from the Visual Geometry Group (VGG), University of Oxford  
**URL:** https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

**Dataset Characteristics:**
- **Total Images:** 8,189
- **Number of Classes:** 102 flower species
- **Image Format:** JPEG
- **Image Resolution:** Variable (typically 500×500 to 1000×1000 pixels)
- **Color Channels:** RGB (3 channels)
- **Average Images per Class:** ~80 images
- **Class Distribution:** Relatively balanced with minor variations

**Dataset Structure:**
```
oxford_flowers_102/
├── jpg/                      # Image files (image_00001.jpg to image_08189.jpg)
├── imagelabels.mat          # Class labels (1-indexed, MATLAB format)
└── setid.mat                # Train/val/test split indices (optional)
```

### 4.2 Data Acquisition & Processing

The dataset was collected from the internet and manually annotated by the VGG research group. Images are high-quality photographs of flowers commonly found in the United Kingdom, captured under varying conditions (indoor/outdoor, different lighting, backgrounds).

**Data Split Strategy:**
The dataset is split using stratified sampling to maintain class distribution across splits:
- **Training Set:** 70% (5,732 images) - Model training
- **Validation Set:** 15% (1,228 images) - Hyperparameter tuning and model selection
- **Test Set:** 15% (1,229 images) - Final evaluation (held-out)

Stratified sampling ensures that each split maintains the original class distribution, preventing bias toward majority classes.

**Data Characteristics:**
- **Intra-class Variation:** High - images of the same class show different poses, scales, backgrounds, and lighting conditions
- **Inter-class Similarity:** Moderate to high - some species are visually similar (e.g., different rose varieties)
- **Background Complexity:** Variable - images contain complex natural backgrounds, which can interfere with feature extraction
- **Image Quality:** Generally high, but some images may have compression artifacts

### 4.3 Data Imbalance Analysis

While the dataset is relatively balanced, minor class imbalance exists (some classes have 40-100 images). The stratified split strategy mitigates this issue. Additionally, data augmentation techniques (discussed in Section 5.1) help balance the effective training distribution.

---

## 5. Methodology / Proposed Approach

The methodology follows a comprehensive pipeline from data preprocessing to model evaluation. The approach combines classical preprocessing techniques with modern deep learning architectures.

**Note:** Code examples and visual diagrams for this section are available in `docs/methodology_code_examples.py` and `docs/diagrams/`. See `docs/methodology_documentation.md` for usage instructions.

### 5.1 Pre-processing

#### 5.1.1 Image Normalization

All images are normalized using ImageNet statistics to match the preprocessing applied during pre-training:
- **Mean:** [0.485, 0.456, 0.406] (RGB channels)
- **Standard Deviation:** [0.229, 0.224, 0.225] (RGB channels)

This normalization ensures compatibility with pre-trained models and stabilizes training.

#### 5.1.2 Resizing & Cropping

- **Training:** Random resized crop to target size (224×224 for ResNet/Baseline, 300×300 for EfficientNet) with scale range [0.7, 1.0] and aspect ratio [0.8, 1.2]
- **Validation/Test:** Resize to (target_size + 32) followed by center crop to target size for consistent evaluation

#### 5.1.3 Data Augmentation (Training Only)

The following augmentation techniques are applied during training to improve generalization:

1. **Geometric Transformations:**
   - Random horizontal flip (p=0.5)
   - Random vertical flip (p=0.3)
   - Random rotation (±30 degrees, p=0.5)
   - Random translation (±10%, p=0.5)
   - Random scaling (0.9-1.1, p=0.5)

2. **Photometric Transformations:**
   - Color jittering (brightness: ±0.3, contrast: ±0.3, saturation: ±0.3, hue: ±0.1, p=0.5)
   - Random brightness/contrast adjustment (p=0.3)
   - Grayscale conversion (p=0.1)

3. **Noise & Blur:**
   - Gaussian blur (kernel size: 3-7, p=0.1)

**Validation/Test Augmentation:** Only normalization and resizing are applied (no augmentation) to ensure fair evaluation.

### 5.2 Feature Extraction / Representation

The project employs two approaches to feature extraction:

#### 5.2.1 Learned Features (CNN)

**Baseline CNN:** The custom architecture learns hierarchical features through convolutional layers:
- Layer 1: 64 filters, 3×3 conv → BatchNorm → ReLU → MaxPool
- Layer 2: 128 filters, 3×3 conv → BatchNorm → ReLU → MaxPool
- Layer 3: 256 filters, 3×3 conv → BatchNorm → ReLU → MaxPool
- Layer 4: 512 filters, 3×3 conv → BatchNorm → ReLU → MaxPool
- Global Average Pooling → 512-dimensional feature vector

**Transfer Learning Models:** Pre-trained models (ResNet50, EfficientNet-B3) extract rich visual features:
- **ResNet50:** 2048-dimensional feature vector from the penultimate layer
- **EfficientNet-B3:** 1536-dimensional feature vector from the penultimate layer

These features capture hierarchical patterns from low-level edges/textures to high-level semantic information.

### 5.3 Core Algorithm / Model

Three model architectures are implemented and compared. Visual architecture diagrams are provided in `docs/diagrams/`:
- `resnet50_architecture.png` - ResNet50 architecture diagram
- `baseline_cnn_architecture.png` - Baseline CNN architecture diagram
- `complete_pipeline.png` - Complete processing pipeline visualization

#### 5.3.1 Baseline CNN (Custom Architecture)

**Architecture:**
```
Input (224×224×3)
    ↓
ConvBlock(64) → 112×112×64
    ↓
ConvBlock(128) → 56×56×128
    ↓
ConvBlock(256) → 28×28×256
    ↓
ConvBlock(512) → 14×14×512
    ↓
Global Average Pooling → 512
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(102) → Softmax
```

**Rationale:** Provides a baseline for comparison and demonstrates learning from scratch. Uses batch normalization for training stability and dropout for regularization.

#### 5.3.2 ResNet50 (Transfer Learning)

**Architecture:**
```
Input (224×224×3)
    ↓
ResNet50 Backbone (Pre-trained on ImageNet, IMAGENET1K_V2 weights)
    ├── Conv + BatchNorm + ReLU
    ├── MaxPool
    ├── Layer1 (3 residual blocks)
    ├── Layer2 (4 residual blocks)
    ├── Layer3 (6 residual blocks)
    ├── Layer4 (3 residual blocks)
    └── AdaptiveAvgPool → 2048 features
    ↓
Custom Classifier Head:
    Dense(2048 → 512) → ReLU → Dropout(0.3)
    ↓
    Dense(512 → 102) → Softmax
```

**Training Strategy (Two-Phase Approach):**

**Phase 1: Classifier Head Training (Frozen Backbone)**
- **Epochs:** 15
- **Learning Rate:** 1×10⁻³ (1e-3)
- **Backbone Status:** Frozen (all ResNet50 layers frozen, only classifier head trainable)
- **Optimizer:** Adam with weight decay (1×10⁻⁴)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1×10⁻⁶)
- **Early Stopping:** Patience=5 epochs
- **Purpose:** Initialize the classifier head using pre-trained ImageNet features without modifying the backbone

**Phase 2: Full Network Fine-Tuning (Unfrozen Backbone)**
- **Epochs:** 32 (completed, out of 35 maximum)
- **Backbone Learning Rate:** 1×10⁻⁴ (1e-4), reduced to 5×10⁻⁵ (5e-5) after epoch 16 via scheduler
- **Classifier Learning Rate:** 2×10⁻⁴ (2e-4), reduced to 1×10⁻⁴ (1e-4) after epoch 16
- **Backbone Status:** Unfrozen (all layers trainable)
- **Optimizer:** Adam with differential learning rates for backbone and classifier
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1×10⁻⁷)
- **Early Stopping:** Patience=10 epochs (not triggered, training completed 32 epochs)
- **Purpose:** Fine-tune the entire network to adapt ImageNet features to the flower classification task
- **Status:** ✅ Completed

**Training Configuration:**
- **Batch Size:** 16 (with gradient accumulation steps=2, effective batch size=32)
- **Image Size:** 224×224 pixels
- **Data Augmentation:** Enabled (Albumentations library)
- **Weighted Sampling:** Enabled (handles minor class imbalance)
- **Class Weights in Loss:** Enabled (weighted CrossEntropyLoss)
- **Mixed Precision Training:** Enabled (AMP - Automatic Mixed Precision for GPU memory efficiency)
- **Gradient Accumulation:** 2 steps (simulates larger batch size within memory constraints)

**Actual Results:**

**Phase 1 (Completed):**
- **Epochs:** 15
- **Best Validation Accuracy:** 92.35% (Epoch 11)
- **Final Validation Accuracy:** 92.02% (Epoch 15)
- **Best Validation Top-5 Accuracy:** 98.94% (Epoch 14)
- **Final Validation Top-5 Accuracy:** 98.78% (Epoch 15)
- **Training Convergence:** Smooth convergence from 29.22% (Epoch 1) to 91.45% (Epoch 15) training accuracy

**Phase 2 (Completed):**
- **Epochs:** 32 (out of 35 planned)
- **Best Validation Accuracy:** 98.78% (Epoch 22 of Phase 2, Epoch 37 overall)
- **Final Validation Accuracy:** 98.62% (Epoch 32 of Phase 2, Epoch 47 overall)
- **Best Validation Top-5 Accuracy:** 100.00% (Epoch 7 of Phase 2, Epoch 22 overall)
- **Final Validation Top-5 Accuracy:** 99.92% (Epoch 32 of Phase 2)
- **Training Accuracy:** 99.53% (final epoch)
- **Training Top-5 Accuracy:** 99.95% (final epoch)
- **Improvement over Phase 1:** +6.43% validation accuracy (from 92.35% to 98.78%)

**Combined Training Results (47 Total Epochs):**
- **Best Overall Validation Accuracy:** 98.78%
- **Final Validation Accuracy:** 98.62%
- **Best Overall Validation Top-5 Accuracy:** 100.00%
- **Final Validation Top-5 Accuracy:** 99.92%

**Rationale:** ResNet50 is a proven architecture for transfer learning, with residual connections enabling effective gradient flow in deep networks. The two-phase training strategy prevents catastrophic forgetting while allowing fine-tuning. Phase 1 establishes a strong baseline by training only the classifier, while Phase 2 refines the entire network with lower learning rates to adapt pre-trained features to the specific flower classification task.

#### 5.3.3 EfficientNet-B3 (Transfer Learning)

**Architecture:**
```
Input (300×300×3)
    ↓
EfficientNet-B3 Backbone (Pre-trained on ImageNet)
    ├── Compound scaling (depth, width, resolution)
    ├── MBConv blocks with squeeze-and-excitation
    └── Global Average Pooling → 1536
    ↓
Dense(512) → ReLU → Dropout(0.3)
    ↓
Dense(102) → Softmax
```

**Rationale:** EfficientNet achieves superior accuracy-efficiency trade-offs through compound scaling, making it ideal for applications requiring both high accuracy and computational efficiency.

**Training Configuration (Planned):**
- **Optimizer:** Adam with learning rate scheduling
- **Loss Function:** Cross-entropy (with class weights if needed)
- **Batch Size:** 32 (or adapted for memory constraints)
- **Epochs:** 50 (with early stopping patience=10)
- **Learning Rate:** 1e-3 (Phase 1), 1e-4 (Phase 2)
- **Note:** Training details will be similar to ResNet50 approach

### 5.4 Post-processing

#### 5.4.1 Prediction

Model outputs are softmax probabilities over 102 classes. The predicted class is obtained via argmax operation:

**Predicted Class = argmax(softmax(logits))**

#### 5.4.2 Confidence Thresholding (Optional)

For deployment scenarios, predictions below a confidence threshold (e.g., 0.5) can be flagged as "uncertain" to improve reliability.

#### 5.4.3 Ensemble Methods (Future Work)

Ensemble predictions from multiple models (Baseline CNN + ResNet50 + EfficientNet) could further improve accuracy, though this is beyond the scope of the current proposal.

---

## 6. Experimental Design

### 6.1 Baseline Methods

1. **Baseline CNN (from scratch):** Trained for 50 epochs to establish baseline performance
2. **Baseline CNN (Improved):** Variant with increased capacity (more channels, deeper layers)

### 6.2 Model Variants

1. **ResNet50 (Transfer Learning) - Two-Phase Training:**
   - **Variant 1 (Phase 1):** Frozen backbone + classifier training (15 epochs, LR=1e-3)
     - Status: ✅ Completed
     - Results: 92.35% best validation accuracy, 98.94% top-5 accuracy
   - **Variant 2 (Phase 2):** Full fine-tuning (32 epochs completed, LR=1e-4 backbone, 2e-4 classifier)
     - Status: ✅ Completed
     - Results: 98.78% best validation accuracy, 100.00% best top-5 accuracy
     - Improvement: +6.43% validation accuracy over Phase 1
     - Purpose: Adapt ImageNet features to flower classification domain - successfully achieved

2. **EfficientNet-B3 (Transfer Learning):** Full fine-tuning approach (planned)

### 6.3 Ablation Studies

To understand the contribution of different components:

1. **Data Augmentation Impact:** Train models with and without augmentation to quantify improvement
2. **Transfer Learning Impact:** Compare ResNet50 with and without pre-training (from scratch vs. transfer learning)
3. **Architecture Comparison:** Compare ResNet50 vs. EfficientNet-B3 under identical training conditions

### 6.4 Evaluation Conditions

Models are evaluated under:

1. **Standard Conditions:** Clean test set with standard preprocessing
2. **Augmented Test Conditions (Future):** Evaluate robustness by applying test-time augmentations (rotations, brightness changes)
3. **Per-Class Performance:** Analyze accuracy for each of the 102 classes to identify challenging categories

### 6.5 Experimental Protocol

1. **Reproducibility:** All experiments use fixed random seeds (seed=42) for data splitting and model initialization
2. **Hardware:** Training performed on GPU (CUDA-compatible, 4GB+ VRAM) with Automatic Mixed Precision (AMP) for memory efficiency
3. **Hyperparameter Configuration:**
   - ResNet50 Phase 1: Learning rate 1e-3, batch size 16 (effective 32 with gradient accumulation)
   - ResNet50 Phase 2: Differential learning rates (1e-4 backbone, 2e-4 classifier), batch size 16
   - Early stopping patience: 5 epochs (Phase 1), 10 epochs (Phase 2)
4. **Optimization Techniques:**
   - Gradient accumulation (2 steps) to simulate larger batch sizes within memory constraints
   - Weighted random sampling and class weights in loss function to handle minor class imbalance
   - Learning rate scheduling: ReduceLROnPlateau with factor=0.5, patience=5
5. **Model Selection:** Best model selected based on validation accuracy (not test accuracy) to prevent overfitting
6. **Training Monitoring:** Comprehensive metrics tracking including top-5 accuracy, loss curves, and learning rate history

---

## 7. Evaluation Metrics

Comprehensive evaluation metrics are computed to assess model performance from multiple perspectives:

### 7.1 Classification Metrics

1. **Overall Accuracy:** Percentage of correctly classified samples
   - Formula: Accuracy = (TP + TN) / Total
   - Interpretation: General classification performance

2. **Precision (Macro & Micro):** 
   - Macro: Average precision across all classes
   - Micro: Overall precision considering all samples
   - Formula: Precision = TP / (TP + FP)
   - Interpretation: Measures correctness of positive predictions

3. **Recall (Macro & Micro):**
   - Macro: Average recall across all classes
   - Micro: Overall recall considering all samples
   - Formula: Recall = TP / (TP + FN)
   - Interpretation: Measures ability to find all positive samples

4. **F1-Score (Macro & Micro):**
   - Harmonic mean of precision and recall
   - Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - Interpretation: Balanced metric considering both precision and recall

5. **Top-5 Accuracy:** Percentage of samples where the true class is among the top 5 predicted classes
   - Interpretation: Useful for applications where multiple candidate predictions are acceptable

6. **Per-Class Metrics:** Precision, recall, and F1-score computed for each of the 102 classes to identify challenging categories

### 7.2 Additional Metrics

7. **Confusion Matrix:** 102×102 matrix showing classification errors
   - Visualization: Heatmap for qualitative analysis
   - Interpretation: Identifies class pairs that are frequently confused

8. **ROC-AUC (Macro):** Area Under the Receiver Operating Characteristic curve (one-vs-rest)
   - Range: [0, 1]
   - Interpretation: Higher values indicate better discriminative ability

9. **Classification Report:** Detailed per-class metrics including support (number of samples per class)

### 7.3 Metric Justification

- **Accuracy:** Primary metric for overall performance comparison
- **Macro-averaged metrics:** Give equal weight to all classes, important for balanced evaluation despite minor class imbalance
- **Micro-averaged metrics:** Useful for understanding overall performance across all samples
- **Top-5 Accuracy:** Relevant for real-world applications where users can select from multiple suggestions
- **ROC-AUC:** Provides threshold-independent evaluation of model discriminative ability
- **Confusion Matrix:** Essential for understanding failure modes and class-specific challenges

These metrics are standard in multi-class classification tasks and enable comprehensive performance assessment and comparison with literature.

---

## 8. Expected Outcomes

### 8.1 Quantitative Results

Based on literature and preliminary experiments:

1. **Baseline CNN:** Expected test accuracy of 65-75%
   - Lower than transfer learning models due to training from scratch on limited data
   - Demonstrates the challenge of fine-grained classification without pre-training

2. **ResNet50 (Transfer Learning):** ✅ Training Completed - Achieved 98.78% validation accuracy
   - **Phase 1 Results (Completed):**
     - Best Validation Accuracy: 92.35% (Epoch 11)
     - Final Validation Accuracy: 92.02% (Epoch 15)
     - Best Validation Top-5 Accuracy: 98.94%
     - Final Validation Top-5 Accuracy: 98.78%
     - Training Accuracy: 91.45% (final epoch)
   - **Phase 2 Results (Completed):**
     - Best Validation Accuracy: 98.78% (Epoch 22 of Phase 2)
     - Final Validation Accuracy: 98.62% (Epoch 32 of Phase 2)
     - Best Validation Top-5 Accuracy: 100.00%
     - Final Validation Top-5 Accuracy: 99.92%
     - Training Accuracy: 99.53% (final epoch)
     - Training Top-5 Accuracy: 99.95% (final epoch)
     - Improvement: +6.43% over Phase 1 best validation accuracy
   - **Combined Results:** 47 total epochs (15 Phase 1 + 32 Phase 2)
     - Best Overall Validation Accuracy: 98.78%
     - Final Validation Accuracy: 98.62%
     - Best Overall Top-5 Accuracy: 100.00%
   - **Expected Test Accuracy:** 97-99% (validation accuracy suggests excellent generalization)
   - Leverages ImageNet pre-training for robust feature extraction
   - Two-phase training strategy (frozen backbone → full fine-tuning) successfully implemented and exceeded expectations

3. **EfficientNet-B3 (Transfer Learning):** Expected test accuracy of 88-94%
   - Superior efficiency-accuracy trade-off compared to ResNet50
   - Expected to achieve highest accuracy with fewer parameters

4. **Top-5 Accuracy:** Expected to exceed 95% for all transfer learning models
   - Indicates models learn meaningful feature representations even if top-1 prediction is incorrect

### 8.2 Qualitative Results

1. **Learning Curves (ResNet50 - Complete Training Observed):**
   - **Phase 1:** Smooth convergence from 29.22% (Epoch 1) to 91.45% (Epoch 15) training accuracy
     - Validation accuracy improved from 58.88% (Epoch 1) to 92.35% (best, Epoch 11)
     - Stable training dynamics with minimal overfitting
     - Top-5 accuracy reached 98.94%
   - **Phase 2:** Continued improvement from 94.40% (Epoch 1) to 99.53% (Epoch 32) training accuracy
     - Validation accuracy improved from 95.93% (Phase 2 Epoch 1) to 98.78% (best, Epoch 22)
     - Excellent generalization: final training accuracy 99.53%, validation accuracy 98.62%
     - Top-5 accuracy reached 100.00% (best), 99.92% (final)
     - Learning rate reduction at epoch 16 (1e-4 → 5e-5) further stabilized training
     - Loss decreased from 0.152 (Phase 2 Epoch 1) to 0.016 (final) for training, 0.062 for validation
   - **Overall:** 47 total epochs, excellent convergence with 98.78% best validation accuracy

2. **Confusion Matrix (Expected):** Most errors occur between visually similar species (e.g., different rose varieties)

3. **Feature Visualizations (Future Work):** Grad-CAM visualizations will show that models focus on flower petals and distinctive morphological features

4. **Failure Cases (Expected):** Misclassifications primarily occur for:
   - Classes with high intra-class variation
   - Visually similar species with subtle distinguishing features
   - Images with heavy background clutter

### 8.3 Hypotheses

1. **Transfer Learning Superiority:** Pre-trained models (ResNet50, EfficientNet) will significantly outperform baseline CNN trained from scratch, validating the effectiveness of transfer learning for fine-grained classification with limited data.

2. **EfficientNet Advantage:** EfficientNet-B3 will achieve comparable or superior accuracy to ResNet50 while using fewer parameters, demonstrating the benefits of compound scaling.

3. **Data Augmentation Impact:** Data augmentation will improve generalization, reducing the gap between training and validation accuracy by 5-10 percentage points.

4. **Class-Specific Performance:** Performance will vary across classes, with visually distinct species (e.g., sunflowers) achieving higher accuracy than similar species (e.g., different rose varieties).

---

## 9. Limitations & Risks

### 9.1 Identified Limitations

1. **Limited Dataset Size:** With ~80 images per class, the dataset is relatively small for deep learning, potentially limiting generalization. However, transfer learning mitigates this issue by leveraging pre-trained features.

2. **Computational Constraints:** Training deep learning models requires GPU resources. Limited access to high-end GPUs may restrict experimentation with larger models or extensive hyperparameter tuning.

3. **Domain Specificity:** Models trained on the Oxford 102 Flower Dataset may not generalize well to:
   - Different geographic regions with different flower species
   - Field conditions with varying lighting, backgrounds, and image quality
   - Other plant organs (leaves, stems) not included in the dataset

4. **Class Imbalance:** Minor class imbalance exists, though stratified sampling and data augmentation help mitigate this issue.

5. **Background Dependency:** Models may learn background features rather than flower-specific features, though data augmentation and transfer learning reduce this risk.

### 9.2 Risks & Mitigation Strategies

1. **Risk: Underperforming Baseline CNN**
   - **Impact:** May not provide meaningful comparison baseline
   - **Mitigation:** Accept lower baseline performance as expected for training from scratch; focus on transfer learning models for primary results

2. **Risk: Overfitting**
   - **Impact:** Poor generalization to test set
   - **Mitigation:** Early stopping, dropout regularization, data augmentation, validation set monitoring

3. **Risk: Training Time Constraints**
   - **Impact:** Insufficient time for hyperparameter tuning or training all model variants
   - **Mitigation:** Prioritize ResNet50 and EfficientNet-B3; use early stopping to reduce training time; leverage pre-trained weights to reduce training epochs

4. **Risk: Hardware Failures**
   - **Impact:** Loss of training progress or results
   - **Mitigation:** Regular model checkpointing, version control for code, cloud backup for results

5. **Risk: Reproducibility Issues**
   - **Impact:** Difficulty reproducing results or comparing with literature
   - **Mitigation:** Fixed random seeds, detailed documentation of hyperparameters, version-controlled code

### 9.3 Fallback Strategies

1. **If Transfer Learning Underperforms:** Increase data augmentation intensity, extend training epochs, or try alternative pre-trained models (e.g., DenseNet, Vision Transformer)

2. **If Computational Resources are Limited:** Reduce batch size, use mixed-precision training, or train on a subset of classes for initial experiments

3. **If Models Fail to Converge:** Adjust learning rates, use learning rate scheduling, or switch to different optimizers (SGD with momentum)

---

## 10. Tools & Implementation Details

### 10.1 Programming Language

- **Primary Language:** Python 3.8+
- **Rationale:** Extensive deep learning ecosystem, excellent libraries (PyTorch, NumPy, scikit-learn), and strong community support

### 10.2 Libraries & Frameworks

**Deep Learning:**
- **PyTorch 2.0+:** Model implementation, training, and inference
- **Torchvision:** Pre-trained models (ResNet50 with IMAGENET1K_V2 weights), data transforms, datasets
- **Albumentations:** Advanced data augmentation (RandomResizedCrop, Affine, ColorJitter, etc.)
- **Mixed Precision Training (AMP):** Automatic Mixed Precision for GPU memory optimization

**Data Processing:**
- **NumPy:** Numerical computations
- **PIL/Pillow:** Image loading and basic manipulation
- **scipy.io:** Loading MATLAB files (.mat format for labels)

**Evaluation & Visualization:**
- **scikit-learn:** Evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC)
- **Matplotlib:** Plotting training curves, confusion matrices
- **Seaborn:** Enhanced visualization (heatmaps)

**Utilities:**
- **tqdm:** Progress bars for training loops
- **PyYAML:** Configuration file management

### 10.3 Hardware

- **GPU:** CUDA-compatible GPU (NVIDIA GPU with CUDA 11.8+)
  - **Usage:** Essential for training deep learning models efficiently
  - **Memory Requirements:** Minimum 4GB VRAM (tested), 8GB+ recommended for larger batch sizes
  - **Current Setup:** ResNet50 training uses batch size 16 with gradient accumulation (effective batch size 32) and AMP (Automatic Mixed Precision) to optimize memory usage on 4GB GPU
- **CPU:** Multi-core CPU for data loading and preprocessing
- **RAM:** 16GB+ recommended for data loading and preprocessing pipelines
- **Storage:** ~5GB for dataset, additional space for model checkpoints and results

### 10.4 Development Environment

- **IDE:** Visual Studio Code / PyCharm
- **Version Control:** Git
- **Environment Management:** Python virtual environment (venv) or Conda
- **Notebooks:** Jupyter Notebooks for exploratory analysis and visualization

### 10.5 Implementation Structure

The codebase follows a modular structure:

```
plant-species-classification/
├── src/
│   ├── data/              # Data loading, preprocessing, augmentation
│   ├── models/            # Model architectures
│   ├── training/          # Training loops, utilities
│   ├── evaluation/        # Evaluation metrics, visualization
│   └── utils/             # Helper functions
├── notebooks/             # Exploratory analysis
├── results/               # Model checkpoints, metrics, plots
└── config.yaml           # Configuration parameters
```

---

### 11.3 Current Progress Summary

**Completed:**
- ✅ Dataset preparation and preprocessing pipeline
- ✅ Baseline CNN implementation and training (72.50% test accuracy)
- ✅ ResNet50 implementation and complete two-phase training:
  - Architecture: ResNet50 backbone (ImageNet pre-trained) + custom classifier (2048→512→102)
  - Training configuration: Batch size 16, gradient accumulation 2, AMP enabled, weighted sampling
  - Phase 1: 15 epochs, frozen backbone, 92.35% best validation accuracy, 98.94% top-5 accuracy
  - Phase 2: 32 epochs, unfrozen backbone, 98.78% best validation accuracy, 100.00% best top-5 accuracy
  - Combined results: 47 total epochs, 98.78% best validation accuracy, excellent generalization
  - Both phases completed successfully with smooth convergence

**In Progress:**
- 🔄 EfficientNet-B3 implementation
- 🔄 EfficientNet-B3 training

**Remaining:**
- ⏳ Comprehensive evaluation and metrics computation
- ⏳ Final report writing
- ⏳ Presentation preparation

---

## 12. References

Chaki, J., & Parekh, R. (2011). Plant leaf recognition using shape based features and neural network classifiers. *International Journal of Advanced Computer Science and Applications*, 2(10), 41-47.

Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. *2009 IEEE conference on computer vision and pattern recognition* (pp. 248-255). IEEE.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).

Lee, S. H., Chan, C. S., Wilkin, P., & Remagnino, P. (2015). Deep-plant: Plant identification with convolutional neural networks. *2015 IEEE international conference on image processing (ICIP)* (pp. 452-456). IEEE.

Nilsback, M. E., & Zisserman, A. (2008). Automated flower classification over a large number of classes. *2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing* (pp. 722-729). IEEE.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE international conference on computer vision* (pp. 618-626).

Söderkvist, O. (2001). *Computer vision classification of leaves from Swedish trees* (Master's thesis, Linköping University).

Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. *International conference on machine learning* (pp. 6105-6114). PMLR.

Wu, S. G., Bao, F. S., Xu, E. Y., Wang, Y. X., Chang, Y. F., & Xiang, Q. L. (2007). A leaf recognition algorithm for plant classification using probabilistic neural network. *2007 IEEE international symposium on signal processing and information technology* (pp. 11-16). IEEE.

Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). Cutmix: Regularization strategy to train strong classifiers with localizable features. *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 6023-6032).

Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *arXiv preprint arXiv:1710.09412*.

---

**Document Version:** 1.2  
**Last Updated:** January 2025  
**Revision Notes:** Updated with complete ResNet50 two-phase training results (Phase 1 & Phase 2). Phase 2 achieved 98.78% validation accuracy, exceeding expectations.

