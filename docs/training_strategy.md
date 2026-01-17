# Training Strategy: Background Removal Impact Analysis

**CT7160NI Computer Vision Coursework**  
**Plant Species Classification - Training Plan**

## Overview

This document outlines the training strategy for evaluating the impact of segmentation mask-based background removal on model performance. This approach provides a comprehensive analysis suitable for coursework requirements while demonstrating both comparative effectiveness and best-case performance.

---

## Chosen Approach: Option B (Strategic Addition)

### Rationale

We selected **Option B** (Strategic Addition) as it provides:
- **Clear before/after comparison**: Direct comparison of Baseline CNN and ResNet50 with and without background removal
- **Demonstrates improvement across model types**: Shows impact on both simple (Baseline CNN) and complex (ResNet50) architectures
- **Best performance demonstration**: EfficientNet-B3 with masks showcases state-of-the-art results
- **Reasonable training time**: Completable within a practical timeframe (~4.5-8.5 hours total)
- **Comprehensive analysis**: Provides statistically meaningful results plus best-case performance for coursework documentation

### Training Plan

We will train **5 model variants** in total (Option B - Strategic Addition):

| Model | Background Removal | Purpose |
|-------|-------------------|---------|
| Baseline CNN | ❌ No | Baseline performance (without masks) |
| Baseline CNN | ✅ Yes | Impact on simple architecture |
| ResNet50 | ❌ No | Baseline for transfer learning |
| ResNet50 | ✅ Yes | Impact on complex architecture |
| EfficientNet-B3 | ✅ Yes | Best-case performance with masks |

**Training Strategy:**
- Baseline CNN and ResNet50 trained both with and without masks for clear before/after comparison
- EfficientNet-B3 trained only with masks to demonstrate best possible performance
- This approach balances comprehensive comparison with practical time constraints

---

## Expected Training Times

Based on RTX 2050 GPU (4GB VRAM) specifications:

| Model | Epochs | Time per Training | Total Time |
|-------|--------|------------------|------------|
| Baseline CNN (no masks) | 50 | ~30-45 min | ~30-45 min |
| Baseline CNN (with masks) | 50 | ~30-45 min | ~30-45 min |
| ResNet50 (no masks) | 50 | ~1-2 hours | ~1-2 hours |
| ResNet50 (with masks) | 50 | ~1-2 hours | ~1-2 hours |
| EfficientNet-B3 (with masks) | 50 | ~1.5-3 hours | ~1.5-3 hours |
| **Total** | - | - | **~4.5-8.5 hours** |

### Training Configuration

- **Batch Size**: 16-32 (Baseline CNN/ResNet50), 8-16 (EfficientNet-B3) - adjusted based on GPU memory
- **Image Size**: 224×224 (Baseline CNN and ResNet50), 300×300 (EfficientNet-B3)
- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau (if needed)
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Enabled with patience=10

---

## Data Preprocessing Status

### No Manual Preprocessing Required ✅

**Important**: The dataset pipeline is configured for **on-the-fly preprocessing**. No manual preprocessing step is required before starting training.

### How It Works

**Automatic On-the-Fly Processing:**
- Images are loaded directly from `data/raw/oxford_flowers_102/102flowers/jpg/` during training
- All preprocessing happens automatically via PyTorch transforms:
  1. **Image Loading**: Images loaded from raw directory as needed
  2. **Resize**: Automatically resized to target size (224×224 or 300×300)
  3. **Background Removal**: Applied using masks if enabled (before transforms)
  4. **Augmentation**: Random augmentations applied during training (rotation, flip, color jitter, etc.)
  5. **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  6. **Tensor Conversion**: Converted to PyTorch tensors

### Required Data Structure

Ensure the following structure exists:
```
data/raw/oxford_flowers_102/
├── 102flowers/
│   └── jpg/              # All flower images (8,189 images)
├── 102segmentations/
│   └── segmim/           # Segmentation masks (8,189 masks)
├── imagelabels.mat       # Image labels
└── setid.mat            # Train/val/test split indices
```

### Verification Checklist

Before starting training, verify:
- ✅ Raw images directory exists: `data/raw/oxford_flowers_102/102flowers/jpg/`
- ✅ Masks directory exists: `data/raw/oxford_flowers_102/102segmentations/segmim/`
- ✅ Labels file exists: `data/raw/oxford_flowers_102/imagelabels.mat`
- ✅ Split file exists: `data/raw/oxford_flowers_102/setid.mat`

### Why No Preprocessing?

**Advantages of On-the-Fly Processing:**
1. **Flexibility**: Easy to change image sizes or augmentation strategies without reprocessing
2. **Storage Efficiency**: No need to store preprocessed images (saves disk space)
3. **Speed**: Modern GPUs and data loaders handle on-the-fly processing efficiently
4. **Mask Integration**: Background removal can be applied dynamically based on training configuration
5. **Standard Practice**: Common approach in PyTorch-based deep learning projects

### Note on `data/processed/` Directory

The `data/processed/` directory exists but is **not used** by the current pipeline. It's empty and can be ignored. This directory would be used if you were to implement batch preprocessing, but it's not required for this project.

### Note on `src/data/preprocessing.py` Module

The `preprocessing.py` file contains utility functions but is **NOT used during training**. Training uses transforms from `augmentation.py` instead. The `preprocessing.py` module is useful for:
- **Inference**: `preprocess_image()` function for preprocessing single images for model predictions
- **Utilities**: Optional functions for batch preprocessing, statistics calculation, etc.

**For training**: Use the on-the-fly transforms (already configured) - do NOT use `preprocessing.py` functions. See `docs/preprocessing_module_explanation.md` for detailed explanation.

### Ready to Train

Once the data structure is verified (checklist above), you can proceed directly to training. The dataset class handles all preprocessing automatically.

---

## EfficientNet-B3 Considerations

### Why Excluded from Option 1?

EfficientNet-B3 is included in Option B (with masks only) for the following reasons:

**1. Best Performance Demonstration:**
- EfficientNet-B3 achieves the highest accuracy (~89-94% with masks)
- Showcases state-of-the-art results for the coursework
- Demonstrates the full potential of the approach

**2. Balanced Time Investment:**
- Training only with masks (not both with/without) keeps additional time reasonable (~1.5-3 hours)
- Total training time remains manageable (~4.5-8.5 hours)
- Good trade-off between comprehensiveness and time constraints

**3. Comprehensive Analysis:**
- Provides performance comparison across all three architectures
- Shows both comparative analysis (Baseline/ResNet50) and best-case performance (EfficientNet)
- Creates well-rounded experimental design

**Note on Resource Constraints:**
- RTX 2050 (4GB VRAM) may require smaller batch sizes (8-16) for 300×300 images
- Memory pressure is manageable with proper batch size adjustment
- Training time may be longer but acceptable for coursework

### Why Option B Was Chosen

**Selected: Option B (Strategic Addition)**

We selected Option B over other alternatives:

**Option A (Not Chosen): Comprehensive - All 6 Variants**
- Train all 6 variants: Baseline CNN, ResNet50, and EfficientNet-B3 (each with/without masks)
- **Total time**: ~6-11.5 hours
- **Not selected because**: Too time-consuming for coursework timeline

**Option B (Chosen): Strategic Addition - 5 Variants**
- Train EfficientNet-B3 **with masks** (best performance)
- Keep Baseline CNN and ResNet50 with/without masks
- **Total variants**: 5
- **Additional time**: ~1.5-3 hours (compared to Option 1)
- **Selected because**: Shows best-case performance while maintaining before/after comparison, balanced time investment

**Option C (Not Chosen): Post-Training Addition**
- Complete Option 1 first, then add EfficientNet-B3 if time permits
- **Not selected because**: Option B provides better structure and planning upfront

### EfficientNet-B3 Specific Configuration

If training EfficientNet-B3, use:

```python
# Different image size required
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    image_size=300,  # EfficientNet-B3 requires 300x300 (not 224x224)
    batch_size=16,   # May need smaller batch size (8-16) due to larger images
    num_workers=0,   # May need to reduce for Windows compatibility
    use_masks=True,  # or False for baseline
    apply_background_removal=True,
    background_color='black'
)

# Model initialization
from src.models import EfficientNetClassifier
model = EfficientNetClassifier(
    num_classes=102,
    model_name='efficientnet_b3',
    pretrained=True,
    freeze_backbone=True,
    dropout=0.3
)
```

### Expected Performance

| Model | Expected Accuracy (without masks) | Expected Accuracy (with masks) | Expected Improvement |
|-------|-----------------------------------|--------------------------------|---------------------|
| EfficientNet-B3 | N/A (not trained without masks) | ~89-94% | (Best Performance) |

**Note**: EfficientNet-B3 is expected to achieve the highest absolute accuracy among all models. While not trained without masks for comparison, based on ResNet50 results, we expect similar relative improvement patterns (2-3% gain) if compared. EfficientNet-B3 with masks represents the best-case performance for this project.

### Chosen Approach: Option B (Strategic Addition)

**Selected Option B** for the following reasons:
1. **Comprehensive comparison**: Baseline CNN and ResNet50 provide clear before/after comparison (with/without masks)
2. **Best performance demonstration**: EfficientNet-B3 with masks shows the best achievable accuracy
3. **Time efficiency**: Completes in reasonable timeframe (~4.5-8.5 hours) - more manageable than full 6-variant approach
4. **Report quality**: Demonstrates the technique's effectiveness while also showcasing state-of-the-art performance
5. **Balanced approach**: Shows both comparative analysis and best-case results without excessive training time

---

## Background Removal Method

### Implementation Details

**Process:**
1. Load segmentation mask from `102segmentations/segmim/` directory
2. Convert mask to binary (foreground = 255, background = 0)
3. Apply mask to original image
4. Replace background with black pixels (foreground preserved)
5. Apply standard transforms (resize, normalize, augment)

**Why Black Background?**
- Common practice in computer vision preprocessing
- Reduces background noise that could confuse the model
- Focuses model attention on foreground objects (flowers)
- Comparable to image cropping but preserves full image dimensions

### Expected Benefits

**Hypothesized Improvements:**
- **2-3% accuracy improvement**: Background removal should help models focus on flower features
- **Better feature learning**: Model learns flower-specific patterns without background distractions
- **More robust predictions**: Less sensitivity to varying backgrounds in test images
- **Improved attention patterns**: Grad-CAM visualizations should align better with actual flower regions

---

## Documentation Plan

### 1. Method Description

**What to include:**
- Explanation of segmentation mask-based background removal
- Rationale for why background removal should improve classification
- Implementation details (mask loading, application, preprocessing pipeline)
- Comparison of preprocessed images (original vs. masked)

**Location in report:**
- Methodology section
- Data preprocessing subsection

### 2. Results Presentation

**What to include:**
- **Performance metrics** for all 5 model variants:
  - Training accuracy and loss curves
  - Validation accuracy and loss curves
  - Test accuracy (final metric)
  - Confusion matrices (optional, if space permits)

- **Comparative tables:**
  ```
  | Model | With Masks | Test Accuracy | Improvement |
  |-------|------------|---------------|-------------|
  | Baseline CNN | No  | 42.94% (validation) | - |
  | Baseline CNN | Yes | ~45-46% (expected) | +2-3% |
  | ResNet50 | No  | X.XX% | - |
  | ResNet50 | Yes | X.XX% | +Y.YY% |
  | EfficientNet-B3 | Yes | X.XX% | (Best Performance) |
  ```

**Location in report:**
- Results section
- Experimental results subsection

### 3. Analysis & Discussion

**What to include:**
- **Quantitative analysis:**
  - Magnitude of improvement for each model
  - Statistical significance (if applicable)
  - Which model benefits more from background removal

- **Qualitative analysis:**
  - Visual inspection of preprocessed images
  - Sample predictions comparison
  - Error analysis (cases where masks help vs. hurt)

- **Discussion points:**
  - Why background removal helps (focus on foreground features)
  - Why improvement might be model-dependent
  - Performance comparison across all three architectures
  - EfficientNet-B3 as the best-performing model (with masks)
  - Limitations of this approach (black backgrounds might not be natural)

**Location in report:**
- Discussion section
- Analysis of results subsection

### 4. Visualizations

**What to include:**

1. **Preprocessing visualization:**
   - Side-by-side: Original image → Mask → Masked image
   - Show 3-5 example images across different classes

2. **Training curves:**
   - Loss curves: Training vs. Validation (with/without masks)
   - Accuracy curves: Training vs. Validation (with/without masks)
   - Comparison plots: Overlay plots showing improvement

3. **Performance comparison:**
   - Bar charts: Test accuracy comparison
   - Improvement percentage visualization

4. **Sample predictions:**
   - Correct predictions: With vs. without masks
   - Error cases: Where masks help vs. where they don't

**Location in report:**
- Results section (figures)
- Appendix (additional visualizations)

---

## Expected Outcomes

### Quantitative Metrics

**Target Improvements:**
- Baseline CNN: +1-2% accuracy improvement (with masks vs. without)
- ResNet50: +2-3% accuracy improvement (with masks vs. without)

**Baseline Expectations (without masks):**
- Baseline CNN: 42.94% validation accuracy (actual achieved) - lower than initially expected due to training from scratch on 102-class dataset
- ResNet50: ~85-90% test accuracy (expected)

**Best Performance (with masks):**
- EfficientNet-B3: ~89-94% test accuracy (highest expected accuracy)

### Qualitative Observations

- Training curves should show smoother convergence with masks
- Validation accuracy should plateau at higher values with masks
- Model confidence (prediction probabilities) may be higher for correct predictions
- Attention visualizations (Grad-CAM) should align better with flower regions

---

## Implementation Notes

### Training Scripts

Each model variant should be saved with descriptive names:
- `baseline_cnn_no_masks_epoch50.pth`
- `baseline_cnn_with_masks_epoch50.pth`
- `resnet50_no_masks_epoch50.pth`
- `resnet50_with_masks_epoch50.pth`

### Configuration

**For Baseline CNN and ResNet50 (with masks):**
```python
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    image_size=224,
    use_masks=True,
    apply_background_removal=True,
    background_color='black'
)
```

**For Baseline CNN and ResNet50 (without masks):**
```python
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    image_size=224,
    use_masks=False,
    apply_background_removal=False
)
```

**For EfficientNet-B3 (with masks):**
```python
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    image_size=300,  # EfficientNet-B3 requires 300x300
    batch_size=16,   # Smaller batch size for larger images
    use_masks=True,
    apply_background_removal=True,
    background_color='black'
)

from src.models import EfficientNetClassifier
model = EfficientNetClassifier(
    num_classes=102,
    model_name='efficientnet_b3',
    pretrained=True,
    freeze_backbone=True
)
```

### Logging

Save training logs for each variant:
- Training history (loss, accuracy per epoch)
- Best validation accuracy
- Training time
- Final test accuracy

---

## Risk Mitigation

### Potential Issues

1. **GPU Memory**: If batch size needs reduction, adjust accordingly
2. **Training Time**: Monitor progress; may need to reduce epochs if time-constrained
3. **No Improvement**: If masks don't help, document and analyze why
4. **Mask Quality**: Some masks may be imperfect; dataset handles missing masks gracefully

### Contingency Plans

- If training takes too long: Reduce epochs to 30-40 (still sufficient for coursework)
- If no improvement: Document findings and discuss in report (negative results are still valid)
- If GPU issues: Use smaller batch sizes or CPU (much slower but functional)

---

## Timeline Estimate

| Phase | Task | Time |
|-------|------|------|
| Setup | Prepare training scripts | 30 min |
| Training 1 | Baseline CNN (no masks) | 30-45 min |
| Training 2 | Baseline CNN (with masks) | 30-45 min |
| Training 3 | ResNet50 (no masks) | 1-2 hours |
| Training 4 | ResNet50 (with masks) | 1-2 hours |
| Training 5 | EfficientNet-B3 (with masks) | 1.5-3 hours |
| Analysis | Evaluate results, generate plots | 1-2 hours |
| **Total** | **Complete training phase** | **~6-10 hours** |

---

## Success Criteria

### Minimum Success

✅ All 5 models train successfully  
✅ Performance metrics recorded for all variants  
✅ Clear comparison showing impact (positive or negative) for Baseline CNN and ResNet50  
✅ EfficientNet-B3 demonstrates best performance  
✅ Visualizations generated  

### Ideal Success

✅ All 5 models train successfully  
✅ Measurable improvement (1-3% accuracy gain) for Baseline CNN and ResNet50  
✅ EfficientNet-B3 achieves highest accuracy (~89-94%)  
✅ Comprehensive visualizations and analysis  
✅ Clear documentation ready for report  

---

## Conclusion

This training strategy provides a balanced approach that:
- Demonstrates thorough experimental methodology
- Shows impact of segmentation-based preprocessing
- Completes within reasonable time constraints
- Generates sufficient data for comprehensive analysis
- Aligns with coursework requirements and expectations

The focus on Baseline CNN and ResNet50 (with and without masks) provides a clear before/after comparison while maintaining feasibility within the project timeline.

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Author**: CT7160NI Computer Vision Coursework

