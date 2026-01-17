# Data Preprocessing and Augmentation Pipeline

**CT7160NI Computer Vision Coursework**  
**Plant Species Classification - Preprocessing and Augmentation Implementation**

---

## Overview

This document explains the **actual data preprocessing and augmentation pipeline** implemented in this project. The project uses **on-the-fly preprocessing** during training, meaning all image transformations are applied automatically as images are loaded, without requiring manual preprocessing steps.

---

## Preprocessing Architecture

### On-the-Fly Processing

**Key Concept:** Images are loaded directly from the raw dataset directory and preprocessed in real-time during training. No preprocessed image storage is required.

**Processing Flow:**
```
Raw Image → Dataset.__getitem__() → Transform → Normalized Tensor → Model
```

**Benefits:**
-  Saves disk space (no duplicate preprocessed images)
-  Flexible (easy to change preprocessing strategies)
-  Efficient (GPU data loading handles on-the-fly processing)
-  Standard practice in PyTorch projects

---

## Module Organization

### Core Preprocessing Modules

| Module | Purpose | Used During Training? |
|--------|---------|----------------------|
| `src/data/dataset.py` | Custom Dataset class - loads images |  **YES** |
| `src/data/augmentation.py` | Defines transforms for training/validation |  **YES** |
| `src/data/data_loader.py` | Creates DataLoaders with transforms |  **YES** |
| `src/data/preprocessing.py` | Utility functions for inference |  **NO** (inference only) |

---

## Data Loading Pipeline

### 1. Dataset Class (`src/data/dataset.py`)

**`FlowerDataset` Class:**
- Loads images from `data/raw/oxford_flowers_102/`
- Handles 70/15/15 stratified train/val/test split
- Applies transforms during `__getitem__()`

**Image Loading Process:**
```python
def __getitem__(self, idx):
    # 1. Load image from path
    image = Image.open(img_path).convert('RGB')
    
    # 2. Apply transform (if provided)
    if self.transform is not None:
        image = self.transform(image)
    
    # 3. Return (image_tensor, label)
    return image, label
```

**Key Features:**
- Automatic RGB conversion
- Transforms applied during loading
- Returns PyTorch tensors ready for model input

### 2. Data Split

**Stratified 70/15/15 Split:**
- **Training**: 70% (~5,700 images)
- **Validation**: 15% (~1,200 images)
- **Test**: 15% (~1,229 images)
- **Stratified**: Maintains class distribution across splits

**Implementation:**
- Uses `sklearn.model_selection.train_test_split`
- Ensures balanced representation of all 102 classes
- Random state=42 for reproducibility

---

## Augmentation Transforms

### Training Transforms (`get_train_transforms()`)

**Library:** Albumentations (primary) or torchvision (fallback)

**Applied Augmentations:**

#### 1. **Random Resized Crop**
```python
RandomResizedCrop(
    size=(image_size, image_size),  # 224×224 or 300×300
    scale=(0.7, 1.0),               # Crop 70-100% of image
    ratio=(0.8, 1.2)                # Aspect ratio range
)
```
**Purpose:** Teaches model to recognize flowers at different scales and positions

#### 2. **Affine Transformations**
```python
Affine(
    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # ±10% translation
    scale=(0.9, 1.1),                                          # ±10% scaling
    rotate=(-30, 30),                                          # ±30° rotation
    p=0.5                                                      # 50% probability
)
```
**Purpose:** Handles variations in image capture (viewpoint, zoom, rotation)

#### 3. **Horizontal Flip**
```python
HorizontalFlip(p=0.5)  # 50% probability
```
**Purpose:** Doubles training data with mirror images (flowers are symmetric)

#### 4. **Vertical Flip**
```python
VerticalFlip(p=0.3)  # 30% probability
```
**Purpose:** Less common but useful for certain flower orientations

#### 5. **Color Jittering**
```python
ColorJitter(
    brightness=0.3,   # ±30% brightness variation
    contrast=0.3,     # ±30% contrast variation
    saturation=0.3,   # ±30% saturation variation
    hue=0.1,          # ±10% hue variation
    p=0.5             # 50% probability
)
```
**Purpose:** Handles lighting variations and color differences

#### 6. **Random Brightness/Contrast**
```python
RandomBrightnessContrast(
    brightness_limit=0.2,
    contrast_limit=0.2,
    p=0.3
)
```
**Purpose:** Additional lighting variation

#### 7. **Grayscale Conversion** (10% probability)
```python
OneOf([
    ToGray(p=1.0),  # Convert to grayscale
    NoOp()
], p=0.1)
```
**Purpose:** Encourages model to focus on shape/texture, not just color

#### 8. **Gaussian Blur**
```python
GaussianBlur(blur_limit=(3, 7), p=0.1)  # 10% probability
```
**Purpose:** Handles out-of-focus images, improves robustness

#### 9. **Normalization**
```python
Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
    std=[0.229, 0.224, 0.225]    # ImageNet RGB stds
)
```
**Purpose:** Standardizes pixel values to match pre-trained models (ResNet50, EfficientNet)

#### 10. **Tensor Conversion**
```python
ToTensorV2()  # Converts numpy array to PyTorch tensor
```
**Purpose:** Converts to PyTorch tensor format (C, H, W)

---

### Validation/Test Transforms (`get_val_transforms()`)

**No Augmentation - Only Preprocessing:**

```python
Compose([
    Resize(height=image_size + 32, width=image_size + 32),  # Resize to 256×256 or 332×332
    CenterCrop(height=image_size, width=image_size),         # Crop to 224×224 or 300×300
    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),        # ImageNet normalization
    ToTensorV2()                                             # Tensor conversion
])
```

**Why No Augmentation:**
- Fair evaluation on original images
- Consistent preprocessing for all validation/test samples
- Center crop ensures all images have same size

---

## Image Size Configuration

### Model-Specific Sizes

| Model | Image Size | Resize + Crop |
|-------|------------|---------------|
| **Baseline CNN** | 224×224 | Resize(256×256) → CenterCrop(224×224) |
| **ResNet50** | 224×224 | Resize(256×256) → CenterCrop(224×224) |
| **EfficientNet-B3** | 300×300 | Resize(332×332) → CenterCrop(300×300) |

**Why Larger Size for EfficientNet?**
- EfficientNet benefits from higher resolution due to compound scaling
- 300×300 is optimal for EfficientNet-B3 architecture
- Better feature extraction at higher resolution

---

## Normalization

### ImageNet Statistics

**Mean:** `[0.485, 0.456, 0.406]` (RGB)  
**Std:** `[0.229, 0.224, 0.225]` (RGB)

**Why ImageNet Statistics?**
- Transfer learning models (ResNet50, EfficientNet) pre-trained on ImageNet
- Using same normalization ensures compatibility with pre-trained weights
- Standard practice for transfer learning

**Normalization Formula:**
```
normalized_pixel = (pixel / 255.0 - mean) / std
```

**Result:**
- Pixel values normalized to approximately [-2, 2] range
- Centered around 0 with unit variance
- Optimized for neural network training

---

## Class Imbalance Handling

### Weighted Random Sampling

**Implementation:**
```python
# Calculate class weights (inverse frequency)
class_counts = np.bincount(train_dataset.labels)
class_weights = 1.0 / class_counts

# Create weighted sampler
WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(train_dataset),
    replacement=True
)
```

**Purpose:**
- Ensures balanced sampling across all 102 classes during training
- Prevents model from overfitting to majority classes
- Improves performance on underrepresented classes

**Used For:**
- Baseline CNN training (102 classes, some imbalance)

---

## Data Loader Configuration

### Typical Configuration

```python
create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    batch_size=16,                    # Adjusted for GPU memory
    image_size=224,                   # 224 for ResNet, 300 for EfficientNet
    num_workers=0,                    # 0 for Windows compatibility
    use_weighted_sampler=True,        # For class imbalance (Baseline CNN)
    use_albumentations=True           # Use Albumentations library
)
```

### DataLoader Features

- **Pin Memory**: `pin_memory=True` for faster GPU transfer
- **Drop Last**: `drop_last=True` for training (consistent batch size)
- **Multi-worker Loading**: `num_workers` for parallel data loading (0 on Windows)

---

## Preprocessing Module (`src/data/preprocessing.py`)

### Purpose: Inference Only

**Important:** `preprocessing.py` is **NOT used during training**. It provides utilities for:

1. **Single Image Inference**
   ```python
   from src.data.preprocessing import preprocess_image
   
   # Preprocess a single image for prediction
   image_tensor = preprocess_image('new_image.jpg', target_size=224)
   prediction = model(image_tensor)
   ```

2. **Batch Preprocessing** (alternative approach, not used)
   - `preprocess_dataset()` - Preprocess entire dataset and save to disk
   - Not used in this project (on-the-fly processing preferred)

3. **Utility Functions**
   - `calculate_channel_stats()` - Calculate custom normalization statistics
   - `visualize_preprocessing()` - Visualize preprocessing steps

**Training Uses:**
-  `augmentation.py` - Training and validation transforms
-  `dataset.py` - Image loading with transforms
-  `data_loader.py` - DataLoader creation

**Inference Uses:**
-  `preprocessing.py` - Single image preprocessing for predictions

---

## Complete Preprocessing Pipeline

### Training Pipeline

```
1. Image Path → FlowerDataset.__getitem__()
2. PIL Image.load() → Convert to RGB
3. Apply Training Transforms:
   ├── Random Resized Crop
   ├── Affine Transformations (rotation, translation, scale)
   ├── Horizontal/Vertical Flip
   ├── Color Jittering
   ├── Random Brightness/Contrast
   ├── Grayscale (10% chance)
   ├── Gaussian Blur (10% chance)
   ├── Normalize (ImageNet statistics)
   └── Convert to Tensor
4. Return (image_tensor, label)
5. DataLoader batches tensors
6. Model receives batch
```

### Validation/Test Pipeline

```
1. Image Path → FlowerDataset.__getitem__()
2. PIL Image.load() → Convert to RGB
3. Apply Validation Transforms:
   ├── Resize (image_size + 32)
   ├── Center Crop (image_size)
   ├── Normalize (ImageNet statistics)
   └── Convert to Tensor
4. Return (image_tensor, label)
5. No shuffling, no augmentation
```

---

## Augmentation Strategy Rationale

### Why These Augmentations?

**1. Geometric Augmentations (Crop, Flip, Rotate):**
- Flowers photographed from various angles
- Handles viewpoint variations
- Improves robustness to image orientation

**2. Color Augmentations (Jitter, Brightness, Contrast):**
- Natural lighting variations
- Different camera settings
- Color variations between flowers of same species

**3. Advanced Augmentations (Blur, Grayscale):**
- Real-world conditions (motion blur, focus issues)
- Encourages shape/texture learning beyond color
- Improves generalization

### Augmentation Intensity

**Moderate Augmentation:**
- Balance between diversity and realism
- Preserves flower characteristics while adding variation
- Prevents overfitting without distorting important features

**Probability-Based:**
- Most augmentations have 30-50% probability
- Ensures mix of original and augmented images
- Gradual learning from varied data

---

## Data Statistics

### Dataset Information

- **Total Images**: 8,189
- **Classes**: 102 flower categories
- **Original Resolution**: Variable (typically 500×500 to 1000×1000)
- **Processed Resolution**: 224×224 (Baseline, ResNet) or 300×300 (EfficientNet)

### Split Distribution

| Split | Samples | Percentage | Purpose |
|-------|---------|------------|---------|
| Training | ~5,700 | ~70% | Model training |
| Validation | ~1,200 | ~15% | Hyperparameter tuning |
| Test | ~1,229 | ~15% | Final evaluation |

**Class Distribution:**
- Stratified split ensures balanced representation
- Each split maintains proportional class distribution
- Prevents bias toward majority classes

---

## Implementation Details

### Albumentations vs Torchvision

**Primary:** Albumentations
- More augmentation options
- Better performance
- Flexible composition
- Active development

**Fallback:** Torchvision
- Standard PyTorch transforms
- Simpler but fewer options
- Used if Albumentations unavailable

### Transform Composition

**Training:**
- Sequential application of augmentations
- Probability-based (each augmentation has `p` parameter)
- Normalization always applied (last step)

**Validation:**
- Minimal transforms (resize, crop, normalize)
- Deterministic (no randomness)
- Consistent preprocessing

---

## Key Differences: Training vs Validation

| Aspect | Training | Validation/Test |
|--------|----------|-----------------|
| **Augmentation** |  Extensive (10+ augmentations) |  None |
| **Randomness** |  Random crops, flips, rotations |  Deterministic |
| **Normalization** |  ImageNet stats |  ImageNet stats |
| **Shuffling** |  Yes (or weighted sampler) |  No |
| **Drop Last** |  Yes (consistent batch size) |  No (use all samples) |

---

## Best Practices Implemented

### 1. On-the-Fly Processing
- No manual preprocessing required
- Flexible and space-efficient
- Standard PyTorch practice

### 2. Appropriate Augmentation
- Augmentations match domain (flowers, natural images)
- Moderate intensity preserves important features
- Probability-based for natural variation

### 3. Consistent Normalization
- ImageNet statistics for transfer learning compatibility
- Applied consistently across all models
- Standard for pre-trained models

### 4. Stratified Splitting
- Maintains class distribution
- Prevents data leakage
- Reproducible (random_state=42)

### 5. Class Imbalance Handling
- Weighted sampling for balanced batches
- Important for 102-class problem
- Improves performance on minority classes

---

## Usage Examples

### Creating DataLoaders

```python
from src.data.data_loader import create_dataloaders

# Create loaders with augmentation
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    batch_size=16,
    image_size=224,  # or 300 for EfficientNet
    num_workers=0,
    use_weighted_sampler=True,
    use_albumentations=True
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### Accessing Transforms Directly

```python
from src.data.augmentation import get_train_transforms, get_val_transforms

# Get transforms
train_transforms = get_train_transforms(image_size=224, use_albumentations=True)
val_transforms = get_val_transforms(image_size=224, use_albumentations=True)

# Use with dataset
dataset = FlowerDataset(
    root_dir='data/raw/oxford_flowers_102',
    split='train',
    transform=train_transforms
)
```

### Single Image Preprocessing (Inference)

```python
from src.data.preprocessing import preprocess_image

# Preprocess image for model prediction
image_tensor = preprocess_image(
    'path/to/image.jpg',
    target_size=224,
    normalize=True
)

# Model prediction
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1)
```

---

## Summary

### Preprocessing Approach

| Aspect | Implementation |
|--------|----------------|
| **Method** | On-the-fly processing during data loading |
| **Training Augmentation** | Extensive (10+ augmentations via Albumentations) |
| **Validation Augmentation** | None (resize, crop, normalize only) |
| **Normalization** | ImageNet statistics (mean, std) |
| **Image Sizes** | 224×224 (Baseline, ResNet) or 300×300 (EfficientNet) |
| **Class Imbalance** | Weighted random sampling (Baseline CNN) |

### Key Modules

- **`dataset.py`** - Image loading with transforms
- **`augmentation.py`** - Transform definitions
- **`data_loader.py`** - DataLoader creation
- **`preprocessing.py`** - Inference utilities (not used in training)

### Benefits

1. **No Manual Preprocessing** - Everything automated
2. **Flexible** - Easy to modify augmentation strategies
3. **Efficient** - No duplicate image storage
4. **Standard Practice** - Follows PyTorch best practices
5. **Reproducible** - Stratified splits with random_state

---

**Document Version**: 2.0 (Based on Actual Implementation)  
**Last Updated**: January 2026
**Author**: Rishav Singh (NP01MS7A240010)
