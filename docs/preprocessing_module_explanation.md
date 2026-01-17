# Preprocessing Module Explanation

**CT7160NI Computer Vision Coursework**  
**Understanding `src/data/preprocessing.py`**

---

## Overview

The `preprocessing.py` module contains utility functions for image preprocessing. However, **it is NOT used during training**. This document explains what's in it and when it might be useful.

---

## What's in `preprocessing.py`?

### Functions Provided

1. **`preprocess_image(image, target_size=224, normalize=True)`**
   - Preprocesses a **single image** for model inference
   - Loads, resizes, normalizes, and converts to tensor
   - **Use case**: Preprocessing images for prediction/inference on new images
   - **Training**: ❌ Not used (training uses transforms from `augmentation.py`)

2. **`resize_image(image_path, output_path, target_size=224)`**
   - Resizes a single image and saves it to disk
   - **Use case**: Manual image resizing if needed
   - **Training**: ❌ Not used

3. **`preprocess_dataset(input_dir, output_dir, target_size=224)`**
   - Preprocesses entire dataset by resizing all images
   - Saves preprocessed images to output directory
   - **Use case**: Batch preprocessing (alternative to on-the-fly processing)
   - **Training**: ❌ Not used (we use on-the-fly processing instead)

4. **`calculate_channel_stats(image_paths)`**
   - Calculates mean and std for each channel across images
   - **Use case**: Computing custom normalization statistics
   - **Training**: ❌ Not used (we use ImageNet stats)

5. **`visualize_preprocessing(image_path, target_size=224)`**
   - Creates visualization of preprocessing steps
   - **Use case**: Understanding/debugging preprocessing
   - **Training**: ❌ Not used

---

## Is `preprocessing.py` Required for Training?

### ❌ **NO - Not Required for Training**

**Training Pipeline Uses:**
- `src/data/augmentation.py` - Provides `get_train_transforms()` and `get_val_transforms()`
- `src/data/dataset.py` - Applies transforms during `__getitem__()`
- **On-the-fly processing** - All preprocessing happens automatically during training

**Training Flow:**
```
Dataset → Load Image → Apply Transforms (from augmentation.py) → Tensor → Model
```

**NOT:**
```
Dataset → preprocess_image() → Tensor → Model  ❌ (This is NOT what happens)
```

---

## When Would You Use `preprocessing.py`?

### 1. **Model Inference/Prediction** ✅

When making predictions on new images (not during training):

```python
from src.data import preprocess_image

# Load and preprocess a single image for prediction
image_tensor = preprocess_image('path/to/new_image.jpg', target_size=224)
prediction = model(image_tensor)
```

**This is useful for:**
- Making predictions on new images
- Testing trained models
- Creating inference scripts

### 2. **Batch Preprocessing (Alternative Approach)** ⚠️

If you wanted to preprocess all images ahead of time (NOT our current approach):

```python
from src.data.preprocessing import preprocess_dataset

# Preprocess entire dataset (would save to data/processed/)
preprocess_dataset(
    input_dir='data/raw/oxford_flowers_102/102flowers/jpg',
    output_dir='data/processed/train',
    target_size=224
)
```

**Why we DON'T use this:**
- On-the-fly processing is more flexible
- Saves disk space
- Easier to change preprocessing strategies

### 3. **Utility Functions** (Optional)

- `calculate_channel_stats()` - If you want custom normalization
- `visualize_preprocessing()` - For understanding preprocessing steps

---

## Current Training Architecture

### What Actually Happens During Training:

```python
# Data loader creation (from data_loader.py)
loaders = create_dataloaders(
    data_dir='data/raw/oxford_flowers_102',
    image_size=224,
    use_masks=True,  # Optional
    apply_background_removal=True  # Optional
)

# Inside create_dataloaders():
train_transform = get_train_transforms(image_size, use_albumentations)  # From augmentation.py
val_transform = get_val_transforms(image_size, use_albumentations)      # From augmentation.py

# Dataset applies transforms automatically (from dataset.py)
dataset = FlowerDataset(
    root_dir=data_dir,
    split='train',
    transform=train_transform  # Applied in __getitem__()
)
```

**Key Point**: `preprocessing.py` is **NOT** called during this process!

---

## Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Required for Training?** | ❌ **NO** | Training uses `augmentation.py` transforms |
| **Required for Inference?** | ✅ **YES** | `preprocess_image()` useful for predictions |
| **Can Delete?** | ⚠️ **Not Recommended** | Useful for inference/utilities, part of API |
| **Should Use for Training?** | ❌ **NO** | Use on-the-fly transforms instead |

---

## Recommendation

**Keep the file** because:
1. `preprocess_image()` is exported in `src/data/__init__.py` (part of module API)
2. Useful for inference/prediction on new images
3. Utilities might be helpful for debugging/analysis
4. Small file, doesn't hurt to keep

**But remember:**
- It's **NOT used during training**
- Training uses `augmentation.py` transforms instead
- The file is optional/utility code, not core training pipeline

---

## Related Files

**Training Pipeline:**
- `src/data/augmentation.py` - ✅ **USED** for training transforms
- `src/data/dataset.py` - ✅ **USED** for dataset loading
- `src/data/data_loader.py` - ✅ **USED** for creating data loaders

**Utility (Not Used in Training):**
- `src/data/preprocessing.py` - ⚠️ Utility for inference/optional preprocessing

---

**Document Version**: 1.0  
**Last Updated**: December 2024

