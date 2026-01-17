# Methodology Code Examples and Diagrams Documentation

This document provides code examples and visualizations for the methodology section of the mid-proposal.

## Generated Diagrams

The following diagrams have been generated and saved in the `diagrams/` directory:

1. **`resnet50_architecture.png`** - Visual representation of the ResNet50 architecture used in this project
2. **`baseline_cnn_architecture.png`** - Visual representation of the Baseline CNN architecture
3. **`complete_pipeline.png`** - Complete processing pipeline from input to output

## Code Examples

All code examples are available in `methodology_code_examples.py`. The following sections are covered:

### 5.1 Pre-processing

#### Noise Reduction
```python
from methodology_code_examples import preprocess_image_noise_reduction

# Apply Gaussian blur
original, gaussian = preprocess_image_noise_reduction('image.jpg', method='gaussian')

# Apply median filter
original, median = preprocess_image_noise_reduction('image.jpg', method='median')

# Apply bilateral filter
original, bilateral = preprocess_image_noise_reduction('image.jpg', method='bilateral')
```

#### Normalization and Resizing
```python
from methodology_code_examples import preprocess_normalization_resize

# Preprocess image with normalization
result = preprocess_normalization_resize('image.jpg', target_size=224)
# Returns: original, resized, normalized images with size information
```

#### Color Space Conversion
```python
from methodology_code_examples import preprocess_color_space_conversion

# Convert to different color spaces
color_spaces = preprocess_color_space_conversion('image.jpg')
# Returns: RGB, HSV, LAB, and Grayscale versions
```

### 5.2 Feature Extraction / Representation

#### Hand-crafted Features: HOG
```python
from methodology_code_examples import extract_handcrafted_features_hog

# Extract HOG features
hog_result = extract_handcrafted_features_hog('image.jpg')
# Returns: features vector, HOG visualization image, feature dimension
```

#### Hand-crafted Features: SIFT
```python
from methodology_code_examples import extract_handcrafted_features_sift

# Extract SIFT features
sift_result = extract_handcrafted_features_sift('image.jpg')
# Returns: keypoints, descriptors, number of keypoints, visualization
```

#### Hand-crafted Features: Harris Corners
```python
from methodology_code_examples import extract_handcrafted_features_harris

# Extract Harris corner features
harris_result = extract_handcrafted_features_harris('image.jpg')
# Returns: Harris response, corner locations, visualization
```

#### Learned Features: CNN Embeddings
```python
from methodology_code_examples import extract_cnn_features
import torch

# Extract features from trained model
model = load_trained_model()  # Your trained ResNet50 model
image_tensor = preprocess_image('image.jpg')  # Preprocessed image tensor
features = extract_cnn_features(model, image_tensor, layer_name='backbone')
# Returns: Feature vector (2048-dim for ResNet50)
```

### 5.3 Core Algorithm / Model

#### Architecture Visualization

The model architectures can be visualized using the provided functions:

```python
from methodology_code_examples import (
    visualize_resnet50_architecture,
    visualize_baseline_cnn_architecture,
    visualize_complete_pipeline
)

# Generate ResNet50 architecture diagram
fig1 = visualize_resnet50_architecture()
fig1.savefig('resnet50_architecture.png', dpi=300, bbox_inches='tight')

# Generate Baseline CNN architecture diagram
fig2 = visualize_baseline_cnn_architecture()
fig2.savefig('baseline_cnn_architecture.png', dpi=300, bbox_inches='tight')

# Generate complete pipeline diagram
fig3 = visualize_complete_pipeline()
fig3.savefig('complete_pipeline.png', dpi=300, bbox_inches='tight')
```

### 5.4 Post-processing

#### Thresholding
```python
from methodology_code_examples import postprocess_thresholding
import numpy as np

# Model output probabilities (102 classes)
probabilities = np.array([0.1, 0.05, ..., 0.85, ...])  # Example

# Apply thresholding
result = postprocess_thresholding(probabilities, threshold=0.5)
# Returns: predicted class, confidence, is_confident flag, message
```

#### Top-K Predictions
```python
from methodology_code_examples import postprocess_top_k_predictions

# Get top-5 predictions
top_k = postprocess_top_k_predictions(probabilities, k=5)
# Returns: top-k classes, probabilities, and pairs
```

#### Temporal Smoothing
```python
from methodology_code_examples import postprocess_smoothing

# Apply smoothing to prediction sequence
predictions = [1, 1, 2, 1, 1, 1, 2, 2, 1, 1]  # Example sequence
smoothed = postprocess_smoothing(predictions, window_size=5)
# Returns: Smoothed prediction sequence
```

## Complete Pipeline Example

Here's a complete example of the entire pipeline:

```python
import torch
from methodology_code_examples import (
    preprocess_normalization_resize,
    extract_cnn_features,
    postprocess_top_k_predictions
)

# 1. Preprocessing
preprocessed = preprocess_normalization_resize('flower_image.jpg', target_size=224)
image_tensor = torch.from_numpy(preprocessed['normalized']).permute(2, 0, 1).unsqueeze(0)

# 2. Feature Extraction (via model forward pass)
model = load_trained_resnet50()
model.eval()
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).numpy()[0]

# 3. Post-processing
top_5 = postprocess_top_k_predictions(probabilities, k=5)
print(f"Top 5 predictions: {top_5['top_k_pairs']}")
```

## Integration with Project Code

These examples are designed to work with the actual project codebase:

- **Preprocessing**: Uses the same normalization and resizing as `src/data/preprocessing.py`
- **Feature Extraction**: Compatible with models in `src/models/`
- **Post-processing**: Can be integrated with `src/evaluation/metrics.py`

## Running the Code

To generate all diagrams:

```bash
cd plant-species-classification/docs
python methodology_code_examples.py
```

This will create:
- `diagrams/resnet50_architecture.png`
- `diagrams/baseline_cnn_architecture.png`
- `diagrams/complete_pipeline.png`

## Dependencies

Required packages:
- numpy
- opencv-python (cv2)
- Pillow (PIL)
- matplotlib
- torch
- torchvision
- scikit-image (for HOG features)
- seaborn (optional, for better styling)

Install with:
```bash
pip install numpy opencv-python pillow matplotlib torch torchvision scikit-image seaborn
```

