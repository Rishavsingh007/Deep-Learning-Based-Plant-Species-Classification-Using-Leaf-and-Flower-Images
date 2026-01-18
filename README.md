# Plant Species Classification using Deep Learning

## CT7160NI Computer Vision Coursework
**London Metropolitan University / Islington College**  
**Autumn Semester 2025**

---

## Project Overview

This project implements a **Deep Learning-based Plant Species Classification System** using flower images from the Oxford 102 Flower Dataset. The system utilizes Convolutional Neural Networks (CNNs) with both custom architectures and transfer learning to accurately classify 102 different flower species.

### Project Objectives
1. Design and implement a functional CNN-based plant classification system
2. Apply transfer learning using pre-trained models (ResNet50, EfficientNet-B3)
3. Compare baseline and advanced model performance
4. Demonstrate practical mastery of computer vision techniques
5. Conduct comprehensive error analysis and model interpretability

---

## Project Structure

```
plant-species-classification/
│
├── src/                        # Source code modules
│   ├── data/                   # Data loading and preprocessing
│   │   ├── dataset.py          # Custom PyTorch Dataset classes
│   │   ├── data_loader.py      # DataLoader utilities
│   │   ├── preprocessing.py    # Image preprocessing functions
│   │   └── augmentation.py     # Data augmentation transforms
│   ├── models/                 # Model architectures
│   │   ├── baseline_cnn.py     # Custom CNN from scratch
│   │   ├── resnet_model.py     # ResNet50 transfer learning
│   │   └── efficientnet_model.py # EfficientNet-B3 implementation
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Training loop
│   │   └── callbacks.py        # Early stopping, checkpoints
│   ├── evaluation/             # Evaluation and visualization
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── visualization.py    # Confusion matrices, ROC curves, etc.
│   │   └── attention_analysis.py # Grad-CAM analysis
│   └── utils/                  # Helper functions
│       ├── config.py           # Configuration management
│       └── helpers.py          # Miscellaneous utilities
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset exploration and analysis
│   ├── 02_data_preprocessing.ipynb    # Data preprocessing pipeline
│   ├── 03_model_training.ipynb        # Centralized model training
│   ├── 04_model_evaluation.ipynb      # Comprehensive model evaluation
│   └── 05_inference_error_analysis.ipynb # Error analysis and interpretability
│
├── results/                    # Generated outputs
│   ├── evaluation/             # Evaluation metrics and summaries
│   ├── figures/                # Generated visualizations
│   │   ├── *_confusion_matrix.png
│   │   ├── *_roc_curves.png
│   │   ├── *_training_curves.png
│   │   └── *_per_class_performance.png
│   └── metrics/                # Training metrics and comparisons
│
├── docs/                       # Documentation
│   ├── training_strategy.md    # Training methodology
│   ├── methodology_documentation.md # Implementation details
│   ├── model_comparison_analysis.md # Model comparison report
│   └── preprocessing_module_explanation.md # Preprocessing details
│
├── config.yaml                 # Project configuration
├── requirements.txt            # Python dependencies
├── train_baseline_cnn_no_masks.py # Training script
├── train_resnet50.py           # Training script
└── README.md                   # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or Google Colab
- Git for version control

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Rishavsingh007/Deep-Learning-Based-Plant-Species-Classification-Using-Leaf-and-Flower-Images.git
cd plant-species-classification
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
```bash
# Oxford 102 Flower Dataset
# Download from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

# Place the downloaded files in:
# data/raw/oxford_flowers_102/
```

---

## Dataset

### Oxford 102 Flower Dataset
- **Source:** [VGG Research Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Size:** 8,189 images
- **Classes:** 102 flower categories
- **Resolution:** Variable (typically 500×500 to 1000×1000 pixels)

### Data Split
| Split | Ratio | Samples | Purpose |
|-------|-------|---------|---------|
| Training | ~70% | ~5,700 | Model training |
| Validation | ~15% | ~1,200 | Hyperparameter tuning |
| Test | ~15% | ~1,229 | Final evaluation |

### Data Preprocessing
The project uses **on-the-fly preprocessing** during training:
- **Resize:** Images resized to 224×224 (Baseline CNN, ResNet50) or 300×300 (EfficientNet-B3)
- **Normalization:** ImageNet mean and standard deviation
- **Augmentation (Training only):**
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jittering
  - Random resized crop

---

##  Model Architectures

### 1. Baseline CNN (Custom)
A custom CNN architecture trained from scratch:

```
Input (224×224×3)
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(128) → BatchNorm → ReLU → MaxPool
Conv2D(256) → BatchNorm → ReLU → MaxPool
Conv2D(512) → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling
    ↓
Dense(256) → Dropout(0.5)
    ↓
Dense(102) → Softmax
```

**Key Features:**
- 4 Convolutional layers with Batch Normalization
- Global Average Pooling for parameter efficiency
- Dropout regularization (0.5)
- Total Parameters: ~11.9M
- Image Size: 224×224

**Performance:**
- Test Accuracy: **86.41%**
- Top-5 Accuracy: **97.80%**
- Precision (Macro): 86.99%
- Recall (Macro): 88.95%
- F1-Score (Macro): 86.88%

### 2. ResNet50 (Transfer Learning)
Pre-trained ResNet50 with fine-tuning:

```
Input (224×224×3)
    ↓
ResNet50 Backbone (ImageNet pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(102) → Softmax
```

**Key Features:**
- Pre-trained on ImageNet
- Two-phase training: frozen backbone + fine-tuning
- Total Parameters: ~24.6M
- Image Size: 224×224

**Performance:**
- Test Accuracy: **97.97%**
- Top-5 Accuracy: **99.51%**
- Precision (Macro): 97.62%
- Recall (Macro): 97.53%
- F1-Score (Macro): 97.43%
- ROC-AUC (Macro): 0.9994

### 3. EfficientNet-B3 (Transfer Learning)
Pre-trained EfficientNet-B3 with compound scaling:

```
Input (300×300×3)
    ↓
EfficientNet-B3 Backbone (ImageNet pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(102) → Softmax
```

**Key Features:**
- Pre-trained on ImageNet
- Compound scaling (width, depth, resolution)
- Efficient MBConv blocks with SE attention
- Total Parameters: ~11.1M
- Image Size: 300×300

**Performance:**
- Test Accuracy: **98.94%**  **Best Performance**
- Top-5 Accuracy: **99.76%**
- Precision (Macro): 99.03%
- Recall (Macro): 98.82%
- F1-Score (Macro): 98.86%
- ROC-AUC (Macro): 0.9993

---

##  Usage

### Training a Model

**Option 1: Using Training Scripts**
```bash
# Train Baseline CNN
python train_baseline_cnn_no_masks.py

# Train ResNet50
python train_resnet50.py
```

**Option 2: Using Jupyter Notebooks**
```bash
jupyter notebook notebooks/
```

Then run `03_model_training.ipynb` for centralized training.

**Option 3: Programmatic Training**
```python
from src.models import BaselineCNN, ResNetModel, EfficientNetModel
from src.training import Trainer

# Initialize model
model = ResNetModel(num_classes=102)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

# Train model
trainer.train(epochs=50)
```

### Evaluating a Model

Run `04_model_evaluation.ipynb` for comprehensive evaluation including:
- Test set metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
- ROC curves
- Per-class performance analysis
- Model comparison

### Error Analysis

Run `05_inference_error_analysis.ipynb` for:
- Misclassification analysis
- Grad-CAM visualizations
- Common failure cases
- Class-specific error patterns

---

## Results Summary

### Performance Comparison

| Model | Test Accuracy | Top-5 Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Parameters | Inference Time |
|-------|---------------|----------------|-------------------|----------------|------------------|---------|------------|----------------|
| **Baseline CNN** | 86.41% | 97.80% | 86.99% | 88.95% | 86.88% | 0.997 | 11.9M | 29.87 ms |
| **ResNet50** | 97.97% | 99.51% | 97.62% | 97.53% | 97.43% | 0.999 | 24.6M | 20.82 ms |
| **EfficientNet-B3** | **98.94%**  | **99.76%** | **99.03%** | **98.82%** | **98.86%** | 0.999 | **11.1M** | 24.10 ms |

### Key Findings

1. **Transfer Learning Advantage:**
   - ResNet50 and EfficientNet-B3 significantly outperform the baseline CNN
   - Pre-trained ImageNet weights provide strong feature representations
   - Transfer learning improves generalization with less training time

2. **Model Efficiency:**
   - EfficientNet-B3 achieves the highest accuracy with the fewest parameters (11.1M vs 24.6M)
   - Demonstrates the effectiveness of compound scaling architecture
   - Best accuracy-to-parameter ratio among all models

3. **Classification Performance:**
   - All models show strong top-5 accuracy (>95%), indicating good class separation
   - High macro-averaged F1-scores suggest balanced performance across all 102 classes
   - ROC-AUC scores close to 1.0 indicate excellent discriminative ability

4. **Inference Speed:**
   - ResNet50 has the fastest inference time (20.82 ms/image)
   - EfficientNet-B3 provides best accuracy with acceptable inference speed (24.10 ms/image)

### Visualization Results

The project includes comprehensive visualizations in `results/figures/`:
- **Confusion Matrices:** Normalized confusion matrices with gamma correction for enhanced visibility
- **ROC Curves:** Macro-average and worst-performing classes analysis
- **Training Curves:** Loss and accuracy over epochs
- **Per-Class Performance:** F1-scores, precision, and recall per class
- **Grad-CAM:** Model attention visualizations
- **Error Analysis:** Misclassified examples and failure patterns

---

## Documentation

- **Training Strategy:** `docs/training_strategy.md` - Complete training methodology
- **Methodology:** `docs/methodology_documentation.md` - Implementation details
- **Model Comparison:** `docs/model_comparison_analysis.md` - Detailed comparison report
- **Preprocessing:** `docs/preprocessing_module_explanation.md` - Data preprocessing pipeline
- **Logbook:** `docs/logbook.md` - Development log and notes

---

## Learning Outcomes

| LO | Description | Addressed In |
|----|-------------|--------------|
| LO1 | Apply mathematical/physical principles in CV | CNN architecture design, transfer learning implementation |
| LO2 | Theoretical understanding of CV systems | Model comparison, error analysis, interpretability |
| LO3 | Use software/hardware tools | PyTorch implementation, GPU training, Jupyter notebooks |
| LO4 | Postgraduate-level analysis and reporting | Comprehensive evaluation, documentation, logbook |
| LO5 | Ethical and professional issues in CV | Dataset usage, model limitations, reproducibility |

---

## Key Technologies

- **Framework:** PyTorch 2.0+
- **Pre-trained Models:** torchvision (ResNet50, EfficientNet-B3)
- **Data Processing:** PIL, torchvision transforms
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** scikit-learn metrics
- **Hardware:** CUDA-compatible GPU (recommended)

---

## References

1. He, K. et al. (2016) 'Deep Residual Learning for Image Recognition', CVPR 2016
2. Tan, M. and Le, Q. (2019) 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks', ICML 2019
3. Nilsback, M-E. and Zisserman, A. (2008) 'Automated Flower Classification over a Large Number of Classes', ICVGIP 2008
4. Selvaraju, R.R. et al. (2017) 'Grad-CAM: Visual Explanations from Deep Networks', ICCV 2017

---

## Author

**Student Name:** Rishav Singh  
**Student ID:** NP01MS7A240010  
**Module:** CT7160NI Computer Vision  
**Institution:** London Metropolitan University / Islington College  
**GitHub:** [Rishavsingh007](https://github.com/Rishavsingh007)

---

## License

This project is submitted as part of academic coursework for CT7160NI Computer Vision module.

---

## Acknowledgments

- Oxford 102 Flower Dataset by VGG Research Group
- PyTorch and torchvision communities
- Pre-trained models from torchvision
- Mr. Juned Alam

---

**Last Updated:** January 2026
