# Project Directory Structure

**CT7160NI Computer Vision Coursework**  
**Plant Species Classification - Complete Project Structure**

---

## Overview

This document provides a comprehensive overview of the project directory structure, explaining the purpose and contents of each directory and key files.

---

## Complete Project Structure

```
plant-species-classification/
│
├── .gitignore                    # Git ignore rules
├── config.yaml                    # Project configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── train_baseline_cnn_no_masks.py # Training script for Baseline CNN
├── train_resnet50.py              # Training script for ResNet50
│
├── src/                           # Source code modules
│   ├── __init__.py
│   │
│   ├── data/                      # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py             # Custom PyTorch Dataset classes
│   │   ├── data_loader.py         # DataLoader creation utilities
│   │   ├── preprocessing.py       # Image preprocessing functions
│   │   └── augmentation.py        # Data augmentation transforms
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py        # Custom CNN from scratch
│   │   ├── resnet_model.py        # ResNet50 implementation
│   │   └── efficientnet_model.py # EfficientNet-B3 implementation
│   │
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   └── callbacks.py           # Early stopping, checkpoints
│   │
│   ├── evaluation/                # Evaluation and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics calculation
│   │   ├── visualization.py       # Plotting and visualization
│   │   └── attention_analysis.py  # Grad-CAM analysis
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       └── helpers.py             # Miscellaneous utilities
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset exploration and analysis
│   ├── 02_data_preprocessing.ipynb    # Data preprocessing pipeline
│   ├── 03_model_training.ipynb        # Centralized model training
│   ├── 04_model_evaluation.ipynb     # Comprehensive model evaluation
│   └── 05_inference_error_analysis.ipynb # Error analysis and interpretability
│
├── results/                       # Generated outputs
│   ├── evaluation/                # Evaluation metrics and summaries
│   │   ├── baseline_cnn_error_analysis.txt
│   │   ├── resnet50_error_analysis.txt
│   │   ├── efficientnet_b3_error_analysis.txt
│   │   ├── evaluation_summary.txt
│   │   ├── inference_error_analysis_summary.txt
│   │   ├── model_comparison.csv
│   │   ├── comprehensive_metrics_comparison.csv
│   │   └── test_metrics.txt
│   │
│   ├── figures/                   # Generated visualizations
│   │   ├── baseline_cnn_*.png     # Baseline CNN visualizations
│   │   ├── resnet50_*.png         # ResNet50 visualizations
│   │   ├── efficientnet_b3_*.png  # EfficientNet-B3 visualizations
│   │   └── model_comparison_*.png # Model comparison plots
│   │
│   └── metrics/                   # Training metrics
│       ├── baseline_cnn_improved_no_masks_training_metrics.txt
│       ├── resnet50_training_metrics.txt
│       ├── efficientnet_b3_training_metrics.txt
│       └── model_training_summary.csv
│
└── docs/                          # Documentation
    ├── directory_structure.md      # This file
    ├── training_strategy.md       # Training methodology
    ├── model_comparison_analysis.md # Model comparison report    
    ├── preprocessing_module_explanation.md # Preprocessing details    
    └── diagrams/                  # Architecture diagrams
        ├── baseline_cnn_architecture.png
        ├── resnet50_architecture.png
        └── model_architectures_comparison.png
```

---

## Directory Descriptions

### Root Directory

**Configuration Files:**
- `config.yaml` - Project configuration (paths, hyperparameters, training settings)
- `requirements.txt` - Python package dependencies
- `README.md` - Main project documentation
- `.gitignore` - Git ignore patterns

**Training Scripts:**
- `train_baseline_cnn_no_masks.py` - Standalone script for Baseline CNN training
- `train_resnet50.py` - Standalone script for ResNet50 training

---

### `src/` - Source Code Modules

Core Python modules organized by functionality.

#### `src/data/` - Data Handling

| File | Purpose |
|------|---------|
| `dataset.py` | Custom PyTorch Dataset classes for Oxford 102 Flowers |
| `data_loader.py` | Functions to create DataLoaders with proper splits |
| `preprocessing.py` | Image preprocessing utilities (resize, normalize) |
| `augmentation.py` | Data augmentation transforms (rotation, flip, color jitter) |

**Key Functions:**
- `create_dataloaders()` - Main function to create train/val/test loaders
- `OxfordFlowers102Dataset` - Custom dataset class
- `get_transforms()` - Get augmentation and preprocessing transforms

#### `src/models/` - Model Architectures

| File | Purpose |
|------|---------|
| `baseline_cnn.py` | Custom 4-layer CNN trained from scratch |
| `resnet_model.py` | ResNet50 with transfer learning |
| `efficientnet_model.py` | EfficientNet-B3 with transfer learning |

**Key Classes:**
- `BaselineCNN` - Custom CNN architecture
- `ResNetModel` - ResNet50 wrapper with custom classifier
- `EfficientNetModel` - EfficientNet-B3 wrapper

#### `src/training/` - Training Utilities

| File | Purpose |
|------|---------|
| `trainer.py` | Main training loop with checkpointing and logging |
| `callbacks.py` | Early stopping and learning rate scheduling callbacks |

**Key Classes:**
- `Trainer` - Centralized training class
- `EarlyStopping` - Early stopping callback
- `ModelCheckpoint` - Checkpoint saving callback

#### `src/evaluation/` - Evaluation and Visualization

| File | Purpose |
|------|---------|
| `metrics.py` | Calculate accuracy, precision, recall, F1, ROC-AUC |
| `visualization.py` | Generate confusion matrices, ROC curves, training plots |
| `attention_analysis.py` | Grad-CAM visualization utilities |

**Key Functions:**
- `calculate_metrics()` - Comprehensive metrics calculation
- `plot_confusion_matrix()` - Confusion matrix visualization
- `plot_roc_curves()` - ROC curve plotting
- `plot_training_history()` - Training curves

#### `src/utils/` - Utilities

| File | Purpose |
|------|---------|
| `config.py` | Configuration loading and management |
| `helpers.py` | Miscellaneous helper functions |

---

### `notebooks/` - Jupyter Notebooks

Interactive notebooks for the complete workflow:

| Notebook | Purpose | Key Contents |
|----------|---------|--------------|
| `01_data_exploration.ipynb` | Dataset analysis | Data statistics, class distribution, sample images |
| `02_data_preprocessing.ipynb` | Preprocessing pipeline | Data loading, augmentation examples, split creation |
| `03_model_training.ipynb` | Model training | Centralized training for all three models |
| `04_model_evaluation.ipynb` | Model evaluation | Test metrics, confusion matrices, ROC curves, comparisons |
| `05_inference_error_analysis.ipynb` | Error analysis | Misclassifications, Grad-CAM, failure patterns |

---

### `results/` - Generated Outputs

All generated results from training and evaluation.

#### `results/evaluation/` - Evaluation Results

**Text Files:**
- `evaluation_summary.txt` - Comprehensive evaluation summary
- `baseline_cnn_error_analysis.txt` - Baseline CNN error analysis
- `resnet50_error_analysis.txt` - ResNet50 error analysis
- `efficientnet_b3_error_analysis.txt` - EfficientNet-B3 error analysis
- `inference_error_analysis_summary.txt` - Combined error analysis
- `test_metrics.txt` - Test set metrics

**CSV Files:**
- `model_comparison.csv` - Model performance comparison
- `comprehensive_metrics_comparison.csv` - Detailed metrics comparison

**PNG Files (Generic):**
- `confusion_matrix.png` - Generic confusion matrix
- `roc_curves.png` - Generic ROC curves
- `per_class_*.png` - Per-class performance plots

#### `results/figures/` - Visualizations

**Model-Specific Visualizations:**

**Baseline CNN:**
- `baseline_cnn_confusion_matrix.png` - Confusion matrix
- `baseline_cnn_roc_curves.png` - ROC curves
- `baseline_cnn_training_curves.png` - Training history
- `baseline_cnn_per_class_performance.png` - Per-class metrics
- `baseline_cnn_per_class_f1.png` - Per-class F1-scores
- `baseline_cnn_misclassified.png` - Misclassified examples
- `baseline_cnn_inference_samples.png` - Sample predictions
- `baseline_cnn_confidence_analysis.png` - Confidence distribution
- `baseline_cnn_confusion_patterns.png` - Confusion patterns
- `baseline_cnn_gradcam_sample_*.png` - Grad-CAM visualizations (5 samples)
- `baseline_cnn_precision_recall_curves.png` - Precision-Recall curves
- `baseline_cnn_tsne.png` - t-SNE feature visualization

**ResNet50:**
- `resnet50_confusion_matrix.png`
- `resnet50_roc_curves.png`
- `resnet50_training_curves.png`
- `resnet50_per_class_performance.png`
- `resnet50_per_class_f1.png`
- `resnet50_misclassified.png`
- `resnet50_inference_samples.png`
- `resnet50_confidence_analysis.png`
- `resnet50_confusion_patterns.png`
- `resnet50_gradcam_sample_*.png` (5 samples)
- `resnet50_precision_recall_curves.png`
- `resnet50_tsne.png`

**EfficientNet-B3:**
- `efficientnet_b3_confusion_matrix.png`
- `efficientnet_b3_roc_curves.png`
- `efficientnet_b3_training_curves.png`
- `efficientnet-b3_per_class_performance.png`
- `efficientnet_b3_per_class_f1.png`
- `efficientnet-b3_misclassified.png`
- `efficientnet-b3_inference_samples.png`
- `efficientnet-b3_confidence_analysis.png`
- `efficientnet-b3_confusion_patterns.png`
- `efficientnet_b3_precision_recall_curves.png`
- `efficientnet_b3_tsne.png`

**Comparison Visualizations:**
- `model_comparison_metrics.png` - Metrics comparison chart
- `model_comparison_radar.png` - Radar chart comparison
- `model_comparison_training.png` - Training curves comparison
- `common_failure_cases_comparison.png` - Error overlap analysis
- `error_overlap_analysis.png` - Shared misclassifications
- `additional_metrics_comparison.png` - Extended metrics comparison

**Data Analysis Visualizations:**
- `sample_images.png` - Sample dataset images
- `augmentation_examples.png` - Data augmentation examples
- `preprocessing_pipeline.png` - Preprocessing visualization
- `image_size_analysis.png` - Image size distribution
- `training_batch_sample.png` - Training batch visualization
- `validation_batch_sample.png` - Validation batch visualization
- `baseline_cnn_architecture_diagram.png` - Architecture diagram

#### `results/metrics/` - Training Metrics

**Training Metrics Files:**
- `baseline_cnn_improved_no_masks_training_metrics.txt` - Baseline CNN training history
- `resnet50_training_metrics.txt` - ResNet50 training history
- `efficientnet_b3_training_metrics.txt` - EfficientNet-B3 training history
- `model_training_summary.csv` - Summary of all training runs

**Metrics File Format:**
- Epoch-by-epoch training and validation metrics
- Loss, accuracy, top-5 accuracy
- Learning rate per epoch
- Best validation accuracy and epoch

---

### `docs/` - Documentation

**Markdown Documentation:**
- `directory_structure.md` - This file (project structure)
- `training_strategy.md` - Training methodology and implementation
- `model_comparison_analysis.md` - Comprehensive model comparison
- `methodology_documentation.md` - Implementation details
- `preprocessing_module_explanation.md` - Preprocessing pipeline details
- `logbook.md` - Development log and notes
- `mid_proposal.md` - Mid-term project proposal

**Diagrams:**
- `diagrams/baseline_cnn_architecture.png` - Baseline CNN architecture
- `diagrams/resnet50_architecture.png` - ResNet50 architecture
- `diagrams/model_architectures_comparison.png` - Architecture comparison
- `diagrams/model_comparison_table.png` - Model comparison table
- `diagrams/*.pdf` - PDF versions of diagrams

---

## File Naming Conventions

### Model Files

**Training Scripts:**
- `train_{model_name}.py` - e.g., `train_baseline_cnn_no_masks.py`

**Model Checkpoints (if saved):**
- `{model_name}_best.pth` - Best model based on validation accuracy
- `{model_name}_final.pth` - Final model after all epochs

### Results Files

**Figures:**
- `{model_name}_{visualization_type}.png`
- Examples:
  - `baseline_cnn_confusion_matrix.png`
  - `resnet50_roc_curves.png`
  - `efficientnet_b3_training_curves.png`

**Metrics:**
- `{model_name}_training_metrics.txt` - Training history
- `{model_name}_error_analysis.txt` - Error analysis

**CSV Files:**
- `model_comparison.csv` - Model comparison table
- `comprehensive_metrics_comparison.csv` - Detailed metrics
- `model_training_summary.csv` - Training summary

---

## Data Directory Structure

**Note:** The data directory is not included in the repository (too large, in `.gitignore`).

**Expected Structure:**
```
data/
└── raw/
    └── oxford_flowers_102/
        ├── 102flowers/
        │   └── jpg/              # All flower images (8,189 images)
        ├── imagelabels.mat       # Image labels
        └── setid.mat            # Train/val/test split indices
```

**Data Loading:**
- Images loaded directly from `data/raw/oxford_flowers_102/`
- No preprocessing required before training
- On-the-fly preprocessing during data loading

---

## Key File Purposes

### Configuration

**`config.yaml`:**
- Project paths (data, models, results)
- Training hyperparameters
- Model configurations
- Evaluation settings

**`requirements.txt`:**
- Python package versions
- Core dependencies: PyTorch, torchvision, numpy, matplotlib, etc.

### Training Scripts

**`train_baseline_cnn_no_masks.py`:**
- Standalone script for Baseline CNN training
- Can be run from command line
- Saves model checkpoints and metrics

**`train_resnet50.py`:**
- Standalone script for ResNet50 training
- Two-phase training (frozen → fine-tuning)
- Saves model checkpoints and metrics

### Notebooks

**`03_model_training.ipynb`:**
- Centralized training for all models
- Can train Baseline CNN, ResNet50, and EfficientNet-B3
- Saves training metrics and checkpoints

**`04_model_evaluation.ipynb`:**
- Comprehensive model evaluation
- Generates all visualizations
- Creates comparison tables and summaries

**`05_inference_error_analysis.ipynb`:**
- Error analysis and interpretability
- Grad-CAM visualizations
- Misclassification patterns

---

## Directory Size Estimates

| Directory | Estimated Size | Contents |
|-----------|----------------|----------|
| `src/` | ~500 KB | Python source code |
| `notebooks/` | ~10-20 MB | Jupyter notebooks (with outputs) |
| `results/figures/` | ~50-100 MB | PNG visualizations |
| `results/metrics/` | ~1-5 MB | Text and CSV files |
| `results/evaluation/` | ~1-2 MB | Evaluation summaries |
| `docs/` | ~5-10 MB | Documentation and diagrams |
| **Total** | **~70-140 MB** | (excluding data and model checkpoints) |

**Model Checkpoints (if saved):**
- Baseline CNN: ~45 MB
- ResNet50: ~94 MB
- EfficientNet-B3: ~42 MB
- **Total**: ~180 MB

---

## Automatic Directory Creation

**Directories Created Automatically:**

The code automatically creates required directories if they don't exist:

```python
# From trainer.py and other modules
Path(save_dir).mkdir(parents=True, exist_ok=True)
```

**Directories Created:**
- `results/models/` - For model checkpoints
- `results/figures/` - For visualizations
- `results/metrics/` - For training metrics
- `results/evaluation/` - For evaluation results

**No Manual Setup Required:**
- All directories are created automatically when needed
- Just ensure you have write permissions

---

## Git Repository Structure

**Tracked Files:**
-  All source code (`src/`)
-  All notebooks (`notebooks/`)
-  Configuration files (`config.yaml`, `requirements.txt`)
-  Documentation (`docs/`)
-  Results summaries (`results/evaluation/`, `results/metrics/`)
-  Key visualizations (`results/figures/`)

**Ignored Files (`.gitignore`):**
-  Data directory (`data/`)
-  Model checkpoints (`results/models/*.pth`)
-  Python cache (`__pycache__/`, `*.pyc`)
-  Jupyter checkpoints (`.ipynb_checkpoints/`)
-  Virtual environments (`venv/`, `env/`)
-  IDE files (`.idea/`, `.vscode/`)

---

## Usage Workflow

### 1. Setup
```bash
cd plant-species-classification
pip install -r requirements.txt
```

### 2. Data Preparation
- Download Oxford 102 Flower Dataset
- Place in `data/raw/oxford_flowers_102/`
- No manual preprocessing needed

### 3. Training
```bash
# Option 1: Training scripts
python train_baseline_cnn_no_masks.py
python train_resnet50.py

# Option 2: Jupyter notebook
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Evaluation
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

### 5. Error Analysis
```bash
jupyter notebook notebooks/05_inference_error_analysis.ipynb
```

---

## Best Practices

### File Organization

1. **Keep source code modular** - Each module in `src/` has a specific purpose
2. **Use notebooks for exploration** - Interactive analysis in `notebooks/`
3. **Save all results** - Keep outputs in `results/` for reproducibility
4. **Document changes** - Update `docs/` and `logbook.md`

### Version Control

1. **Commit source code** - All `src/` files should be tracked
2. **Commit notebooks** - Include notebooks (without large outputs if possible)
3. **Commit key results** - Important visualizations and summaries
4. **Ignore large files** - Model checkpoints, data, cache files

### Backup Recommendations

**Critical Files to Backup:**
- Model checkpoints (`results/models/*.pth`)
- Training metrics (`results/metrics/*.txt`)
- Key visualizations (`results/figures/`)
- Source code (`src/`)
- Notebooks (`notebooks/`)

---

**Document Version**: 2.0 (Based on Actual Implementation)  
**Last Updated**: January 2026  
**Author**: Rishav Singh (NP01MS7A240010)
