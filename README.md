# 🌿 Plant Species Classification using Deep Learning

## CT7160NI Computer Vision Coursework
**London Metropolitan University / Islington College**  
**Autumn Semester 2025**

---

## 📋 Project Overview

This project implements a **Deep Learning-based Plant Species Classification System** using leaf and flower images. The system utilizes Convolutional Neural Networks (CNNs) with transfer learning to accurately classify plant species from the Oxford 102 Flower Dataset.

### Project Objectives
1. Design and implement a functional CNN-based plant classification system
2. Apply transfer learning using pre-trained models (ResNet50, EfficientNet)
3. Compare baseline and advanced model performance
4. Demonstrate practical mastery of computer vision techniques

---

## 🏗️ Project Structure

```
plant-species-classification/
│
├── data/
│   ├── raw/                    # Original datasets
│   │   └── oxford_flowers_102/ # Oxford 102 Flower Dataset
│   └── processed/              # Preprocessed images
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training utilities
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Helper functions
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_visualization.ipynb
│
├── results/
│   ├── models/                 # Saved model weights
│   ├── figures/                # Generated plots
│   ├── metrics/                # Performance metrics
│   └── logs/                   # Training logs
│
├── docs/                       # Documentation and reports
├── presentation/               # Viva presentation materials
│
├── config.yaml                 # Project configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or Google Colab
- Git for version control

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
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

## 📊 Dataset

### Oxford 102 Flower Dataset
- **Source:** [VGG Research Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Size:** 8,189 images
- **Classes:** 102 flower categories
- **Resolution:** Variable (500x500 to 1000x1000)

### Data Split
| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

### Data Preprocessing
**⚠️ Important: No manual preprocessing required!**

The dataset pipeline uses **on-the-fly preprocessing**:
- Images are loaded directly from raw directory during training
- All preprocessing (resize, normalization, augmentation) happens automatically via transforms
- See `docs/training_strategy.md` for complete preprocessing details

**Required Data Structure:**
```
data/raw/oxford_flowers_102/
├── 102flowers/jpg/          # Flower images (or jpg/ folder)
├── imagelabels.mat          # Labels
└── setid.mat               # Train/val/test splits (optional, uses custom split)
```

---

## 🧠 Model Architectures

### 1. Baseline CNN (Custom)
- 4 Convolutional layers with Batch Normalization
- Global Average Pooling
- Fully connected layers with Dropout
- **Achieved Accuracy:** 42.94% (validation)
- **Note:** Lower than initial expectations due to training from scratch on a challenging 102-class dataset. This serves as a baseline for comparison with transfer learning approaches.

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned top layers
- **Expected Accuracy:** 80-88%

### 3. EfficientNet-B3 (Transfer Learning)
- Pre-trained on ImageNet
- Efficient compound scaling
- **Expected Accuracy:** 85-92%

---

## 🚀 Usage

### Training a Model

```python
from src.models import BaselineCNN, ResNetModel
from src.training import Trainer

# Initialize model
model = ResNetModel(num_classes=102)

# Create trainer
trainer = Trainer(model, train_loader, val_loader)

# Train model
trainer.train(epochs=50)
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

### Evaluating a Model

```python
from src.evaluation import evaluate_model

# Load trained model
model = load_model('results/models/resnet50_best.pth')

# Evaluate
results = evaluate_model(model, test_loader)
print(f"Test Accuracy: {results['accuracy']:.2f}%")
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Baseline CNN | 72.50% | - | - | - | ✅ Trained |
| Baseline CNN (Improved) | 59.90% | - | - | - | ✅ Trained |
| ResNet50 | ~85% | ~84% | ~85% | ~84% | ⏳ Expected |
| EfficientNet-B3 | ~89% | ~88% | ~89% | ~88% | ⏳ Expected |

**Note:** The Baseline CNN achieved 42.94% validation accuracy, which is lower than initially expected (60-70%). This is typical for models trained from scratch on complex multi-class problems. Transfer learning models (ResNet50, EfficientNet) are expected to achieve significantly higher accuracy due to pre-training on ImageNet.

---

## 📝 Documentation

- **Training Strategy:** `docs/training_strategy.md` - Complete training plan
- **Logbook:** `docs/logbook.md`
- **Project Proposal:** `docs/project_proposal.pdf`
- **Mid-term Report:** `docs/mid_term_report.pdf`
- **Final Report:** `docs/final_report.pdf`

### Training Strategy Overview

This project implements **Option B (Strategic Addition)** training strategy:
- **5 Model Variants:** 
  - Baseline CNN (with and without background removal)
  - ResNet50 (with and without background removal)
  - EfficientNet-B3 (best performance)
- **Expected Time:** ~4.5-8.5 hours total training time
- **Purpose:** Comprehensive model comparison and best-case performance demonstration
- **Best Performance:** EfficientNet-B3 expected to achieve ~89-94% accuracy

See `docs/training_strategy.md` for complete details.

---

## 🎓 Learning Outcomes

| LO | Description | Addressed In |
|----|-------------|--------------|
| LO1 | Apply mathematical/physical principles in CV | Implementation, Report |
| LO2 | Theoretical understanding of CV systems | Report, Viva |
| LO3 | Use software/hardware tools | Code, Implementation |
| LO4 | Postgraduate-level analysis and reporting | Report, Logbook |
| LO5 | Ethical and professional issues in CV | Report, Viva |

---

## 📚 References

1. He, K. et al. (2016) 'Deep Residual Learning for Image Recognition'
2. Tan, M. and Le, Q. (2019) 'EfficientNet: Rethinking Model Scaling'
3. Lee, S.H. et al. (2015) 'Deep-plant: Plant Identification with CNNs'
4. Szeliski, R. (2021) 'Computer Vision: Algorithms and Applications'

---

## 👤 Author

**Student Name:** Rishav Singh
**Student ID:** NP01MS7A240010
**Module:** CT7160NI Computer Vision  
**Institution:** London Metropolitan University / Islington College

---

## 📄 License

This project is submitted as part of academic coursework for CT7160NI Computer Vision module.

---

**Last Updated:** December 2025

