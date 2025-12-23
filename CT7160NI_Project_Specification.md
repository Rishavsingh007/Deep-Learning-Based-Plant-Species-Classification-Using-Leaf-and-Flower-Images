# CT7160NI Computer Vision Coursework
## Deep Learning-Based Plant Species Classification Using Leaf and Flower Images

**Module Code:** CT7160NI  
**Module Title:** Computer Vision  
**Coursework Weight:** 50% of total module grades  
**Submission Date:** Sunday, 25 January 2026  

---

## 📋 Executive Summary

This document provides a complete project specification for implementing a **Deep Learning-Based Plant Species Classification System** using leaf and flower images. The project falls under the **Image Classification** category and will utilize CNN architectures with transfer learning approaches.

---

## 🎯 Project Overview

### Project Title
**Deep Learning-Based Plant Species Classification Using Leaf and Flower Images**

### Project Category
✅ **Image Classification** – Implement or fine-tune CNN architectures for object categorization

### Project Objectives
1. Design and implement a functional CNN-based plant classification system
2. Apply transfer learning with pre-trained models (ResNet50, EfficientNet)
3. Compare baseline and advanced model performance
4. Demonstrate practical mastery of computer vision techniques
5. Document the research process with comprehensive evaluation metrics

---

## 📊 Assessment Alignment

| Component | Marks | How This Project Addresses It |
|-----------|-------|-------------------------------|
| Abstract, Literature Review & Background Research | 10 | Review of 7+ papers on CNN and plant classification |
| Methodology, Architecture & Experimentation | 20 | Multiple architectures: Baseline CNN, ResNet50, EfficientNet |
| Code Implementation & System Functionality | 30 | Fully functional Python prototype with PyTorch |
| Results, Analysis & Evaluation | 10 | Comprehensive metrics, confusion matrices, Grad-CAM |
| Conclusion and Recommendations | 10 | Reflective analysis with future work proposals |
| Reflective Logbook | 10 | Weekly documented entries |
| Viva Presentation & Demonstration | 10 | Live demo with prepared presentation |
| **Total** | **100** | |

---

## 📂 Dataset Selection

### Primary Dataset: Oxford 102 Flower Dataset ⭐

| Attribute | Details |
|-----------|---------|
| **Source** | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ |
| **Size** | 8,189 images |
| **Classes** | 102 flower categories |
| **Resolution** | Variable (500x500 to 1000x1000) |
| **Split** | Predefined train/val/test splits available |

**Why This Dataset?**
- ✅ Manageable size for experimentation
- ✅ Standard benchmark in computer vision literature
- ✅ High-quality, curated images
- ✅ Good class balance
- ✅ Challenging fine-grained classification task
- ✅ Well-documented with existing baselines for comparison

### Secondary Dataset: Flavia Leaf Dataset (Optional Extension)

| Attribute | Details |
|-----------|---------|
| **Source** | http://flavia.sourceforge.net/ |
| **Size** | 1,907 images |
| **Classes** | 32 plant species |
| **Resolution** | 1600x1200 |

**Purpose:** To demonstrate multi-organ classification capability (flowers + leaves)

### Data Split Strategy

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

---

## 🏗️ Project Directory Structure

```
plant-species-classification/
│
├── 📁 data/
│   ├── raw/                          # Original downloaded datasets
│   │   ├── oxford_flowers_102/       # Oxford Flower Dataset
│   │   │   ├── jpg/                  # Image files
│   │   │   ├── imagelabels.mat       # Labels
│   │   │   └── setid.mat             # Train/val/test splits
│   │   └── flavia_leaves/            # Optional: Flavia dataset
│   │
│   ├── processed/                    # Preprocessed images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── augmented/                    # Augmented training data (if saved)
│
├── 📁 src/
│   ├── __init__.py
│   │
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Custom PyTorch Dataset classes
│   │   ├── data_loader.py            # DataLoader utilities
│   │   ├── preprocessing.py          # Image preprocessing functions
│   │   └── augmentation.py           # Data augmentation transforms
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py           # Custom CNN from scratch
│   │   ├── resnet_model.py           # ResNet50 transfer learning
│   │   ├── efficientnet_model.py     # EfficientNet transfer learning
│   │   └── ensemble.py               # Ensemble methods
│   │
│   ├── 📁 training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── callbacks.py              # Early stopping, checkpoints
│   │   └── scheduler.py              # Learning rate schedulers
│   │
│   ├── 📁 evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Accuracy, F1, precision, recall
│   │   ├── confusion_matrix.py       # Confusion matrix generation
│   │   └── visualization.py          # Grad-CAM, t-SNE, plots
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logger.py                 # Logging utilities
│       └── helpers.py                # Miscellaneous utilities
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and dataset analysis
│   ├── 02_baseline_model.ipynb       # Custom CNN training
│   ├── 03_transfer_learning.ipynb    # ResNet & EfficientNet
│   ├── 04_hyperparameter_tuning.ipynb# Grid search experiments
│   ├── 05_evaluation.ipynb           # Comprehensive evaluation
│   └── 06_visualization.ipynb        # Grad-CAM and feature maps
│
├── 📁 results/
│   ├── models/                       # Saved model weights (.pth)
│   │   ├── baseline_cnn_best.pth
│   │   ├── resnet50_finetuned.pth
│   │   └── efficientnet_b3.pth
│   ├── figures/                      # Generated plots and charts
│   │   ├── training_curves/
│   │   ├── confusion_matrices/
│   │   ├── roc_curves/
│   │   └── gradcam/
│   ├── metrics/                      # Performance metrics (JSON/CSV)
│   └── logs/                         # Training logs
│
├── 📁 docs/
│   ├── project_proposal.pdf          # Milestone 1 (Week 7)
│   ├── mid_term_report.pdf           # Milestone 2 (Week 10)
│   ├── final_report.pdf              # Final submission
│   └── logbook.md                    # Reflective logbook
│
├── 📁 presentation/
│   ├── viva_slides.pptx              # Viva presentation
│   └── demo_script.md                # Demo walkthrough
│
├── config.yaml                       # Project configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Git ignore rules
└── LICENSE                           # License file
```

---

## 🛠️ Technical Stack & Dependencies

### requirements.txt

```
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0                    # Pre-trained models (EfficientNet)

# Computer Vision
opencv-python>=4.8.0
Pillow>=10.0.0
albumentations>=1.3.0          # Advanced augmentation

# Data Science
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Experiment Tracking (Optional)
tensorboard>=2.13.0
# wandb>=0.15.0                # Weights & Biases (optional)

# Utilities
tqdm>=4.65.0                   # Progress bars
PyYAML>=6.0                    # Config management
jupyter>=1.0.0                 # Notebooks

# Grad-CAM Visualization
grad-cam>=1.4.8
captum>=0.6.0                  # PyTorch interpretability
```

### Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | Google Colab (free) | NVIDIA GTX 1080+ / RTX 3060+ |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 20 GB |
| Python | 3.8+ | 3.10+ |

---

## 🧠 Model Architecture Design

### Model 1: Baseline CNN (Custom Architecture)

```
Input (224x224x3)
    │
    ▼
┌─────────────────────────┐
│ Conv2D(64, 3x3) + BN    │
│ ReLU + MaxPool(2x2)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Conv2D(128, 3x3) + BN   │
│ ReLU + MaxPool(2x2)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Conv2D(256, 3x3) + BN   │
│ ReLU + MaxPool(2x2)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Conv2D(512, 3x3) + BN   │
│ ReLU + MaxPool(2x2)     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ GlobalAveragePooling    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Dense(512) + ReLU       │
│ Dropout(0.5)            │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Dense(102) + Softmax    │
│ (Output Layer)          │
└─────────────────────────┘

Parameters: ~2.5M
Expected Accuracy: 60-70%
```

### Model 2: ResNet50 (Transfer Learning)

```
Input (224x224x3)
    │
    ▼
┌──────────────────────────────┐
│ ResNet50 Backbone            │
│ (Pre-trained on ImageNet)    │
│ ─────────────────────────    │
│ Layers 1-45: FROZEN          │
│ Layers 46+: TRAINABLE        │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ AdaptiveAvgPool2D(1,1)       │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Dense(512) + ReLU            │
│ Dropout(0.3)                 │
│ Dense(102) + Softmax         │
└──────────────────────────────┘

Parameters: ~23.5M (trainable: ~2M)
Expected Accuracy: 80-88%
```

### Model 3: EfficientNet-B3 (Transfer Learning)

```
Input (300x300x3)
    │
    ▼
┌──────────────────────────────┐
│ EfficientNet-B3 Backbone     │
│ (Pre-trained on ImageNet)    │
│ ─────────────────────────    │
│ MBConv blocks with SE        │
│ Compound scaling             │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Global Average Pooling       │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Dense(256) + ReLU            │
│ Dropout(0.3)                 │
│ Dense(102) + Softmax         │
└──────────────────────────────┘

Parameters: ~10.7M
Expected Accuracy: 85-92%
```

---

## 📅 Project Milestones & Timeline

### Milestone Overview

| Milestone | Week | Deliverable | Weight |
|-----------|------|-------------|--------|
| **Milestone 1** | Week 7 | Project Proposal (500 words) | Formative |
| **Milestone 2** | Week 10 | Partial Implementation + Mid-term Report | Formative |
| **Final Submission** | Week 12 | Complete Report + Code + Logbook | 100% |

---

### Detailed Weekly Plan

#### **Phase 1: Foundation (Weeks 5-6)**

##### Week 5: Project Setup & Literature Review
| Task | Hours | Deliverable |
|------|-------|-------------|
| Read 5 core research papers | 4 | Annotated bibliography |
| Set up development environment | 2 | Working Python environment |
| Initialize GitHub repository | 1 | Version control setup |
| Download Oxford 102 dataset | 1 | Raw data downloaded |
| Create project structure | 2 | Directory skeleton |
| **Total** | **10** | |

**Key Papers to Read:**
1. He et al. (2016) - Deep Residual Learning (ResNet)
2. Tan & Le (2019) - EfficientNet
3. Lee et al. (2015) - Deep-plant identification
4. Mohanty et al. (2016) - Plant disease detection
5. Selvaraju et al. (2017) - Grad-CAM

##### Week 6: Data Exploration & Preprocessing
| Task | Hours | Deliverable |
|------|-------|-------------|
| Exploratory Data Analysis | 3 | EDA notebook |
| Implement preprocessing pipeline | 3 | preprocessing.py |
| Create data augmentation transforms | 2 | augmentation.py |
| Build custom Dataset class | 2 | dataset.py |
| Create DataLoaders | 1 | data_loader.py |
| Document data characteristics | 1 | Logbook entry #1 |
| **Total** | **12** | |

**Data Augmentation Pipeline:**
```python
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.RandomRotation(limit=20),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

#### **Phase 2: Model Development (Weeks 7-8)**

##### Week 7: Baseline CNN + Milestone 1 📌
| Task | Hours | Deliverable |
|------|-------|-------------|
| Implement baseline CNN | 4 | baseline_cnn.py |
| Create training loop | 3 | trainer.py |
| Train baseline model (50 epochs) | 2 | Trained model |
| Evaluate baseline performance | 2 | Initial metrics |
| **Write Project Proposal (500 words)** | 3 | **MILESTONE 1** |
| **Total** | **14** | |

**✅ MILESTONE 1 DELIVERABLE: Project Proposal (500 words)**
- Topic selection with justification
- Research outline and objectives
- Initial design diagrams
- Methodology overview

##### Week 8: Transfer Learning Models
| Task | Hours | Deliverable |
|------|-------|-------------|
| Implement ResNet50 model | 3 | resnet_model.py |
| Implement EfficientNet model | 3 | efficientnet_model.py |
| Train both models (Phase 1: frozen) | 3 | Preliminary models |
| Fine-tune models (Phase 2: unfrozen) | 3 | Fine-tuned models |
| Compare architectures | 2 | Comparison table |
| Update logbook | 1 | Logbook entry #2 |
| **Total** | **15** | |

---

#### **Phase 3: Optimization & Evaluation (Weeks 9-10)**

##### Week 9: Hyperparameter Tuning & Optimization
| Task | Hours | Deliverable |
|------|-------|-------------|
| Grid search: learning rates | 3 | Tuning results |
| Grid search: batch sizes | 2 | Tuning results |
| Implement learning rate scheduler | 2 | scheduler.py |
| Implement early stopping | 1 | callbacks.py |
| Train optimized models | 3 | Best models |
| Update logbook | 1 | Logbook entry #3 |
| **Total** | **12** | |

**Hyperparameter Search Space:**
```python
hyperparameters = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'optimizer': ['Adam', 'SGD'],
    'weight_decay': [1e-4, 1e-5],
    'epochs': [50, 100]
}
```

##### Week 10: Comprehensive Evaluation + Milestone 2 📌
| Task | Hours | Deliverable |
|------|-------|-------------|
| Calculate all metrics | 3 | metrics.py |
| Generate confusion matrices | 2 | Confusion matrix plots |
| Create Grad-CAM visualizations | 3 | Grad-CAM images |
| Plot ROC curves | 2 | ROC/AUC plots |
| **Write Mid-term Report Draft** | 4 | **MILESTONE 2** |
| **Total** | **14** | |

**✅ MILESTONE 2 DELIVERABLES:**
- Partial implementation (working models)
- Preliminary training results
- Mid-term report draft
- Initial evaluation metrics

---

#### **Phase 4: Documentation & Submission (Weeks 11-12)**

##### Week 11: Report Writing
| Task | Hours | Deliverable |
|------|-------|-------------|
| Write Abstract (150 words) | 1 | Abstract section |
| Write Introduction (300 words) | 2 | Introduction section |
| Write Literature Review (500 words) | 3 | Background section |
| Write Methodology (600 words) | 4 | Methodology section |
| Write Results (500 words) | 3 | Results section |
| Write Discussion & Conclusion | 2 | Final sections |
| Format references (Harvard) | 1 | Reference list |
| **Total** | **16** | |

##### Week 12: Final Submission 📌
| Task | Hours | Deliverable |
|------|-------|-------------|
| Finalize report (2500 words) | 3 | final_report.pdf |
| Complete code documentation | 2 | Documented code |
| Write README.md | 2 | README.md |
| Finalize requirements.txt | 1 | requirements.txt |
| Complete reflective logbook | 2 | logbook.md |
| Prepare viva presentation | 3 | viva_slides.pptx |
| Test code reproducibility | 2 | Verified code |
| **Submit all deliverables** | 1 | **FINAL SUBMISSION** |
| **Total** | **16** | |

**✅ FINAL SUBMISSION DELIVERABLES:**
- Full Report (2500 words, PDF format)
- Complete source code with documentation
- requirements.txt with all dependencies
- Reflective logbook
- Trained model weights (optional)

---

## 📈 Evaluation Metrics Framework

### Primary Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Overall Accuracy | (TP + TN) / Total | > 85% |
| Precision (Macro) | Avg(TP / (TP + FP)) | > 83% |
| Recall (Macro) | Avg(TP / (TP + FN)) | > 83% |
| F1-Score (Macro) | 2 × (P × R) / (P + R) | > 84% |
| Top-5 Accuracy | Correct in top 5 / Total | > 95% |

### Expected Results Table

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Training Time |
|-------|----------|-----------|--------|----------|------------|---------------|
| Baseline CNN | ~68% | ~67% | ~68% | ~67% | 2.5M | ~30 min |
| ResNet50 | ~85% | ~84% | ~85% | ~84% | 23.5M | ~45 min |
| EfficientNet-B3 | ~89% | ~88% | ~89% | ~88% | 10.7M | ~50 min |
| Ensemble | ~91% | ~90% | ~91% | ~90% | - | - |

### Required Visualizations

1. **Training Curves** - Loss and accuracy over epochs
2. **Confusion Matrix** - Heatmap for all 102 classes (or top 20)
3. **ROC Curves** - Multi-class ROC with AUC
4. **Grad-CAM Heatmaps** - Model attention visualization
5. **t-SNE Embeddings** - Feature space clustering
6. **Per-class Performance** - Bar chart of per-class accuracy
7. **Misclassification Examples** - Grid of error cases

---

## 📝 Report Structure (2500 Words)

### Word Count Distribution

| Section | Words | Percentage |
|---------|-------|------------|
| Abstract | 150 | 6% |
| Introduction | 300 | 12% |
| Background & Literature Review | 500 | 20% |
| Methodology | 600 | 24% |
| Implementation | 300 | 12% |
| Results & Evaluation | 500 | 20% |
| Discussion & Conclusion | 150 | 6% |
| **Total** | **2500** | **100%** |

### Section Details

#### 1. Abstract (150 words)
- Problem statement
- Methodology overview
- Key results (quantitative)
- Main conclusion

#### 2. Introduction (300 words)
- Context and motivation (importance of plant classification)
- Problem statement and scope
- Research objectives (3-4 specific goals)
- Contribution summary

#### 3. Background & Literature Review (500 words)
- CNN fundamentals (convolution, pooling, activation)
- Transfer learning concepts
- Review of 5-7 related papers
- Gap analysis and research justification

#### 4. Methodology (600 words)
- Dataset description (Oxford 102 Flowers)
- Preprocessing pipeline
- Model architectures (with diagrams)
- Training strategy and hyperparameters

#### 5. Implementation (300 words)
- Experimental setup (hardware, software)
- Training procedure and challenges
- Hyperparameter selection process

#### 6. Results & Evaluation (500 words)
- Quantitative results table
- Confusion matrix analysis
- Grad-CAM interpretation
- Comparison with literature benchmarks

#### 7. Discussion & Conclusion (150 words)
- Strengths and limitations
- Key findings
- Future work recommendations

---

## 📓 Reflective Logbook Template

### Entry Format

```markdown
## Logbook Entry #X
**Date:** [DD/MM/YYYY]
**Week:** [X]
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Task 1
- [ ] Task 2

### Key Achievements
- Achievement 1
- Achievement 2

### Challenges Faced
- Challenge 1: [Description]
  - Solution: [How you resolved it]

### Learning Outcomes
- What I learned about [topic]
- New skill acquired: [skill]

### Next Steps
- Planned task 1
- Planned task 2

### Reflection
[Personal reflection on progress, 50-100 words]
```

### Required Entries

| Entry # | Week | Focus Area |
|---------|------|------------|
| 1 | 5-6 | Setup and data exploration |
| 2 | 7 | Baseline model development |
| 3 | 8 | Transfer learning implementation |
| 4 | 9 | Optimization and tuning |
| 5 | 10 | Evaluation and metrics |
| 6 | 11 | Report writing |
| 7 | 12 | Final submission and reflection |

---

## 🎤 Viva Preparation Guide

### Presentation Structure (10-15 slides)

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title & Introduction | 1 min |
| 2 | Problem Statement & Objectives | 1 min |
| 3 | Dataset Overview | 1 min |
| 4 | Methodology Overview | 2 min |
| 5-6 | Model Architectures | 2 min |
| 7 | Training Process | 1 min |
| 8-9 | Results & Metrics | 3 min |
| 10 | Visualizations (Grad-CAM) | 2 min |
| 11 | Demo | 2 min |
| 12 | Conclusion & Future Work | 1 min |
| 13 | Q&A | 4 min |

### Potential Viva Questions

**Technical Questions:**
1. Why did you choose ResNet50 over other architectures?
2. Explain how transfer learning improved your results
3. How does Grad-CAM work and what does it show?
4. What augmentation techniques did you use and why?
5. How did you handle class imbalance (if any)?

**Conceptual Questions:**
1. What are the key differences between your baseline and transfer learning models?
2. Explain the convolution operation in CNNs
3. What is the purpose of batch normalization?
4. Why is dropout used as regularization?

**Critical Analysis:**
1. What are the limitations of your approach?
2. How would you improve your system given more time?
3. How does your work compare to state-of-the-art?
4. What ethical considerations apply to your project?

---

## ✅ Final Submission Checklist

### Documents (Submit to MST Portal)

- [ ] **Final Report** (PDF format, 2500 words)
  - [ ] Abstract
  - [ ] Introduction
  - [ ] Literature Review
  - [ ] Methodology
  - [ ] Implementation
  - [ ] Results & Evaluation
  - [ ] Discussion
  - [ ] Conclusion & Future Work
  - [ ] References (Harvard format)

- [ ] **Reflective Logbook** (PDF or MD format)
  - [ ] Minimum 7 entries
  - [ ] Weekly progress documented
  - [ ] Challenges and solutions
  - [ ] Personal reflections

### Code Submission

- [ ] **Source Code** (ZIP archive)
  - [ ] All Python source files
  - [ ] Jupyter notebooks
  - [ ] requirements.txt
  - [ ] README.md with setup instructions
  - [ ] Configuration files

- [ ] **Code Quality**
  - [ ] Clear documentation/comments
  - [ ] Reproducible results
  - [ ] Modular structure
  - [ ] Version control (Git history)

### Optional

- [ ] Trained model weights (.pth files)
- [ ] Sample predictions
- [ ] Viva presentation slides

---

## 📚 References (Harvard Format Examples)

### Core References

1. He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep residual learning for image recognition', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770-778.

2. Tan, M. and Le, Q. (2019) 'EfficientNet: Rethinking model scaling for convolutional neural networks', *International Conference on Machine Learning*, pp. 6105-6114.

3. Lee, S.H., Chan, C.S., Wilkin, P. and Remagnino, P. (2015) 'Deep-plant: Plant identification with convolutional neural networks', *2015 IEEE International Conference on Image Processing (ICIP)*, pp. 452-456.

4. Mohanty, S.P., Hughes, D.P. and Salathé, M. (2016) 'Using deep learning for image-based plant disease detection', *Frontiers in Plant Science*, 7, p. 1419.

5. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) 'Grad-CAM: Visual explanations from deep networks via gradient-based localization', *Proceedings of the IEEE International Conference on Computer Vision*, pp. 618-626.

6. Szeliski, R. (2021) *Computer Vision: Algorithms and Applications*. 2nd edn. Springer.

7. Gonzalez, R.C., Woods, R.E. and Eddins, S.L. (2008) *Digital Image Processing*. 3rd edn. Pearson.

---

## 🎯 Success Criteria Summary

### To Achieve First Class (70%+):

| Area | Requirements |
|------|-------------|
| **Literature** | 7+ papers reviewed, critical analysis |
| **Methodology** | Multiple architectures, clear rationale |
| **Implementation** | Working code, reproducible results |
| **Results** | Comprehensive metrics, visualizations |
| **Analysis** | Error analysis, benchmark comparison |
| **Documentation** | Clear report, reflective logbook |
| **Viva** | Confident presentation, technical depth |

### Learning Outcomes Addressed

| Learning Outcome | How Addressed |
|-----------------|---------------|
| LO1: Apply mathematical/physical principles | CNN architecture, optimization |
| LO2: Theoretical understanding of CV | Literature review, methodology |
| LO3: Use software/hardware tools | PyTorch, OpenCV implementation |
| LO4: Postgraduate-level analysis | Report, evaluation metrics |
| LO5: Ethical and professional issues | Discussion of dataset ethics, bias |

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Module:** CT7160NI Computer Vision  
**Institution:** London Metropolitan University / Islington College

