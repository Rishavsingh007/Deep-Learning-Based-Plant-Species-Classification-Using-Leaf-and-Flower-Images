# CT7160NI Computer Vision Coursework
# Reflective Logbook

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Project Title:** Deep Learning-Based Plant Species Classification Using Leaf and Flower Images  
**Module:** CT7160NI Computer Vision  
**Academic Year:** 2025/26

---

## Logbook Entry #1: Project Setup & Literature Review
**Date:** [DD/MM/YYYY]  
**Week:** 5-6  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Read core research papers on plant classification
- [ ] Set up development environment
- [ ] Initialized GitHub repository
- [ ] Downloaded Oxford 102 Flower Dataset
- [ ] Created project directory structure

### Key Achievements
- Successfully set up Python environment with PyTorch and required libraries
- Understood the fundamentals of CNN architectures for image classification
- Gained insight into transfer learning approaches from literature

### Challenges Faced
- **Challenge:** [Describe any challenge]
  - **Solution:** [How you resolved it]

### Key Learnings
- [What you learned about CNNs, transfer learning, etc.]
- [Technical skills acquired]

### Papers Reviewed
1. He et al. (2016) - ResNet: Key insights on residual connections
2. [Paper 2] - [Key insights]
3. [Paper 3] - [Key insights]

### Reflection
[Personal reflection on progress, what worked well, what could be improved]

### Next Steps
- Begin exploratory data analysis
- Implement data preprocessing pipeline
- Start baseline CNN implementation

---

## Logbook Entry #2: Data Exploration & Preprocessing
**Date:** [DD/MM/YYYY]  
**Week:** 6  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Completed exploratory data analysis
- [ ] Analyzed class distribution
- [ ] Implemented data augmentation pipeline
- [ ] Created custom Dataset class
- [ ] Tested DataLoader functionality

### Key Achievements
- [Achievement 1]
- [Achievement 2]

### Data Analysis Findings
- Dataset contains 8,189 images across 102 flower categories
- Class distribution: [observations]
- Image quality assessment: [observations]

### Data Augmentation Strategy
Applied the following augmentations:
- Random rotation (±20°)
- Horizontal flip
- Color jittering
- Random resized crop

### Challenges Faced
- **Challenge:** [e.g., Dataset loading issues]
  - **Solution:** [How resolved]

### Reflection
[Thoughts on data quality, preprocessing decisions, etc.]

### Next Steps
- Implement baseline CNN architecture
- Begin training experiments

---

## Logbook Entry #3: Baseline Model Development
**Date:** [DD/MM/YYYY]  
**Week:** 7  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Designed baseline CNN architecture
- [ ] Implemented training loop
- [ ] Trained baseline model
- [ ] Evaluated initial performance
- [ ] Completed Milestone 1: Project Proposal

### Model Architecture
```
[Describe your baseline architecture]
Conv2D(64) -> BatchNorm -> ReLU -> MaxPool
Conv2D(128) -> BatchNorm -> ReLU -> MaxPool
...
```

### Training Results
| Metric | Value |
|--------|-------|
| Final Training Accuracy | XX% |
| Final Validation Accuracy | XX% |
| Training Time | XX minutes |

### Key Observations
- [What worked well]
- [Areas for improvement]

### Challenges Faced
- **Challenge:** [e.g., Overfitting]
  - **Solution:** [Applied dropout, data augmentation]

### Milestone 1 Completion
Submitted project proposal covering:
- Topic selection and justification
- Research objectives
- Initial methodology
- Architecture diagrams

### Reflection
[Thoughts on baseline performance, learning experience]

### Next Steps
- Implement transfer learning with ResNet50
- Compare performance with baseline

---

## Logbook Entry #4: Transfer Learning Implementation
**Date:** [DD/MM/YYYY]  
**Week:** 8  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Implemented ResNet50 transfer learning
- [ ] Implemented EfficientNet-B3
- [ ] Trained both models with frozen backbone
- [ ] Fine-tuned with unfrozen layers
- [ ] Compared architectures

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | XX% | XX% | XX% | XX% |
| ResNet50 | XX% | XX% | XX% | XX% |
| EfficientNet-B3 | XX% | XX% | XX% | XX% |

### Transfer Learning Strategy
1. Phase 1: Frozen backbone, trained classification head
2. Phase 2: Unfroze backbone, fine-tuned with lower learning rate

### Key Observations
- Transfer learning significantly improved accuracy
- [Other observations]

### Challenges Faced
- **Challenge:** [e.g., GPU memory issues]
  - **Solution:** [Reduced batch size, used gradient accumulation]

### Reflection
[Thoughts on transfer learning effectiveness, model comparison]

### Next Steps
- Hyperparameter optimization
- Generate comprehensive evaluation metrics

---

## Logbook Entry #5: Optimization & Hyperparameter Tuning
**Date:** [DD/MM/YYYY]  
**Week:** 9  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Grid search for learning rates
- [ ] Experimented with batch sizes
- [ ] Implemented learning rate scheduling
- [ ] Applied early stopping
- [ ] Optimized best model

### Hyperparameter Experiments
| Experiment | Learning Rate | Batch Size | Accuracy |
|------------|---------------|------------|----------|
| 1 | 1e-3 | 32 | XX% |
| 2 | 1e-4 | 32 | XX% |
| 3 | 1e-4 | 64 | XX% |

### Best Configuration
- Learning Rate: [value]
- Batch Size: [value]
- Optimizer: [Adam/SGD]
- Scheduler: [ReduceLROnPlateau]

### Challenges Faced
- **Challenge:** [e.g., Finding optimal hyperparameters]
  - **Solution:** [Systematic grid search]

### Reflection
[Thoughts on hyperparameter importance, tuning process]

### Next Steps
- Comprehensive evaluation with all metrics
- Generate visualizations

---

## Logbook Entry #6: Evaluation & Visualization
**Date:** [DD/MM/YYYY]  
**Week:** 10  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Calculated comprehensive metrics
- [ ] Generated confusion matrix
- [ ] Created Grad-CAM visualizations
- [ ] Plotted ROC curves
- [ ] Analyzed misclassifications
- [ ] Completed Milestone 2

### Final Results
| Metric | Baseline | ResNet50 | EfficientNet |
|--------|----------|----------|--------------|
| Accuracy | XX% | XX% | XX% |
| Precision | XX% | XX% | XX% |
| Recall | XX% | XX% | XX% |
| F1-Score | XX% | XX% | XX% |
| Top-5 Accuracy | XX% | XX% | XX% |

### Key Visualizations Created
1. Training/Validation curves
2. Confusion matrix
3. ROC curves
4. Grad-CAM heatmaps
5. t-SNE embeddings (optional)

### Error Analysis Findings
- Most common misclassifications: [describe]
- Challenging classes: [list]
- Grad-CAM insights: [what model focuses on]

### Milestone 2 Completion
Submitted:
- Partial implementation with working models
- Preliminary results and graphs
- Mid-term report draft

### Reflection
[Thoughts on model performance, visualization insights]

### Next Steps
- Complete final report writing
- Prepare for viva presentation

---

## Logbook Entry #7: Final Report & Submission
**Date:** [DD/MM/YYYY]  
**Week:** 11-12  
**Hours Spent:** [X hours]

### Tasks Completed
- [ ] Completed 2500-word report
- [ ] Finalized all code documentation
- [ ] Created comprehensive README
- [ ] Prepared requirements.txt
- [ ] Prepared viva presentation
- [ ] Final submission

### Report Sections Completed
- [ ] Abstract (150 words)
- [ ] Introduction (300 words)
- [ ] Literature Review (500 words)
- [ ] Methodology (600 words)
- [ ] Implementation (300 words)
- [ ] Results (500 words)
- [ ] Conclusion (150 words)
- [ ] References (Harvard format)

### Final Model Performance
Best Model: [EfficientNet-B3]
- Test Accuracy: XX%
- Test F1-Score: XX%

### Comparison with Literature
| Study | Dataset | Accuracy |
|-------|---------|----------|
| Lee et al. (2015) | Oxford 102 | 80% |
| Our ResNet50 | Oxford 102 | XX% |
| Our EfficientNet | Oxford 102 | XX% |

### Project Achievements
1. Successfully implemented 3 CNN architectures
2. Achieved XX% accuracy (improvement over baseline)
3. Created comprehensive evaluation framework
4. Demonstrated transfer learning effectiveness

### Challenges Throughout Project
1. [Major challenge 1] - [How overcome]
2. [Major challenge 2] - [How overcome]
3. [Major challenge 3] - [How overcome]

### Overall Reflection
[Comprehensive reflection on the entire project journey]
- What I learned about computer vision
- What I learned about deep learning
- Skills developed
- What I would do differently
- How this relates to my career goals

### Learning Outcomes Achieved
- LO1: Applied mathematical principles (convolution, optimization)
- LO2: Demonstrated theoretical understanding (literature review, methodology)
- LO3: Used software tools effectively (PyTorch, OpenCV)
- LO4: Conducted postgraduate-level analysis (report, evaluation)
- LO5: Considered ethical issues (data usage, bias)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Hours Spent | [XX hours] |
| Number of Experiments | [XX] |
| Papers Reviewed | [XX] |
| Lines of Code | [XX] |
| Final Accuracy | [XX%] |

---

**Logbook Completed:** [Date]  
**Student Signature:** [Your Name]

---

*This logbook documents my individual work on the CT7160NI Computer Vision coursework. All code and analysis presented is my own work unless otherwise cited.*

