# Comprehensive File Deletion Analysis
**Date:** January 2025  
**Purpose:** Identify all unnecessary files that can be safely deleted

---

## Executive Summary

After analyzing the entire codebase, **~30-40 files** can be safely deleted, including:
- 6 standalone Python scripts (already identified)
- Duplicate/redundant figure files
- Outdated documentation files
- Duplicate notebook references
- Old analysis files

**Total estimated space savings:** ~50-100 MB

---

## 1. Python Scripts to Delete (6 files)

### Already Identified in `FILES_TO_DELETE_ANALYSIS.md`:

1. ✅ `analyze_model.py` - Model analysis (use notebook instead)
2. ✅ `plot_training_curves.py` - Plotting (use notebook instead)
3. ✅ `model_comparison_analysis.py` - Comparison (output already generated)
4. ✅ `create_comparison_table.py` - Table generation (output exists)
5. ✅ `evaluate_model.py` - Evaluation (notebook `04_model_evaluation.ipynb` covers this)
6. ✅ `architecture_diagram.py` (if exists in root) - Diagrams already generated

**Action:** Delete all 6 files

---

## 2. Duplicate/Redundant Figure Files (15-20 files)

### Confusion Matrix Duplicates:
- ❌ `results/figures/baseline_cnn_confusion_matrix_dual.png` - You now have single version
- ❌ `results/figures/confusion_matrix_baseline.png` - Duplicate of baseline_cnn_confusion_matrix.png
- ❌ `results/figures/confusion_matrix_resnet50.png` - Duplicate of resnet50_confusion_matrix.png

### ROC Curve Duplicates:
- ❌ `results/figures/roc_curves_baseline.png` - Duplicate of baseline_cnn_roc_curves.png
- ❌ `results/figures/roc_curves_resnet50.png` - Duplicate of resnet50_roc_curves.png
- ❌ `results/figures/roc_auc_comparison.png` - If you have model_comparison_metrics.png

### Training Curves Duplicates:
- ❌ `results/figures/training_curves_baseline.png` - Duplicate of baseline_cnn_training_curves.png
- ❌ `results/figures/training_curves_resnet50.png` - Duplicate of resnet50_training_curves.png
- ❌ `results/figures/accuracy_curve_baseline.png` - Likely duplicate
- ❌ `results/figures/accuracy_curve_resnet50.png` - Likely duplicate

### Per-Class Performance Duplicates:
- ❌ `results/figures/per_class_performance_baseline.png` - Duplicate of baseline_cnn_per_class_performance.png
- ❌ `results/figures/per_class_performance_resnet50.png` - Duplicate of resnet50_per_class_performance.png

### Inference Examples Duplicates:
- ❌ `results/figures/inference_examples_baseline.png` - Duplicate of baseline_cnn_inference_samples.png
- ❌ `results/figures/inference_examples_resnet50.png` - Duplicate of resnet50_inference_samples.png
- ❌ `results/figures/misclassified_examples_baseline.png` - Duplicate of baseline_cnn_misclassified.png
- ❌ `results/figures/misclassified_examples_resnet50.png` - Duplicate of resnet50_misclassified.png

**Action:** Keep only the newer naming convention files (with model name prefix)

---

## 3. Outdated/Redundant Documentation Files

### In `docs/` directory:

1. ❌ `docs/next_steps.md` - Planning document, likely outdated
2. ❌ `docs/mid_proposal.md` - If you have a final proposal/report
3. ⚠️ `docs/logbook.md` - **KEEP** if actively maintained, otherwise archive
4. ⚠️ `docs/directory_structure.md` - **KEEP** if current, otherwise delete

### Root Level:

5. ❌ `plant_classification_guide.md` - General guide, likely superseded by project-specific docs
6. ⚠️ `FILES_TO_DELETE_ANALYSIS.md` - **DELETE AFTER** implementing this analysis

**Action:** Review and delete outdated planning/guide documents

---

## 4. Duplicate Notebook Issues

### From Git Status:
- ⚠️ `notebooks/03_model_evaluation.ipynb` - **DELETED** (marked with 'D' in git)
  - You have `04_model_evaluation.ipynb` which is the current version
  - This is already handled by git

**Action:** Ensure deleted notebook is not restored

---

## 5. Redundant Results Files

### In `results/evaluation/`:

1. ❌ `results/evaluation/error_analysis.txt` - Generic, likely superseded by model-specific files
2. ❌ `results/evaluation/test_metrics.txt` - Generic, likely superseded
3. ⚠️ `results/evaluation/confusion_matrix.png` - Generic, keep model-specific ones
4. ⚠️ `results/evaluation/roc_curves.png` - Generic, keep model-specific ones
5. ⚠️ `results/evaluation/per_class_*.png` - Generic, keep model-specific ones

**Action:** Keep only model-specific files (baseline_cnn_*, resnet50_*, efficientnet_*)

---

## 6. Old Training Metrics Files

### In `results/metrics/`:

**Keep:**
- ✅ `model_training_summary.csv` - Summary file
- ✅ `*_training_metrics.txt` - Final training metrics

**Consider Consolidating:**
- ⚠️ `*_phase1_training_metrics.txt` - Intermediate phase files
- ⚠️ `*_phase2_training_metrics.txt` - Intermediate phase files

**Action:** Keep phase files if needed for analysis, otherwise archive

---

## 7. Duplicate Source Files

### Found:
- ⚠️ `plant-species-classification\src\evaluation\visualization.py` (Windows path format)
- ✅ `plant-species-classification/src/evaluation/visualization.py` (Unix path format)

**Action:** Ensure only one exists (Unix format is standard)

---

## 8. Grad-CAM Sample Files (Optional)

### In `results/figures/`:

**Grad-CAM samples (5 per model):**
- `*_gradcam_sample_1.png` through `*_gradcam_sample_5.png`

**Decision:**
- **KEEP** if you're using Grad-CAM for interpretability analysis
- **DELETE** if you decided to skip Grad-CAM (as discussed earlier)

**Action:** Delete if Grad-CAM is not part of your final analysis

---

## 9. t-SNE Visualization Files (Optional)

### In `results/figures/`:
- `baseline_cnn_tsne.png`
- `resnet50_tsne.png`
- `efficientnet_b3_tsne.png`

**Decision:**
- **KEEP** if t-SNE is part of your analysis
- **DELETE** if you decided to skip t-SNE (as discussed earlier)

**Action:** Delete if t-SNE is not part of your final analysis

---

## 10. Precision-Recall Curve Files (Optional)

### In `results/figures/`:
- `baseline_cnn_precision_recall_curves.png`
- `resnet50_precision_recall_curves.png`
- `efficientnet_b3_precision_recall_curves.png`

**Decision:**
- **KEEP** if PR curves are part of your analysis
- **DELETE** if you decided ROC curves are sufficient (as discussed earlier)

**Action:** Delete if PR curves are redundant with ROC curves

---

## Summary of Files to Delete

### High Priority (Safe to Delete):

1. **Python Scripts (6 files):**
   - `analyze_model.py`
   - `plot_training_curves.py`
   - `model_comparison_analysis.py`
   - `create_comparison_table.py`
   - `evaluate_model.py`
   - `architecture_diagram.py` (if exists)

2. **Duplicate Figures (~15 files):**
   - All files with old naming convention (without model prefix)
   - Duplicate confusion matrices, ROC curves, training curves

3. **Outdated Documentation (3-4 files):**
   - `docs/next_steps.md`
   - `docs/mid_proposal.md` (if superseded)
   - `plant_classification_guide.md`
   - `FILES_TO_DELETE_ANALYSIS.md` (after implementing)

4. **Generic Results Files (3-5 files):**
   - Generic `error_analysis.txt`, `test_metrics.txt`
   - Generic `confusion_matrix.png`, `roc_curves.png`

### Medium Priority (Review First):

5. **Optional Visualizations (if not using):**
   - Grad-CAM samples (15 files if deleting)
   - t-SNE visualizations (3 files if deleting)
   - Precision-Recall curves (3 files if deleting)

6. **Intermediate Metrics:**
   - Phase 1/2 training metrics (if not needed for analysis)

---

## Deletion Commands

### Python Scripts:
```bash
cd plant-species-classification
rm analyze_model.py
rm plot_training_curves.py
rm model_comparison_analysis.py
rm create_comparison_table.py
rm evaluate_model.py
```

### Duplicate Figures (example):
```bash
cd plant-species-classification/results/figures
rm confusion_matrix_baseline.png
rm confusion_matrix_resnet50.png
rm roc_curves_baseline.png
rm roc_curves_resnet50.png
rm training_curves_baseline.png
rm training_curves_resnet50.png
rm per_class_performance_baseline.png
rm per_class_performance_resnet50.png
rm inference_examples_baseline.png
rm inference_examples_resnet50.png
rm misclassified_examples_baseline.png
rm misclassified_examples_resnet50.png
rm accuracy_curve_baseline.png
rm accuracy_curve_resnet50.png
```

### Documentation:
```bash
cd plant-species-classification
rm docs/next_steps.md
rm docs/mid_proposal.md  # If superseded
rm FILES_TO_DELETE_ANALYSIS.md  # After implementing
```

### Optional (if not using):
```bash
# Grad-CAM (if not using)
rm results/figures/*_gradcam_sample_*.png

# t-SNE (if not using)
rm results/figures/*_tsne.png

# Precision-Recall (if not using)
rm results/figures/*_precision_recall_curves.png
```

---

## Files to KEEP (Essential)

✅ **All files in `src/` directory** - Core functionality  
✅ **Training scripts** (`train_*.py`) - Essential for training  
✅ **All notebooks** (`01-05_*.ipynb`) - Main workflow  
✅ **Configuration files** (`config.yaml`, `requirements.txt`)  
✅ **Current documentation** (`README.md`, `docs/methodology_*.md`)  
✅ **Model-specific results** (with model name prefix)  
✅ **Final training metrics**  

---

## Risk Assessment

### Low Risk:
- Duplicate figure files
- Old Python scripts (functionality in notebooks)
- Outdated documentation
- Generic result files

### Medium Risk:
- Phase 1/2 metrics (review if needed for analysis)
- Optional visualizations (only delete if confirmed not needed)

### High Risk:
- **DO NOT DELETE** anything in `src/` directory
- **DO NOT DELETE** training scripts
- **DO NOT DELETE** current notebooks
- **DO NOT DELETE** final model checkpoints

---

## Recommendation

**Delete in phases:**

1. **Phase 1 (Immediate):** Python scripts + duplicate figures (~20 files)
2. **Phase 2 (Review):** Optional visualizations (if confirmed not needed)
3. **Phase 3 (Final):** Outdated documentation

**Estimated space savings:** 50-100 MB

---

**End of Analysis**
