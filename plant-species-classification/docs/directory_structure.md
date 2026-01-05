# Results Directory Structure

**CT7160NI Computer Vision Coursework**  
**Directory Organization for Training Results**

---

## Directory Structure

```
results/
├── models/              # Saved model checkpoints
│   ├── baseline_cnn_no_masks_best.pth
│   ├── baseline_cnn_no_masks_final.pth
│   ├── baseline_cnn_with_masks_best.pth
│   ├── resnet50_no_masks_best.pth
│   ├── resnet50_with_masks_best.pth
│   ├── efficientnet_b3_with_masks_best.pth
│   └── checkpoints/     # Optional: Additional checkpoints
│
├── figures/             # Generated visualizations
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── gradcam/
│   └── comparisons/
│
├── metrics/             # Performance metrics (JSON/CSV)
│   ├── baseline_cnn_no_masks_metrics.json
│   ├── performance_comparison.csv
│   └── training_history.json
│
└── logs/                # Training logs
    ├── baseline_cnn_no_masks.log
    └── training_summary.txt
```

---

## What Gets Saved Where

### `results/models/`

**Saved by Trainer:**
- `{model_name}_best.pth` - Best model based on validation accuracy
- `{model_name}_final.pth` - Final model after all epochs

**Checkpoint Contents:**
- Model state dict (weights)
- Optimizer state dict
- Scheduler state dict (if used)
- Training history
- Best validation accuracy
- Epoch number

**Example Files:**
- `baseline_cnn_no_masks_best.pth`
- `baseline_cnn_no_masks_final.pth`
- `baseline_cnn_with_masks_best.pth`
- `resnet50_no_masks_best.pth`
- `resnet50_with_masks_best.pth`
- `efficientnet_b3_with_masks_best.pth`

### `results/figures/`

**Saved by Evaluation/Visualization:**
- Training curves (loss and accuracy plots)
- Confusion matrices
- Performance comparison charts
- Grad-CAM visualizations
- Sample predictions
- Attention analysis plots

**Note**: Currently contains mask test visualizations from testing phase.

### `results/metrics/`

**Saved by Evaluation:**
- Performance metrics (accuracy, precision, recall, F1)
- Training history (JSON format)
- Comparison tables (CSV format)
- Per-class performance metrics

### `results/logs/`

**Saved by Training:**
- Training logs (if logging enabled)
- Training summaries
- Error logs (if any)

---

## Automatic Directory Creation

**Good News**: The Trainer class automatically creates directories if they don't exist!

From `trainer.py`:
```python
self.save_dir = Path(save_dir)
self.save_dir.mkdir(parents=True, exist_ok=True)  # Creates if missing
```

**This means:**
- ✅ If `results/models/` doesn't exist, it will be created
- ✅ If `results/models/checkpoints/` doesn't exist, it will be created
- ✅ No manual directory creation needed

---

## Current Status

**All Required Directories Exist:**
- ✅ `results/models/` - Ready for model checkpoints
- ✅ `results/figures/` - Ready for visualizations (already has mask test images)
- ✅ `results/metrics/` - Ready for metrics files
- ✅ `results/logs/` - Ready for training logs

**Optional:**
- `results/models/checkpoints/` - Will be created if needed by trainer

---

## Disk Space Considerations

**Estimated Space Requirements:**

| Item | Size per Model | Total (5 models) |
|------|----------------|------------------|
| Model Checkpoints | ~50-200 MB | ~250 MB - 1 GB |
| Training History | ~1-5 MB | ~5-25 MB |
| Visualizations | ~1-10 MB | ~5-50 MB |
| Metrics Files | <1 MB | <5 MB |
| **Total** | - | **~300 MB - 1.1 GB** |

**Note**: Model sizes vary:
- Baseline CNN: ~50-100 MB
- ResNet50: ~100-200 MB
- EfficientNet-B3: ~150-250 MB

---

## Verification Before Training

**Quick Check:**
```python
from pathlib import Path

# Check required directories
required = [
    'results/models',
    'results/figures',
    'results/metrics',
    'results/logs'
]

for dir_path in required:
    path = Path(dir_path)
    if path.exists():
        print(f"✅ {dir_path} exists")
    else:
        print(f"⚠️ {dir_path} missing (will be created automatically)")
```

---

## File Naming Convention

**Model Files:**
- `{model_name}_best.pth` - Best validation accuracy
- `{model_name}_final.pth` - After all epochs
- `{model_name}_epoch{N}.pth` - Optional: Per-epoch checkpoints

**Naming Pattern:**
- `baseline_cnn_no_masks`
- `baseline_cnn_with_masks`
- `resnet50_no_masks`
- `resnet50_with_masks`
- `efficientnet_b3_with_masks`

---

## Backup Recommendations

**Before Training:**
- Ensure sufficient disk space (~1-2 GB free)
- Consider backing up existing results if important

**After Training:**
- Backup model checkpoints (most important)
- Save training history for analysis
- Keep visualizations for report

---

**Document Version**: 1.0  
**Last Updated**: December 2024

