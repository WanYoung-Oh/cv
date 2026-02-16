# Gap Detector Memory - CV Project

## Project Context
- PyTorch Lightning + Hydra + WanDB document image classification
- 17 classes, 1,570 images
- Target: F1-Macro 0.88+
- Achieved: F1-Macro 0.993 (Step 1 Baseline only)

## Key File Paths
- Design: `docs/02-design/data-augmentation-strategy.md`
- Analysis: `docs/03-analysis/CV.analysis.md`
- Data config: `configs/data/baseline_aug.yaml`
- Training config: `configs/training/baseline_768.yaml`
- DataModule: `src/data/datamodule.py`
- Train script: `src/train.py`
- Model module: `src/models/module.py`

## Analysis History
- 2026-02-14: Code quality analysis (79 -> 85 points)
- 2026-02-15: Data Augmentation Step 1 gap analysis (Match Rate: 95%)

## Key Findings
- Step 1 Baseline achieved F1 0.993, far exceeding design prediction of 0.82-0.85
- 7 additional document-specific augmentations (CLAHE, Perspective, etc.) drove performance
- Step 2 (Bucketing) and Step 3 (TTA/Ensemble) deemed unnecessary given results
- Config uses model.model_name (not model.name) to avoid Python reserved word conflicts
