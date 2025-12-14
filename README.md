# CRISP-DM Image Classification Pipeline

This project implements the CRISP-DM process for a 3-class image classification problem using the existing training folders: `train/Black`, `train/Brown`, and `train/White`.

## Steps (CRISP-DM)
- Business Understanding: Classify images into Black/Brown/White categories.
- Data Understanding: Scan dataset, count per class, save summary and distribution plot.
- Data Preparation: Resize, normalize; extract HOG or raw features; split train/val/test.
- Modeling: Train SVM baseline; save model.
- Evaluation: Confusion matrix and classification report saved to `reports/`.
- Deployment: Simple CLI (`src/predict.py`) predicts class for a new image.

## Quick Start

### Install dependencies
```powershell
pip install -r requirements.txt
```

### Run full pipeline
```powershell
python -m src.main
```

Artifacts will be saved under `reports/` and `models/`.

### Predict on a new image
```powershell
python -m src.predict "path/to/image.jpg"
```

## Configuration
Edit `configs/config.yaml` to adjust image size, features, and model parameters.
