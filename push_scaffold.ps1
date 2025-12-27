<#
PowerShell script to create the "P2 — Vision for Quality Inspection" scaffold,
create branch `project-structure` (or checkout existing), commit and push to origin.

Usage (from repo root in PowerShell):
1) Allow script execution for this session:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

2) Run:
   .\push_scaffold.ps1

Notes:
- Run this in the local clone of your repo (where .git exists).
- Git must be installed and authenticated (SSH or HTTPS + credentials).
- If the remote branch already exists, the script will checkout/pull it.
#>

$ErrorActionPreference = 'Stop'

$BRANCH = "project-structure"
$REMOTE = "origin"

function Write-Info($m){ Write-Host $m -ForegroundColor Cyan }
function Write-Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Write-Ok($m){ Write-Host $m -ForegroundColor Green }

# Basic checks
if (-not (Test-Path ".git")) {
    Write-Error "This directory does not look like a git repository. Run this from your repo root."
    exit 1
}

try {
    git --version > $null 2>&1
} catch {
    Write-Error "Git CLI not found. Please install Git and ensure it's on your PATH."
    exit 1
}

Write-Info "Fetching remote..."
git fetch $REMOTE --prune

# Check if remote branch exists
$remoteBranchInfo = git ls-remote --heads $REMOTE $BRANCH 2>$null
if ($remoteBranchInfo) {
    Write-Info "Remote branch '$BRANCH' exists. Setting up local tracking branch..."
    # If local branch exists, checkout and pull; otherwise create tracking branch
    git show-ref --verify --quiet "refs/heads/$BRANCH"
    if ($LASTEXITCODE -eq 0) {
        git checkout $BRANCH
        git pull $REMOTE $BRANCH
    } else {
        git checkout -b $BRANCH --track $REMOTE/$BRANCH
    }
} else {
    Write-Info "Creating new local branch '$BRANCH'..."
    # If local exists, checkout; else create new
    git show-ref --verify --quiet "refs/heads/$BRANCH"
    if ($LASTEXITCODE -eq 0) {
        git checkout $BRANCH
    } else {
        git checkout -b $BRANCH
    }
}

Write-Info "Creating directories..."
$dirs = @(
    "data\neu\train", "data\neu\test", "data\mvtec",
    "notebooks", "src\data", "src\features", "src\models", "src\evaluation", "src\visualization",
    "configs", "reports\figures", "assets\screenshots", "models", "scripts"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Info "Adding .gitkeep placeholders..."
$gitkeeps = @(
    "notebooks\.gitkeep",
    "configs\.gitkeep",
    "reports\figures\.gitkeep",
    "assets\screenshots\.gitkeep",
    "src\data\.gitkeep",
    "src\features\.gitkeep",
    "src\models\.gitkeep",
    "src\visualization\.gitkeep"
)
foreach ($f in $gitkeeps) {
    if (-not (Test-Path $f)) {
        Set-Content -Path $f -Value "# keep this directory in git" -Encoding UTF8
    }
}

Write-Info "Writing files..."

$readme = @'
# P2 — Vision for Quality Inspection

Mục tiêu
- Stage 1 (quick win): defect classification trên NEU Surface Defect Database
- Stage 2 (stronger): anomaly detection / segmentation trên MVTec AD

Dataset khuyến nghị
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- NEU Surface Defect Database: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Approach gợi ý
- Baseline: OpenCV features + classical ML (SVM) hoặc simple CNN
- Main: CNN / U-Net (segmentation) hoặc anomaly methods (feature embedding + distance map)

Evaluation
- Classification: accuracy, macro-F1, confusion matrix
- Segmentation/anomaly: AUROC + pixel-level metrics (nếu có mask)

Deliverables bắt buộc
- reports/figures/: sample predictions (good vs defect) + confusion matrix
- reports/report.pdf: mô tả data split, augmentations, metric, failure cases
- src/evaluation/: script xuất bảng metrics (CSV)
'@

Set-Content -Path "README.md" -Value $readme -Encoding UTF8

$license = @'
MIT License

Copyright (c) 2025 nonreaction123

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
'@
Set-Content -Path "LICENSE" -Value $license -Encoding UTF8

$gitignore = @'
# Python
__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints
data/
models/
*.pth
.DS_Store
.vscode/
'@
Set-Content -Path ".gitignore" -Value $gitignore -Encoding UTF8

$requirements = @'
numpy
pandas
opencv-python
scikit-image
scikit-learn
matplotlib
seaborn
torch
torchvision
tqdm
jupyter
albumentations
joblib
'@
Set-Content -Path "requirements.txt" -Value $requirements -Encoding UTF8

$dataReadme = @'
# data/

Suggested layout:

data/
  neu/
    train/
    test/
  mvtec/
    <category>/
      train/
      test/
      ground_truth/

Links:
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- NEU: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
'@
Set-Content -Path "data\README.md" -Value $dataReadme -Encoding UTF8

$assetsReadme = @'
# assets/

Add demo GIF and screenshots manually:
- assets/demo.gif
- assets/screenshots/*
'@
Set-Content -Path "assets\README.md" -Value $assetsReadme -Encoding UTF8

$reportsReadme = @'
# reports/

- reports/figures/: sample predictions, confusion matrices, ROC curves
- reports/report.pdf: add manually (final report)
- reports/metrics_classification.csv: will be produced by baseline
'@
Set-Content -Path "reports\README.md" -Value $reportsReadme -Encoding UTF8

# src/evaluation/metrics.py
$metricsPy = @'
"""Evaluation utilities: classification and segmentation metrics + CSV export."""
from typing import Dict, Any
import csv, json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def compute_classification_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist()}

def compute_segmentation_auroc(y_true_mask_flat, y_score_flat) -> Dict[str, Any]:
    try:
        auroc = float(roc_auc_score(y_true_mask_flat, y_score_flat))
    except Exception:
        auroc = float("nan")
    return {"pixel_auroc": auroc}

def save_metrics_csv(metrics: Dict[str, Any], csv_path: str):
    keys = list(metrics.keys())
    values = [json.dumps(metrics[k]) if not isinstance(metrics[k], (int, float, str, bool, type(None))) else metrics[k] for k in keys]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerow(values)
'@
Set-Content -Path "src\evaluation\metrics.py" -Value $metricsPy -Encoding UTF8

# src/models/baseline_svm.py
$baselinePy = @'
"""Baseline: HOG + LinearSVC for NEU dataset."""
import os, glob, joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.io import imread
from src.evaluation.metrics import compute_classification_metrics, save_metrics_csv

def extract_hog(image, pixels_per_cell=(16,16), cells_per_block=(2,2)):
    if image is None:
        return None
    if image.ndim == 3:
        image = image.mean(axis=2)
    return hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)

def load_dataset(root_dir):
    pattern = os.path.join(root_dir, '*', '*')
    files = sorted(glob.glob(pattern))
    X, y = [], []
    for fp in files:
        try:
            img = imread(fp)
            feats = extract_hog(img)
            if feats is None: continue
            X.append(feats)
            y.append(os.path.basename(os.path.dirname(fp)))
        except Exception as e:
            print("Failed", fp, e)
    if len(X) == 0:
        return np.zeros((0,)), np.array([])
    return np.vstack(X), np.array(y)

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    X_train, y_train = load_dataset(os.path.join('data','neu','train'))
    X_test, y_test = load_dataset(os.path.join('data','neu','test'))
    if X_train.size == 0 or X_test.size == 0:
        print("Place NEU dataset under data/neu/train and data/neu/test")
        return
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    metrics = compute_classification_metrics(y_test, y_pred, labels=np.unique(np.concatenate([y_test, y_pred])))
    save_metrics_csv(metrics, os.path.join('reports','metrics_classification.csv'))
    joblib.dump({'model': clf, 'scaler': scaler}, os.path.join('models','baseline_svm.joblib'))
    print("Done. metrics saved to reports/metrics_classification.csv")
if __name__ == "__main__":
    main()
'@
Set-Content -Path "src\models\baseline_svm.py" -Value $baselinePy -Encoding UTF8

# helper shell script (optional)
$shellHelper = @'
#!/usr/bin/env bash
mkdir -p data/neu/train data/neu/test data/mvtec
mkdir -p notebooks src/data src/features src/models src/evaluation src/visualization configs reports/figures assets/screenshots models
touch notebooks/.gitkeep configs/.gitkeep reports/figures/.gitkeep assets/screenshots/.gitkeep src/data/.gitkeep
echo "Scaffold created"
'@
Set-Content -Path "scripts\create_scaffold.sh" -Value $shellHelper -Encoding UTF8

# Stage: git add / commit / push if there are changes
Write-Info "Checking git status..."
$porcelain = git status --porcelain
if ($porcelain) {
    Write-Info "Staging changes..."
    git add .
    $commitMsg = "Add P2 — Vision for Quality Inspection scaffold (MIT, torch, baseline SVM, evaluation scripts)"
    git commit -m $commitMsg
    Write-Info "Pushing to $REMOTE/$BRANCH..."
    git push -u $REMOTE $BRANCH
    Write-Ok "Pushed scaffold to $REMOTE/$BRANCH"
} else {
    Write-Warn "No changes to commit (working tree clean). Nothing was pushed."
}

Write-Ok "Done. If you want, open a Pull Request from '$BRANCH' -> 'main' on GitHub."