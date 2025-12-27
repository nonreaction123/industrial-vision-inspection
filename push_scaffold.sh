#!/usr/bin/env bash
set -euo pipefail

BRANCH="project-structure"
REMOTE="origin"

if [ ! -d ".git" ]; then
  echo "Không thấy .git — hãy chạy trong thư mục repo đã clone."
  exit 1
fi

git fetch ${REMOTE} --prune

# Checkout or create branch (handles existing remote branch)
if git ls-remote --heads ${REMOTE} ${BRANCH} &>/dev/null; then
  echo "Remote branch ${BRANCH} exists. Checking out..."
  if git show-ref --verify --quiet refs/heads/${BRANCH}; then
    git checkout ${BRANCH}
    git pull ${REMOTE} ${BRANCH}
  else
    git checkout -b ${BRANCH} --track ${REMOTE}/${BRANCH}
  fi
else
  echo "Creating new local branch ${BRANCH}..."
  git checkout -b ${BRANCH}
fi

echo "Creating scaffold files and directories..."

# Directories
mkdir -p data/neu/train data/neu/test data/mvtec
mkdir -p notebooks src/data src/features src/models src/evaluation src/visualization configs reports/figures assets/screenshots models scripts

# .gitkeep
touch notebooks/.gitkeep configs/.gitkeep reports/figures/.gitkeep assets/screenshots/.gitkeep src/data/.gitkeep src/features/.gitkeep src/models/.gitkeep src/visualization/.gitkeep

# README.md
cat > README.md <<'EOF'
# P2 — Vision for Quality Inspection

Mục tiêu
- Stage 1: defect classification (NEU)
- Stage 2: anomaly detection / segmentation (MVTec AD)

Datasets:
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- NEU: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Deliverables:
- reports/figures/: sample predictions + confusion matrix
- reports/report.pdf: final report
- src/evaluation/: script export metrics CSV
EOF

# LICENSE
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2025 nonreaction123

Permission is hereby granted, free of charge, to any person obtaining a copy...
EOF

# .gitignore
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints
data/
models/
*.pth
.DS_Store
.vscode/
EOF

# requirements.txt
cat > requirements.txt <<'EOF'
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
EOF

# data README
cat > data/README.md <<'EOF'
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
EOF

# assets README
cat > assets/README.md <<'EOF'
# assets/

Add demo GIF and screenshots manually:
- assets/demo.gif
- assets/screenshots/*
EOF

# reports README
cat > reports/README.md <<'EOF'
# reports/

Place sample predictions and figures in reports/figures/.
Add final report as reports/report.pdf (manually).
EOF

# src/evaluation/metrics.py
mkdir -p src/evaluation
cat > src/evaluation/metrics.py <<'EOF'
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
EOF

# src/models/baseline_svm.py
mkdir -p src/models
cat > src/models/baseline_svm.py <<'EOF'
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
EOF

# helper script
mkdir -p scripts
cat > scripts/create_scaffold.sh <<'EOF'
#!/usr/bin/env bash
mkdir -p data/neu/train data/neu/test data/mvtec
mkdir -p notebooks src/data src/features src/models src/evaluation src/visualization configs reports/figures assets/screenshots models
touch notebooks/.gitkeep configs/.gitkeep reports/figures/.gitkeep assets/screenshots/.gitkeep src/data/.gitkeep
echo "Scaffold created"
EOF
chmod +x scripts/create_scaffold.sh

git add .
git commit -m "Add P2 — Vision for Quality Inspection scaffold (MIT, torch, baseline SVM, evaluation scripts)"
git push -u ${REMOTE} ${BRANCH}

echo "Pushed scaffold to ${REMOTE}/${BRANCH}."