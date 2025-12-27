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
