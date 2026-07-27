"""Shared evaluation: AUPRC (primary), ROC-AUC, recall at fixed FPR."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "roc_auc": float("nan"), "recall_at_5fpr": float("nan")}
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_at_5fpr": recall_at_fpr(y_true, y_score, 0.05),
    }
