"""Tabular ML evaluation metrics for binary classification."""
from __future__ import annotations
from typing import Dict
import numpy as np


def safe_auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_score).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y_true, y_score) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_score).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob).ravel()
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def background_rejection_at_signal_eff(y_true, y_score, signal_eff: float = 0.5) -> float:
    """Physics figure of merit: 1 / FPR at the threshold that yields the
    target true-positive rate. Higher is better."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_score).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.searchsorted(tpr, signal_eff))
    idx = max(1, min(idx, len(fpr) - 1))
    f = float(fpr[idx])
    return float("inf") if f == 0 else 1.0 / f


def confusion_at_threshold(y_true, y_prob, threshold: float = 0.5) -> Dict[str, int]:
    y = np.asarray(y_true).ravel().astype(int)
    pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def compute_split_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Aggregate binary-classification metrics for one split."""
    from sklearn.metrics import (
        accuracy_score, log_loss, matthews_corrcoef,
        precision_score, recall_score, f1_score, fbeta_score,
    )
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(y_prob).ravel()
    pred = (p >= threshold).astype(int)
    out: Dict[str, float] = {
        "n": int(len(y)),
        "auc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "ece": expected_calibration_error(y, p),
        "accuracy": float(accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": (float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
                     if len(np.unique(y)) > 1 else float("nan")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2.0, zero_division=0)),
        "positive_prevalence_true": float(np.mean(y)),
        "positive_prevalence_pred": float(np.mean(pred)),
        "br_at_50sig": background_rejection_at_signal_eff(y, p, 0.5),
        "br_at_70sig": background_rejection_at_signal_eff(y, p, 0.7),
        "br_at_90sig": background_rejection_at_signal_eff(y, p, 0.9),
    }
    cm = confusion_at_threshold(y, p, threshold)
    out.update({f"cm_{k}": v for k, v in cm.items()})
    return out
