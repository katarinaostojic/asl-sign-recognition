"""Evaluation metrics (NumPy / scikit-learn only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def summarize(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = np.arange(n_classes)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "precision_macro": float(p[present].mean()),
        "recall_macro": float(r[present].mean()),
        "f1_macro": float(f[present].mean()),
        "per_class": {"precision": p, "recall": r, "f1": f, "support": support},
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def top_confusions(cm: np.ndarray, names: list[str], k: int = 10) -> list[tuple[str, str, int]]:
    """Most frequent (true -> predicted) mistakes."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] > 0:
            out.append((names[i], names[j], int(off[i, j])))
    return out


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> tuple[float, dict]:
    """ECE of the top-1 prediction plus per-bin data for a reliability diagram."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = {"conf": [], "acc": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        bins["conf"].append(float(conf[m].mean()))
        bins["acc"].append(float(correct[m].mean()))
        bins["count"].append(int(m.sum()))
    return float(ece), bins


def risk_coverage(probs: np.ndarray, y_true: np.ndarray, n_points: int = 50):
    """Accuracy when only the `coverage` most confident predictions are kept."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    order = np.argsort(-conf)
    correct = correct[order]
    n = len(correct)
    cov = np.linspace(0.05, 1.0, n_points)
    acc = [correct[: max(1, int(round(c * n)))].mean() for c in cov]
    return cov, np.array(acc)


def confidence_summary(probs: np.ndarray, y_true: np.ndarray) -> dict:
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    return {
        "mean_conf_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
        "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
        "frac_wrong_below_0.9": float((conf[~correct] < 0.9).mean()) if (~correct).any() else float("nan"),
        "frac_correct_above_0.9": float((conf[correct] >= 0.9).mean()) if correct.any() else float("nan"),
    }
