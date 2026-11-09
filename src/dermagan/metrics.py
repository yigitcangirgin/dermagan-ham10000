"""Evaluation metrics used in the thesis: accuracy, precision, recall,
F1-score (weighted average), plus specificity and ROC AUC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dermagan.labels import CLASSES


def specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean one-vs-rest specificity TN / (TN + FP) over classes present."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    values = []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        if tn + fp > 0:
            values.append(tn / (tn + fp))
    return float(np.mean(values)) if values else float("nan")


def compute_metrics(y_true, y_pred, y_prob=None, n_classes: int = len(CLASSES)) -> dict[str, float]:
    """Return a flat dict of metrics. `y_prob` (n_samples, n_classes) is optional."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    common = {"labels": labels, "zero_division": 0}
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", **common),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", **common),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **common),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **common),
        "specificity_mean": specificity(y_true, y_pred, n_classes),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        present = np.unique(y_true)
        if len(present) > 1:
            # ROC AUC is only defined for classes that appear in y_true.
            probs = y_prob[:, present]
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
            result["roc_auc_weighted"] = roc_auc_score(
                y_true, probs, multi_class="ovr", average="weighted", labels=present
            )
    return {k: round(float(v), 4) for k, v in result.items()}


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def plot_confusion_matrix(y_true, y_pred, path: str | Path, title: str = "") -> None:
    """Save a row-normalised confusion matrix as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
