from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from prodml.metrics import reliability_bins


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig = Figure(figsize=(10, 9))
    ax = fig.subplots()
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (validation)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def save_reliability_diagram(
    probs_before: np.ndarray, probs_after: np.ndarray, labels: np.ndarray, path: Path
) -> None:
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for probs, name in ((probs_before, "before temperature"), (probs_after, "after")):
        confidence, accuracy, share = reliability_bins(probs, labels)
        used = share > 0
        ax.plot(confidence[used], accuracy[used], marker="o", label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
