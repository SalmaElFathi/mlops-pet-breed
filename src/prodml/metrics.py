from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def top1(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    scores = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per confidence bin: (mean confidence, accuracy, share of samples)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf = np.zeros(n_bins)
    accuracy = np.zeros(n_bins)
    share = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            mean_conf[i] = conf[mask].mean()
            accuracy[i] = correct[mask].mean()
            share[i] = mask.mean()
    return mean_conf, accuracy, share


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: weighted gap between confidence and accuracy."""
    mean_conf, accuracy, share = reliability_bins(probs, labels, n_bins)
    return float(np.sum(share * np.abs(accuracy - mean_conf)))


def selective_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Coverage (share answered) and accuracy on the answered samples only."""
    accepted = probs.max(axis=1) >= threshold
    coverage = float(accepted.mean())
    if not accepted.any():
        return coverage, float("nan")
    correct = probs.argmax(axis=1) == labels
    return coverage, float(correct[accepted].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature that minimises the negative log-likelihood (grid search)."""
    best_t, best_nll = 1.0, float("inf")
    rows = np.arange(len(labels))
    for t in np.linspace(0.5, 5.0, 91):
        p = softmax(logits, float(t))[rows, labels]
        nll = float(-np.mean(np.log(p + 1e-12)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    return cm
