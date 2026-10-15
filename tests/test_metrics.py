import numpy as np
import pytest

from prodml.metrics import (
    confusion_matrix,
    ece,
    fit_temperature,
    macro_f1,
    selective_metrics,
    softmax,
    top1,
)


def test_softmax_rows_sum_to_one() -> None:
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_top1() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert top1(probs, np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_macro_f1_known_value() -> None:
    preds = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 1, 1])
    assert macro_f1(preds, labels, 2) == pytest.approx((2 / 3 + 0.8) / 2)


def test_selective_metrics() -> None:
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    coverage, acc = selective_metrics(probs, np.array([0, 1]), threshold=0.7)
    assert coverage == 0.5
    assert acc == 1.0


def test_ece_is_zero_for_a_perfectly_calibrated_toy_case() -> None:
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert ece(probs, np.array([0, 0])) == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    probs = np.array([[0.95, 0.05]] * 10)
    labels = np.array([0] * 5 + [1] * 5)  # right half of the time
    assert ece(probs, labels) == pytest.approx(0.45)


def test_overconfident_logits_get_a_temperature_above_one() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 5)) * 6.0
    labels = rng.integers(0, 5, size=500)  # unrelated to logits
    assert fit_temperature(logits, labels) > 1.0


def test_confusion_matrix() -> None:
    cm = confusion_matrix(np.array([0, 1, 1]), np.array([0, 0, 1]), 2)
    assert cm.tolist() == [[1, 1], [0, 1]]
