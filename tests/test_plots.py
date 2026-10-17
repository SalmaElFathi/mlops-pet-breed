import numpy as np

from prodml.plots import save_confusion_matrix, save_reliability_diagram


def test_plots_write_non_empty_files(tmp_path) -> None:
    save_confusion_matrix(np.eye(3, dtype=int), tmp_path / "cm.png")
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    save_reliability_diagram(probs, probs, np.array([0, 1]), tmp_path / "rel.png")
    assert (tmp_path / "cm.png").stat().st_size > 0
    assert (tmp_path / "rel.png").stat().st_size > 0
