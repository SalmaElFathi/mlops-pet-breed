import json

from prodml.train import write_metrics_file


def test_write_metrics_file_rounds_and_handles_nan(tmp_path) -> None:
    path = tmp_path / "out" / "metrics.json"
    write_metrics_file({"top1": 0.9497281234, "selective_accuracy": float("nan")}, path)
    data = json.loads(path.read_text())
    assert data == {"top1": 0.949728, "selective_accuracy": None}
