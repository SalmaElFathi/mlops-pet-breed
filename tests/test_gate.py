import json
import subprocess

from prodml import gate


def _metrics(tmp_path, top1):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"top1": top1}))
    return str(path)


def _base(monkeypatch, value):
    monkeypatch.setattr(gate, "load_baseline", lambda ref, path: value)


def test_passes_when_better(tmp_path, monkeypatch) -> None:
    _base(monkeypatch, {"top1": 0.95})
    assert gate.main(["--metrics", _metrics(tmp_path, 0.951)]) == 0


def test_passes_within_tolerance(tmp_path, monkeypatch) -> None:
    _base(monkeypatch, {"top1": 0.95})
    assert gate.main(["--metrics", _metrics(tmp_path, 0.9475)]) == 0


def test_fails_on_regression(tmp_path, monkeypatch) -> None:
    _base(monkeypatch, {"top1": 0.95})
    assert gate.main(["--metrics", _metrics(tmp_path, 0.94)]) == 1


def test_skipped_without_baseline(tmp_path, monkeypatch) -> None:
    _base(monkeypatch, None)
    assert gate.main(["--metrics", _metrics(tmp_path, 0.5)]) == 0


def test_load_baseline_handles_a_missing_file(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(gate.subprocess, "check_output", boom)
    assert gate.load_baseline("origin/main", "reports/metrics.json") is None
