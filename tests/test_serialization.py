from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from prodml.config import settings
from prodml.export import export_onnx, main
from prodml.model import PetClassifier

NAMES = [f"breed_{i:02d}" for i in range(37)]


@pytest.fixture(scope="module")
def model() -> PetClassifier:
    torch.manual_seed(0)
    return PetClassifier("resnet18", NAMES, pretrained=False).eval()


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_onnx_logits_match_pytorch_on_200_inputs(model: PetClassifier, tmp_path: Path) -> None:
    path = export_onnx(model, tmp_path / "model.onnx")
    torch.manual_seed(1)
    x = torch.randn(200, 3, 224, 224)
    with torch.inference_mode():
        expected = model(x).numpy()
    actual = _session(path).run(None, {"image": x.numpy()})[0]
    assert actual.shape == (200, 37)
    assert np.allclose(expected, actual, atol=1e-4)


def test_onnx_model_accepts_a_batch_of_one(model: PetClassifier, tmp_path: Path) -> None:
    path = export_onnx(model, tmp_path / "model.onnx")
    out = _session(path).run(None, {"image": torch.zeros(1, 3, 224, 224).numpy()})[0]
    assert out.shape == (1, 37)


def test_export_script_writes_the_onnx_file_next_to_the_checkpoint(
    model: PetClassifier, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    monkeypatch.setattr(settings, "model_path", checkpoint)
    main()
    assert checkpoint.with_suffix(".onnx").exists()
