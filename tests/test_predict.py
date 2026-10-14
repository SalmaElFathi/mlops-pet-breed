from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms as T

from prodml.config import params
from prodml.dataset import PetDataset
from prodml.predict import Predictor
from prodml.transforms import build_eval_transform


def _training_path_logits(predictor: Predictor, image_path: Path, tmp_path: Path) -> torch.Tensor:
    """Logits obtained the way validation does it: manifest, PetDataset, eval transform."""
    names = predictor.model.class_names
    manifest = tmp_path / "manifest.json"
    label_map = tmp_path / "label_map.json"
    record = {"image_id": "s", "path": str(image_path), "breed": names[0], "split": "val"}
    manifest.write_text(json.dumps([record]))
    label_map.write_text(json.dumps({n: i for i, n in enumerate(names)}))

    t = params["train"]
    transform = build_eval_transform(t["image_size"], t["resize_size"])
    x, _ = PetDataset(manifest, label_map, "val", transform)[0]
    with torch.inference_mode():
        return predictor.model(x.unsqueeze(0))


def test_confidence_is_a_probability_and_top3_is_sorted(
    predictor: Predictor, sample_image_path: Path
) -> None:
    with Image.open(sample_image_path) as im:
        result = predictor.predict(im)
    confidences = [s.confidence for s in result.top_3]
    assert isinstance(result.confidence, float)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1.0 + 1e-6


def test_prediction_is_deterministic(predictor: Predictor, sample_image_path: Path) -> None:
    with Image.open(sample_image_path) as im:
        first = predictor.predict(im)
        second = predictor.predict(im)
    assert first == second


def test_training_path_and_api_path_agree(
    predictor: Predictor, client: TestClient, sample_image_path: Path, tmp_path: Path
) -> None:
    model = predictor.model
    train_logits = _training_path_logits(predictor, sample_image_path, tmp_path)

    with Image.open(sample_image_path) as im:
        api_input = model.preprocess(im).unsqueeze(0)
    with torch.inference_mode():
        api_logits = model(api_input)
    assert torch.allclose(train_logits, api_logits, atol=1e-4)

    # Same image through the real HTTP endpoint.
    response = client.post(
        "/predict", files={"file": ("a.jpg", sample_image_path.read_bytes(), "image/jpeg")}
    )
    probs = torch.softmax(train_logits / model.temperature, dim=1)[0]
    top = probs.topk(3)
    best = response.json()["top_3"][0]
    assert best["breed"] == model.class_names[int(top.indices[0])]
    assert best["confidence"] == pytest.approx(float(top.values[0]), abs=1e-4)


def test_a_preprocessing_mismatch_would_be_detected(
    predictor: Predictor, sample_image_path: Path
) -> None:
    """Negative control: forgetting the ImageNet normalization must change the logits."""
    model = predictor.model
    with Image.open(sample_image_path) as im:
        rgb = im.convert("RGB")
        good = model.preprocess(rgb).unsqueeze(0)
        skewed = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])(rgb).unsqueeze(0)
    with torch.inference_mode():
        assert not torch.allclose(model(good), model(skewed), atol=1e-4)


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, "confident"), (1.01, "uncertain")])
def test_decision_follows_the_abstention_threshold(
    predictor: Predictor,
    sample_image_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threshold: float,
    expected: str,
) -> None:
    monkeypatch.setattr(predictor, "threshold", threshold)
    with Image.open(sample_image_path) as im:
        assert predictor.predict(im).decision == expected


def test_confidence_equal_to_the_threshold_is_confident(
    predictor: Predictor, sample_image_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Image.open(sample_image_path) as im:
        confidence = predictor.predict(im).confidence
        monkeypatch.setattr(predictor, "threshold", confidence)
        assert predictor.predict(im).decision == "confident"
