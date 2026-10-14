from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from prodml.dataset import PetDataset
from prodml.model import BACKBONES, PetClassifier, build_backbone
from prodml.schemas import ManifestRecord
from prodml.transforms import build_eval_transform, build_train_transform

NAMES = [f"breed_{i:02d}" for i in range(37)]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_backbone_outputs_one_logit_per_class(backbone: str) -> None:
    model = PetClassifier(backbone, NAMES, pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 37)


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("vgg16", 37, pretrained=False)


def test_augmentation_is_random_in_training_and_absent_in_eval() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    evaluate, train = build_eval_transform(), build_train_transform()
    assert torch.equal(evaluate(image), evaluate(image))
    assert not torch.equal(train(image), train(image))


def test_dataset_keeps_only_the_requested_split(tmp_path: Path) -> None:
    Image.new("RGB", (300, 200)).save(tmp_path / "a.jpg")
    record = {"image_id": "a", "path": str(tmp_path / "a.jpg"), "breed": "Bengal"}
    manifest = [{**record, "split": "val"}, {**record, "split": "train"}]
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "l.json").write_text(json.dumps({"Bengal": 3}))
    dataset = PetDataset(tmp_path / "m.json", tmp_path / "l.json", "val", build_eval_transform())
    x, y = dataset[0]
    assert len(dataset) == 1
    assert y == 3
    assert x.shape == (3, 224, 224)


def test_manifest_record_defaults_describe_a_clean_image() -> None:
    record = ManifestRecord(
        image_id="Abyssinian_1",
        path="data/raw/Abyssinian_1.jpg",
        breed="Abyssinian",
        species="cat",
        class_index=0,
        split="train",
        width=394,
        height=500,
    )
    assert record.corruption is None
    assert record.severity == 0
