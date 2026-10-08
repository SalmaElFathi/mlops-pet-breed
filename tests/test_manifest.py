"""Validate the dataset before training (handbook, Step 0, item 5)."""

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from prodml.config import settings

REFERENCE_LABEL_MAP = Path(__file__).parent / "reference_label_map.json"


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return json.loads(settings.manifest_path.read_text())


@pytest.fixture(scope="module")
def label_map() -> dict[str, int]:
    return json.loads(settings.label_map_path.read_text())


def test_every_manifest_path_exists_and_opens(manifest):
    for r in manifest:
        with Image.open(r["path"]) as im:
            im.verify()


def test_no_image_id_in_two_splits(manifest):
    seen: dict[str, str] = {}
    for r in manifest:
        assert seen.setdefault(r["image_id"], r["split"]) == r["split"], r["image_id"]


def test_label_map_has_37_entries(label_map):
    assert len(label_map) == 37


def test_label_map_matches_reference(label_map):
    reference = json.loads(REFERENCE_LABEL_MAP.read_text())
    assert label_map == reference


def test_every_class_has_at_least_50_train_images(manifest):
    counts = Counter(r["breed"] for r in manifest if r["split"] == "train")
    assert len(counts) == 37
    assert min(counts.values()) >= 50


def test_no_image_under_32x32(manifest):
    assert min(min(r["width"], r["height"]) for r in manifest) >= 32