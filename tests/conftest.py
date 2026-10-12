from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings
from prodml.model import PetClassifier

CLASS_NAMES = [f"breed_{i:02d}" for i in range(37)]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An untrained ResNet-18 checkpoint: enough to test the code, no files needed."""
    path = tmp_path_factory.mktemp("model") / "model.pt"
    PetClassifier("resnet18", CLASS_NAMES, pretrained=False).save(path)
    return path


@pytest.fixture(scope="session")
def species_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("species") / "species.json"
    species = {name: "cat" if i % 2 else "dog" for i, name in enumerate(CLASS_NAMES)}
    path.write_text(json.dumps(species))
    return path


@pytest.fixture(scope="session")
def client(tiny_checkpoint: Path, species_file: Path) -> Iterator[TestClient]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "model_path", tiny_checkpoint)
        mp.setattr(settings, "species_map_path", species_file)
        with TestClient(app) as c:
            yield c
