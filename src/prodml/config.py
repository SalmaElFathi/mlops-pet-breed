import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = Path(os.environ.get("PRODML_PARAMS_PATH", ROOT / "params.yaml"))
params = yaml.safe_load(PARAMS_PATH.read_text())


class Settings(BaseSettings):
    data_path: Path = Path("data/raw")
    test_size: float = params["split"]["val_fraction"]
    random_state: int = params["split"]["seed"]
    label_map_path: Path = Path("data/label_map.json")
    species_map_path: Path = Path("data/species_map.json")
    manifest_path: Path = Path("data/manifest.json")
    corrupted_root: Path = Path("data/corrupted")
    corrupted_manifest_path: Path = Path("data/manifest_corrupted.json")
    model_path: Path = Path("models/model.pt")
    model_version: str = "0.0.0"
    max_upload_bytes: int = 5 * 1024 * 1024
    abstention_threshold: float = params["calibration"]["abstention_threshold"]


settings = Settings()
