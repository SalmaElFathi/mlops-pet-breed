from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

params = yaml.safe_load(Path("params.yaml").read_text())


class Settings(BaseSettings):
    data_path: Path = Path("data/raw")
    test_size: float = params["split"]["val_fraction"]
    random_state: int = params["split"]["seed"]
    label_map_path: Path = Path("data/label_map.json")
    manifest_path: Path = Path("data/manifest.json")
    corrupted_root: Path = Path("data/corrupted")
    corrupted_manifest_path: Path = Path("data/manifest_corrupted.json")


settings = Settings()