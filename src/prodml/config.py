from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: Path = Path("data/raw")
    test_size: float = 0.2
    random_state: int = 42
    label_map_path: Path = Path("data/label_map.json")
    manifest_path: Path = Path("data/manifest.json")
    corrupted_root: Path = Path("data/corrupted")
    corrupted_manifest_path: Path = Path("data/manifest_corrupted.json")


settings = Settings()