from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PetDataset(Dataset):
    """Reads one split of a manifest. Labels come from the committed label map."""

    def __init__(
        self,
        manifest_path: Path,
        label_map_path: Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.label_map: dict[str, int] = json.loads(label_map_path.read_text())
        records = json.loads(manifest_path.read_text())
        self.records = [r for r in records if r["split"] == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.records[i]
        with Image.open(r["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.label_map[r["breed"]]