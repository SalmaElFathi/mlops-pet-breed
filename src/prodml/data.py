from collections.abc import Sequence
from pathlib import Path

from torchvision.datasets import OxfordIIITPet


def load_data(
    root: str | Path = "data/raw",
    split: str = "trainval",
    target_types: str | Sequence[str] = "category",
) -> OxfordIIITPet:
    return OxfordIIITPet(
        root=root,
        split=split,
        target_types=target_types,
        download=True,
    )