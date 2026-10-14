from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models

from prodml.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform

BACKBONES = ("resnet50", "resnet18", "mobilenet_v3_small")


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {name!r}, choose from {BACKBONES}")
    return m


class PetClassifier(nn.Module):
    """Backbone + exported eval transform, saved and loaded as one artifact."""

    def __init__(
        self,
        backbone: str,
        class_names: list[str],
        image_size: int = 224,
        resize_size: int = 256,
        temperature: float = 1.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.class_names = class_names
        self.image_size = image_size
        self.resize_size = resize_size
        self.temperature = temperature
        self.net = build_backbone(backbone, len(class_names), pretrained)
        self.transform = build_eval_transform(image_size, resize_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """The one preprocessing path used at serving time."""
        return self.transform(image.convert("RGB"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "meta": {
                    "backbone": self.backbone_name,
                    "class_names": self.class_names,
                    "image_size": self.image_size,
                    "resize_size": self.resize_size,
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                    "temperature": self.temperature,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> PetClassifier:
        ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        meta = ckpt["meta"]
        model = cls(
            backbone=meta["backbone"],
            class_names=meta["class_names"],
            image_size=meta["image_size"],
            resize_size=meta["resize_size"],
            temperature=meta["temperature"],
            pretrained=False,
        )
        model.net.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()
