from __future__ import annotations

import logging
from pathlib import Path

import torch

from prodml.config import settings
from prodml.logging_conf import setup_logging
from prodml.model import PetClassifier

logger = logging.getLogger(__name__)


def export_onnx(model: PetClassifier, path: Path) -> Path:
    """Export the network to ONNX with a dynamic batch dimension."""
    model.eval()
    dummy = torch.zeros(1, 3, model.image_size, model.image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def main() -> None:
    setup_logging()
    model = PetClassifier.load(settings.model_path)
    out = export_onnx(model, settings.model_path.with_suffix(".onnx"))
    logger.info("exported ONNX model to %s", out)


if __name__ == "__main__":
    main()
