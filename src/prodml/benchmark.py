from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from prodml.config import settings
from prodml.logging_conf import setup_logging
from prodml.model import PetClassifier

logger = logging.getLogger(__name__)

N_IMAGES = 200
WARMUP = 10


def load_val_inputs(model: PetClassifier, n: int) -> list[torch.Tensor]:
    """Preprocess n validation images once, as batches of one."""
    records = [r for r in json.loads(settings.manifest_path.read_text()) if r["split"] == "val"]
    inputs = []
    for r in records[:n]:
        with Image.open(r["path"]) as im:
            inputs.append(model.preprocess(im).unsqueeze(0))
    return inputs


def measure(run: Callable[[torch.Tensor], object], inputs: list[torch.Tensor]) -> dict[str, float]:
    """Time `run` on each input after a warm-up. Returns mean and p95 in milliseconds."""
    for x in inputs[:WARMUP]:
        run(x)
    times_ms = []
    for x in inputs:
        start = time.perf_counter()
        run(x)
        times_ms.append((time.perf_counter() - start) * 1000)
    arr = np.array(times_ms)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95))}


def main() -> None:
    setup_logging()
    model = PetClassifier.load(settings.model_path)
    inputs = load_val_inputs(model, N_IMAGES)
    sess = ort.InferenceSession(
        str(settings.model_path.with_suffix(".onnx")), providers=["CPUExecutionProvider"]
    )

    @torch.inference_mode()
    def run_torch(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    def run_onnx(x: torch.Tensor) -> object:
        return sess.run(None, {"image": x.numpy()})

    results = {
        "pytorch": measure(run_torch, inputs),
        "onnxruntime": measure(run_onnx, inputs),
    }
    for name, r in results.items():
        logger.info("%s: mean %.2f ms, p95 %.2f ms", name, r["mean_ms"], r["p95_ms"])


if __name__ == "__main__":
    main()
