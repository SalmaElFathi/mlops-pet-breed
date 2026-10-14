from __future__ import annotations

import torch
from PIL import Image

from prodml.api.schemas import BreedScore, PetBreedResponse
from prodml.model import PetClassifier
from prodml.utils import timed


class Predictor:
    """Holds the model (loaded once) and turns an image into a decision."""

    def __init__(
        self,
        model: PetClassifier,
        species: dict[str, str],
        threshold: float,
        model_version: str,
    ) -> None:
        self.model = model
        self.species = species
        self.threshold = threshold
        self.model_version = model_version

    @property
    def num_classes(self) -> int:
        return len(self.model.class_names)

    @timed
    @torch.inference_mode()
    def predict(self, image: Image.Image) -> PetBreedResponse:
        x = self.model.preprocess(image).unsqueeze(0)
        probs = torch.softmax(self.model(x) / self.model.temperature, dim=1)[0]
        conf, idx = probs.topk(3)
        top = [
            BreedScore(breed=self.model.class_names[int(i)], confidence=float(c))
            for c, i in zip(conf, idx, strict=True)
        ]
        best = top[0]
        return PetBreedResponse(
            breed=best.breed,
            species=self.species[best.breed],
            confidence=best.confidence,
            top_3=top,
            decision="confident" if best.confidence >= self.threshold else "uncertain",
            model_version=self.model_version,
        )
