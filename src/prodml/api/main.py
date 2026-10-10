from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from prodml.api.schemas import HealthResponse, PetBreedResponse
from prodml.config import settings
from prodml.model import PetClassifier
from prodml.predict import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model = PetClassifier.load(settings.model_path)
    species = json.loads(settings.species_map_path.read_text())
    app.state.predictor = Predictor(
        model, species, settings.abstention_threshold, settings.model_version
    )
    yield


app = FastAPI(title="Pet breed classifier", lifespan=lifespan)


def _read_image(file: UploadFile) -> Image.Image:
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=422, detail="file too large")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="file is not a valid image") from exc


@app.post("/predict", response_model=PetBreedResponse)
def predict(request: Request, file: UploadFile = File(...)) -> PetBreedResponse:
    return request.app.state.predictor.predict(_read_image(file))


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    p: Predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy", model_version=p.model_version, num_classes=p.num_classes
    )