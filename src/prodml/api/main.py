from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from prodml.api.schemas import HealthResponse, PetBreedResponse
from prodml.config import settings
from prodml.logging_conf import correlation_id, setup_logging
from prodml.model import PetClassifier
from prodml.predict import Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    model = PetClassifier.load(settings.model_path)
    species = json.loads(settings.species_map_path.read_text())
    app.state.predictor = Predictor(
        model, species, settings.abstention_threshold, settings.model_version
    )
    logger.info("model loaded: %d classes", len(model.class_names))
    yield


app = FastAPI(title="Pet breed classifier", lifespan=lifespan)


@app.middleware("http")
async def add_correlation_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    cid = str(uuid.uuid4())
    token = correlation_id.set(cid)
    try:
        logger.info("request received: %s %s", request.method, request.url.path)
        response = await call_next(request)
        response.headers["X-Request-ID"] = cid
        return response
    finally:
        correlation_id.reset(token)


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
    result = request.app.state.predictor.predict(_read_image(file))
    logger.info("prediction served: breed=%s decision=%s", result.breed, result.decision)
    return result


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    p: Predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy", model_version=p.model_version, num_classes=p.num_classes
    )
