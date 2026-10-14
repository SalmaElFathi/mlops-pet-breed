from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
        try:
            response = await call_next(request)
        except Exception:
            # Full traceback goes to the logs, never to the client.
            logger.exception("unhandled error")
            response = JSONResponse(status_code=500, content={"detail": "internal server error"})
        response.headers["X-Request-ID"] = cid
        return response
    finally:
        correlation_id.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    messages = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    logger.error("validation rejected: %s", messages)
    return JSONResponse(status_code=422, content={"detail": messages})


def _reject(detail: str) -> HTTPException:
    logger.error("validation rejected: %s", detail)
    return HTTPException(status_code=422, detail=detail)


def _decode_image(data: bytes) -> Image.Image:
    if len(data) > settings.max_upload_bytes:
        raise _reject("file too large")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise _reject("file is not a valid image") from exc


@app.post("/predict", response_model=PetBreedResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PetBreedResponse:
    data = await file.read(settings.max_upload_bytes + 1)
    image = await run_in_threadpool(_decode_image, data)
    result = await run_in_threadpool(request.app.state.predictor.predict, image)
    logger.info("prediction served: breed=%s decision=%s", result.breed, result.decision)
    return result


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    p: Predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy", model_version=p.model_version, num_classes=p.num_classes
    )
