"""Where the served model comes from: the registry alias, or a local file as fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from prodml.config import settings
from prodml.model import PetClassifier

logger = logging.getLogger(__name__)


def load_from_registry(retries: int = 10, delay_s: float = 3.0) -> tuple[PetClassifier, str]:
    """Load the version carrying `settings.model_alias`. Returns (model, registry_version).

    Retries so the API can start before MLflow is ready. If the registry stays
    unreachable it raises: silently serving another model is worse than not starting.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
            dst = Path(settings.model_cache_dir) / f"v{version.version}"
            dst.mkdir(parents=True, exist_ok=True)
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=version.source, dst_path=str(dst)
            )
            model = PetClassifier.load(Path(local))
            logger.info(
                "model loaded from registry: %s@%s = version %s (run %s)",
                settings.model_name,
                settings.model_alias,
                version.version,
                version.run_id,
            )
            return model, str(version.version)
        except Exception as exc:
            last_error = exc
            logger.warning("registry not ready (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay_s)
    raise RuntimeError(
        f"Could not load {settings.model_name}@{settings.model_alias} from "
        f"{settings.mlflow_tracking_uri}"
    ) from last_error


def load_model() -> tuple[PetClassifier, str]:
    """Return (model, model_version) according to USE_REGISTRY."""
    if settings.use_registry:
        return load_from_registry()
    logger.warning("USE_REGISTRY is off: serving the local file %s", settings.model_path)
    return PetClassifier.load(settings.model_path), settings.model_version
