from __future__ import annotations

import logging
import sys

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from prodml.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "PetBreedClassifier"
ARTIFACT_PATH = "model.pt"
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def get_version_by_alias(
    client: MlflowClient, alias: str, name: str = MODEL_NAME
) -> ModelVersion | None:
    """Return the model version carrying `alias`, or None if the alias is unset."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _metric(client: MlflowClient, run_id: str, metric: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    if metric not in metrics:
        raise KeyError(f"Metric '{metric}' not found on run {run_id}")
    return float(metrics[metric])


def promote_if_better(
    candidate_run_id: str,
    metric: str = "top1",
    margin: float = 0.0,
    alias: str = ALIAS_STAGING,
    name: str = MODEL_NAME,
) -> bool:
    """Register the candidate and give it `alias` if it beats @production.

    Higher is better (top1). The candidate must exceed Production by more than `margin`.
    Returns True if promoted, False if rejected. Never touches @production.
    """
    if alias == ALIAS_PRODUCTION:
        raise ValueError("Auto-promotion to Production is forbidden: use a human approval.")

    client = _client()
    candidate_score = _metric(client, candidate_run_id, metric)
    production = get_version_by_alias(client, ALIAS_PRODUCTION, name)

    if production is not None:
        production_score = _metric(client, production.run_id, metric)
        if candidate_score <= production_score + margin:
            logger.info(
                "REJECTED run %s: %s=%.4f vs Production v%s=%.4f (margin %.4f)",
                candidate_run_id,
                metric,
                candidate_score,
                production.version,
                production_score,
                margin,
            )
            return False
    else:
        logger.info("No Production model yet: candidate accepted by default.")

    existing = client.search_model_versions(f"name='{name}' and run_id='{candidate_run_id}'")
    if existing:
        version = existing[0].version
    else:
        version = mlflow.register_model(f"runs:/{candidate_run_id}/{ARTIFACT_PATH}", name).version

    client.set_registered_model_alias(name, alias, version)
    logger.info(
        "PROMOTED run %s -> @%s (v%s, %s=%.4f)",
        candidate_run_id,
        alias,
        version,
        metric,
        candidate_score,
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_id = sys.argv[1]
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    promote_if_better(run_id, margin=margin)
    sys.exit(0)
