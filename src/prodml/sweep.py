from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Any

import mlflow

from prodml.config import params, settings
from prodml.model import BACKBONES
from prodml.train import EXPERIMENT, data_version, git_commit, run_experiment

logger = logging.getLogger(__name__)


def sample_configs(base: dict[str, Any], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Random search with a fixed seed, so the sweep is reproducible."""
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        configs.append(
            {
                **base,
                "lr": float(f"{10 ** rng.uniform(-4, -2):.3g}"),
                "batch_size": rng.choice([16, 32, 64, 128]),
                "weight_decay": rng.choice([0.0, 1e-5, 1e-4, 1e-3]),
                "epochs": rng.choice([5, 10, 20]),
            }
        )
    return configs


def run_sweep(backbone: str, n_trials: int, seed: int) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    base = {**params["train"], "backbone": backbone}
    base.setdefault("weight_decay", 1e-4)
    configs = sample_configs(base, n_trials, seed)

    with mlflow.start_run(run_name=f"sweep-{backbone}"):
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_version": data_version(),
                "author": os.environ.get("USER", "unknown"),
                "framework": "pytorch",
            }
        )
        mlflow.log_params(
            {
                "backbone": backbone,
                "n_trials": n_trials,
                "sweep_seed": seed,
                "search": "random",
                "selection_metric": "top1 on validation",
            }
        )
        results = []
        for i, cfg in enumerate(configs, start=1):
            name = f"{backbone}-trial-{i:02d}-lr{cfg['lr']:g}-bs{cfg['batch_size']}"
            run_id = run_experiment(cfg, run_name=name, nested=True)
            top1 = mlflow.get_run(run_id).data.metrics["top1"]
            results.append((top1, run_id, cfg))
            logger.info("trial %d/%d: top1=%.4f", i, n_trials, top1)

        best_top1, best_id, best_cfg = max(results, key=lambda r: r[0])
        mlflow.log_metric("best_top1", best_top1)
        mlflow.set_tag("best_run_id", best_id)
        mlflow.log_params(
            {f"best_{k}": best_cfg[k] for k in ("lr", "batch_size", "weight_decay", "epochs")}
        )
        logger.info("best: top1=%.4f run=%s", best_top1, best_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Nested-run hyperparameter sweep")
    parser.add_argument("--backbone", default="resnet18", choices=BACKBONES)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_sweep(args.backbone, args.trials, args.seed)


if __name__ == "__main__":
    main()
