from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prodml import metrics
from prodml.config import params, settings
from prodml.dataset import PetDataset
from prodml.model import BACKBONES, PetClassifier
from prodml.plots import save_confusion_matrix, save_reliability_diagram

logger = logging.getLogger(__name__)

EXPERIMENT = "pet-breed-classifier"
CACHE_DIR = Path(".cache/features")


def get_head(net: nn.Module, backbone: str) -> nn.Module:
    return net.fc if backbone.startswith("resnet") else net.classifier[3]


def set_head(net: nn.Module, backbone: str, head: nn.Module) -> None:
    if backbone.startswith("resnet"):
        net.fc = head
    else:
        net.classifier[3] = head


@torch.no_grad()
def extract_features(clf: PetClassifier, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone and keep what the classification head receives."""
    original = get_head(clf.net, clf.backbone_name)
    set_head(clf.net, clf.backbone_name, nn.Identity())
    clf.net.eval()
    feats, labels = [], []
    try:
        for x, y in loader:
            feats.append(clf.net(x))
            labels.append(y)
    finally:
        set_head(clf.net, clf.backbone_name, original)
    return torch.cat(feats), torch.cat(labels)


def cached_features(
    clf: PetClassifier, split: str, version: str
) -> tuple[torch.Tensor, torch.Tensor, float]:
    key = f"{clf.backbone_name}-{split}-{clf.image_size}-{clf.resize_size}-{version}"
    path = CACHE_DIR / f"{key}.pt"
    if path.exists():
        data = torch.load(path, weights_only=True)
        return data["feats"], data["labels"], 0.0
    dataset = PetDataset(settings.manifest_path, settings.label_map_path, split, clf.transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=2)
    start = time.perf_counter()
    feats, labels = extract_features(clf, loader)
    elapsed = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, path)
    return feats, labels, elapsed


def train_head(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    lr: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(seed)
    head = nn.Linear(feats.shape[1], num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(feats, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(head(x), y).backward()
            optimizer.step()
    return head


def evaluate(logits: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """All the Track C metrics. Temperature is fitted on these same logits."""
    num_classes = logits.shape[1]
    temperature = metrics.fit_temperature(logits, labels)
    raw = metrics.softmax(logits)
    calibrated = metrics.softmax(logits, temperature)
    coverage, selective = metrics.selective_metrics(calibrated, labels, threshold)
    return {
        "top1": metrics.top1(calibrated, labels),
        "f1_macro": metrics.macro_f1(calibrated.argmax(axis=1), labels, num_classes),
        "ece_before": metrics.ece(raw, labels),
        "ece": metrics.ece(calibrated, labels),
        "temperature": temperature,
        "coverage": coverage,
        "selective_accuracy": selective,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_version() -> str:
    """The DVC hash of the manifest, falling back to a hash of the file itself."""
    lock = Path("dvc.lock")
    if lock.exists():
        stages = yaml.safe_load(lock.read_text()).get("stages", {})
        for out in stages.get("build_manifest", {}).get("outs", []):
            if out.get("path") == "data/manifest.json" and out.get("md5"):
                return str(out["md5"])[:12]
    return hashlib.md5(settings.manifest_path.read_bytes()).hexdigest()[:12]


def installed_packages() -> str:
    lines = {f"{d.metadata['Name']}=={d.version}" for d in distributions()}
    return "\n".join(sorted(lines, key=str.lower)) + "\n"


def run_experiment(cfg: dict[str, Any], run_name: str | None = None, nested: bool = False) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    label_map = json.loads(settings.label_map_path.read_text())
    class_names = sorted(label_map, key=label_map.get)
    backbone = cfg["backbone"]
    version = data_version()
    clf = PetClassifier(
        backbone, class_names, cfg["image_size"], cfg["resize_size"], pretrained=True
    )
    name = run_name or f"{backbone}-lr{cfg['lr']}-bs{cfg['batch_size']}"

    with mlflow.start_run(run_name=name, nested=nested) as active:
        mlflow.log_params(
            {
                "backbone": backbone,
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "weight_decay": cfg["weight_decay"],
                "seed": cfg["seed"],
                "image_size": cfg["image_size"],
                "resize_size": cfg["resize_size"],
                "split_seed": params["split"]["seed"],
                "val_fraction": params["split"]["val_fraction"],
                "abstention_threshold": settings.abstention_threshold,
                "data_version": version,
                "training_mode": "linear_probe",
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_version": version,
                "author": os.environ.get("USER", "unknown"),
                "framework": "pytorch",
            }
        )

        train_x, train_y, t_train = cached_features(clf, "train", version)
        val_x, val_y, t_val = cached_features(clf, "val", version)

        start = time.perf_counter()
        head = train_head(
            train_x,
            train_y,
            len(class_names),
            cfg["lr"],
            cfg["batch_size"],
            cfg["epochs"],
            cfg["weight_decay"],
            cfg["seed"],
        )
        head_seconds = time.perf_counter() - start

        with torch.no_grad():
            logits = head(val_x).numpy()
        labels = val_y.numpy()
        scores = evaluate(logits, labels, settings.abstention_threshold)

        set_head(clf.net, backbone, head)
        clf.temperature = scores["temperature"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            clf.save(out / "model.pt")
            calibrated = metrics.softmax(logits, scores["temperature"])
            save_confusion_matrix(
                metrics.confusion_matrix(calibrated.argmax(axis=1), labels, len(class_names)),
                out / "confusion_matrix.png",
            )
            save_reliability_diagram(
                metrics.softmax(logits), calibrated, labels, out / "calibration.png"
            )
            (out / "requirements.txt").write_text(installed_packages())
            scores["model_size_mb"] = (out / "model.pt").stat().st_size / 1e6
            scores["head_train_s"] = head_seconds
            scores["feature_extraction_s"] = t_train + t_val
            mlflow.log_metrics({k: v for k, v in scores.items() if not np.isnan(v)})
            mlflow.log_artifacts(str(out))

        logger.info("%s -> %s", name, {k: round(v, 4) for k, v in scores.items()})
        return active.info.run_id


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    base = params["train"]
    parser = argparse.ArgumentParser(description="Train one run and track it in MLflow")
    parser.add_argument("--backbone", default=base["backbone"], choices=BACKBONES)
    parser.add_argument("--lr", type=float, default=base["lr"])
    parser.add_argument("--batch-size", type=int, default=base["batch_size"])
    parser.add_argument("--epochs", type=int, default=base["epochs"])
    parser.add_argument("--weight-decay", type=float, default=base.get("weight_decay", 1e-4))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    cfg = {
        **base,
        "backbone": args.backbone,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
    }
    run_experiment(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
