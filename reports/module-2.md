# Module 2 report

## 1. Tracking stack

- Services (Docker Compose, profile `mlops`): PostgreSQL 16 (backend store), MinIO (artifact store), MLflow 3.17.1 server. The API still starts alone with `docker compose up`.
- MLflow is served with `--serve-artifacts`: the client only knows the tracking URI, and the server talks to MinIO.
- MinIO image: `minio/minio` and `minio/mc` are no longer published on Docker Hub, so I use the Chainguard build, pinned by digest (MinIO `RELEASE.2026-09-22T19-25-18Z`). The `mc` client being gone, the bucket `mlflow` is created by a small `boto3` script (`scripts/create_bucket.py`), idempotent.
- MLflow server image: built from `python:3.12-slim` with `psycopg2-binary` and `boto3` (1.33 GB). The backend URI names the driver explicitly (`postgresql+psycopg2://`), otherwise SQLAlchemy looks for `psycopg`, which is not installed.
- Acceptance check: `mlflow.search_experiments()` lists the experiments through `settings.mlflow_tracking_uri`. A test run was found in PostgreSQL (params, metrics) and in the MinIO bucket (artifact).

## 2. Experiment tracking

### Setup

- Training mode: **linear probe**. The pretrained backbone is frozen, its features are extracted once and cached (keyed by backbone, split, image size and data version), and only the classification head is trained. This keeps every run fast on a CPU-only laptop, at the cost of the few points a full fine-tuning would add.
- Data: 2,944 train and 736 validation images, split seed 42, zero overlap (checked). The test split is not used.
- Every run logs params (backbone, lr, batch size, epochs, weight decay, seeds, data version), metrics (top-1, macro-F1, ECE before and after temperature scaling, temperature, coverage and selective accuracy at threshold 0.7, model size, head training time), artifacts (`model.pt`, confusion matrix, reliability diagram, `requirements.txt`) and tags (`git_commit`, `data_version`, `author`, `framework`).
- The experiment holds 36 runs: 3 base runs, 3 sweep parents and 30 nested trials. Re-running the same configuration gave identical metrics (e.g. MobileNet top-1 0.8940 twice), so training is deterministic.

### 2.1 Comparison 1: three backbones

Same data, same hyperparameters (lr 0.001, batch 32, 5 epochs). Only `backbone` differs (checked with "Show diff only").

| Backbone | top-1 | macro-F1 | ECE (calibrated) | Temperature | Coverage | Selective acc. | Size (MB) |
|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | 0.894 | 0.894 | 0.0214 | 0.7 | 0.867 | 0.953 | 6.3 |
| ResNet-18 | 0.917 | 0.918 | 0.0176 | 0.6 | 0.875 | 0.969 | 44.9 |
| ResNet-50 | 0.939 | 0.939 | 0.0185 | 0.6 | 0.917 | 0.973 | 94.6 |

![Backbone comparison](images/mlflow-compare-backbones.png)

ResNet-50 has the highest top-1 (0.939), but it is also the largest model (94.6 MB). Compared with MobileNet it gains 4.5 points of top-1 for a file about 15 times bigger. ResNet-18 sits in between. Calibration works for all three: ECE after temperature scaling is close to 0.02 (0.0176 for ResNet-18, 0.0185 for ResNet-50, 0.0214 for MobileNet). All temperatures are below 1, so the heads were under-confident before calibration, which is expected after only 5 epochs.

Feature extraction time (measured on the first, uncached run): about 39 s for MobileNet and 70 s for ResNet-50. Which backbone to ship depends on latency and size budgets, which Module 3 will measure.

### 2.2 Comparison 2: one sweep per backbone

Each backbone got the same random search: 10 nested trials, same seed (so the same 10 configurations), selection on validation top-1. Total: 3 base runs, 3 sweep parents, 30 trials (36 runs).

| Backbone | Best trial (lr, batch, weight decay, epochs) | top-1 | ECE | Gain over base run |
|---|---|---|---|---|
| ResNet-50 | 0.00191, 32, 0.001, 10 | **0.9497** | 0.0223 | +1.1 pt (0.9389) |
| ResNet-18 | 0.00191, 32, 0.001, 10 | 0.9280 | 0.0199 | +1.1 pt (0.9171) |
| MobileNetV3-Small | 0.00853, 128, 0.001, 10 | 0.9008 | 0.0344 | +0.7 pt (0.8940) |

![ResNet-50 sweep](images/mlflow-compare-sweep.png)

The ranking is the same before and after tuning (ResNet-50 > ResNet-18 > MobileNet). Both ResNets picked the same configuration. Tuning adds about one point of top-1, which is at the edge of what 736 validation images can resolve (standard error around 0.8 point), so the sweep did not find a clearly better setup than the defaults. In the ResNet-50 parallel-coordinates plot the best runs tend to be trained for 10 or 20 epochs and the worst for 5, but `lr`, `batch_size`, `weight_decay` and `epochs` all vary together, so the effect of the epoch count alone cannot be isolated. The tuned ResNet-50 has a slightly higher ECE than its base run (0.0223 vs 0.0185): accuracy went up, calibration did not.

All figures are validation figures: the same 736 images
