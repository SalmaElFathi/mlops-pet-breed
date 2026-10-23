# mlops-pet-breed

[![CI](https://github.com/SalmaElFathi/mlops-pet-breed/actions/workflows/ci.yml/badge.svg)](https://github.com/SalmaElFathi/mlops-pet-breed/actions/workflows/ci.yml)

A FastAPI service that takes a pet photo and returns the predicted breed (37 cat and dog
breeds, Oxford-IIIT Pet) with a confidence score. Low-confidence predictions are flagged as
`uncertain` instead of being guessed. It runs in a non-root Docker container and writes
structured JSON logs with one correlation ID per request.

Dataset, splits and corruption suite: see [data/README.md](data/README.md).

> **Status:** the published image `0.1.0` contains an untrained checkpoint, used to build and
> test the serving system. Its predictions are essentially random. A model trained in Module 2
> (ResNet-50 linear probe, top-1 0.9497 on the validation set) is registered in the MLflow
> registry as `PetBreedClassifier` and will ship as image `0.2.0`. The images built by the CI also
> contain an untrained demo checkpoint (see [CI](#ci-and-continuous-training)).

## Quick start (3 commands)

```bash
docker pull salma201/pet-breed:0.1.0
docker run -d --name pet-breed -p 8000:8000 salma201/pet-breed:0.1.0
curl -F "file=@photo.jpg" http://localhost:8000/predict
```

Replace `photo.jpg` with any cat or dog photo. Stop the service with
`docker stop pet-breed && docker rm pet-breed`.

## Example response

Values rounded to 4 decimals for readability.

```json
{
  "breed": "Scottish Terrier",
  "species": "dog",
  "confidence": 0.1037,
  "top_3": [
    {"breed": "Scottish Terrier", "confidence": 0.1037},
    {"breed": "Leonberger", "confidence": 0.0834},
    {"breed": "Chihuahua", "confidence": 0.0675}
  ],
  "decision": "uncertain",
  "model_version": "0.0.0"
}
```

`decision` is `confident` when the top score reaches the abstention threshold, and
`uncertain` otherwise. `model_version` is `0.0.0` for the untrained checkpoint; in registry mode it
is the version number of the registered model.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Multipart image, returns the breed, top-3 and decision |
| `/health` | GET | `{status, model_version, num_classes}` |
| `/docs` | GET | Interactive documentation |

Invalid uploads return `422` (not an image, file over 5 MB). Unexpected errors return `500`
with a generic message; the traceback is only written to the logs.

## Development

```bash
git clone https://github.com/SalmaElFathi/mlops-pet-breed.git
cd mlops-pet-breed
uv sync --extra dev --extra train
pre-commit install
```

```bash
pre-commit run --all-files   # lint and format (same as the CI lint job)
pytest                       # tests, 70% coverage gate
```

The 4 data-validation tests (`tests/test_manifest.py`) are skipped when `data/manifest.json` is not on
disk. In the CI the DVC remote is a MinIO on my laptop, unreachable from GitHub runners, so these tests
run locally only.

To run the API from source, `models/model.pt` is needed (it is not in Git). This creates an
untrained demo checkpoint:

```bash
python -c "
import json
from pathlib import Path
from prodml.model import PetClassifier
lm = json.loads(Path('data/label_map.json').read_text())
PetClassifier('resnet18', sorted(lm, key=lm.get), pretrained=False).save(Path('models/model.pt'))
"
uvicorn prodml.api.main:app --port 8000
```

## Tracking stack (Module 2)

Copy `.env.example` to `.env` and set the passwords and the MLflow version (the same as in
`pyproject.toml`), then start PostgreSQL, MinIO and MLflow:

```bash
cp .env.example .env
docker compose --profile mlops up -d
```

MLflow is at http://localhost:5000 and the MinIO console at http://localhost:9001. Without
`--profile mlops`, `docker compose up` starts only the API (it needs `models/model.pt`).

## Training and registry

```bash
uv run python -m prodml.train --backbone resnet50 --run-name test      # one tracked run
uv run python -m prodml.sweep --backbone resnet18 --trials 10          # nested-run sweep
uv run python -m prodml.registry <run_id> [margin]                     # promote to @staging if better
USE_REGISTRY=true docker compose --profile mlops up -d api             # serve @production
```

The training is a linear probe: the pretrained backbone is frozen and only the classification head is
trained. In registry mode the API loads the version of `PetBreedClassifier` that carries the alias
`production`; moving the alias and restarting the container changes the served model without any code
change or rebuild.

## Data pipeline (DVC)

The data lives in the `dvc` bucket of the MinIO above. The access keys are not in Git: put them in
`.dvc/config.local`.

```bash
set -a; . ./.env; set +a
dvc remote modify --local storage access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local storage secret_access_key "$MINIO_ROOT_PASSWORD"
dvc pull       # fetch the data
dvc repro      # rebuild what changed (manifest, corruptions, train)
dvc metrics show
```

## Infrastructure as code (Terraform)

`infra/` declares the same MLflow stack with the Docker provider, on other ports than the Compose stack
(MLflow on 15000, MinIO on 19000 and 19001). The two passwords come from `.env` and are copied into a
file that Git ignores:

```bash
cd infra
(set -a; . ../.env; set +a; printf 'postgres_password = "%s"\nminio_password = "%s"\n' "$POSTGRES_PASSWORD" "$MINIO_ROOT_PASSWORD" > secrets.auto.tfvars)
terraform init && terraform plan && terraform apply
```

## CI and continuous training

- `.github/workflows/ci.yml` runs on every pull request to `main` and on every push to `main`: lint, tests
  with the 70% coverage gate, a quality gate that fails if top-1 dropped by more than 0.005 compared with
  `main`, and the image build. The image is pushed to Docker Hub as `sha-<commit>` and `latest` only on
  `main`; because `models/model.pt` is not in Git, it contains an untrained demo checkpoint.
- `.github/workflows/continuous-training.yml` retrains on a schedule, on a manual run or on a
  `repository_dispatch` event, applies a gate against the model in Production, promotes to Staging and
  builds an image. Production always needs a human approval (GitHub Environment `production`). It runs on a
  self-hosted runner, because the MLflow server and the data are on my laptop.

Details, measurements and differences from the course guide: [reports/module-2.md](reports/module-2.md).

## Repository layout

```text
src/prodml/          package: config, data, model, training, sweep, registry, quality gate, API, logging
tests/               pytest suite and reference label map
reports/             measurements and reports (module-1.md, module-2.md, images/)
data/                manifest and corruption suite (DVC-tracked)
scripts/             bucket creation and a tracking smoke test
infra/               Terraform (Docker provider): the MLflow stack as code
.github/workflows/   ci.yml and continuous-training.yml
Dockerfile           multi-stage image of the API (non-root, healthcheck; needs models/model.pt)
Dockerfile.single    single-stage variant, kept for the image size comparison
mlflow.Dockerfile    MLflow server image (PostgreSQL and S3 drivers)
docker-compose.yml   the API, and with --profile mlops PostgreSQL, MinIO and MLflow
.env.example         variables for Compose and Terraform (copy to .env)
params.yaml          tracked parameters (split, corruptions, calibration, training)
dvc.yaml, dvc.lock   data and training pipeline
pyproject.toml       package metadata and dependencies
.pre-commit-config.yaml   lint and format hooks
```
