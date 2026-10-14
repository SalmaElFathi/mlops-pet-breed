# mlops-pet-breed

A FastAPI service that takes a pet photo and returns the predicted breed (37 cat and dog
breeds, Oxford-IIIT Pet) with a confidence score. Low-confidence predictions are flagged as
`uncertain` instead of being guessed. It runs in a non-root Docker container and writes
structured JSON logs with one correlation ID per request.

Dataset, splits and corruption suite: see [data/README.md](data/README.md).

> **Status:** the published model is an untrained checkpoint, used to build and test the
> serving system. Its predictions are essentially random. A trained model comes with the
> next release.

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
`uncertain` otherwise.

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
uv sync --extra dev
pre-commit install
```

```bash
ruff check src tests && black --check src tests   # lint
pytest                                            # tests, 70% coverage gate
```

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

## Repository layout

```text
src/prodml/          package: config, data, model, prediction, API, logging
tests/               pytest suite and reference label map
reports/             measurements and reports (module-1.md)
data/                manifest and corruption suite (DVC-tracked)
Dockerfile           multi-stage image of the API (non-root, healthcheck)
Dockerfile.single    single-stage variant, kept for the image size comparison
docker-compose.yml   runs the API with one command
params.yaml          tracked parameters (split, corruptions, training)
dvc.yaml, dvc.lock   data pipeline
pyproject.toml       package metadata and dependencies
.pre-commit-config.yaml   lint and format hooks
```
