# Module 1 Report

## Model quality

Pending: the model has not been trained yet. The first top-1 and macro-F1 will come from the
MLflow runs (Module 2). Current checkpoint: untrained ResNet-18, used to build and test the
serving system.

## Serialization

- Export: PyTorch checkpoint (`model.pt`) to ONNX (opset 17, dynamic batch axis).
- Parity, manual check: 200 validation images, max abs diff 1.19e-06, passes at atol 1e-4.
- Parity, committed test (`tests/test_serialization.py`): 200 random tensors, same tolerance.

## Latency: PyTorch vs ONNX Runtime

- Hardware: Intel Core i7-1165G7 @ 2.80GHz, CPU only, WSL2
- Model: ResNet-18, untrained checkpoint (latency depends on the architecture, not the weights)
- Protocol: 200 validation images, batch of 1, 10 warm-up calls, preprocessing excluded
- Two independent runs, reported separately

| Runtime | Run 1 mean / p95 (ms) | Run 2 mean / p95 (ms) |
|---|---|---|
| PyTorch | 36.40 / 47.78 | 37.62 / 54.98 |
| ONNX Runtime | 20.48 / 27.18 | 18.63 / 26.42 |

ONNX Runtime is roughly 1.8 to 2x faster than PyTorch on both mean and p95.
The p95 of PyTorch moved by about 15% between runs: with only 200 samples the
tail is noisy, so the mean is the more reliable comparison.
To redo with the trained model.

## Serialization formats compared

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No (optional JSON Schema) | Yes (data only), but not suited to model weights |
| Protobuf | No (binary) | Yes | Yes (`.proto` file) | Yes (data only) |
| Pickle | No (binary) | No (Python only) | No | **No: executes arbitrary code on load** |
| ONNX | No (binary, inspectable with tools) | Yes | Yes (protobuf-based graph and operator spec) | Mostly: it describes a graph, not Python code, but still check the source |

**Decision:** the service serves the PyTorch checkpoint (`model.pt`) for now, because it
carries the network, the class names, the temperature and the eval transform settings in a
single artifact, and because it gives the "before" baseline for the Module 3 comparison.
ONNX Runtime is about 1.8 to 2x faster on CPU in my benchmark, so the ONNX file is the
format the optimized CPU path will serve in Module 3.

## Docker image size

| Variant | Disk usage (`docker images`) | Compressed (`docker image inspect`) |
|---|---|---|
| Multi-stage (shipped) | 1.72 GB | 396,476,229 bytes |
| Single-stage | 1.72 GB | 397,763,398 bytes |

Multi-stage saves about 1.3 MB compressed (0.3%): negligible. The build stage has no
compilers or build caches to discard, since every dependency is a prebuilt wheel and pip
runs with `--no-cache-dir`. The weight is the Python environment (about 1.14 GB in
`docker history`), dominated by PyTorch CPU. Shrinking the image would mean installing
fewer packages, not restructuring the build.

## Effect of `.dockerignore`

Two extra builds, measured with BuildKit. They are 131 bytes smaller than the published
image, probably because `README.md` was removed from the Dockerfile in between.

| Variant | Build context sent | Compressed (`docker image inspect`) |
|---|---|---|
| Without `.dockerignore` | 2.10 kB | 396,476,098 bytes |
| With `.dockerignore` | 2.10 kB | 396,476,098 bytes |

The Dockerfile copies only named files, so the ignore file changes neither the build
context nor the image here. It matters if a broad `COPY . .` is ever added: it would then
keep `.venv` (1.4 GB), `data/` (4.9 GB) and `.git` (3.3 MB), measured with `du -sh`, out of
the build.

## Docker Hub

- Image: https://hub.docker.com/r/salma201/pet-breed
- Tags: `0.1.0`, `latest` (same digest `sha256:6310eede...`)
- The published model is an untrained checkpoint; a trained one will ship as `0.2.0`.

## Maturity self-assessment

**Current level:** The code is packaged, tested (74% coverage, 70% gate),
containerized as a non-root image and published on Docker Hub, and the data is versioned
with DVC. But training is still run by hand: there is no experiment tracking, no model
registry, and no CI pipeline that checks model quality.

**What is missing for the next level:** experiment tracking and a model registry (MLflow),
a reproducible training pipeline with `dvc repro`, and CI with a quality gate that blocks a
regression in top-1. This is Module 2.
