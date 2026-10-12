# Module 1 Report

## Serialization

- Export: PyTorch checkpoint (`model.pt`) to ONNX (opset 17, dynamic batch axis).
- Parity test: 200 validation images, max abs diff 1.19e-06, passes at atol 1e-4.

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
