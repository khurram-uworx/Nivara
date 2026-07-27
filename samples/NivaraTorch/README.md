# NivaraTorch — Per-Layer PyTorch Validation

Formal A/B validation of every NN layer type in Nivara's AutoDiff engine. PyTorch generates reference tensors via `gen_reference.py`, Nivara reproduces them to machine precision.

## What's here

- **`gen_reference.py`** — Generates PyTorch reference fixtures (float32 binary + JSON manifest) for all 21+ layer types. Run `python gen_reference.py` to regenerate.
- **Tests live in `tests/Nivara.Tests/NivaraTorch/`** — NUnit test files organized by layer type, comparing C# output against PyTorch fixtures.

## Fixture data

All fixtures are stored in `samples/data/torch-comparison/` (not in this directory). The generator writes 45 test cases covering:

| Layer type | Configs | Notes |
|---|---|---|
| Conv2d | 5 | 3x3, 1x1, depthwise, stride 2, with bias |
| Conv1d | 4 | kernel 3/5/7, stride 2 |
| BatchNorm2d | 3 + 3 bug-case | Eval mode with running stats |
| BatchNorm1d | 2 | 2D and 3D input |
| ReLU/ReLU6 | 2 | 1D and 4D |
| LeakyReLU | 2 | slope=0.01 |
| Sigmoid | 2 | 1D and 4D |
| Tanh | 2 | 1D and 4D |
| MaxPool2d | 2 | 3x3 stride 2, 2x2 stride 2 |
| AdaptiveAvgPool2d | 2 | Large and small feature maps |
| Linear | 2 | 128→64, 512→1000 |
| Embedding | 2 | Single and batch lookup |
| Dropout | 1 | Eval mode passthrough |
| RMSNorm | 2 | Per-row, 2D and 3D |
| LayerNorm | 2 | 2D and 3D |
| Softmax / LogSoftmax | 2 | Over dim 1 |
| MatMul | 1 | 4×8 @ 8×16 |
| BCEWithLogitsLoss | 2 | Sum and mean reduction |
| CrossEntropyLoss | 1 | With integer targets |
| MSELoss | 2 | Sum and mean reduction |
| L1Loss | 1 | Sum reduction |

## Layout notes

- **Conv1d weight layout**: PyTorch stores `[outChannels, inChannels, kernelSize]`. Nivara's kernel expects `[outChannels, kernelSize, inChannels]`. Tests transpose the fixture weight when loading.
- **RMSNorm**: `ReverseGradOperations.RMSNorm` normalizes over the entire flattened tensor. For per-row normalization (matching PyTorch), use `ReverseGradOperations.PerRowRMSNorm`.

## How to run tests

```bash
dotnet test --filter "FullyQualifiedName~NivaraTorch"
```

## Regenerating fixtures

Requires Python with PyTorch:

```bash
python samples/NivaraTorch/gen_reference.py
```
