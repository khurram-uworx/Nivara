# NivaraOnnxExport - Train in Nivara, deploy via ONNX Runtime

A minimal sample demonstrating the export-only ONNX bridge: train a small MLP in Nivara, export it to `.onnx`, then run inference through ONNX Runtime and verify parity with Nivara's own output.

## Prerequisites

- .NET 11 SDK
- `Microsoft.ML.OnnxRuntime` 1.31.0 (resolved automatically by the project)

## Quick start

```bash
dotnet run --project samples/NivaraOnnxExport -c Release
```

Expected output (approx):

```
epoch  1/20: mse = ...
epoch 20/20: mse = ...
exported: .../nivara-mlp.onnx (N bytes)
ONNX Runtime output shape: [8, 3]
max abs diff vs Nivara: 1.234e-06
OK: Nivara / ONNX Runtime parity within 1e-4
```

## What it shows

- Manual training loop wrapped in `using (GradientUtils.Grad())`.
- `Module<T>.Eval()` for inference mode.
- `OnnxExporter.Export<T>(module, stream, options)` with `InputShape`.
- Parity check via `InferenceSession` from `Microsoft.ML.OnnxRuntime`.

## Limitations

- Only ops supported by `IOnnxExportContext<T>` can be exported (see `docs/ONNX.md`).
- Nivara does not import or run ONNX models; use ONNX Runtime / ML.NET for that.
