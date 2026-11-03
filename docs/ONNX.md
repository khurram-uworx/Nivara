# ONNX Export

Export a Nivara-trained `Module<T>` to a `.onnx` file and run it anywhere ONNX Runtime or ML.NET runs. Nivara owns the training; the platform owns inference.

## Scope and non-goals

- **Export only.** Nivara does not import or execute ONNX models. Use ONNX Runtime or ML.NET (`ApplyOnnxModel`) over the exported file or over frame data.
- No ONNX runtime dependency in core; all export code lives in `Nivara.Extensions` under `Nivara.Onnx`.
- Unsupported operations fail loudly (they throw during export). There is no silent fallback.

## Prerequisites

- .NET 11 SDK
- `Nivara` (core) and `Nivara.Extensions`
- For verification only: `Microsoft.ML.OnnxRuntime` 1.31.0

## API surface

- `OnnxExporter.Export<T>(Module<T> module, Stream stream, OnnxExportOptions? options = null)`
- `OnnxExportOptions`: `InputShape`, `OpsetVersion` (default 20), `ProducerName`, `ProducerVersion`, `GraphName`, `NativeDtype` / `UseNativeDtype` / `ExportNativeDtype`
- `Module<T>.ExportToOnnx(IOnnxExportContext<T>, string inputName)` implemented by your model
- Context emitters: `Linear`, `Linear<TWeight>`, `Conv1d`, `Conv2d`, `Embedding`, `Embedding<TWeight>`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `TransformerBlock`, `MultiheadAttention`, `Sequential`, `Relu`, `Gelu`, `Sigmoid`, `Tanh`, `Add`

ONNX element type follows the module element type: `float`, `double`, `Half`, or `BFloat16`. By default weights are widened to F32 for runtime compatibility; set native dtype to preserve `Half`/`BFloat16`.

## Workflow

```csharp
using var scope = GradientUtils.Grad();
// ... train ...
model.Eval();

var ms = new System.IO.MemoryStream();
OnnxExporter.Export(model, ms, new OnnxExportOptions
{
    InputShape = new[] { rows, features }
});
File.WriteAllBytes("model.onnx", ms.ToArray());
```

## Diagnostics

Export decisions and unsupported operations are recorded through `OperationDiagnostics` when `DiagnosticsTracker.IsEnabled` or `OnnxExportDiagnostics.Enabled` is true. Disabled by default (zero per-emitter overhead).

## Verification commands

```bash
dotnet build Nivara.slnx
dotnet run --project samples/NivaraOnnxExport -c Release
```

The sample exports a tiny MLP and asserts ONNX Runtime output matches Nivara within 1e-4.

## Limits

- Only modules implementing `ExportToOnnx` can be exported; traversal covers the common module set (Linear, Embedding, Conv1d/Conv2d, BatchNorm, LayerNorm/RMSNorm, TransformerBlock/MultiheadAttention, activations, Sequential).
- Input shape is required and must match the model's expected input.
- No grouped/asymmetric parameter surprises: grouped and asymmetric Conv2d are validated during export.
