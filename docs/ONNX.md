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
- Context emitters: `Linear`, `Linear<TWeight>`, `Conv1d`, `Conv2d`, `Embedding`, `Embedding<TWeight>`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `TransformerBlock`, `MultiheadAttention`, `Sequential`, `Relu`, `Gelu`, `Sigmoid`, `Tanh`, `Add`, `TransposeAxes`, `Reshape`, `Concat`, `ReduceMean`

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

## Classifier model support

NivaraClassifier exports both linear and convolutional text classifiers to ONNX successfully. Both models expect token ID inputs of shape `[batch, maxSeqLen]`.

### TextClassifierModel (linear)
Emitted graph (from `Nivara.AutoDiff.Nn.TextClassifierModel<T>`):
1. `Embedding` from input token IDs
2. `ReduceMean` over axis `1` (mean pooling) with `keepDims: false`
3. `Linear` + `Relu`
4. `Linear` (logits)

Emitters used: `Embedding`, `ReduceMean`, `Linear`, `Relu`.

### ConvTextClassifierModel (convolutional)
Emitted graph (from `NivaraClassifier.ConvTextClassifierModel<T>`):
1. `Embedding` from input token IDs
2. `TransposeAxes` to convert from NLC to NCL layout
3. Three parallel `Conv1d` branches (kernel sizes 3, 5, 7) with `Relu`
4. `Reshape` each branch output to flatten spatial dimensions
5. `Concat` branches along axis `1`
6. `Linear` + `Relu`
7. `Linear` (logits)

Emitters used: `Embedding`, `TransposeAxes`, `Conv1d`, `Relu`, `Reshape`, `Concat`, `Linear`. `Dropout` is identity in Eval mode and is omitted from the exported graph.

## Support matrix and known gaps

| Operator | Status | Notes |
| --- | --- | --- |
| Embedding/Gather | Supported | Input indices cast to int64 if needed |
| Linear/Gemm | Supported | Weights/biases exported as initializers |
| Conv1d/Conv2d | Supported | Grouped and asymmetric Conv2d validated during export |
| TransposeAxes | Supported | Used by conv classifier models |
| Reshape | Supported | Used by conv classifier models |
| Concat | Supported | Used by multi-branch conv models |
| ReduceMean | Supported | Used for mean pooling in linear classifier |
| Relu/Gelu/Sigmoid/Tanh | Supported | Activation functions |
| BatchNorm1d/BatchNorm2d | Supported | Requires `trackRunningStats` |
| LayerNorm/RMSNorm | Supported | Normalization layers |
| MultiheadAttention/TransformerBlock | Supported | Covered by parity tests |
| Dropout | Supported (identity) | Passes through unchanged; omitted in Eval/export by design |
| Add | Supported | Element-wise addition (used by residuals) |
| MaxPool/AvgPool | Not supported | Not present in `IOnnxExportContext`; would need to be added for future conv variants |

Any unsupported operator desired by future models should be recorded as a linked issue rather than silently missing. Export fails loudly for unsupported operations.

## Diagnostics

Export decisions and unsupported operations are recorded through `OperationDiagnostics` when `DiagnosticsTracker.IsEnabled` or `OnnxExportDiagnostics.Enabled` is true. Disabled by default (zero per-emitter overhead).

## Verification commands

```bash
dotnet build Nivara.slnx
dotnet run --project samples/NivaraClassifier -c Release -- --command onnx-compare
```

The sample exports a tiny MLP and asserts ONNX Runtime output matches Nivara within 1e-4.

## Limits

- Only modules implementing `ExportToOnnx` can be exported; traversal covers the common module set (Linear, Embedding, Conv1d/Conv2d, BatchNorm, LayerNorm/RMSNorm, TransformerBlock/MultiheadAttention, activations, Sequential).
- Input shape is required and must match the model's expected input.
- No grouped/asymmetric parameter surprises: grouped and asymmetric Conv2d are validated during export.
