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

ONNX element type follows the module element type: `float`, `double`, `Half`, or `BFloat16`. Mixed-precision weights (e.g. `Linear<T, TWeight>`) are widened to the module element type `T` during export. `NativeDtype`/`UseNativeDtype`/`ExportNativeDtype` are reserved for future use and currently have no effect.

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
| Embedding/Gather | Supported | Input indices cast to int64 if needed (float inputs are truncated toward zero) |
| Linear/Gemm | Supported | Weights/biases exported as initializers |
| Conv1d | Supported | Group=1 only; symmetric kernel/stride; non-negative padding |
| Conv2d | Supported | Grouped + asymmetric padding validated during export; square kernel, single stride for both axes; channels divisible by groups |
| TransposeAxes | Supported | Used by conv classifier models |
| Reshape | Supported | Used by conv classifier models |
| Concat | Supported | Used by multi-branch conv models |
| ReduceMean | Supported | Used for mean pooling in linear classifier |
| Relu/Gelu/Sigmoid/Tanh | Supported | Activation functions |
| BatchNorm1d/BatchNorm2d | Supported | Requires `trackRunningStats`; epsilon exported as a float attribute |
| LayerNorm | Supported | Normalized (last) dimension must match `NormalizedShape`; epsilon exported as a float attribute; affine optional |
| RMSNorm | Supported | Normalization layer |
| TransformerBlock | Supported | Inlines multi-head self-attention; 2D input `[seqLen, embdDim]` (no batch dimension); dropout ignored (Eval mode assumed). Covered by parity tests |
| MultiheadAttention | Not supported | Standalone module; export throws `NotSupportedException`. Use `TransformerBlock` instead |
| Dropout | Supported (identity) | Passes through unchanged; omitted in Eval/export by design |
| Add | Supported | Element-wise addition (used by residuals) |
| MaxPool/AvgPool | Not supported | Not present in `IOnnxExportContext`; would need to be added for future conv variants |

Any unsupported operator desired by future models should be recorded as a linked issue rather than silently missing. Export fails loudly for unsupported operations.

## Constraints and unsupported operations

- `MultiheadAttention`, `ConvTranspose2d`, and any module without an `ExportToOnnx` override are **not** exportable and throw `NotSupportedException` during export (no silent fallback).
- `Conv1d` is group=1 only. `Conv2d` requires groups > 0 with channels divisible by groups, positive kernel size and stride, and non-negative per-side padding.
- `BatchNorm1d`/`BatchNorm2d` require `TrackRunningStats`.
- `TransformerBlock` assumes a 2D input `[seqLen, embdDim]` (no batch dimension) and is exported in Eval mode (dropout ignored).
- `Embedding` inputs are treated as integer indices; non-integer floats are truncated toward zero by the ONNX `Cast`.
- Epsilon for `BatchNormalization` and `LayerNormalization` is exported as a float attribute (ONNX spec).

## Verification

Parity against ONNX Runtime (which also backs ML.NET's `ApplyOnnxModel`/`OnnxTransformer`) is covered by:
- `tests/Nivara.Tests/Onnx/OnnxExportMlpTests.cs` — `Linear -> ReLU -> Linear`, tolerance 1e-4
- `tests/Nivara.Tests/Onnx/OnnxExportTransformerTests.cs` — `TransformerBlock`, tolerance 5e-4
- `tests/Nivara.Tests/Onnx/OnnxExportUnsupportedTests.cs` — unsupported modules throw

Run: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Onnx"`

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
