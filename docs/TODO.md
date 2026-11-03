# ONNX export for Nivara-trained modules (#866)

## Problem

Nivara can train AutoDiff `Module<T>` instances but only serialize them as JSON state dicts
(ModelSerializer) or import SafeTensors. The interchange gap is exporting to `.onnx` so trained
models can be deployed via ONNX Runtime / ML.NET. Import/inference stay first-party ML.NET
concerns and are deliberately out of scope.

## Scope and acceptance

Export a trained `Module<T>` to a `.onnx` file covering the common module set (`Linear`,
`Embedding`, `Conv1d`/`Conv2d`, `BatchNorm`, `LayerNorm`/`RMSNorm`,
`TransformerBlock`/`MultiheadAttention`, activations, `Sequential`). The exported file must load
in ONNX Runtime and reproduce Nivara inference within float tolerance.

Non-goals: no ONNX import/inference inside Nivara; no runtime ONNX dependency in core; no cloud.

## Grounding summary (G1)

- Microsoft Learn: `Microsoft.ML.OnnxRuntime.InferenceSession` is the parity runtime; ML.NET
  `OnnxExportExtensions.ConvertToOnnx` only converts ML.NET `ITransformer`, not Nivara modules.
- No managed ONNX model-builder API. The preferred build-time route is Google.Protobuf plus
  Grpc.Tools proto generation from a vendored `onnx.proto`.
- Code-memory symbol/relationship scan of `Module`, `Sequential`, `Linear`, `Embedding`,
  `Conv*`, `BatchNorm`, `LayerNorm`, `RMSNorm`, `TransformerBlock`, `MultiheadAttention`,
  `Parameter`, and `StateDict` (see AISTACK roadmap context).

## Proposed changes

- Add `src/Nivara.Extensions/Onnx/` exporting ONNX as a protobuf graph in
  `Nivara.Onnx`, depending only in Extensions (core keeps `System.Numerics.Tensors`).
- Vendor and generate ONNX protobuf types via Grpc.Tools; storage dtype defaults to F32.
- Implement per-module emitters that consume an IR/graph builder and preserve module weights
  as initializers:
  1. Linear via Gemm (transB=1) with weight/bias parameters.
  2. Activations (ReLU, Gelu, Sigmoid, Tanh as available).
  3. Sequential chain assembly.
  4. Embedding via Gather.
  5. Conv1d/Conv2d via Conv.
  6. BatchNorm via BatchNormalization.
  7. LayerNorm via LayerNormalization.
  8. RMSNorm via a decomposed subgraph.
  9. Transformer attention and blocks via the primitives above.
- Apply inference mode (`Eval()` semantics) when exporting: dropout becomes identity/no-op.
- Throw a clear `NotSupportedException` for modules or ops without an emitter, per "no silent fallback".
- Record export decisions through `OperationDiagnostics`.
- Add round-trip parity tests in `tests/Nivara.Tests/Onnx/` using `Microsoft.ML.OnnxRuntime`
  `InferenceSession`, comparing outputs within float tolerance.

## Blast radius

- Files touched: `src/Nivara.Extensions/Onnx/*`, `src/Nivara.Extensions/Nivara.Extensions.csproj`,
  new tests `tests/Nivara.Tests/Onnx/*`, `tests/Nivara.Tests/Nivara.Tests.csproj` package refs.
- Downstream callers: none existing; new public API under `Nivara.Onnx`.
- Risk areas: attention subgraph shape correctness; exporter graph naming/init shapes; mixed
  precision (`Linear<TModel,TWeight>`) weight widening; docs pins may need adjustment when new docs/tests land.

## Planned commits

1. `docs: plan #866 ONNX export in TODO.md`
2. `feat: add ONNX protobuf generation scaffolding (Onnx namespace)`
3. `feat: add OnnxExporter graph builder and Linear/activations/Sequential`
4. `feat: add Conv, BatchNorm, LayerNorm, RMSNorm, Embedding emitters`
5. `feat: add TransformerBlock/MultiheadAttention export`
6. `test: add ONNX round-trip parity tests`
7. `chore: record ONNX export diagnostics`
8. `docs: update AISTACK plans for ONNX completion`
9. `test: remove TODO.md (plan executed)` (gate dependent)

## GitHub issues log

- (none yet - append immediately if anything deferred)

## Verification plan

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Onnx"` with confirmation
- Parity: export module, run `InferenceSession`, compare to Nivara `Forward` output
- Clean up temp scaffolding; keep tests in existing test projects