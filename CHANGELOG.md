# Changelog

All notable changes to Nivara are documented here. Released versions are published to NuGet via the tag-triggered CD workflow (`v*` tags on `main`).

## Unreleased

### Breaking changes

- **`NivaraSeries<T>.Sum()` / `Min()` / `Max()` removed** (AutoDiff refactor, Task 9): NivaraSeries is now a labeled-column wrapper and keeps only `Average()`. Use the null-aware column reductions `NivaraColumn<T>` extensions `Sum` / `Min` / `Max` (`Nivara.Tensors`, `INumber<T>`-constrained) via `series.Values`; empty-column `Sum` throws, all-null `Sum` returns `T.Zero`, all-null `Min`/`Max` throw. Non-numeric (string/object) Min/Max/Sum are no longer supported.
- **`NivaraTensorExtensions` stripped to column reductions** (AutoDiff refactor, Task 8): the column-level activations/gradients/MatMul/Transpose/GELU family extension methods were deleted (they now live in `GradKernels` as span kernels) along with the obsolete Series extensions (`AddTensor`, `MultiplyTensor`, `SumTensor`, `DotProduct`, `Norm`, `TransformTensor`) and `MatrixMultiply`. Remaining members: `Sum`, `Mean`, `Min`, `Max`. `NivaraColumn<T>.Subtract(NivaraColumn<T>)` / `Divide(NivaraColumn<T>)` / `Divide(T)` were promoted from extensions to first-class members.
- **Model/checkpoint serialization format bumped to `nivara-ss-v2` / `nivara-ckpt-v2`** (AutoDiff, ADR-001): the null-mask persistence (`HasNulls` / `NullMask` on parameter entries, `ParameterData<T>.NullMask`) was removed from the AutoDiff non-nullable domain. Deserialize now uses the zero-copy `CreateFromOwnedArray` path. v1 files are rejected loudly with an "unsupported format" error instead of being silently misread.
- `ArrowConversionOptions.UseZeroCopy` removed — the option defaulted to `true` but every zero-copy interop path was a placeholder that silently copied. Nivara does not advertise unsupported capability; real zero-copy returns with ARROW-ROADMAP Phase D (adding real APIs then).

### Storage Consolidation

- `Nivara.Storage.MemoryStorage<T>` renamed to `Nivara.Storage.ColumnStorage<T>` and moved to sole-owner contiguous `T[]` backing with an optional `bool[]` null mask (`null` mask ⇒ non-nullable column). `Data`/`NullMaskMemory`/`AsSpan()`/`TryGetSpan`/`Slice` keep their zero-copy, shared-buffer semantics.
- New internal lazy `ColumnStorage<T>.AsTensor()` returns a zero-copy `Tensor<T>` view over the storage's backing array (unmanaged `T` only — `Half`/`BFloat16` pass; reference-containing types throw). Slices are supported via `Tensor.Create(array, start, lengths, strides)`.
- `ColumnStorageFactory` now builds `ColumnStorage<T>` directly for every type — vectorizable primitives no longer route to `TensorStorage<T>`. The tensor/memory split helpers (`createTensorStorage`, `CreateTensorStorageForType`, `CreateTensorStorageForOwnedArray`, `CreateTensorStorageForNullableType`) and the duplicate `IsUnmanagedType<T>()` type list were deleted; the runtime unmanaged guard lives on `ColumnStorage<T>.AsTensor()` via `RuntimeHelpers.IsReferenceOrContainsReferences<T>()`. `IsVectorizable<T>()` is retained for `KernelSelector` heuristics.
- `Nivara.Storage.TensorStorage<T>` deleted and `StorageType`/`StorageType`-based dispatch removed from the storage contract (`IColumnStorage<T>`), `ColumnDiagnostics`, and `NivaraColumn`. All storage is the single `ColumnStorage<T>`; span access is always a genuine zero-copy view (`ProvidesZeroCopySpanAccess` dropped), and the `NivaraColumn` vectorized scalar kernels now operate directly on the storage's zero-copy span instead of pooling + copying the tensor-backed buffers. The scalar-comparison dead branches that threw for unsupported combinations were removed along with the tensor path.
- Storage consolidation onto a single `ColumnStorage<T>` is **complete**: `NivaraColumn` dispatch path collapse, AutoDiff boundary hardening (runtime ADR-001 throws), and the benchmark gate all landed. Before/after results (baseline vs post-consolidation) are captured in `tests/Nivara.PerformanceTests/README.md`.
- **AutoDiff boundary (ADR-001) enforced at runtime**: `ReverseGradTensor`/`ForwardGradTensor` constructors now throw `AutoGradException` (message contains "ADR-001") when the input column `HasNulls` (previously only a stripped-in-Release `Debug.Assert`); `ForwardGradTensor` tangent columns are guarded identically.
- **AutoDiff enter path is zero-copy**: `FromColumn`/`FromSeries` wrap the column without copying; `FromArray`/`FromMatrix` now wrap the caller's array via `CreateFromOwnedArray` — **breaking contract change**, callers must not mutate the source array afterward. `GradTensor.AsTensor()` returns a zero-copy `ColumnStorage<T>.AsTensor()` view sharing the backing array instead of a flattened copy; `NivaraColumn.AsTensorView()` backs it. `ModuleHelpers.GetSpan` fallback copy removed (`TryGetSpan` now always succeeds for AutoDiff tensors).

### Fixed

- **Owned-array contract documented on remaining factory surfaces** (#106): `Parameter(string, T[], bool)` and `GradientUtils.Constant(T[])` wrap caller arrays zero-copy; XML docs now state ownership transfers and that the source array must not be mutated afterward, matching the `FromArray`/`FromMatrix` contract.

### Added

- **Public zero-copy tensor view** (#107): `NivaraColumn<T>.AsTensorView()` and `NivaraSeries<T>.AsTensorView()` are now public (previously internal). They return a lazy `Tensor<T>` view sharing the column's/series' backing array with no copy; null-containing columns and reference element types throw `InvalidOperationException`. Callers must treat the view as read-only.

### Query Engine

- `OrderBy`/`OrderByDescending` support computed sort keys (`OrderBy(x => x["A"] + x["B"])`) via a materialized-key `SortByExpressionOperation` — no longer throws `NotSupportedException`; null placement and direction match `Sort` semantics
- `ThenBy`/`ThenByDescending` compose secondary sorts lexicographically with a preceding `OrderBy`/`Sort`: `NivaraFrame` string overloads and LINQ `QueryFrame` lambda overloads, both computed-key capable. Column-reference keys merge into the efficient multi-key `SortOperation`; computed keys merge into a multi-key `SortByExpressionOperation`. Without a preceding sort they act as a primary sort

## [1.1.0] - 2026-07-31

### Automatic Differentiation (inference-default)

- Reverse-mode graph construction is opt-in via `GradientUtils.Grad()`; inference is the default and records no graph nodes
- Type constraint relaxed from `INumber<T>` to `IFloatingPointIeee754<T>` — `float`, `double`, `Half`/F16 and BFloat16 pass runtime validation
- All differentiable operations span-ified over `TensorPrimitives` (no `NivaraColumn.Data` access)
- ADR-001 non-nullable domain cleanup: null-mask infrastructure removed from AutoDiff ops and hot paths; `Debug.Assert` boundary guards in `ReverseGradTensor` and `ComputationGraph.AddNode`

### NN Module System

- `Conv1d<T>` — im2col + `TensorPrimitives.Dot` kernel, PyTorch-compatible weight layout
- `Conv2d<T>` — tiled im2col, PatchLocation lookup, grouped convolution, 1x1 fast path, InputGrad specializations; `ConvTranspose2d<T>`
- `BatchNorm1d<T>` (2D `[N,C]` and 3D `[B,C,L]` inputs) and `BatchNorm2d<T>` — fused span kernels
- `LayerNorm<T>` (SIMD `TensorPrimitives.Dot`), `DepthwiseSeparableConv2d<T>`, `TransformerBlock<T>` (RMSNorm/LayerNorm + GELU), `MultiheadAttention<T>` (self/cross/causal, padding mask)
- `ConvVAE<T>`, `VAE<T>` (optional conditioning), `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, `GELU` activation
- `RMSNormKernel<T>` consolidating duplicated per-row RMSNorm logic

### Performance

- SIMD-accelerated kernels via TensorPrimitives chains: Adam, AdamW, PerRowRMSNorm backward, LayerNorm sum-of-squares, GELU forward/backward
- ArrayPool-backed buffer management in hot paths: `AccumulateGradient`, Gather backward, Adam/AdamW state
- `Gather` zero-copy forward path + ArrayPool backward path; `Embedding` lookup via Gather (replaces one-hot + MatMul)

### Training & Serialization

- Optimizers `SGD`, `Adam`, `AdamW` with SIMD kernels; `BCEWithLogitsLoss` fused backward; `MSELoss` `reduceToMean`
- `TrainingLoop<T>`, `DataParallelTrainer<T>`, `TensorDataset<T>`
- `ModelSerializer` JSON/binary save-load; `StateDict()` / `LoadStateDict()` module state

### Samples & Interop

- `samples/NivaraInference` — MobileNetV2/ResNet-18 inference with `SafeTensorsLoader` (I32/I64/F16/BF16/F32 dtype-aware)
- `samples/NivaraFineTuning` — DistilBERT fine-tuning on GLUE SST-2
- `samples/NivaraTimeSeries` — time-series anomaly detection
- `samples/NivaraTorch` — 55 PyTorch-validated functional tests across 21+ layer types (`gen_reference.py` fixtures)
- Generic dtype-aware weight loading for `DistilBertModel`, `MiniLMDistilled`, `SafeTensorsLoader`

### Documentation

- README, GETTING-STARTED, ARCHITECTURE, docs/AUTODIFF updated for the inference-default AutoDiff direction and new modules

## [1.0.0]

- Initial stable release of the columnar DataFrame core: typed immutable columns/frames, LINQ-like query engine with lazy/eager/streaming/parallel strategies, tensor-accelerated kernels, explicit null masks, join/group-by/aggregation, CSV/JSON sources, Parquet/Arrow/ML.NET interop (Extensions), performance diagnostics and buffer pooling
- Reverse-mode AutoDiff (initial), VAE/ConvVAE samples
