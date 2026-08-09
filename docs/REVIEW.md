# Nivara API Surface Review

Date: 2026-08-09
Reviewer: senior .NET engineering review (assisted)

This document captures the findings of a full review of the Nivara library and its
public API surface. It is a snapshot of the tree at the time of writing — it is
*not* an action plan. See the "Prioritized Cleanup Ideas" section at the end for a
starting point if this ever becomes one.

## Scope

- Core library: `src/Nivara/` (columnar engine, query/execution, AutoDiff, IO, diagnostics)
- Extensions library: `src/Nivara.Extensions/` (CSV, Arrow, Parquet, ML.NET, AI)
- Public surface only: every public type/member inventoried; internal storage verified leak-free

## Overall Assessment

Nivara is an ambitious library — effectively **three products in one package**:

1. a pandas-style columnar DataFrame engine,
2. a query planner/executor with four execution strategies,
3. a full autograd + NN module system.

The core is genuinely well-engineered in places: storage is cleanly internal, the
null-mask semantics are rigorous, and the core stays dependency-light (single
`System.Numerics.Tensors` reference). But the API surface shows the strain of that
ambition: three parallel query paradigms, inconsistent naming across analogous
members, ~80% dead public exception surface, and several silent API lies
(parameters accepted but ignored, `Half` admitted but not served, `ToList()` that
returns a frame).

## Issue Tracking

The High and Medium findings below were filed as GitHub issues (2026-08-09) for
the engineering team to discuss and prioritize. Lower-priority findings are not
yet tracked as issues.

| # | Finding | Priority | Issue |
|---|---|---|---|
| 1 | `NivaraSeries<T>.Average()` throws for `Half`/`nint`/`nuint`/`Int128`/`UInt128` | High | [#172](https://github.com/khurram-uworx/Nivara/issues/172) |
| 2 | Frame `Take`/`Skip`/`Slice` slice columns via reflection | High | [#173](https://github.com/khurram-uworx/Nivara/issues/173) |
| 3 | `NivaraResourceManager` per-allocation overhead + permanent timer | High | [#174](https://github.com/khurram-uworx/Nivara/issues/174) |
| 4 | Three parallel query paradigms, no declared primary API | Medium | [#175](https://github.com/khurram-uworx/Nivara/issues/175) |
| 5 | `QueryFrame.ToList()` returns `NivaraFrame` | Medium | [#176](https://github.com/khurram-uworx/Nivara/issues/176) |
| 6 | AutoDiff weight access inconsistent across modules | Medium | [#177](https://github.com/khurram-uworx/Nivara/issues/177) |
| 7 | Dead public surface in AutoDiff (GradKernels, ComputationGraph, exceptions, legacy initializers) | Medium | [#178](https://github.com/khurram-uworx/Nivara/issues/178) |
| 8 | Silent API lies in AutoDiff (Softmax.dim, Half, CanBackward, BatchNorm NRE, BatchBackward) | Medium | [#179](https://github.com/khurram-uworx/Nivara/issues/179) |
| 9 | Loss API: no common base, no `Reduction` enum | Medium | [#180](https://github.com/khurram-uworx/Nivara/issues/180) |
| 10 | Optimizer API: split learning-rate mutation paths, SGD-only functional entry | Medium | [#181](https://github.com/khurram-uworx/Nivara/issues/181) |

## What's Genuinely Good

- **Storage encapsulation.** `ColumnStorage<T>`, `ColumnStorageFactory`,
  `IColumnStorage<T>` are all internal; no internal type leaks into any public
  signature. `NivaraColumn<T>.AsTensorView()` / `TryGetSpan` / `CreateFromOwnedArray`
  are a clean zero-copy story with proper ownership contracts.
- **Null semantics are first-class and consistent.** Explicit bool masks, OR
  propagation, `TryGetSpan` returns `false` on nulls — more principled than
  pandas' NaN-overloading.
- **`NivaraRow` is a well-designed readonly struct** — allocation-free, explicit
  `default(NivaraRow)` guard, typed `GetValue<T>` with `ColumnTypeMismatchException`,
  `TryGetValue` escape hatch.
- **AD-weighted exception hierarchy in IO** (`NivaraIOException` with `FilePath`,
  `UnsupportedTypeException` with `SuggestedAlternatives`) and `[CollectionBuilder]`
  support on both column and series are modern touches.
- **Inference-default AutoDiff** (`GradientUtils.Grad()`) is a deliberate, documented
  product decision with runtime enforcement (ADR-001) and regression guards.

---

## High-Priority Findings (functional bugs / hot-path perf)

### 1. `NivaraSeries<T>.Average()` throws for 5 of 17 supported numeric types

> **Resolved 2026-08-09 (issue #172, PR #182):** `divideByCount` gained the 5 missing
> arms and the public `Average()` guard now accepts the full `GetNumericTypes()` domain
> (bool remains rejected by the sum dispatch). Covered by
> `tests/Nivara.Tests/NivaraSeriesAggregateTests.cs`.

`sumTensorPrimitive` (`NivaraSeries.cs:69-97`) dispatches 17 types, but
`divideByCount` (`NivaraSeries.cs:102-183`) covers only 12 and throws
`NotSupportedException` for **`Half`/`nint`/`nuint`/`Int128`/`UInt128`** — after
the SIMD sum already succeeded. `NivaraSeries<Half>.Average()` computes the sum,
then throws. This is a runtime functional gap, not style. (Also:
`NivaraTensorExtensions.Mean<T>` on the column side has no such gap — an
inconsistency between sibling APIs.)

### 2. Every frame slice/take/skip is reflection-based

> **Resolved 2026-08-09 (issue #173, branch khurram/173):** `NivaraFrame.sliceColumn`
> now calls `IColumn.Slice(int, int)` directly (`NivaraFrame.cs:50`). The same
> reflection pattern in the query engine's `SliceOperation.SliceColumn` was fixed
> as well. Both unreachable `ColumnFilterHelper.CreateFilteredColumn` fallbacks
> were deleted; the helper remains for the Distinct/Filter/SelectRows/FilterByMask
> paths. A `Frame Slice [10k x 128]` scenario was added to the perf harness so the
> removed per-call `object[]` allocations are measurable.

`NivaraFrame.sliceColumn` (`NivaraFrame.cs:50`) calls
`columnType.GetMethod("Slice", ...)` + `MethodInfo.Invoke` on the `IColumn` for
**every** `Take`/`Skip`/`Slice` (`NivaraFrame.cs:1018,1063,1106`). This is a hot
path in a library that advertises high performance. The interface `IColumn`
already declares `Slice(int, int)` — cast to `IColumn` and call it directly. No
reflection needed.

### 3. Global hidden resource-manager overhead on every construction

> **Resolved 2026-08-09 (issue #174, branch khurram/174):** `NivaraResourceManager`
> tracking is now opt-in via `NivaraResourceManager.Enable()` / `IsEnabled` (default
> OFF). The 30-second cleanup timer is created lazily only on `Enable()`, so default
> hosts never run a timer thread; `TrackResource` / `UntrackResource` / the timer
> callback are guarded no-ops when disabled. Column ctor only computes
> `estimateMemoryUsage()` inside the enabled branch. Public surface unchanged
> (`MemoryRecommendations`, `ResourceStatistics`, `NivaraFrame.GetMemoryRecommendations`).
> Covered by `tests/Nivara.Tests/ResourceManagementPropertyTests.cs` including a
> default-off assertion.

Every `NivaraColumn<T>` ctor (and frame, presumably) calls
`NivaraResourceManager.TrackResource` — inserting into a
`ConcurrentDictionary<WeakReference, ResourceInfo>` with a `DateTime`, plus a
**process-lifetime 30-second Timer** running in all hosts
(`NivaraResourceManager.cs:11-18`). That's a per-column heap allocation
(WeakReference + boxed ResourceInfo) and global state that monitors GC
collectability but never prevents collection — pure overhead on the hot path,
with no opt-out. For a "high-performance" library this is a design smell. The
feature would be better as opt-in telemetry or dropped.

---

## Medium-Priority Findings (API design inconsistencies)

### 4. Three parallel query paradigms with overlapping surface

- `QueryFrame` fluent API (string + `ColumnExpression` based)
- `NivaraLinqExtensions.Where/Select/OrderBy` on `QueryFrame` via
  `Func<RowExpressionBuilder, ColumnExpression>` — a *third* DSL
- `NivaraTypedLinqExtensions.Query<T>()` → `NivaraQuery<T>` expression-tree layer

Three ways to filter a frame, with different laziness, discoverability, and error
models. This needs a deliberate story (one primary, others relegated), not
organic growth.

### 5. `QueryFrame.ToList()` returns `NivaraFrame`

`NivaraLinqExtensions.ToList(this QueryFrame)` (`NivaraLinqExtensions.cs:140`)
returns a `NivaraFrame` — `ToList` universally means "materialize to a list" in
.NET. This is a naming trap; callers will read it as a collection of rows. It's a
documented alias for `Collect()`, but the name should be `ToNivaraFrame()` (which
also exists!) and `ToList` removed or made `[Obsolete]`.

### 6. AutoDiff weight access is three different contracts for the same concept

- `Linear<T>`: `Weight` (tensor) **and** `WeightParam` (parameter)
- `Conv1d/2d/Transpose`: `WeightParam`/`BiasParam` only
- `BatchNorm1d/2d`, `LayerNorm`: `Weight`/`Bias` are `Parameter<T>?` (nullable)
- `Embedding`: both `Weight` + `WeightParam`

Same member, three names, three nullability contracts. This will confuse every
consumer and every serialization path.

### 7. Dead public surface in AutoDiff

- **4 of 5 sealed exception types are never thrown.** `CircularDependencyException`,
  `InvalidBackwardCallException`, `TypeValidationException`,
  `GradientComputationException` (and most of `AutoGradException`'s rich members)
  are never constructed; the hot path throws plain `InvalidOperationException`
  instead. Public surface that can never occur.
- **`GradKernels` is public raw span math** with no AD semantics and no null
  guards — internal machinery that leaked. Contrast: `AttentionKernels`,
  `BatchNormKernel<T>`, `ModuleHelpers<T>` are correctly internal.
- **`ComputationGraph` is public with zero public members** (everything
  `internal static`).
- **Dual initializer APIs**: legacy static `KaimingNormal.Init<T>(Dictionary<...>)`
  (which *replaces* tensors in the dict, breaking `Parameter` identity) alongside
  `IInitializer<T>` instance classes — both public.
- **`DefaultInitializers.Bias<T>()` exists to return null.**

### 8. Silent API lies in AutoDiff

- **`Softmax<T>.dim` / `LogSoftmax<T>.dim` ctor param is stored, never read.**
  `Forward` always calls the dim-less op. Passing `dim=0` on 2D input silently
  gives last-dim semantics.
- **`Half` is admitted** (`IFloatingPointIeee754<T>` + `TypeValidator`) but the
  fast kernels and conversions (`ToFloat`/`ToDouble`, no `ToHalf`) don't cover it
  — supported-type story is overstated. Doc comments still say `INumber<T>`.
- **`GradientUtils.CanBackward` returns `RequiresGrad && Length == 1`** while
  `Backward()` explicitly supports non-scalar with an explicit gradient — the
  helper understates real capability.
- **`BatchNorm*.RunningMean/RunningVar/NumBatchesTracked` are non-nullable**
  backed by `!` — NRE if `trackRunningStats: false`.
- **`NivaraAutoGradExtensions.BatchBackward(tensors, loss)` ignores `tensors`** —
  calls only `loss.Backward()`.

### 9. Loss API is a grab bag

`BCELoss`/`L1Loss` (no reduction option), `MSELoss`/`BCEWithLogitsLoss`
(bool-flag overload), `CrossEntropyLoss` (always mean). No common base, no
`Reduction` enum, inconsistent ctor styles. `BCELoss(eps)` is stateful, the
others stateless.

### 10. Optimizer mental-model split

`Optimizer<T>.LearningRate` is get-only, but nested `ParameterGroup.LearningRate`
is settable and `SetGroupLearningRate` mutates. `SGD<T>` exposes a functional
`SgdUpdate`, Adam/AdamW don't.

---

## Lower-Priority Findings (naming / typing / discoverability)

### 11. Core types

- **`NivaraColumn<T>` operators are asymmetric**: only `*` and `+`; `-` and `/`
  are methods only.
- **`NivaraColumn<T>.CreateFromNullable(Array)`** takes a non-generic `Array`
  (boxing + reflection); the null-mask copy path allocates two arrays per call.
- **`NivaraSeries<T>.Index` is `NivaraColumn<object>`** — every default series
  pays a boxed int per position and an object-typed index column.
  `TopKDescending` then silently nulls non-string labels.
- **`NivaraSeries` has no instance `Sum`/`Min`/`Max`** — only `Average()`.
  `series.Values.Sum()` exists via extension but `series.Sum()` doesn't.
- **Int/label indexer ambiguity** is documented (`this[int]` = position,
  `this[object]` = label) but boxed ints silently route to the label path — the
  `GetByLabel` escape hatch exists but the trap is foot-gun-prone.

### 12. IO / Extensions

- **`Json` and `Csv` each expose 8 methods where 4 are duplicates**
  (`Scan`≡`ScanJson`, `Read`≡`ReadJson`, ...).
- **`JsonOptions.Default` / `CsvOptions.Default` are mutable singletons** —
  mutating one silently changes every default-path caller.
- **`CsvOptions.TrimOptions` is a `bool`** named like an options object;
  **`ParquetWriteOptions.Compression` is a magic string** `"snappy"` instead of
  an enum.
- **Sample POCOs public**: `TwoColumnData`, `ThreeColumnData`, `GenericData` in
  `MLNetInterop`.
- **`ModelIntegration.TrainAndEvaluate` returns `object Metrics`**;
  `TensorConversions.ReshapeToArray<T>`/`FlattenFromTensor(Array)` traffic in
  non-generic `Array` while core uses `Tensor<T>`.
- **Overlapping APIs**: `frame.ToParquet()` vs
  `ParquetWriter.WriteParquet(frame, path)`; `ArrowInterop.ToArrowArray` vs
  `NivaraSeriesExtensions.ToArrowArray`; `MLNetInterop.ToNivaraFrame(IDataView,
  MLContext)` vs `MLNetExtensions.ToNivaraFrame(this MLContext, IDataView)` — the
  parameter order flips between the two.

### 13. Schema

- **`ColumnMetadata.With()` cannot clear values** — `defaultValue ?? DefaultValue`
  means you can never reset `DefaultValue`/`Description`/`Properties` to
  null/empty.
- **`Schema.Equals` ignores `ColumnMetadata`** (name+type only).
- **Numeric domain is defined three different ways**: `Schema.areTypesCompatible`,
  `TypeExtensions.IsNumericType`, and `TypeCompatibilityValidator.GetNumericTypes`
  disagree (e.g. `Schema` omits `Half`/`nint`/`nuint`/`Int128`/`UInt128` that the
  #168 work added elsewhere).

### 14. Hygiene

- `public` members on internal classes (`ColumnStorageFactory.Create*`,
  `TensorsHelper.*`, `RankKernel.Compute`) — demoted classes whose members
  weren't tightened.
- `TensorsHelper.RowNorms` is orphaned dead code, contradicting AGENTS.md's claim
  that it was deleted; `RowDot`/`RowCosineSimilarity` survive on the frame.
- `TrainingResult.PrintSummary()` writes directly to `Console` — a side effect in
  a library API.
- `TrainingLoop<T>`/`DataParallelTrainer<T>` are unsealed against the repo's
  sealed-by-default rule, with inconsistent `Dispose(bool)` visibility.
- `SchemaInferenceRecords` (JSON) vs `SchemaInferenceRows` (CSV) — same concept,
  different word.
- `public` members on internal classes and `public static` members on internal
  static classes suggest classes that were demoted to internal without tightening
  member visibility.

---

## AutoDiff Public Surface Inventory

All types below are `where T : struct, IFloatingPointIeee754<T>` unless noted.

| Type | Kind | File |
|---|---|---|
| `GradTensor<T>` | `public class : IDisposable` (non-sealed) | `AutoDiff\GradTensor.cs` |
| `ReverseGradTensor<T>` | `public sealed class : GradTensor<T>` | `AutoDiff\ReverseGradTensor.cs` |
| `ForwardGradTensor<T>` | `public sealed class : GradTensor<T>` | `AutoDiff\ForwardGradTensor.cs` |
| `ComputationGraph` | `public sealed class` (no public members) | `AutoDiff\ComputationGraph.cs` |
| `OpNode<T>` | **internal** | `AutoDiff\OpNode.cs` |
| `GradientUtils`, `TypeValidator`, `TypeConverter` | `public static class` | `AutoDiff\Utilities\` |
| `AutoGradException` + 5 sealed derived | public exceptions | `AutoDiff\Exceptions\AutoGradExceptions.cs` |
| `Module<T>` (abstract), `Parameter<T>`, `Sequential<T>`, `Linear<T>`, `Conv1d/2d/Transpose2d<T>`, `BatchNorm1d/2d<T>`, `LayerNorm<T>`, `Dropout<T>`, `Embedding<T>`, `SparseEmbedding<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, `MultiheadAttention<T>`, `TransformerBlock<T>`, `VAE<T>`, `ConvVAE<T>`, `DepthwiseSeparableConv2d<T>`, `Sampler<T>`, `TextTokenizer` | NN modules | `AutoDiff\Nn\` |
| `BCELoss<T>`, `BCEWithLogitsLoss<T>`, `CrossEntropyLoss<T>`, `MSELoss<T>`, `L1Loss<T>`, `Softmax<T>`, `LogSoftmax<T>` | functional losses (no common base) | `AutoDiff\Nn\Functional\` |
| `IInitializer<T>` + 6 `*Initializer<T>` + 6 legacy static `*` + `DefaultInitializers` | initializers (two generations) | `AutoDiff\Nn\Initializers\` |
| `ReverseGradOperations` (39 ops), `ForwardGradOperations` (24 ops), `GradKernels` (**public**) | operations | `AutoDiff\Operations\` |
| `Optimizer<T>`, `SGD<T>`, `Adam<T>`, `AdamW<T>` | optimizers | `AutoDiff\Optimizer\` |
| `TrainingLoop<T>` (unsealed), `TrainingResult<T>`, `EpochResult<T>`, `DataLoader<T>`, `TensorDataset<T>`, `Batch<T>`, `DataParallelTrainer<T>` (unsealed), `DataParallelTrainingResult<T>`, `DataParallelEpochResult<T>` | training | `AutoDiff\Training\` |
| `ModelSerializer`, `Checkpoint<T>`, `ParameterData<T>` | serialization | `AutoDiff\Serialization\` |
| `NivaraAutoGradExtensions` | extensions | `AutoDiff\Extensions\` |
| `AutoDiffDiagnostics` | internal static class, members `public` | `AutoDiff\AutoDiffDiagnostics.cs` |

Known coverage asymmetry: `ForwardGradOperations` is missing `AddBias`,
`MatMulTransposedB`, `MultiHeadAttention`, `BatchedMultiHeadAttention`,
`TransposeAxes`, `MeanPool`, `GeluExact`, `Pow`, `Slice`, `Concat`, `RMSNorm`,
`PerRowRMSNorm`, `SparseEmbeddingBag`, `Gather`, `BroadcastMultiply`,
`BroadcastAdd` — forward mode cannot express what reverse mode can.

---

## Prioritized Cleanup Ideas

These are starting points if this review becomes an action plan. Grouped by
whether the change is safe (no breaking change) or requires a major bump.

### Safe cleanups (non-breaking)

1. Fix the `NivaraSeries<T>.Average()` divide-by-count coverage gap (add
   `Half`/`nint`/`nuint`/`Int128`/`UInt128` cases or route through a shared
   kernel). Add a `NivaraSeries<Half>` test.
2. ~~Replace the reflection-based `NivaraFrame.sliceColumn` with a direct
   `IColumn.Slice(int, int)` call.~~ — Done 2026-08-09 (issue #173): direct call in
   both `NivaraFrame.sliceColumn` and `SliceOperation.SliceColumn`.
3. Internalize `GradKernels`, `ComputationGraph`, `AutoDiffDiagnostics`, and the
   orphaned `TensorsHelper.RowNorms`; tighten `public` members on already-internal
   classes (`ColumnStorageFactory`, `RankKernel`, etc.).
4. Mark `QueryFrame.ToList()` `[Obsolete]` in favor of `ToNivaraFrame()`.
5. Delete the legacy static initializer API (`KaimingNormal.Init`, etc.) and
   `DefaultInitializers.Bias<T>()`.
6. Fix `NivaraAutoGradExtensions.BatchBackward` to honor `tensors` (or remove it).
7. Remove the dead `dim` field behavior in `Softmax<T>`/`LogSoftmax<T>` — either
   implement dim support or drop the parameter.
8. Correct AGENTS.md's stale `RowNorms` claim.

### Breaking-change items (2.0 candidate)

9. Unify AutoDiff weight access (`Weight`/`WeightParam`/`Parameter<T>?`) into one
   contract.
10. Introduce a `Reduction` enum and a common loss base; align reduction overloads.
11. Make `JsonOptions.Default`/`CsvOptions.Default` immutable or remove them.
12. Replace `CsvOptions.TrimOptions` bool and `ParquetWriteOptions.Compression`
    string with proper enums.
13. Move sample POCOs (`TwoColumnData`, etc.) out of the public surface.
14. Type `ModelIntegration.TrainAndEvaluate` metrics result; reconcile the `Array`
    vs `Tensor<T>` conversions.
15. Reconcile the duplicated JSON/CSV method suffixes and the overlapping
    Parquet/Arrow/ML.NET extension APIs.
16. Decide the query-paradigm story (one primary API; obsolete or remove the other
    two).
17. ~~Decide the `NivaraResourceManager` overhead: opt-in telemetry or removal.~~ —
    Decided 2026-08-09 (issue #174): opt-in telemetry via
    `NivaraResourceManager.Enable()` / `IsEnabled`, default OFF, lazy timer.
18. Reconsider whether the DataFrame engine, query engine, and AutoDiff should be
    one package or separate packages.

---

## Notes

- All file/line references are as of 2026-08-09 and may drift.
- `docs/REVIEW.md` is a snapshot; update it in place rather than duplicating when
  findings change.
- The AutoDiff subsystem and the query/execution layer have their own focused
  documents (`docs/AUTODIFF.md`, `docs/LINQ.md`).
