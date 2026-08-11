# Changelog

All notable changes to Nivara are documented here. Released versions are published to NuGet via the tag-triggered CD workflow (`v*` tags on `main`).

## [Unreleased]

### Added

- **Window-function expressions in the expression DSL (#159)** — rolling / cumulative / shift / lead / rank windows are now first-class `ColumnExpression`s (`WindowExpression` + `ColumnExpressions` factories `RollingSum`/`RollingMean`/`RollingMin`/`RollingMax`, `CumulativeSum`/`Max`/`Min`/`Product`/`Count`, `Shift`/`Lead`, `RowNumber`/`Rank`/`DenseRank`/`PercentRank`). A window expression can be embedded in `Select`/`Filter`/`SortBy` and composed with elementwise math (e.g. `Select(RollingSum(Col("Salary"), 2) * 2)`); the fused evaluator rewrites window nodes bottom-up via the existing kernels and injects synthetic columns, so nested windows compose and a standalone window stays a single materialization. Window ops in the lazy pipeline accept computed sources and keys: `RollingSum(Col("A") * 2, "r", 2)`, `CumulativeSum(expr, ...)`, `Shift(expr, ...)`, `Lead(expr, ...)`, and `Rank(resultColumn, orderBy: [SortExpressionKey(Col("B") * 1)], partitionBy: [Col("Dept")])` — `RollingOperation`/`CumulativeOperation`/`ShiftOperation` gain an optional `SourceExpression` (`Source` is now `string?`) and `RankOperation` an expression-key constructor, with schema/result types (`WindowFunctionHelpers.GetResultType`) shared between the AST and the ops. The plan layer routes `OperationType.Rolling`/`.Cumulative`/`.Shift`/`.Rank` to a new `VisitWindow` hook in both `QueryPlanVisitorBase` and `QueryPlanTransformerBase<T>` (previously "unknown"), and `QueryPlan.GetOperationDetails` describes window/rank operations in `GenerateDiagnosticInfo`. Note: the pre-existing ambiguous `QueryFrame.RowNumber(string)` overload was removed so `RowNumber(expr, resultColumn, orderBy, partitionBy)` is unambiguous.

- **Typed LINQ object model `frame.Query<T>()` (#130)** — an ergonomic typed layer over the expression engine. `NivaraFrame.Query<T>()` (requires `T : class, new()`) returns an immutable `NivaraQuery<T>` supporting `Where`/`Select`/`OrderBy`/`OrderByDescending`/`ThenBy`/`ThenByDescending`/`Skip`/`Take`/`Slice`/`GroupBy`/`Collect`/`ToObjects`/`ToList`/`ToRows`. Predicates and projections are `Expression<Func<...>>` lambdas translated at build time to `ColumnExpression` by `TypedExpressionTranslator` (property access, literals, arithmetic, comparisons, `&&`/`||`/`!`); method calls, closures, nested access, and ternary fail fast with `UnsupportedQueryExpressionException`. `GroupBy` accepts aggregate `Select` (`g.Key`, `g.Average/Sum/Count/Min/Max`) via `Grouping<TKey,T>` or a bare `Collect` of distinct keys. `Collect()`/`ToList()` return a `NivaraFrame`; `ToObjects()`/`ToRows()` materialize `IReadOnlyList<TResult>` through a compiled, cached per-type row factory (`TypedRowFactory`). Also fixes the expression engine's `And`/`Or` evaluation to produce boolean columns with SQL-like null masking instead of object-typed columns. Known limitation: `p.City == null` comparisons fail at `Collect` time (literal coercion only supports non-null constants); null-check semantics (`IsNull`/`IsNotNull`) are a follow-up.
- **Window functions: rolling / cumulative / shift / lead (#135)** — delivered at three layers with a single per-aggregate method shape. Column primitives (`src/Nivara/Tensors/WindowFunctions.cs`) add `RollingSum/Mean/Min/Max`, `CumulativeSum/Max/Min/Product/Count`, `Shift`, and `Lead` extensions on `NivaraColumn<T>` with explicit null-mask semantics: rolling output is null until the window holds `minPeriods` valid values (default full window); cumulative ops skip nulls with carry-forward; `Shift`/`Lead` move in nulls (or `fillValue`) at boundaries; an optional `nullHandler` replaces each null so it participates and every position satisfies the window. Eager `NivaraFrame` extensions (`src/Nivara/WindowFrameExtensions.cs`) and lazy `QueryFrame` members expose the identical shape, appending a result column while preserving all inputs. In the query pipeline the ops run as `OperationType.Rolling` / `.Cumulative` / `.Shift` (`src/Nivara/Operations/WindowOperations.cs`, `WindowNode`), and are marked non-parallelizable and non-streamable.
- **Rank-family window functions: row_number / rank / dense_rank / percent_rank (#156)** — SQL `OVER (PARTITION BY ... ORDER BY ...)` semantics at the same three layers. Column primitives (`src/Nivara/Tensors/RankFunctions.cs`) drive a shared `RankKernel` that partitions via grouping, orders with `SortKey` direction/null ordering, and emits `RowNumber`/`Rank`/`DenseRank` as `long` and `PercentRank` as `double`. Eager `NivaraFrame` extensions (`src/Nivara/WindowFrameExtensions.cs`) and lazy `QueryFrame` members (`RowNumber`/`Rank`/`DenseRank`/`PercentRank`) expose the same shape; `Rank`/`DenseRank`/`PercentRank` require at least one order key while `RowNumber` allows none. A null order key yields null output for that row and it is excluded from numbering and the percent-rank denominator. In the pipeline the ops run as `OperationType.Rank` (`src/Nivara/Operations/RankOperation.cs`) and are non-parallelizable/non-streamable.
- **Row-wise frame scoring (`NivaraFrame.RowDot` / `RowCosineSimilarity`, #138/#141/#142)** — a scoped tensor interop convenience: each row of a frame is scored against a `NivaraSeries<T>` query vector. `TensorsHelper` gains internal row-slice `TensorPrimitives` kernels (`RowDot`, `RowCosineSimilarity`, `RowNorms`, `ValidateRowKernelArgs`, `AnyTrue`) over a row-major buffer + null mask; the public frame methods materialize row-major through a pooled blocked transpose and return a `NivaraSeries<T>`. SQL-like null semantics: a null in a row masks only that row's score, a null in the query masks all scores, and the result always carries a null mask. `Nivara.PerformanceTests` gains four row-scoring scenarios (per-row status quo, frame API, raw kernels) as the regression gate; the frame API runs ~2.5× faster than the per-row status quo on the 10k × 128 benchmark. No public `RowNorms`/`ColumnNorms`/`Dot`/`CosineSimilarity` were re-added — the removed tensor-axis APIs stay removed (see 1.2.0).
- **`NivaraFrameExtensions.Standardize` (z-score alias, #143)** — data-prep promoted from `Nivara.MLNet` into core frame extensions (`src/Nivara/NivaraFrameExtensions.cs`). `Normalize`/`Standardize` now use `TensorPrimitives` (`Average`/`StdDev`/`Subtract`/`Divide`) for SIMD statistics and transform, compute mean/stddev over non-null values only, and preserve the null mask in the result (`CreateFromSpans`). Auto-select (`Normalize()`/`Standardize()` with no arguments) now normalizes all float/double columns instead of returning an unchanged frame (a latent bug in the old `??=` fallback). `IsNumericColumn` narrowed to float/double; explicitly naming an unsupported column throws `NotSupportedException`.
- **`Normalize`/`Standardize` full `INumber<T>` surface (#144)** — supersedes the float/double-only dispatch from #143. Support is now interface-based: a schema type is normalized when it implements `INumber<>` and is not in the explicit blocklist (`char`, `BigInteger`, `Int128`, `UInt128`). `int`, `long`, `short`, `byte`, `uint`, `ushort`, `sbyte`, `nint`, `nuint`, and `decimal` columns are now z-scored too, with the result promoted to `NivaraColumn<double>` (`TensorPrimitives.ConvertChecked<T,double>` → `Average<double>`/`StdDev<double>` → `Subtract`/`Divide`); `float`/`double`/`Half` keep the in-place SIMD path (`TensorsHelper.TryNormalizeInPlace`). `NormalizeColumn` dispatches through a per-type cached compiled delegate (`ConcurrentDictionary<Type, Func<...>>` + `MakeGenericMethod` + `Expression.Lambda`), so the interface predicate runs once per column type instead of per call. Auto-select now normalizes every supported numeric column; explicitly naming an unsupported column still throws `NotSupportedException`. Null-skip statistics and zero-variance-unchanged semantics carry over unchanged.
- **Mixed-type numerics use the typed promoted path in expression evaluation** — the fused evaluator no longer falls back to a boxed `Convert.ToDouble` path for mixed numeric operands (`double + int`, `decimal + int`, `byte + int`, `Col("A") + 1`, `Col("A") > 5`). Operand pairs are widened to the C# binary-numeric-promotion common type (`NumericPromoter.GetPromotedType`) and the operation runs through the compiled typed kernel with null-OR propagation, producing a typed `NivaraColumn<TResult>` result instead of `NivaraColumn<object?>`. C#-rejected pairs (`ulong` + signed, `decimal` + float/double) resolve to `double`, matching the previous boxed behavior; non-numeric and non-promotable pairs (Guid, string, etc.) are rejected with `NotSupportedException` (the legacy boxed fallback was removed). Integer division remains integral for integral results, matching the same-type typed path.
- **`NivaraRow` typed row view (#154)** — a public readonly struct passed to `NivaraFrame.Where(Func<NivaraRow, bool>)` predicates. Allocation-free over the frame's columns: `GetValue<T>` / `TryGetValue<T>` / `IsNull` / indexer / `RowIndex`, with case-insensitive name lookup, `ColumnNotFoundException` / `ColumnTypeMismatchException` on bad access, and a clear `InvalidOperationException` from the `default` state.
- **Modulo (`%`) arithmetic (#152)** — added to the expression DSL (`ColumnExpression` binary + scalar `%` operators, `BinaryOperator.Modulo`) and the typed LINQ translator (`p.Age % 2`). Runs through the fused compiled evaluator (`Expression.Modulo`) and the generic node-tree kernel with C# numeric promotion and null-OR mask propagation; `byte + byte` produces a `NivaraColumn<int>` like the rest of the DSL. `docs/LINQ.md` updated — `%` is now supported, not fail-fast.
- **Dim-aware `Softmax`/`LogSoftmax` (#179)** — the `dim` parameter (default `-1` = true last dim) is now honored across arbitrary axes via strided kernels (`GradKernels.SoftmaxDim`/`LogSoftmaxDim`/`SoftmaxDimGradient`/`LogSoftmaxDimGradient`) dispatched from `ReverseGradOperations.Softmax`/`LogSoftmax` and the `Softmax<T>`/`LogSoftmax<T>` modules. Negative dims are normalized against the input rank; out-of-range dims throw `ArgumentOutOfRangeException`; layout mismatches throw `ArgumentException`. Rank-2 last-dim behavior is unchanged, so existing callers (CrossEntropyLoss, samples) are unaffected.
- **`ReverseGradTensor<T>.ToHalf()` / `TypeConverter.ToHalf<T>` (#179)** — completes the conversion surface so `Half` is fully served alongside `float`/`double`. `Half` SIMD fast paths added to `RMSNormKernel` (per-row forward/backward), `Adam`, and `AdamW` via `TensorPrimitives` chains over `MemoryMarshal.Cast<T, Half>` views. Stale `INumber<T>` doc comments corrected to `IFloatingPointIeee754<T>`.
- **`GradientUtils.CanBackward(tensor, gradient)` overload (#179)** — checks `RequiresGrad`, matching length, and matching shape for `Backward(gradient)` calls (the seed-less scalar helper is unchanged). `DescribeTensor` now reports `Can Backward (no seed)`.
- **`NivaraColumn<T>` arithmetic generic-math collapse (#157)** — the six `NivaraColumn<T>` arithmetic kernel helpers (scalar `Multiply`/`Divide`, column `Multiply`/`Add`/`Subtract`/`Divide`) now dispatch `decimal`, `Half`, `nint`, `nuint`, `Int128`, and `UInt128` through the `INumber<T>`-constrained `NumericTensorKernels<T>` typed switch, matching `NivaraSeries`. These types previously threw (`InvalidOperationException` for `Half`/`nint`/`nuint`/`Int128`/`UInt128` via `validateTypeSupportsOperation`, `NotSupportedException` for `decimal` at kernel dispatch). On .NET 10 `TensorPrimitives` runs the six types via SIMD (`Half` widening, `nint`/`nuint`) or the operator-based software fallback (`decimal`/`Int128`/`UInt128`). `IsNumericType()` recognizes the five previously-rejected types so validation no longer blocks them; non-numeric types (`string`/`Guid`/`DateTime`) still throw the clear validation error. `KernelSelector` still reports `KernelType.Scalar` for the six, so diagnostics stay accurate.

### Fixed

- **Dynamic column creation covers the extended CLR domain (#158)** - the five
  dynamic column-creation dispatch sites used fixed type switches that fell through to a
  `NivaraColumn<object>` for less common types: `AggregationFunction.CreateColumnFromValues`
  and `GroupByOperation.CreateColumnFromValues` missed `Half`, `nint`/`nuint`,
  `Int128`/`UInt128`, `sbyte`/`ushort`/`uint`/`char`, `DateOnly`/`TimeOnly`,
  `DateTimeOffset`, `Guid`, and `TimeSpan`; `JoinOperation` coalesce/gather and
  `FusedExpressionEvaluator.CreateConstantColumn` had the same gap. A new `ColumnFactory`
  (`src/Nivara/Helpers/ColumnFactory.cs`) centralizes dispatch behind a cached
  `MakeGenericMethod` over null-safe kernels (`CreateFromNullable` for value types,
  `CreateForReferenceType` for reference types, `Nullable<T>` unwrapping) and is used by all
  four sites; join coalesce/gather dispatch directly onto the existing generic kernels, and
  the object-column fallbacks were removed. The existing `Cast<T>()`-based creation also
  threw on null values - the new kernel is null-mask safe. Window operations
  (`WindowFrameExtensions` rolling/cumulative/count/shift) now accept the full `INumber<T>`
  numeric domain (`byte`..`Half`, `nint`/`nuint`, `Int128`/`UInt128`, `char`) instead of
  throwing `NotSupportedException`, and `adaptNullHandler`/`convertFillValue` no longer use
  `Convert.ChangeType` (which throws for `Half`/`nint`/`nuint`/`Int128`/`UInt128`); typed
  fill/null values use a direct cast and string values use a cached `TryParse`. Out of scope:
  Parquet/Arrow/ML interop and CSV/JSON value conversion keep their format-specific type
  systems.

- **`NivaraResourceManager` tracking is now opt-in (#174)** — column/frame/QueryFrame
  construction no longer registers a `WeakReference` + boxed `ResourceInfo` in a global
  `ConcurrentDictionary`, and the process-lifetime 30-second cleanup `Timer` is gone from
  default hosts. Tracking is disabled by default (performance-first); hosts that want
  resource diagnostics opt in via `NivaraResourceManager.Enable()` (internal), which lazily
  creates the timer. `TrackResource` / `UntrackResource` / the timer callback are guarded
  no-ops when disabled, and the column ctor only computes `estimateMemoryUsage()` inside the
  enabled branch. Public surface (`MemoryRecommendations`, `ResourceStatistics`,
  `NivaraFrame.GetMemoryRecommendations`) is unchanged. Behavioral note: `QueryFrame`
  abandoned-lazy-source cleanup (its `CleanupAction`) is now opt-in too.

- **Same-type small-integral promotion in `NumericPromoter` (#152)** — `GetPromotedType` returned the operand type for equal operand pairs, so `byte + byte` produced `byte` instead of the C# spec §12.4.7.3 rule 1 result `int`. Same-type `sbyte`/`byte`/`short`/`ushort`/`char` pairs now promote to `int`; other same-type pairs (`decimal`, `uint`, `float`, `double`, `Half`, …) keep their type. This flows through `ExpressionTypeInferer` plan types and the compiled kernel target, fixing schema/result divergence for small-integral expressions.

- **Row-major hot loops no longer re-evaluate `NivaraFrame.RowCount`** (`columns.Values.FirstOrDefault()` LINQ allocates ~40 B per access): `CopyToRowMajor`, `ToNullableTensor`, and the new `materializeRowMajor` cache `RowCount`/`ColumnCount` in locals. On a 10k × 128 frame this was ~51 MB/op of pure garbage; the fix drops `Frame RowDot` allocation to ~452 KB/op (dominated by result-series construction, not the kernel).

- **`Sum`/`Mean` group-by aggregation for the full numeric domain (#169)** — `SumAggregation` and `MeanAggregation` only handled `int`/`byte`/`short`/`long`/`float`/`double`/`decimal`; `uint`/`ushort`/`sbyte`/`ulong`/`char`/`bool` passed validation then threw in `Apply`, and `Half`/`nint`/`nuint`/`Int128`/`UInt128` were rejected at validation. Both now accept the full 17-type `GetNumericTypes()` domain plus `bool`. Sum promotes per `NivaraSeries` rules (small integrals/`char`/`bool` → `long`, `ulong` → `ulong`, `nint` → `Int128`, `nuint` → `UInt128`, `Int128` → `Int128`, `UInt128` → `UInt128`, `float`/`Half` → `double`, `decimal` → `decimal`); widening now uses typed `TResult.CreateChecked` instead of `Convert.ChangeType` (which throws for `Half`, which has no `IConvertible`). Mean converts widened sums to `double` through a typed `ToDouble` switch (boxed `Int128`/`UInt128` are not `IConvertible`). Group-by sums produce typed result columns for `ulong`/`Int128`/`UInt128`.

- **`NivaraSeries<T>.Average()` for the extended numeric domain (#172)** — `divideByCount` only handled 12 of the 17 types the sum dispatch supports, so `NivaraSeries<Half/nint/nuint/Int128/UInt128>.Average()` computed a SIMD sum then threw `NotSupportedException`; the public `Average()` guard also rejected those types via `IsNumericType()` before the kernel path ran. `divideByCount` gains the 5 missing arms (same-type truncating division, matching the existing integral arms) and the guard accepts the full `GetNumericTypes()` domain (bool remains rejected by the sum dispatch).

- **Frame `Take`/`Skip`/`Slice` no longer slice columns via reflection (#173)** — `NivaraFrame.sliceColumn` called `GetMethod("Slice", ...)` + `MethodInfo.Invoke` on every column for every `Take`/`Skip`/`Slice` (an `object[]` boxing allocation plus a dictionary lookup per column per call). It now calls the `IColumn.Slice(int, int)` interface method directly; the unreachable `ColumnFilterHelper.CreateFilteredColumn` fallback was deleted. The query engine's `SliceOperation.SliceColumn` had the identical reflection pattern and was fixed the same way. A `Frame Slice [10k x 128]` scenario was added to `Nivara.PerformanceTests` so the removed allocations are measurable.
- **BatchNorm running-stats NRE fixed (#179)** — `BatchNorm1d<T>`/`BatchNorm2d<T>.RunningMean`/`RunningVar`/`NumBatchesTracked` threw `NullReferenceException` (via `!`) when the module was created with `trackRunningStats: false`. They now throw a clear `InvalidOperationException` explaining the constructor option, and a new `TrackRunningStats` property exposes the flag. `StateDict`/`LoadStateDict` are unaffected.
- **`BatchBackward` now honors its tensor list (#179)** — `NivaraAutoGradExtensions.BatchBackward(tensors, loss)` previously ignored `tensors` and only ran `loss.Backward()`. It now verifies every listed requires-grad tensor received a gradient after backward and throws `InvalidOperationException` listing the offending keys (constants, i.e. `RequiresGrad == false`, are exempt). `ToGradientFrame` xmldoc clarifies its intentional asymmetry vs `ToFrame` (gradient columns are skipped when null).

### Breaking changes

- **Legacy `ExpressionEvaluator` removed (#152)** — the per-operator boxed evaluator (`src/Nivara/Helpers/ExpressionEvaluator.cs`) and its tests are deleted. Every production query op (`FilterOperation`, `SelectOperation`, `SortByExpressionOperation`) and `ParallelExecutionStrategy` already routed through the fused evaluator (`FusedExpressionEvaluator` + `FusedKernel` + `ExpressionTypeInferer`), which is now the sole engine; unsupported operand combinations throw `NotSupportedException` instead of silently falling back to boxed evaluation. The performance benchmark's fused-vs-multi-pass comparison scenario was dropped.
- **`MLNetExtensions.Normalize` removed from `Nivara.MLNet`** — moved to core `NivaraFrameExtensions.Normalize`/`Standardize` (same signature, namespace `Nivara`). Update `using` if you relied on `Nivara.MLNet` for this helper.
- **`NivaraFrame.Where(Func<dynamic, bool>)` removed (#154)** — the last public `dynamic` surface in the core library. It built an `ExpandoObject` per row plus a reflection `Item` lookup per element (`CreateDynamicRow`), both deleted. The overload is now `Where(Func<NivaraRow, bool>)`; a predicate that boxed `dynamic` member access (e.g. `row => row.Age > 25`) must switch to typed accessors (`row => row.GetValue<int>("Age") > 25`). Predicate exceptions now propagate unwrapped instead of being rethrown as `InvalidOperationException`.
- **AutoDiff weight access unified on `Parameter<T>?` `Weight`/`Bias` (#177)** — every leaf module that owns learnable weights now exposes `Weight`/`Bias` of type `Parameter<T>?` (null = parameter omitted via `bias: false` / `affine: false`), matching PyTorch's `module.weight`/`module.bias`. `Linear<T>`, `Embedding<T>`, `SparseEmbedding<T>` lose their `WeightParam` accessor and their tensor-typed `Weight` member; `Conv1d`/`Conv2d`/`ConvTranspose2d` lose `WeightParam`/`BiasParam`. Consumers reach the tensor via `Weight!.Tensor` / `Bias!.Tensor`; `GetParameters()` / `StateDict()` keys are unchanged.

## [1.2.0] - 2026-08-05

### Breaking changes

- **`NivaraFrame.Dot<T>` / `CosineSimilarity<T>` / `ColumnNorms<T>` / `RowNorms<T>` removed** (AutoDiff refactor, Task 10): the four deprecated frame tensor-axis methods are deleted rather than relocated — they had no production callers. Use `TensorPrimitives.Dot` / `TensorPrimitives.CosineSimilarity` / `TensorPrimitives.Norm` on column spans (via `TryGetSpan`) or on row-major spans assembled through `CopyToRowMajor`. The `TensorsHelper.RowNorms` kernel (only consumer was `frame.RowNorms`) was removed with them.
- **`NivaraSeries<T>.Sum()` / `Min()` / `Max()` removed** (AutoDiff refactor, Task 9): NivaraSeries is now a labeled-column wrapper and keeps only `Average()`. Use the null-aware column reductions `NivaraColumn<T>` extensions `Sum` / `Min` / `Max` (`Nivara.Tensors`, `INumber<T>`-constrained) via `series.Values`; empty-column `Sum` throws, all-null `Sum` returns `T.Zero`, all-null `Min`/`Max` throw. Non-numeric (string/object) Min/Max/Sum are no longer supported.
- **`NivaraTensorExtensions` stripped to column reductions** (AutoDiff refactor, Task 8): the column-level activations/gradients/MatMul/Transpose/GELU family extension methods were deleted (they now live in `GradKernels` as span kernels) along with the obsolete Series extensions (`AddTensor`, `MultiplyTensor`, `SumTensor`, `DotProduct`, `Norm`, `TransformTensor`) and `MatrixMultiply`. Remaining members: `Sum`, `Mean`, `Min`, `Max`. `NivaraColumn<T>.Subtract(NivaraColumn<T>)` / `Divide(NivaraColumn<T>)` / `Divide(T)` were promoted from extensions to first-class members.
- **`TextClassifierModel<T>` / `TokenClassifierModel<T>` moved out of core** — the two pre-built NLP classification modules now live in `samples/Nivara.Samples` (`TextClassifierModel.cs`, `TokenClassifierModel.cs`). `TextTokenizer` remains in core (`src/Nivara/AutoDiff/Nn/TextTokenizer.cs`). Samples and docs were updated to reference the new home.
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
- **AutoDiff initializers and `TensorDataset<T>` enter the graph zero-copy** (Task 11): all 13 initializer implementations wrap freshly allocated weight arrays with `NivaraColumn<T>.CreateFromOwnedArray` instead of copying through `Create`; `TensorDataset<T>.GetBatch` now slices column spans via `TryGetSpan` and throws ADR-001 when a source column contains nulls (previously the null-mask path always threw at the tensor constructor, so behavior is unchanged).

### AutoDiff (GradKernels & inference fast paths)

- **`GradKernels<T>` span-kernel layer** (ADR-002, Tasks 1–6): all `ReverseGradOperations` and `ForwardGradOperations` now delegate to shared `GradKernels<T>` span kernels (`Span<T>`/`ReadOnlySpan<T>` + `TensorPrimitives`), replacing per-op duplicated column math and eliminating `NivaraColumn.Data` access. Results wrap once via `NivaraColumn<T>.CreateFromOwnedArray` (no copy). ADR-002 records the span boundary as the canonical AutoDiff architecture.
- **Inference-only fast paths**: `Gelu`, `GeluExact`, `LayerNorm` run single-path inference kernels that never construct graph nodes outside `GradientUtils.Grad()` (verified by `InferenceGraphTests`/`InferenceFastPathTests`); conv bias tracking is gated on `Grad()` scope so inference builds no graph. AutoDiff diagnostics are gated behind a static toggle for zero-cost inference.
- **Linear inference & transposed-weight cache** (#87): forward inference passes the raw weight to the kernel's `MatMulTransposedB` path (zero transposes); training reuses a version-stamped transposed-weight cache invalidated only on `Parameter<T>.Version` change.
- **New ops**: `AddBias` row-broadcast (Linear bias), `MatMulTransposedB` (transposed-B matmul), `GeluExact` (exact erf GELU for BERT-family activations, SIMD `TensorPrimitives`), `BatchedMultiHeadAttention` — fused `[B, L, D]` batch attention with per-batch additive `[B, qLen, kvLen]` masks, single `OpNode` VJP producing dQ/dK/dV (PyTorch-parity fixtures + perf scenarios).
- **BCL-tuned MatMul kernels**: `MultiplyCore` optimized against `TensorPrimitives.Dot` (BCL swap-target annotations in `TensorsHelper`); rank-2 backward transpose buffers now pooled via `ArrayPool<T>.Shared` instead of per-call allocations.
- **Enter path is zero-copy** (Task 11): all 13 initializers wrap freshly allocated weight arrays with `CreateFromOwnedArray`; `TensorDataset<T>.GetBatch` slices column spans via `TryGetSpan` and throws ADR-001 on null-containing columns.

### Training & Serialization

- **`Optimizer<T>.StateDict()` / `LoadStateDict()`** — optimizers now expose their moment/velocity buffers for incremental-training scenarios (matching the module `StateDict`/`LoadStateDict` contract).
- **Optimizer state persisted in checkpoints**: `Checkpoint<T>.OptimizerState` added and `ModelSerializer.SaveCheckpoint`/`LoadCheckpoint` now round-trip optimizer state alongside model parameters, so a checkpoint is a full training resume point.
- **Epoch-aware `DataLoader<T>.GetBatches(epoch, skipBatches)`** — yields a single epoch's batches with skip support, enabling incremental/online training loops (`NivaraChat --online-learning` uses it).

### Fixed

- **Owned-array contract documented on remaining factory surfaces** (#106): `Parameter(string, T[], bool)` and `GradientUtils.Constant(T[])` wrap caller arrays zero-copy; XML docs now state ownership transfers and that the source array must not be mutated afterward, matching the `FromArray`/`FromMatrix` contract.
- **Storage consolidation doc debt** (#108): 7 planning/review docs reconciled with the single `ColumnStorage<T>` design; public zero-copy claims aligned with the post-consolidation span semantics (Task 7).

### Added

- **Public zero-copy tensor view** (#107): `NivaraColumn<T>.AsTensorView()` and `NivaraSeries<T>.AsTensorView()` are now public (previously internal). They return a lazy `Tensor<T>` view sharing the column's/series' backing array with no copy; null-containing columns and reference element types throw `InvalidOperationException`. Callers must treat the view as read-only.
- **`NivaraEmbeddingGenerator<TInput>`** in `Nivara.Extensions` (AI): wraps any `IEmbeddingGenerator<TInput, Embedding<float>>` as a label column generator for `NivaraFrame.FromRows`; brings `Microsoft.Extensions.AI.Abstractions` into Extensions. Powers the `NivaraChat --embed` and `--rag`/`--rag-agent` modes.

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
