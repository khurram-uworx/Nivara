# Changelog

## [Unreleased]

### Breaking changes (migration)

- `ResourceStatistics` (removed) → no replacement. The type was public but only ever reachable
  through the internal `NivaraResourceManager`, which has been deleted (#1168). Delete any
  reference; no shipped code path produced an instance.
- `QueryFrame.Dispose()` / `QueryFrame.DisposeAsync()` (and the typed `NivaraQuery<T>` /
  `NivaraGroupedQuery<TKey, T>` equivalents) no longer swallow a source-disposal failure:
  a failed close/flush now propagates out of `Dispose`/`DisposeAsync`. `Dispose` stays idempotent
  (a second call is a no-op, including after a failure), so `using`/`await using` cleanup can
  surface an I/O error. `ParquetLazySource.Dispose()` likewise propagates a reader-close failure
  instead of swallowing it. (#1170)

### Removed

- Internal abandoned-resource tracking subsystem (`NivaraResourceManager`: `Enable`/`Disable`/
  `ForceCleanup`/`GetResourceStatistics`/`TrackResource`/`UntrackResource`, the 30 s cleanup timer,
  and the tracking dictionary). It was internal and unreachable from a consumer, and its `Untrack`
  was an O(n) scan under a global lock on every frame/column dispose. Explicit `Dispose`/`Release`
  is now the only release path for frames, columns and query sources — the same one-source-one-release
  contract `FileStream` has.
- `NivaraColumn`/`NivaraFrame` no longer register themselves for abandoned-resource cleanup, and
  `NivaraFrame`'s constructor no longer computes the per-column memory estimate that was discarded
  when tracking was disabled.

### Changed

- `MemoryRecommendations` / `MemoryWarningLevel` moved to `src/Nivara/Helpers/MemoryRecommendations.cs`
  and `GetMemoryRecommendations` to the internal `MemoryAdvisor` class. Public API (`NivaraFrame.
  GetMemoryRecommendations`) and behaviour are unchanged.

### Fixed

- `NivaraBlas.UseBlas(path)` now honours `path` on the process's first OpenBLAS load and returns
  `false` with the path in `NivaraBlas.LastDiagnostic` when the file is missing or cannot be used;
  the previously dead `OpenBlasInterop.ExplicitPath` branch is live.
- The real reason a registration failed (ILP64 build, bad image, missing file) is now surfaced via
  `NivaraBlas.LastDiagnostic` instead of being swallowed and reported as "libopenblas.dll not
  found".

### Changed

- OpenBLAS thread count is applied once, at registration, instead of inside every `MatMul`/`Gemv`.
- `NivaraBlas.UseManaged()` and the `IDisposable` returned by `NivaraBlas.EnterBlas` restore the
  previous registration and the previous OpenBLAS pool size.
- `BlasOptions.OpenBlasThreads` is documented as process-wide; `BlasOptions.DllPath` is now wired
  through to the loader.

All notable changes to Nivara are documented here. Released versions are published to NuGet via the tag-triggered CD workflow (`v*` tags on `main`).

## [1.5.0] - 2026-11-07

### Breaking changes (migration)

Nivara is pre-2.0 and 1.5.0 removes/renames several public types without an `[Obsolete]`
deprecation window. Migrate as follows:

- `DataFrameSchemaValidationException` (removed) → catch `QuerySchemaValidationException`
  (query engine) or `DataSchemaValidationException` (I/O) instead.
  `catch (DataFrameSchemaValidationException)` → `catch (QuerySchemaValidationException)` /
  `catch (DataSchemaValidationException)`.
- `SchemaValidationException` (ambiguous, removed) → `QuerySchemaValidationException` (query
  engine, ~60 call sites) or `DataSchemaValidationException` (Parquet reader/writer).
- `NivaraColumn` (non-generic factory, renamed) → `NivaraColumnFactory`.
  `NivaraColumn.CreateFromNullable<T>(values)` → `NivaraColumnFactory.CreateFromNullable<T>(values)`.
- Legacy `ExpressionEvaluator` (removed) → `FusedExpressionEvaluator`.
  `new ExpressionEvaluator()` → `new FusedExpressionEvaluator()`.
- Dead AutoDiff exceptions removed (`GradientComputationException`, `CircularDependencyException`,
  `InvalidBackwardCallException`, `TypeValidationException`) → `AutoGradException` or the standard
  `InvalidOperationException` / `ArgumentException` thrown by the hot paths.
- Array-based tensor conversions removed → span-based helpers in `TensorInteropExtensions`
  (`FromTensor`, `FromTensorSpan`, `ToTensor`, `FlattenFromTensor`, `ReshapeToTensor`).
- ILGPU removed from the shipped stack → Silk (OpenCL) backend (`Nivara.Silk`). See `docs/SILK.md`.

### Added

- **Silk (OpenCL) GPU backend + BERT/AI layer are first-class** — moved from samples into
  `src/Nivara.Extensions/Silk` (`Nivara.Silk`) and `src/Nivara.Extensions/AI` (`Nivara.AI`): runners, sessions, kernels, BERT/ModernBERT/Laya models, tokenizers, loaders, and the GPU seam.
- **Embeddings convention and Microsoft.Extensions.AI glue** — `NivaraEmbeddingExtensions` now maps between `NivaraColumn<float[]>`, `E0..E{n-1}` frames and `Microsoft.Extensions.AI` embeddings; null policies are explicit and named.
- **Arrow zero-copy export for real** — opt-in `ArrowConversionOptions.UseZeroCopy` aliases flat primitive column buffers (null-bearing columns rebuild validity bitmap); unsupported types throw `UnsupportedTypeException`.
- **Reachable streaming memory budget** — `QueryFrame.AsStream(NivaraExecutionContext)` plus `NivaraQuery<T>` overloads; context cloning; fixed default 10,000-row chunk.
- **ModernBERT benchmark levers** — `--seq`, `--warmup`, `--iters` on both CPU and GPU benchmark paths, selectable pass counts.
- **Qwen on Silk and faster decode** — distillation runs on the Silk backend; batched prefill and a fused per-token decoder; transposed-GEMV and fused GQA decode attention fast paths.
- **Seedable AutoDiff initializers and module overrides** — every initializer gained `seed`/`Random` constructors; `Conv*`/`VAE` accept custom initializers; seedable Kaiming init drives reproducible samples.
- **Documentation integrity gates** — every `File.cs:NN` reference, every gated csharp fence, the ADR index, and the docs navigation are machine-checked; 141 previously-ungated fences now have an authoring contract.

### Changed

- **Nullable columns are always mask-based** — `NivaraColumn<T>.Create` extracts a null mask and reports the underlying `ElementType`; null-nullable type checks removed.
- **Silk cached decode fuses projection norms** — Q/K/V and gate/up in one RMSNorm launch; no normalized-row materialization.
- **Conv defaults standardized** to He-uniform `sqrt(6/fanIn)` and fan-in aligned to weight shape.
- **Type-name disambiguation** — `SchemaValidationException` split into `QuerySchemaValidationException`/`DataSchemaValidationException`; `NivaraColumn` factory renamed `NivaraColumnFactory`.
- **Rename: `DataFrameSchemaValidationException` removed** — never thrown; dead public surface.
- **BFloat16 and integer math** — BFloat16 admitted to AutoDiff types; mixed-type numerics use typed promotion.

### Fixed

- Flaky Streamix backpressure bridge test replaced ownership-gated overflow (see RELEASE15 review).
- 15 stale `CreateFromNullable` doc call sites and 84 dead doc references corrected.
- NivaraInference CLI hardened: rejects inert options before disk access; `--quick` alias for `--benchmark --iterations 1`.
- Streaming budget formulas no longer wrap around 2.1 TB; budget-derived sizing is reachable.
- Python vision benchmarks now pin `torch.set_num_threads`.
- BatchNorm, attention-mask, and fused kernel parity defects in Qwen path fixed (details in AUTODIFF addendum / QWEN doc).

### Removed

- **ILGPU** from the shipped stack as a first-class path; historical reference remains in tests/docs. Inference runs on Silk (OpenCL).
- Legacy `ExpressionEvaluator`, dead public AutoDiff exceptions, and array-based tensor conversions.

## [1.4.0] - 2026-08-21

### Summary

- Public streaming API (`QueryFrame.AsStream`, `ScanAsQueryFrame` factories, lazy factories), Over/WindowSpec window functions, fused expression engine with kernel IR, genuinely async `CollectAsync`, conditional expressions in LINQ, new aggregations (Quantile, Median, StdDev, Variance), public query/execution APIs, Parquet predicate pushdown, window overlap and streaming budget tracking.

### Notes

- AutoDiff promoted to inference-default with `GradientUtils.Grad()` opt-in; nullable domain enforced (ADR-001); spanified ops (ADR-002).
- .NET 10 → .NET 11 upgrade across core and extensions (System.Numerics.Tensors 11.x).

## [1.3.0] - 2026-08-14

### Summary

- Canonical `Json`/`Csv`/`Parquet` file-source entry points; immutable I/O options with `With()` builders; metadata-aware schema equality; `NivaraSeries<T>` virtual positional index and restored aggregates; Parquet/Arrow extended-domain round-trip; ML.NET faithful conversion; optimizer and loss APIs.

## [1.2.0] - 2026-08-05

### Summary

- Storage consolidated to single `ColumnStorage<T>`; tensor view zero-copy; AutoDiff boundary hardening and graph-building cleanup.
- Query engine: typed LINQ `frame.Query<T>()`, window expressions as first-class column expressions, rank family windows.
- Null-handling and numeric promotion across mixed types promoted to typed path.

### Breaking changes (high level)

- `NivaraFrame.Dot`/`CosineSimilarity`/`ColumnNorms`/`RowNorms` removed; `NivaraSeries.Sum/Min/Max` removed (use column reductions).
- Model/checkpoint serialization bumped to `nivara-ss-v2`/`nivara-ckpt-v2`.
- `UseZeroCopy` (Arrow) removed (returned in 1.5).

## [1.1.0] - 2026-07-31

### Summary

- Inference-default AutoDiff (reverse graph via `Grad()`), relaxed type constraint to `IFloatingPointIeee754<T>`.
- NN module system: Conv1d/2d/Transpose, BatchNorm, LayerNorm, DepthwiseSeparableConv, TransformerBlock, MHA, VAE/ConvVAE, MaxPool, AdaptiveAvgPool.
- Optimizers SGD/Adam/AdamW, training loop, data parallelism, serialization.
- Samples: NivaraInference (MobileNet/ResNet), NivaraFineTuning, NivaraTimeSeries, NivaraTorch (55 PyTorch validated tests).

## [1.0.0]

### Summary

- Initial stable release of the columnar DataFrame core: typed columns/frames, LINQ query engine (lazy/eager/streaming/parallel), null masks, join/group/aggregation, CSV/JSON, Parquet/Arrow/ML.NET interop, diagnostics and pooling.
- Initial reverse-mode AutoDiff and VAE/ConvVAE samples.
