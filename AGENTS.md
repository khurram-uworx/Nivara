# Guidance for AI-assisted coding

**GitHub repo:** `https://github.com/khurram-uworx/Nivara` — use `--repo khurram-uworx/Nivara` for `gh` commands.

## Facts & Research

- Nivara AutoDiff product direction: inference is the default/common path; reverse-mode training is opt-in via `using (GradientUtils.Grad())`. Do not implement NoGrad as the primary API. Built-in training loops should enter Grad() internally, while manual training examples/docs should wrap forward/loss/backward/optimizer code in Grad().
- Nivara AutoDiff ADR-001 (non-nullable domain) is fully implemented. Type constraint relaxed from `INumber<T>` to `IFloatingPointIeee754<T>`, which passes Half/F16 and BFloat16 through runtime validation alongside float/double. All AutoDiff ops are span-ified (no NivaraColumn.Data access; Span<T> + TensorPrimitives).
- Key BCL .NET 10 tensor patterns: TensorPrimitives now generic (200+ overloads for any INumber/IRootFunctions T), ReadOnlyTensorSpan<T> with TryGetSpan, implicit conversion from T[] to ReadOnlyTensorSpan<T>, Tensor<T> stable in .NET 10. Spans are the currency.
- Nivara v1.4.0 shipped: public streaming API (`QueryFrame.AsStream`, `ScanAsQueryFrame` factories), `Over()`/`WindowSpec` window functions, fused expression engine (`FusedExpressionEvaluator` with SIMD backend + flat IR fallback), genuinely async `CollectAsync`, conditional expressions (`?:` in LINQ DSL), new aggregations (Quantile, Median, StdDev, Variance), public QueryPlan/ExecutionEngine/IExecutionStrategy/QueryDiagnostics/ExecutionProgress.

## Shell environment (Windows with GNU coreutils)

This environment has GNU coreutils at `C:\Program Files\coreutils\bin\` on PATH. Most Linux commands work directly (`grep`, `find`, `touch`, `sort`, `head`, `tail`, `wc`, `cat`, `ls`, `rm`, `mv`, `cp`). PowerShell aliases map `rm`/`mv`/`cp`/`cat`/`ls` to their PowerShell cmdlet equivalents, which behave similarly for basic file operations. Use normal command syntax — avoid verbose PowerShell idioms like `Remove-Item -LiteralPath`.

**GitHub CLI body gotcha:** Always write issue/PR bodies to a temp file and use `--body-file` instead of inline `--body`. PowerShell interprets backslash sequences in double-quoted strings (e.g., `\t` → tab, `\n` → newline), and special characters (backticks, quotes, backslashes) silently break or truncate `gh` inline bodies. This applies to `gh issue create`, `gh issue comment`, `gh pr create`, and `gh pr edit`. Pattern:
```
write the body to a temp file → gh issue comment N --repo ... --body-file /path/to/file.md
```
Do NOT rely on inline `--body "..."` for anything beyond trivial one-liners.

**Solution file:** This repo uses `.slnx` (XML-based solution format), not `.sln`. Build with `dotnet build Nivara.slnx`.

## Where to look (implementation map)

- **Storage:** `src/Nivara/Storage/ColumnStorage.cs` (single unified storage: sole-owner `T[]` + optional `bool[]` null mask; zero-copy Slice; lazy `AsTensor()`), `ColumnStorageFactory.cs` (`IsVectorizable<T>()`, `Create<T>()`)
- **Column kernels:** `src/Nivara/NivaraColumn.cs` (arithmetic, comparison, mask propagation, `TensorPrimitives` for float/double)
- **Tensor helpers:** `src/Nivara/Tensors/TensorInteropExtensions.cs`, `TensorsHelper.cs` (MatMul, SoftMax, Sigmoid, Tanh, Transpose)
- **Kernel selection:** `src/Nivara/KernelSelector.cs` (`DetermineKernelType()`)
- **Execution engine:** `src/Nivara/Execution/` — `ExecutionEngine.cs`, `ExecutionStrategyBase.cs`, `LazyExecutionStrategy.cs`, `EagerExecutionStrategy.cs`, `StreamingExecutionStrategy.cs`, `ParallelExecutionStrategy.cs`, `ParallelExecutionHelper.cs`
- **Interfaces:** `src/Nivara/Interfaces.cs`, `src/Nivara/Query/IQueryInterfaces.cs`, `src/Nivara/Query/OperationType.cs`
- **Window functions:** `src/Nivara/Operations/WindowSpec.cs`, `src/Nivara/Tensors/PartitionedWindowEngine.cs`
- **Fused expressions:** `src/Nivara/Expressions/FusedExpressionEvaluator.cs`
- **Frame interop:** `src/Nivara/NivaraFrame.cs` (`ToTensors`, `TryGetRowMajorSpan`, `CopyToRowMajor`, `AsQueryFrame()`), `src/Nivara/Query/QueryFrame.cs` (`AsStream`, `CollectAsync`, lazy factories)
- **AutoDiff:** See [docs/AUTODIFF.md](AUTODIFF.md) for all operations, modules, optimizers, forward-mode AD, and DataFrame integration. Core: `src/Nivara/AutoDiff/` (ReverseGradTensor, GradNode, ReverseGradOperations, ForwardGradOperations, Optimizer/, Nn/, Training/, Serialization/)
- **Factory:** `src/Nivara/Storage/ColumnStorageFactory.cs`, `src/Nivara/Tensors/NivaraTensorExtensions.cs`

## Key rules for AI Agents

1. **Storage selection:** Call `ColumnStorageFactory.IsVectorizable<T>()` to know whether `T` supports vectorized kernels. All columns use the single `ColumnStorage<T>`; kernel dispatch is decided by `KernelSelector`, never by the storage class.
2. **Zero-copy preference:** Use `NivaraColumn<T>.TryGetSpan` (zero-copy read-only view when no nulls) or `AsTensorView()` (lazy zero-copy `Tensor<T>` for unmanaged types; check `HasNulls` first). When creating a `TensorSpan`/`Tensor<T>` from an existing span, ensure there are no nulls; otherwise throw or fallback to copy.
3. **Kernel use:** Use `TensorPrimitives` for `float` and `double` arithmetic/comparisons. Prefer generic `TensorPrimitives` overloads when available. Provide robust scalar fallbacks using `Span<T>` loops or `INumber<T>` where available.
4. **Null-mask semantics:** resultNullMask = leftNullMask OR rightNullMask (or leftNullMask for scalar op). Where result is boolean (comparisons), null positions should be represented in the result null mask and the boolean output at those positions should be false (SQL-like semantics). Always keep the mask.
5. **Minimize allocations:** Avoid calling `Tensor.FlattenTo` repeatedly in hot loops. For temporary buffers larger than 1024 elements, rent arrays from `ArrayPool<T>.Shared` (core) or `BufferPool` (Extensions) and return promptly. `AsTensorView()` caches the lazy `Tensor<T>` view inside `ColumnStorage<T>`; reuse it.
6. **Kernel selection heuristics:** Implement or reuse `DetermineKernelType()` that considers `IsVectorizable` (type level), `Vector.IsHardwareAccelerated`, and `Length >= vectorSize * 4` (heuristic threshold). If kernel selection resolves to scalar, avoid preparing tensor copies.
7. **Safe type dispatch:** When converting spans/arrays to typed `ColumnStorage<T>`, convert to arrays first and then use `ColumnStorageFactory.Create<T>(...)` or `CreateFromOwnedArray<T>`. Avoid `MemoryMarshal.Cast` unless `T` is unmanaged. Keep explicit type-switch branches for each supported primitive.
8. **Consolidating duplicate logic:** When the same helper logic is duplicated across multiple files, promote one authoritative implementation as an extension method on the most natural receiver type, then remove all copies and update call sites. Place the extension in the same assembly to avoid cross-project visibility concerns.
9. **Testing & diagnostics:** Add unit tests covering null-mask propagation across arithmetic and comparisons. Validate tensor conversions keep correct shape (`tensor.Lengths` is `nint[]`) and use casts `(int)tensor.Lengths[0]` in tests. Record `Nivara.Diagnostics.OperationDiagnostics` for kernel selection and include in performance tests. Use `ColumnDiagnostics`, `DiagnosticsTracker`, and `QueryDiagnostics` when changing kernel selection, query execution, or optimization behavior.

## Common gotchas (lint-like checks)

- `ReadOnlyMemory<T>?` has `HasValue == true` for empty memory; always check `.Length > 0` to decide if mask exists.
- Slicing null masks: always check `.Length > 0` before slicing.
- `Tensor.Create(..., [length])` should use `new nint[] { length }` for dimensions.
- `Tensor.Lengths` is `nint[]`, not `int[]`.
- Avoid reflection/emits that attempt to pass `Span<T>` to `MethodInfo.Invoke` — convert to arrays first.
- Nullable generics & static constraints (CS0080): avoid `where T : struct` on static methods in generic classes. Validate at runtime and throw clear exceptions.
- MemoryMarshal.Cast requires unmanaged constraints; use explicit type switch with `(T)(object)` casting for safe conversion.
- Tensor interop: zero-copy is limited — `NivaraColumn` doesn't expose underlying data as `Span`; interop requires element-by-element copying.
- Series indexer ambiguity: boxed `int` routes to label indexer. Use explicit casts or `GetByLabel()` to disambiguate.
- Method overload resolution: disambiguate 1D vs 2D tensor methods with explicit parameters.
- Expression Equals/GetHashCode: always override when adding custom equality operators to expression types.
- `Memory<T>` disposal: implement `IDisposable` consistently for frames, columns, and data sources.
- `NivaraColumn.TryGetSpan` returns `ReadOnlySpan<T>` (immutable), diverging from BCL's `Tensor<T>.TryGetSpan` which returns `Span<T>` (mutable). This is deliberate — Nivara columns are immutable. Use `CopyTo(Span<T>, T)` for the explicit-fill path.
- `DataFrameOperation` no longer has strategy-switch dispatch or `Strategy` property — it was simplified to a single `Execute()` abstract method. Strategy dispatch is the `ExecutionEngine`'s responsibility via `IExecutionStrategy`.

## Engineering Guidelines

See [docs/GUIDELINES.md](GUIDELINES.md) for transferable engineering principles (null semantics, lazy execution, zero-copy, testing strategy, etc.).

## Code Style

- `.editorconfig` at repo root is authoritative; follow it over any convention below.
- **Private fields:** `camelCase` without `_` prefix (`logger`, not `_logger`).
- **Member ordering:** inner classes → constructors → properties → methods; static before instance; private → protected → internal → public.
- **Primary constructors:** preferred for service/DI classes over classic constructor with field assignment.
- **Sealed by default:** use `sealed class` for non-abstract classes unless inheritance is explicitly designed.
- **Collection expressions:** `[]` for empty/static collections; `new List<T>()` or `new Dictionary<K,V>()` for mutable ones.
- **Nullable reference types:** enabled; do not introduce avoidable warnings.
- **Omit braces** from single-line `if`/`else` bodies when the body fits one line and is on the same line as the condition.
- **No comments** in generated code unless explaining a non-obvious design decision.
- **No Hungarian notation** — no prefixes encoding scope or mutability.

## Testing Conventions

- **Framework:** NUnit 4.x — `[Test]`, `Assert.That(...)`, `Assert.ThrowsAsync`, no `[TestCase]`
- **Naming:** `Method_Scenario_ExpectedBehavior` PascalCase
- **Pattern:** Arrange-Act-Assert (AAA); no explicit comments needed
- **Organization:** one test class per source class, `*Tests.cs` suffix; split by behavior when a class is large
- **Mocking:** prefer real implementations where feasible; use `Substitute.For<T>()` only when external dependencies require it
- Ask before running `dotnet test` or any long-running test/verification command; wait for explicit confirmation before starting it.
- The `tests/` directory holds three projects: `Nivara.Tests` (NUnit 4.x unit tests), `Nivara.PerformanceTests` (standalone stopwatch harness, `dotnet run -c Release`), and `Nivara.SimdProbe` (self-contained SIMD probe, `dotnet run -c Release -- correctness|benchmark`). Each non-NUnit project has its own `README.md`.
- Avoid `[TestCase]` with null arrays; use regular `[Test]` with inline arrays.
- Reflection cannot pass `Span<T>` via `MethodInfo.Invoke` — convert to array first.
- Test for key phrases in error messages rather than exact message strings.
- Property-like tests: implement with parameterized NUnit test suites rather than full FsCheck (see [docs/GUIDELINES.md](GUIDELINES.md) for rationale).
- Native integer types (`nint`): use `nint` for test assertions when comparing tensor dimensions.
- Resource-management tests that depend on weak-reference cleanup may force multiple GC cycles; avoid GC forcing in normal code paths.
- Code examples: see [docs/AGENT-CODE-EXAMPLES.md](docs/AGENT-CODE-EXAMPLES.md).

## I/O & Interop Guidance

- Keep third-party dependencies in `Nivara.Extensions`; core stays dependency-free.
- Map CLR ↔ Arrow ↔ Parquet with explicit dictionaries and fallback suggestions.
- Handle nullable value types by extracting underlying types via `Nullable.GetUnderlyingType()`.
- **Arrow:** Build arrays using builders and individual `Append`/`AppendNull` calls. Convert `DateTime` to UTC and use `DateTimeOffset` for Timestamp arrays. Handle chunked arrays by iterating `chunkedArray.ArrayCount`. Create valid empty schemas/record batches for empty tables.
- **Parquet:** Validate schema first, then reconstruct columns — use `CreateFromNullable` for value types, build arrays preserving nulls for reference types. `Parquet.Net DataColumn` expects non-nullable arrays matching `DataField<T>` generic type; pass `default(T)` for nulls and set field as nullable. Preserve string nulls as null, not empty string.
- **CSV/JSON:** Lazy sources: `IsLazy = true`, infer schema from samples (e.g., 100 rows). Eager sources wrap lazy ones and materialize immediately. Conservative type detection: int → double → string; fallback to string in ambiguous cases. Lazy sources should validate structure/schema early, collect scan errors while traversing data, and throw during `Collect()` with source and operation context.
- **Dependencies (Extensions only):** CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.10

## Performance & Optimization Thresholds

- **Buffer pooling threshold**: rent arrays >1024 elements from `BufferPool` (in Extensions).
- **Default memory budget for streaming**: 256 MB (configurable).
- **Streaming chunk size**: derived from memory budget when unset (`clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)`); row-group aligned for Parquet. Full contract in `docs/STREAMING.md`.
- **Vectorization overhead threshold**: prefer only when `Length >= vectorSize * 4` (heuristic).
- **Fused expression engine**: primary path compiles `ColumnExpression` AST to cached delegates (SIMD auto-vectorized). `FusedKernel` fallback for non-compilable expressions. No boxed fallback — non-fusible expressions throw.
- **FlattenTo**: cache flattened tensor data if multiple accesses needed; use single `FlattenTo` for one-time access.
- **StreamingBufferManager**: use bounded buffer manager (in Extensions) for large datasets with memory budgets and GC triggers.
- **Vectorization checks**: verify `Vector.IsHardwareAccelerated` and type vectorizability before using SIMD kernels.
- **Unmanaged constraint**: `ColumnStorage<T>.AsTensor()` / `NivaraColumn<T>.AsTensorView()` require unmanaged `T` (`int`, `float`, `double`, `long`, `bool`, etc.).
- **Resource management**: implement object-disposed guards and dispose frames, columns, and data sources consistently.
- **Diagnostics**: preserve diagnostic context when wrapping kernel, query, optimization, and I/O failures.

## Known Issues & Follow-ups

- **Parquet round-trip**: nullable value type null preservation may degrade — investigate (high priority).
- **Zero-copy Arrow arrays**: removed from the public API (claims-integrity triage, see CHANGELOG); real zero-copy returns with `ARROW-ROADMAP` Phase D (issue #94).
- **Tensor interop**: investigate more efficient conversion patterns for large datasets (element-wise `Series`/`Frame` ↔ `Tensor<T>` interop still copies); `AsTensorView()` covers the flat, null-free column case.
- **NivaraSeries TopKDescending**: added in Phase 3 on `NivaraSeries<T>` (not `NivaraFrame`), returns labeled results with null-propagating scores; threshold-based optimization not yet implemented.
- **ConvTranspose2d**: no grouped convolution support; grouped transpose would require new kernel paths.
- **ConvTranspose2d**: direct scatter produces zero-padded interior positions (stride > 1); test verified numerically correct but may look unexpected.
- **BatchNorm2d**: uses generic per-element kernel (not the fused `BatchNormKernel<T>` span path); functionally correct but slightly slower than optimal.
- **PerRowLayerNorm**: delegates to `LayerNormKernel` with per-row slicing instead of a fused multi-row kernel; functionally correct but not optimal for large row counts.
- **GPU GEMM throughput**: the iGPU is still 1.9–2.2× slower than PyTorch CPU; the leading GPU item is #440 (tile-32/2×2 GEMM). See [docs/ACCELERATION.md](ACCELERATION.md).
- **Vision gap**: convolution kernels are naive nested loops with no SIMD and no parallelism; tracked as issue #457.
- **CPU-side performance**: the ~5.6× Nivara-CPU-vs-PyTorch deficit is not a GEMM problem at Laya's shapes; tracked as #458.

## Quick Reference

- **Vectorizable types (confirmed)**: `int`, `float`, `double`, `long`, `short`, `byte`, `uint`, `ulong`, `ushort`, `sbyte`, `bool` (requires unmanaged constraint)
- **Target framework**: .NET 10.0 with System.Numerics.Tensors 10.0.10
- **Common deps (Extensions only)**: CsvHelper 33.1.0, Apache.Arrow 23.0.0, Parquet.Net 6.0.3, Microsoft.ML 5.0.0, System.Numerics.Tensors 10.0.10
- **Useful helpers**: `ColumnDiagnostics`, `DiagnosticsTracker`, `ColumnStorageFactory.IsVectorizable<T>()`, `NivaraColumn<T>.CreateFromNullable(T?[])`, `Tensor.Create(array)` + `FlattenTo(buffer)`, `KernelSelector.DetermineKernelType()`, `SGD<T>`, `Adam<T>`, `AdamW<T>`, `Linear<T>`, `Sequential<T>`, `Module<T>.StateDict()`, `Module<T>.LoadStateDict()`, `TrainingLoop<T>`, `DataParallelTrainer<T>`, `ModelSerializer`, `Loss<T>`/`Reduction`, `Activation.Gelu`, `ReverseGradOperations.Gelu`
- **AutoDiff type constraint**: `IFloatingPointIeee754<T>` (float, double, Half) — ADR-001 non-nullable domain
- **Storage**: single `ColumnStorage<T>` for all types (sole-owner `T[]` + optional `bool[]` null mask; zero-copy Slice; lazy `AsTensorView()`); vectorization decided by `KernelSelector`
- **Null handling**: explicit boolean masks, no NaN-based semantics
- **Query execution**: lazy by default, multiple strategies (eager, streaming, parallel)

## Architectural Decisions (ADRs)

See `docs/adr/` for recorded decisions:

- **ADR-001** (`docs/adr/001-autodiff-nonnullable-domain.md`): AutoDiff is a non-nullable domain. Null boundary enforced at domain entry points (`NivaraColumn<T>` → `ReverseGradTensor<T>` conversion). All AutoDiff ops assume non-null data. Storage layer remains nullable.
- **ADR-002** (`docs/adr/002-autodiff-span-boundary.md`): AutoDiff uses span-based boundaries. All ops are span-ified with `TensorPrimitives`.
- **ADR-003** (`docs/adr/003-batch-fused-ops-not-rank-n-primitives.md`): Batch fused ops preferred over rank-N primitives.
- **ADR-004** (`docs/adr/004-fused-expression-engine-kernel-ir-span-backends.md`): Fused expression engine uses kernel IR with span backends.

## Agent Framework Workflow Patterns

See the "Agent Framework integration patterns" section of `samples/NivaraChat/README.md` for Nivara-specific integration notes from the NivaraChat sample.
