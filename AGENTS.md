# Guidance for AI-assisted coding

**GitHub repo:** `https://github.com/khurram-uworx/Nivara` — use `--repo khurram-uworx/Nivara` for `gh` commands.

## Facts & Research

- Nivara AutoDiff product direction: inference is the default/common path; reverse-mode training is opt-in via `using (GradientUtils.Grad())`. Do not implement NoGrad as the primary API. Built-in training loops should enter Grad() internally, while manual training examples/docs should wrap forward/loss/backward/optimizer code in Grad().
- Nivara AutoDiff ADR-001 (non-nullable domain) is fully implemented. Type constraint relaxed from `INumber<T>` to `IFloatingPointIeee754<T>`, which passes Half/F16 and BFloat16 through runtime validation alongside float/double. All AutoDiff ops are span-ified (no NivaraColumn.Data access; Span<T> + TensorPrimitives).
- Key BCL .NET 11 tensor patterns: TensorPrimitives now generic (200+ overloads for any INumber/IRootFunctions T), ReadOnlyTensorSpan<T> with TryGetSpan, implicit conversion from T[] to ReadOnlyTensorSpan<T>, Tensor<T> stable (no longer experimental). Spans are the currency.
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
- `Tensor<T>` has **no** `Span`, `Memory`, `AsSpan()` or `AsMemory()`. To read one row of a rank-2 tensor use `GetSpan(startIndexes, length)` — and supply **one index per dimension**: `GetSpan([i, 0], dims)`. `GetSpan([i], dims)` compiles and then throws `ArgumentOutOfRangeException`, which the snippet gate cannot catch because it compiles but never runs. Its `length` parameter is `int` while `Lengths` is `nint[]`, and it is overloaded on both `ReadOnlySpan<nint>` and `ReadOnlySpan<NIndex>`, so a collection expression (`[i, 0]`) is what resolves the ambiguity. **Before writing a `Tensor<T>` call from a doc or an issue, run the probe** — `dotnet run -c Release --project tests/Nivara.SimdProbe -- tensor-api` — which prints the real surface, asserts the members above are absent, and runs each candidate form against a flat row-major reference. Issues #524–#532 are eight more "the doc names an API that does not exist", and the recurring finding is that the filed prescription is itself wrong; check rather than trust the issue body.
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

See [GUIDELINES.md](GUIDELINES.md) for transferable engineering principles (null semantics, lazy execution, zero-copy, testing strategy) and, in its "Verification, Evidence & Claim Discipline" section, the rules that keep a passing result from meaning less than it appears to (gate coverage, one failure reason per catch, exactness over tolerance, commit scope, superseded-measurement propagation, and asserting the exit status of the process you care about).

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
- **Reach for an existing probe project before writing a throwaway harness.** The `tests/` directory holds three projects: `Nivara.Tests` (NUnit 4.x unit tests), `Nivara.PerformanceTests` (standalone stopwatch harness, `dotnet run -c Release`), and `Nivara.SimdProbe` (self-contained probe, `dotnet run -c Release -- correctness|benchmark|scalar|transpose`). Each non-NUnit project has its own `README.md`. Before diagnosing anything performance- or kernel-shaped in a temp directory, **check whether a mode already exists** — `Nivara.SimdProbe` exists precisely to answer "is this hand-written kernel worth keeping in `src/Nivara`, and does the BCL beat it?", which is the question behind most kernel investigations, and `Nivara.PerformanceTests` is the home for repeatable benchmark harnesses and gate policy. Add a mode there (follow the `Correctness`/`Benchmark` pattern, document it in that project's `README.md`) instead of building a scratch project. A throwaway temp harness is only acceptable once you have confirmed no existing mode covers the question. If you do use temp for initial diagnosis, say so explicitly and plan to promote or delete it — do not leave a scratch project behind.
- The gate/skip conventions above are not optional: CI's `--filter "Category!=Performance"` is what keeps timing assertions out of shared runners, so a new timing test that omits `[Category("Performance")]` **will** run in CI. A test taking more than 2 seconds is `Performance` — the rule and the re-measure command are under Performance & Optimization Thresholds.
- Avoid `[TestCase]` with null arrays; use regular `[Test]` with inline arrays.
- Reflection cannot pass `Span<T>` via `MethodInfo.Invoke` — convert to array first.
- Test for key phrases in error messages rather than exact message strings.
- Property-like tests: implement with parameterized NUnit test suites rather than full FsCheck (see [GUIDELINES.md](GUIDELINES.md) for rationale).
- A gate must state its own coverage in every summary, and reduced coverage is a failure, not a neutral event. Keep unhostable-skip and compile-failure on separate paths, and count each verdict in a separate counter so a structural failure is never reported as a numeric one.
- When a new implementation must reproduce a reference bit-for-bit, assert that instead of a tolerance band. A tolerance turns a structural defect into a judgement call; exactness does not.
- One commit, one reason. Do not alter a test's inputs, thresholds or counts inside a commit about something else — it silently invalidates the baselines the next person compares against.
- **Do not sign commits or PRs.** No `Co-Authored-By`, `Signed-off-by`, `Reviewed-by`, or AI-attribution trailers, and no `git commit -S`. The human has not asked for attribution and a trailer naming a model is a false claim about who wrote the code. Describe the change and its reason in the message body; nothing else. If you find a trailer you added earlier, say so rather than quietly leaving it.
- When a measurement is superseded, move every site that carries any part of it in one change, then check the arithmetic: parts of a partition should sum to the whole.
- Capture the exit status of the process you care about, not of a filter in a pipeline — a filtered command reports the filter's status. Never report a check you have not seen fail.
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
- **Dependencies (Extensions only):** CsvHelper, Apache.Arrow, Parquet.Net, Microsoft.ML, Microsoft.Extensions.AI.Abstractions, Streamix, System.Numerics.Tensors — .NET 11 latest, versions deliberately unpinned. Read the csproj.

## Performance & Optimization Thresholds

- **Always measure performance in Release — never Debug.** `dotnet test` and `dotnet run` default to Debug, where `Nivara.dll` is compiled **unoptimized** while `System.Numerics.Tensors` ships **ReadyToRun** and stays optimized regardless. A Debug run therefore compares an unoptimized handwritten kernel against optimized framework code and reports a false regression — the #482 transpose gate failed 3/5 runs on a clean tree purely because of this. Use `dotnet test -c Release` and `dotnet run -c Release --project tests/...`. CI already does this correctly (`dotnet test --no-build --configuration Release`).
- **Timing assertions need an optimized-build guard.** Any NUnit test that asserts on wall-clock must call `TimingGuards.RequireOptimizedBuildForTiming()` (see `tests/Nivara.Tests/TimingGuards.cs`) so a Debug run reports "not measured" instead of a fake pass/fail, and must carry `[Category("Performance")]` so CI's `--filter "Category!=Performance"` excludes it. The same category is required of any test that measures wall-clock or allocations, and of any test that takes more than **2 seconds** on a shared runner, whether or not it asserts on time. The filter is opt-in: a slow test that forgets the category does not fail, it makes every push pay for it. The filtered suite's median is under 1 ms and its p99 is about 0.1 s (Release, 2026-10-06, 3873 tests), so 2 s is far above the noise floor rather than a judgement call. Re-measure with `dotnet test -c Release --filter "Category!=Performance" --logger "trx;LogFileName=d.trx" --results-directory TestResults` and read `duration` on each `UnitTestResult`. The trx is uploaded as the `test-durations` artefact on every CI run. `SlowTestCategorisationTests` fails when a test measures wall-clock or allocations without the category; it cannot see a slow test that measures nothing, which is why the 2-second rule is stated here and not only in that gate.
- **The documentation snippet gate stays in the filtered suite.** `EveryGatedBlock_CompilesWithoutErrors` took 13.6 s on the 2026-10-06 Release run, over the 2-second line, and it is not `Performance`. It compiles every gated fenced csharp block. Categorising it would make a green CI run say nothing about whether the docs still compile. It is the written exception to the 2-second rule (#545).
- **Compare distributions, not single estimates.** Warm up both routes, interleave A/B rounds, alternate which side is measured first so load drift cannot systematically penalise the first-measured route, then assert on the median ratio with a bound set well above the observed noise floor. Best-of-N taken separately per route and compared once is what made #482 flaky even in Release. The reusable implementation is `TransposeKernelProbe` (`transpose` mode) in `tests/Nivara.SimdProbe`.
- **Buffer pooling threshold**: rent arrays >1024 elements from `BufferPool` (in Extensions).
- **Default memory budget**: 1 GB (`1024 * 1024 * 1024`, `NivaraExecutionContext.cs:17`) for the core query pipeline, pinned by `ExecutionContextTests.Constructor_DefaultValues_SetsExpectedDefaults`. Do not confuse it with the 256 MB default owned by the IO-layer `StreamingBufferManager` (`Nivara.Extensions`), which is intentionally not wired into query execution.
- **Streaming chunk size**: `AsStream` defaults to 10,000 rows (its own parameter default, not budget-derived). The budget-derived fallback `clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)` applies only when `NivaraExecutionContext.ChunkSize` is null — on the public API that means `ExecutionEngine.Execute(plan, context)` with `Strategy = Streaming`. Row-group aligned for Parquet. Full contract in `docs/STREAMING.md`.
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
- **GPU GEMM throughput**: at 128 tokens / 66 M params the iGPU is still 1.9–2.2× slower than PyTorch CPU, but that deficit is shape-dependent — at Laya's 512 tokens / 1.4 B it **ties** PyTorch (3.00 s vs 2.97 s). GEMM tiling (#440) was the nominated lever and is now **measured null** — every wider tile lost to Row4, because shared-memory capacity per group dominates shared-memory traffic per MAC on this iGPU. (That measurement is now conditioned: Row4's denominator moved ~20% when it was migrated off two `SharedMemory.Allocate2D` calls (#468) while the #440 geometries stayed on the 1-D form, so the family has not been re-measured on one addressing form — see the supersession note in `tests/Nivara.PerformanceTests/README.md`.) The leading GPU item is `BatchedAttention` at **57% of the Laya forward** (#447). See [docs/ACCELERATION.md](ACCELERATION.md) §5.2.
- **Vision gap**: convolution kernels are naive nested loops with no SIMD and no parallelism; tracked as issue #457.
- **CPU-side performance**: the ~5.6× Nivara-CPU-vs-PyTorch deficit is not a GEMM problem at Laya's shapes; tracked as #458.

## Quick Reference

- **Vectorizable types (confirmed)**: `int`, `float`, `double`, `long`, `short`, `byte`, `uint`, `ulong`, `ushort`, `sbyte`, `bool` (requires unmanaged constraint)
- **Target framework**: .NET 11, latest NuGet packages (versions unpinned — read the csproj)
- **Common deps (Extensions only)**: same set as above — .NET 11 latest, unpinned
- **Useful helpers**: `ColumnDiagnostics`, `DiagnosticsTracker`, `ColumnStorageFactory.IsVectorizable<T>()`, `NivaraColumnFactory.CreateFromNullable<T>(T?[])`, `Tensor.Create(array)` + `FlattenTo(buffer)`, `KernelSelector.DetermineKernelType()`, `SGD<T>`, `Adam<T>`, `AdamW<T>`, `Linear<T>`, `Sequential<T>`, `Module<T>.StateDict()`, `Module<T>.LoadStateDict()`, `TrainingLoop<T>`, `DataParallelTrainer<T>`, `ModelSerializer`, `Loss<T>`/`Reduction`, `Activation.Gelu`, `ReverseGradOperations.Gelu`
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
- **ADR-005** (`docs/adr/005-snippet-gate-authoring-contract.md`): Read before editing a fenced ```csharp block in a gated doc, or adding one. Each fence compiles alone; `<!-- gate -->` keys are `mode`/`preamble`/`locals`/`exclude`/`reason`; `preamble:` is a label, not a resolver.

## Agent Framework Workflow Patterns

See the "Agent Framework integration patterns" section of `samples/NivaraChat/README.md` for Nivara-specific integration notes from the NivaraChat sample.
