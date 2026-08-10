# POLARS-ROADMAP — the road to a native-.NET columnar engine

**Status:** planning reference · **Scope:** the columnar engine (`src/Nivara`) · **Lens:** `docs/POLARS-REVIEW.md`

This is the travel plan for turning Nivara into what a columnar engine *would* be if Polars had been conceived in .NET — not a port, a native design.

---

## 0. Vision

> **Feels like LINQ. Executes like a columnar engine. Born in .NET, not ported.**

Concretely, when this roadmap is done:

- A query like `frame.Where(r => r["Salary"] * 1.1 > 100)` runs as a schema-typed, null-aware, single-pass span kernel — with zero `object?` boxing on the hot path.
- Computed keys feed `OrderBy`, window functions exist (`Over`, rolling, cumulative), and streaming is async-first.
- Roslyn source generators emit typed frame accessors and specialized query builders — capability a Rust/Python stack structurally cannot have.
- Arrow and Parquet remain *interchange* at the boundaries; internal storage stays `Tensor<T>`/`Memory<T>`.

### Where we are today

The column path is already native and fast: `NivaraColumn<T>` arithmetic is typed and vectorized (`TensorPrimitives` in `src/Nivara/NivaraColumn.cs`), storage is tensor/memory-backed and pooled, and the optimizer (`QueryOptimizer` + pushdown/fusion rules in `src/Nivara/Optimization/`) is real.

The query *expression* hot path is now typed and fused: the boxed `object?` interpreter in `ExpressionEvaluator` was replaced by a fused evaluator (compiled-first `Expression.Compile` target over `T[]` arrays with a generic node-tree fallback in `src/Nivara/Expressions/`), `Filter`/`Select`/`SortByExpression` route through it, `OrderBy` computes typed keys, `MultiColumnComparer` compares without boxing, and unsupported type/operator combinations throw a clear `NotSupportedException` instead of silently boxing. No `object?` per-element dispatch remains on the numeric/vectorizable query path. The legacy `ExpressionEvaluator` interpreter file was fully removed in #152; the fused evaluator is the sole expression engine.

Row-level filters are typed too: the last public `dynamic` surface in the core (`frame.Where(Func<dynamic,bool>)` via `ExpandoObject` + reflection) was removed (#154) in favor of `frame.Where(Func<NivaraRow,bool>)`, where `NivaraRow` is an allocation-free readonly struct over the frame's columns with `GetValue<T>`/`TryGetValue<T>`/`IsNull` accessors. The row view is a bridge for user predicates; the engine's own filter kernels stay columnar and fused.

### Non-goals (explicit)

- **No Arrow-internal storage rewrite.** Internal stays `Tensor<T>`/`Memory<T>`; Arrow stays an interchange boundary (Pillar 7).
- **No unsafe hand-rolled SIMD asm.** Use `TensorPrimitives` / `System.Runtime.Intrinsics`; hand-assembly is not a native-.NET move.
- **No Polars DSL clone.** The front end is C#/LINQ; the `ColumnExpression` DSL stays a thin ergonomic layer, not a product surface.
- **No cross-language bindings** (Python, etc.). A .NET-native engine is .NET-first; Arrow handles cross-language exchange.

---

## 1. The roadmap

### Phase 1 — Unified typed expression engine *(make-or-break)*

**Motivation:** The boxed interpreter is the single largest contradiction to the vision. Both front ends (`ColumnExpression` operators and `RowExpressionBuilder`) already produce the same AST, so fixing the back end fixes both surfaces at once — the highest-leverage change in the project.

**Status (delivered, #153):** The boxed interpreter is gone. A typed fast path plus a typed numeric-promotion path replaced it, with no `object?` per-element dispatch on the numeric/vectorizable query path. The typed interpreter (then `ExpressionEvaluator`) was superseded by the fused evaluator and its file removed in #152; the public front ends are unchanged and `OrderBy` computed keys feed the fused evaluator (Phase 2).

**Scope:**
- Replace the interpreter in `ExpressionEvaluator` with a typed lowering pass: `ColumnExpression` AST → typed evaluation that produces `NivaraColumn<T>` results by reusing the existing typed column kernels (no `object?` result columns on the happy path).
- Keep the public front ends unchanged (`Where`, `Select`, `.AsQueryFrame()`), so this is an internal rewrite with a compatibility contract.
- Make expressions first-class typed values so computed results can feed downstream operators (feeds Phase 2 `OrderBy` work).

**Key files:** `src/Nivara/Helpers/ExpressionEvaluator.cs`, `src/Nivara/Expressions/ColumnExpression.cs`, `src/Nivara/Query/QueryFrame.cs`, `src/Nivara/Query/QueryExecutor.cs`.

**Dependencies:** none (foundation).

**Acceptance criteria:**
- `frame.Where(r => r["Salary"] * 1.1 > 100)` returns a typed result with null masks intact.
- No `object?` per-element dispatch on the query hot path for numeric/vectorizable columns.
- Existing `QueryOptimizationPropertyTests` and `QueryExecutionPropertyTests` stay green; results identical to today.

**Risks:** type-unification of heterogeneous literals/columns (mitigate with typed lowering + explicit numeric promotion), regression risk on non-vectorizable columns (string/object keep a scalar fallback).

---

### Phase 2 — Kernel fusion + generic-math collapse

**Motivation:** Fixing the interpreter gets us *typed* and *vectorized per operator*, but an expression like `(Salary * 1.1) + 1000` still materializes intermediate columns. Native design fuses it into one span pass. In parallel, collapse the column layer's `float`/`double` type-switches onto generic math — AutoDiff already proves the pattern.

**Status (delivered):** Expression trees fuse into a single pass through `FusedExpressionEvaluator` (`src/Nivara/Expressions/FusedExpressionEvaluator.cs`, `FusedKernel.cs`, `ExpressionTypeInferer.cs`) with a compiled-first `Expression.Compile` target over `T[]` arrays plus a generic sealed node-tree fallback; `Filter`/`Select`/`SortByExpression` and the parallel sort route through it. The boxed/`dynamic` fallbacks in `ExpressionEvaluator`, `NivaraColumn<T>` arithmetic, and `NivaraSeries` Sum/Average are removed (unsupported combinations throw), the numeric domain is extended (`Half`, `decimal`, `nint`/`nuint`, `Int128`/`UInt128`), and `MultiColumnComparer` sorts without boxing. Fused vs multi-pass benchmark at 1M rows: ~11.6x faster, ~44% less allocation. The `NivaraColumn<T>` arithmetic kernels now dispatch the full numeric domain (`Half`, `decimal`, `nint`/`nuint`, `Int128`/`UInt128` included) through the `INumber<T>`-constrained `NumericTensorKernels<T>` typed switch (#157).

**Scope:**
- **Kernel fusion:** lower expression trees to fused single-pass kernels over `ReadOnlySpan<T>`, with two compile targets:
  - Generic `INumber<T>` / `IFloatingPointIeee754<T>` static kernels via SAIS (the native monomorphization).
  - `Expression.Compile` to a span-consuming delegate as the fallback for non-generic-math types.
- **OrderBy computed keys:** teach the sort layer to accept an evaluated key column (remove the `NotSupportedException` in `NivaraLinqExtensions.OrderBy`), routing complex expressions through the fused evaluator.
- **Generic-math collapse:** replace the explicit `float`/`double` branches in `NivaraColumn<T>` arithmetic (`src/Nivara/NivaraColumn.cs:57-76`, `:188-208`, and operator overloads) with `INumber<T>` generic paths, keeping `TensorPrimitives` on the vectorizable fast path.

**Scope (remaining):** `BFloat16`-typed kernels stay deferred to the net11 migration (#137); the fused compiled target runs over `T[]` arrays rather than spans (ref-structs are excluded from expression trees — span-capable target tracked as #155).

**Key files:** `src/Nivara/Expressions/FusedExpressionEvaluator.cs` (successor to the removed `src/Nivara/Helpers/ExpressionEvaluator.cs`), `src/Nivara/NivaraColumn.cs`, `src/Nivara/Operations/SortOperation.cs`, `src/Nivara/Linq/NivaraLinqExtensions.cs`, `src/Nivara/Optimization/OperationFusionRule.cs`, `src/Nivara/KernelSelector.cs`.

**Dependencies:** Phase 1.

**Acceptance criteria:**
- Fused expression output is bit-equivalent to the per-operator result (null-mask preserving).
- Benchmark shows single-pass vs multi-pass for chained arithmetic; fused path wins on length ≥ vector threshold (`KernelSelector` heuristics).
- `OrderBy(r => r["Salary"] * 1.1)` works and is vectorized.
- `Half`/`BFloat16` columns execute through the generic path (mirroring AutoDiff's `IFloatingPointIeee754<T>` validation).

**Risks:** fused-kernel edge cases (division by zero, overflow semantics, null short-circuit ordering), `Expression.Compile` delegate allocation caching, regression on non-vectorizable columns.

---

### Phase 3 — Window functions

**Motivation:** The largest analytical feature gap versus Polars. No `Over`/rolling/cumulative/lag/rank exists anywhere in `src/`.

**Status (core set + rank family delivered, #135/#156):** Rolling min/max/mean/sum over fixed windows, cumulative sum/max/min/product/count, `Shift`/`Lead`, and the rank family (`RowNumber`/`Rank`/`DenseRank`/`PercentRank` over partitions with `SortKey` ordering) now ship on `NivaraColumn<T>` (`src/Nivara/Tensors/WindowFunctions.cs`, `src/Nivara/Tensors/RankFunctions.cs`), eager `NivaraFrame` extensions (`src/Nivara/WindowFrameExtensions.cs`), and the lazy `QueryFrame` pipeline (`src/Nivara/Operations/WindowOperations.cs` and `src/Nivara/Operations/RankOperation.cs`, `OperationType.Rolling`/`.Cumulative`/`.Shift`/`.Rank`). Semantics: nulls ignored by default with output gated on `minPeriods` (default full window); cumulative ops skip nulls with carry-forward; `Shift`/`Lead` boundary positions are null or `fillValue`; an optional `nullHandler` replaces nulls so every position satisfies the window; a null rank order key yields null output and is excluded from numbering/denominator. Documented in `docs/LINQ.md`; covered by `tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`, `tests/Nivara.Tests/Tensors/RankFunctionsTests.cs`, `tests/Nivara.Tests/Query/WindowOperationTests.cs`, and `tests/Nivara.Tests/Query/RankOperationTests.cs`.

**Scope (remaining):**
- Null-aware windowing consistent with the project's explicit null-mask model (ADR-001 boundary, no NaN semantics).
- Built on the Phase 2 fused expression engine so window expressions compose with ordinary expressions.

**Key files:** new `src/Nivara/Operations/` window operators, `src/Nivara/Query/OperationType.cs`, `src/Nivara/Expressions/` (window expression nodes), plan schema propagation.

**Dependencies:** Phases 1–2.

**Acceptance criteria:**
- Window results match a documented reference semantics (Polars window semantics as the spec, adapted to explicit-null model).
- Null masks propagate per window frame; empty/partial frames defined.
- Property tests (parameterized NUnit) covering partitioning, ordering ties, and null positions.

**Risks:** semantic ambiguity (null ordering, frame boundaries), performance on large partitions (mitigate with vectorized cumulative kernels + pooled scratch).

---

### Phase 4 — Async-first streaming

**Motivation:** The async seams already exist on `IQuerySource` (`ReadChunkAsync`, `ToAsyncEnumerable` in `src/Nivara/Query/IQueryInterfaces.cs:34-52`), and strategies are implemented — but the pipeline is a synchronous chunk puller with async wrappers. Native design is async-native, with cancellation and bounded memory as first-class properties.

**Scope:**
- Make the streaming path genuinely `IAsyncEnumerable`-driven end to end (lazy sources → operators → `CollectAsync`), with `CancellationToken` threading.
- Channel-based buffering with the existing memory-budget machinery in `StreamingExecutionStrategy` / `StreamingBufferManager` (Extensions).
- Add `CollectAsync`/`ToListAsync` style public entry points; keep synchronous `Collect()` as a thin blocking wrapper.

**Key files:** `src/Nivara/Execution/StreamingExecutionStrategy.cs`, `src/Nivara/Query/IQueryInterfaces.cs`, `src/Nivara/Query/QueryFrame.cs`, lazy sources in `src/Nivara.Extensions/IO/`.

**Dependencies:** Phases 1–2 (so async operators consume typed fused expressions).

**Acceptance criteria:**
- Streaming results equal eager results (property tests over chunk sizes).
- Cancellation is respected and produces clean `OperationCanceledException`.
- Memory stays within the configured budget under load; no unbounded buffering.

**Risks:** interaction between operator fusion and streaming boundaries (fused operators must still work chunk-at-a-time), resource disposal across async hops.

---

### Phase 5 — Source generators *(the uniquely-.NET differentiator)*

**Motivation:** Roslyn can emit typed schema accessors and specialized query builders at build time — structurally impossible for a Rust engine fronted by Python (Pillar 8). This is Nivara's native trump card, and it is currently untouched.

**Scope:**
- New `Nivara.Generators` project (Roslyn incremental generator, `IIncrementalGenerator`).
- **Typed frame accessors:** from a schema (or a source-declared record), generate `Frame.RowCount`, strongly typed column getters, and typed row projections — no reflection on the hot path.
- **Specialized query builders:** generate per-schema `Where`/`Select` overloads that compile to direct typed kernels (bypassing name-based lookup).
- Optional: generated serialization / Arrow-schema emission.

**Key files:** new generator project; consumed by `src/Nivara` (or `Nivara.Extensions`); sample + tests in `tests/Nivara.Tests/`.

**Dependencies:** Phases 1–2 (generated builders lower into the typed expression engine). Can be split into accessors-first / builders-later.

**Acceptance criteria:**
- Generated accessors compile cleanly and are used on a hot path with zero reflection.
- Generated query builder results equal the name-based query results.
- No incremental-generator correctness regressions across build clean/rebuild.

**Risks:** generator complexity (incremental caching, incremental steps), analyzer compatibility across Roslyn versions, design-time/build-time coupling.

---

### Post-roadmap (aspirational)

- **Pillar 9 — NativeAOT + `LibraryImport` escape hatch:** publish path for zero-runtime deployment; optional P/Invoke into battle-tested native kernels when BCL kernels are insufficient.
- **GPU offload** via the tensor boundary (`Tensor<T>` has a path to `GPUTensor` in the BCL direction) — explicitly *not* an internal rewrite, an interchange addition.

---

## 2. Sequencing rationale

**Why the expression engine is first:** every later phase (window functions, async, generators) consumes *typed, fused expressions*. Shipping it first de-risks the whole roadmap and is also the single biggest user-visible performance win.

**Quick wins vs long horizon:**

| Item | Effort | When |
| --- | --- | --- |
| Kill the boxed interpreter (Phase 1) | Medium | ✅ Delivered (#153) |
| `OrderBy` computed keys (Phase 2) | Small | ✅ Delivered |
| Generic-math collapse of column arithmetic (Phase 2) | Medium | ✅ Delivered |
| Fused single-pass kernels (Phase 2) | Medium-High | ✅ Delivered |
| Window functions, core set (Phase 3) | High | ✅ Delivered (#135) |
| `Over`/`Rank`/`DenseRank` (Phase 3 remainder) | High | ✅ Delivered (#156) |
| Async-native streaming (Phase 4) | Medium | Phase 4 |
| Source generators (Phase 5) | High, splittable | Phase 5 |

**What we leverage, not reinvent:** `TensorsHelper` (SIMD kernels), `KernelSelector.DetermineKernelType()`, the `OptimizationEngine` rule set, AutoDiff's proven `IFloatingPointIeee754<T>` + span + `ArrayPool` techniques, `BufferPool`, `docs/LINQ.md` (plan-layer spec), and `docs/AUTODIFF.md` (kernel patterns).

---

## 3. Cross-cutting conventions (every phase)

- **Diagnostics:** record `OperationDiagnostics` / `ExecutionEngine.LastDiagnostics` for kernel selection, fusion decisions, and window execution.
- **Testing:** NUnit 4.x, `Method_Scenario_ExpectedBehavior` naming, parameterized property tests for null-mask propagation and plan-rewrite equivalence; performance regression tests for fused vs per-operator paths.
- **Null semantics:** ADR-001 non-nullable boundary holds; null masks are authoritative (mask OR on binary results, false at nulls in comparisons).
- **No comments in code** beyond non-obvious design decisions; `.editorconfig` is authoritative.

---

## 4. Definition of done

The vision at §0 holds, verified by:

- Query path performance is within noise of the column path on the same data (`OperationDiagnostics` + benchmarks).
- Window functions exist with documented, tested semantics.
- Streaming is async-native with bounded memory and cancellation.
- Source generators ship typed accessors/builders used on a hot path.
- Full `dotnet build Nivara.slnx` + `dotnet test` green; no new `object?`-boxing expression paths.

---

## Related documents

- `docs/POLARS-REVIEW.md` — the lens, the nine pillars, and the per-pillar scorecard this roadmap closes out.
- `docs/ARROW-REVIEW.md` — the second lens (Arrow-inspired columnar physics) and its scorecard.
- `docs/ARROW-ROADMAP.md` — the columnar-physics roadmap (chunked columns, layouts, bitmap validity, zero-copy) that this roadmap converges with at the streaming/kernel merge points.
- `docs/IDEA.md` — original product vision (already Polars-inspired). Retired 2026-08-06; the outstanding items it described are tracked as GitHub issues: [benchmark API #128](https://github.com/khurram-uworx/Nivara/issues/128), [typed LINQ `Query<T>()` #130](https://github.com/khurram-uworx/Nivara/issues/130), [window functions #134](https://github.com/khurram-uworx/Nivara/issues/134), [observability #129](https://github.com/khurram-uworx/Nivara/issues/129).
- `docs/TENSORS.md` — tensor-vs-Polars strategic framing (Nivara's standing and committed direction).
- `docs/LINQ.md` — plan-layer and query-engine specification.
- `docs/AUTODIFF.md` — kernel and generic-math patterns the roadmap reuses.
