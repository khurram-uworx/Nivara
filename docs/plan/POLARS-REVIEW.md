# If Polars originated in .NET — the lens and the review

**Status:** architecture reference · **Scope:** the columnar engine (`src/Nivara`) · **Companion:** `docs/POLARS-ROADMAP.md`

This document records a specific lens for evaluating Nivara's columnar engine: **"What if Polars had been conceived in .NET instead of Rust?"** It is deliberately *not* "how close are we to a Polars port." A port copies a foreign architecture. A lens derives a native one.

The short version is: Nivara is already closer to the native-.NET design than to a Polars clone, and the highest-leverage gaps are concentrated in one place — the query *expression* path.

---

## 1. Why a lens instead of a port

Polars is a product of its substrate. It is Rust + Arrow wearing a query DSL:

- Zero-copy ownership semantics come from Rust's borrow checker, not from a design decision about dataframes.
- The Arrow IPC memory layout is the interchange standard Polars adopted, and it also happens to be its internal format.
- Trait-based monomorphization generates one specialized kernel per type.
- The Python-facing DSL exists because Python is the front end; the expression model had to be invented.

A .NET-native columnar engine does not inherit any of that. It inherits a garbage-collected runtime, `Span<T>`/`Memory<T>`, generic math (statically abstract interface members), `System.Numerics.Tensors`, Roslyn source generators, and LINQ. Those are different levers, and a native design pulls every one of them. The result is *not* Polars with C# syntax — it is a different architecture that lands in the same product category.

The pillars below are the concrete translation.

---

## 2. The nine pillars of a .NET-native columnar engine

### Pillar 1 — GC-aware pooled memory, not ownership-based zero-copy

- **Principle:** zero-copy is a *safety* property, not a memory-management property. In .NET it is achieved by immutability plus structural sharing, backed by pooled buffers.
- **.NET substrate:** `ReadOnlyMemory<T>` as the column currency, `Span<T>` for kernels, `ArrayPool<T>.Shared` / pooled buffer manager for scratch, validity masks as contiguous memory, no refcounts anywhere.
- **Polars instead:** ownership (Rust) — zero-copy columns shared across the graph because the borrow checker proves safety.

### Pillar 2 — One typed expression front end, compiled to fused span kernels

- **Principle:** users write familiar C#; the engine lowers to a *typed* AST that *compiles* to fused, typed span kernels — never an interpreter that boxes.
- **.NET substrate:** `Expression<T>` / `Expression.Compile`, generic `INumber<T>` kernels, `TensorPrimitives`.
- **Polars instead:** its own DSL (`pl.col("x") * 1.1 + 1000`) is necessary because the front end is a different language (Python). A native engine's front end is C# itself, so the DSL layer is optional.
- **Cardinal sin to avoid:** evaluating expressions by boxing every element into `object?` and dispatching per value. See §4.

### Pillar 3 — Generic math (SAIS) as the replacement for monomorphization

- **Principle:** one generic kernel, JIT-instantiated per `T`, instead of hand-written type switches per primitive.
- **.NET substrate:** `INumber<T>`, `IFloatingPointIeee754<T>`, `T.CreateChecked`, static abstract interface members.
- **Polars instead:** Rust traits monomorphize at compile time; C# gets the same effect from the generic JIT.

### Pillar 4 — BCL tensor infrastructure as the kernel substrate

- **Principle:** the runtime team ships `Tensor<T>` and `TensorPrimitives`; a native engine treats them as a co-developed dependency, not a competitor.
- **.NET substrate:** `System.Numerics.Tensors`, `System.Runtime.Intrinsics` (AVX-512 / AdvSimd) for the escape hatch.
- **Polars instead:** hand-rolled kernels and its own SIMD wrappers in Rust.

### Pillar 5 — Plan-based execution with a real optimizer

- **Principle:** queries are plans; plans are inspected, validated, and rewritten (predicate/projection pushdown, operator fusion, column elimination) before execution.
- **.NET substrate:** immutable plan nodes, schema propagation, cost estimates, explain-style diagnostics.
- **Shared with Polars:** Polars' `LazyFrame` optimizer is the same idea; both benefit.

### Pillar 6 — Async-first streaming

- **Principle:** a native engine's streaming pipeline is `IAsyncEnumerable`/`ValueTask`/channels, with bounded memory and cancellation — because .NET has first-class async.
- **.NET substrate:** `IAsyncEnumerable<T>`, `ValueTask`, `Channels`, `CancellationToken`.
- **Polars instead:** pull-based streaming with a separate scheduler; no async in the Python surface.

### Pillar 7 — Arrow as the interchange boundary, not the internal format

- **Principle:** internally, columns are `Memory<T>`/`Tensor<T>` (GC-safe, pooled). Arrow IPC is the *handoff* format at the edges — cross-language exchange, Parquet, zero-copy import/export.
- **.NET substrate:** `MemoryMarshal`/borrowed buffers at the conversion boundary; Arrow and Parquet live in `Nivara.Extensions`.
- **Polars instead:** Arrow is the internal format; the interchange is an implementation detail of the engine.

### Pillar 8 — Source generators as the uniquely-.NET differentiator

- **Principle:** Roslyn can emit code at build time — typed schema accessors, specialized query builders, serializers. This is structurally impossible for a Rust engine fronted by Python.
- **.NET substrate:** Roslyn incremental generators (`IIncrementalGenerator`).
- **Polars instead:** reflection/interpreter at runtime, or a codegen step in the build tooling — neither is a source generator.

### Pillar 9 — NativeAOT + `LibraryImport` as the escape hatch

- **Principle:** when the BCL kernels are insufficient, drop to `System.Runtime.Intrinsics` or P/Invoke into a battle-tested native library (simdjson, DataFusion, Velox) — with `NativeAOT` for zero-runtime deployment.
- **.NET substrate:** `[LibraryImport]` source generation, `NativeAOT` publish.
- **Polars instead:** the native library *is* the product; there is no managed layer to escape from.

---

## 3. Nivara scorecard, pillar by pillar

| Pillar | Status | Evidence | Roadmap |
| --- | --- | --- | --- |
| 1 · GC-aware pooled memory | ✅ Native | Sole-owner `ColumnStorage<T>` (`src/Nivara/Storage/ColumnStorage.cs`); zero-copy `AsTensor()` view cached; `BufferPool`/`ArrayPool` in hot paths (Adam/AdamW, AccumulateGradient); immutability + structural sharing | Done |
| 2 · Typed expression → fused kernels | ❌ Violated | Boxed interpreter in `src/Nivara/Helpers/ExpressionEvaluator.cs`; see §4 | **Phase 1–2** |
| 3 · Generic math (SAIS) | 🟡 Partial | AutoDiff already `IFloatingPointIeee754<T>` + generic `TensorPrimitives`; but `NivaraColumn` arithmetic still float/double type-switches (`src/Nivara/NivaraColumn.cs:57-76`, `:188-208`) | Phase 2 |
| 4 · BCL tensor substrate | ✅ Native | `TensorPrimitives` kernels in `TensorsHelper.cs`, `NivaraTensorExtensions.cs`; `Tensor<T>` storage | Done |
| 5 · Plan + optimizer | ✅ Strong | `QueryPlan`, `QueryOptimizer`, `OptimizationEngine`, pushdown/fusion/elimination rules (`src/Nivara/Optimization/`) | Phase 2 (kernel fusion) |
| 6 · Async-first streaming | 🟡 Partial | Async seams exist on `IQuerySource` (`src/Nivara/Query/IQueryInterfaces.cs:34-52`); strategies implemented, but pipeline is pull-chunk, not async-native | Phase 4 |
| 7 · Arrow as interchange | ✅ Native | `ToArrowTable`/`FromArrow` in `src/Nivara.Extensions/IO/ArrowInterop.cs`; Parquet; internal stays `Tensor`/`Memory` | Done |
| 8 · Source generators | ❌ Absent | No analyzer/generator project exists | Phase 5 |
| 9 · NativeAOT + LibraryImport | ❌ Absent | Not exercised; aspirational | Post-roadmap |

Legend: ✅ native-aligned · 🟡 partially aligned · ❌ gap.

---

## 4. The one fatal flaw: the boxed expression evaluator

`src/Nivara/Helpers/ExpressionEvaluator.cs` is the single largest contradiction to Pillar 2 in the codebase.

Every query predicate or projection that goes through `QueryFrame` (`.Where(...)`, `.Select(...)`) is evaluated by this interpreter:

- `ApplyBinaryOperation` allocates an `object?[]` and calls `left.GetValue(i)`/`right.GetValue(i)` per row (`ExpressionEvaluator.cs:220-235`).
- `AddValues`/`SubtractValues`/... pattern-match boxed values and fall back to `Convert.ToDouble` (`ExpressionEvaluator.cs:262-320`).
- Comparisons go through `IComparable` (`ExpressionEvaluator.cs:345-391`).

**Consequences:**
- No SIMD, no `TensorPrimitives`, no `Span<T>` — the exact path Polars compiles to fused kernels is a per-row, boxing, allocating interpreter here.
- Meanwhile `NivaraColumn<T>` arithmetic is fully typed and vectorized (`TensorPrimitives.Multiply(xFloat, yFloat, destFloat)`). The query path and the column path are two different worlds of performance.
- Result columns degrade to `NivaraColumn<object?>` in many cases, leaking the untyped fallback into the result schema.

**Contributing factor — the `OrderBy` gap:** `NivaraLinqExtensions.OrderBy` only supports direct column references or simple name-based expressions and throws `NotSupportedException` for complex keys (`src/Nivara/Linq/NivaraLinqExtensions.cs:64-88`). The sort layer expects a column name, so the expression engine cannot feed it computed keys. This is a symptom of the same disease: expressions are not first-class typed values.

**Contributing factor — two front ends, one weak back end:** both `ColumnExpression` operator overloads (the DSL, `src/Nivara/Expressions/ColumnExpression.cs`) and `RowExpressionBuilder` (the LINQ-ish surface, `src/Nivara/Linq/NivaraLinqExtensions.cs`) ultimately produce the *same* `ColumnExpression` AST. The split is not two ASTs — it is two ergonomic front ends converging on one AST that is then interpreted through the boxed evaluator. Fixing the back end fixes both front ends at once. This is the highest-leverage change in the project.

---

## 5. Secondary gaps

1. **Operator-level fusion only.** `OperationFusionRule` fuses *plan operators* (`src/Nivara/Optimization/OperationFusionRule.cs`). There is no *kernel-level* expression fusion — `(Salary * 1.1) + 1000` still materializes intermediate columns instead of running as one span pass.
2. **No window functions.** No `Over`/rolling/cumulative/lag/rank anywhere in `src/` (confirmed by grep). This is the largest *analytical feature* gap versus Polars.
3. **Generic-math collapse pending.** `NivaraColumn<T>` arithmetic branches on `float`/`double` explicitly. AutoDiff proves the generic `IFloatingPointIeee754<T>` pattern works; the column layer has not adopted it.
4. **Async not first-class in streaming.** The seams exist; the streaming strategy is still a synchronous chunk puller with async wrappers.
5. **No source generators.** The uniquely-.NET differentiator is untouched (see Pillar 8, Phase 5).

---

## 6. What is already right (and should not be changed)

- **Storage is native, not a port.** Sole-owner `ColumnStorage<T>` + pooled buffers is exactly Pillar 1. Do not rewrite it to an Arrow-internal format; Arrow belongs at the boundary (Pillar 7).
- **The optimizer is real.** Pushdown + fusion + elimination rules with plan inspection and diagnostics is a genuine native asset.
- **AutoDiff is the proof-of-pattern.** It already demonstrates generic-math SAIS kernels, span-first implementations, `ArrayPool` discipline, and the inference-default model. The column layer should copy its techniques, not invent new ones.
- **Diagnostics everywhere.** `OperationDiagnostics`, `ExecutionEngine.LastDiagnostics`, `QueryPlan` inspection — observability is a native strength no Python/Rust stack matches ergonomically.

---

## 7. Summary

Nivara is ~75% a "Polars for .NET" and ~80% a "columnar + Arrow-inspired + AI infrastructure in .NET style," but those numbers are not the point. The lens says something sharper:

> Nivara already makes the right native-.NET choices on **memory**, **tensor substrate**, **Arrow positioning**, **optimization**, and **observability**. It violates the native model in exactly one architectural place — the expression engine boxes instead of compiling — and it is missing the native-only differentiators (source generators) and the analytical breadth (window functions).

The roadmap to close those gaps is in **`docs/POLARS-ROADMAP.md`**.

## Related documents

- `docs/POLARS-ROADMAP.md` — the query/expression roadmap that closes out this review.
- `docs/ARROW-REVIEW.md` — the second lens: Arrow-inspired columnar *physics* (validity bitmaps, layouts, chunked columns, zero-copy), where the same storage/columns are evaluated differently.
- `docs/ARROW-ROADMAP.md` — the roadmap for that physics gap.
- `docs/AISTACK-REVIEW.md` — the third lens: complementing the local .NET AI stack (ONNX, ML.NET, MEAI, AutoDiff).
- `docs/AISTACK-ROADMAP.md` — the roadmap for that AI-data last mile.
- `docs/IDEA.md` (retired 2026-08-06; its query/expression-adjacent items are tracked as GitHub issues: [benchmark API #128](https://github.com/khurram-uworx/Nivara/issues/128), [observability #129](https://github.com/khurram-uworx/Nivara/issues/129), [typed LINQ `Query<T>()` #130](https://github.com/khurram-uworx/Nivara/issues/130), [window functions #134](https://github.com/khurram-uworx/Nivara/issues/134)), `docs/TENSORS.md`, `docs/LINQ.md`, `docs/AUTODIFF.md` — product vision, strategic framing, plan-layer spec, and kernel patterns.
