# Execution Engine — Multi-Phase Plan

## Purpose

Refactor the `src/Nivara/Execution/` directory to eliminate duplicated execution patterns, implement real parallel and streaming strategies, add diagnostics integration, clean up dead API surface, and achieve comprehensive test coverage.

## Architectural Summary (Current State)

The execution layer has **two overlapping patterns** with zero code connecting them:

### Pattern A: ExecutionEngine → IExecutionStrategy → QueryExecutor

```
QueryFrame.Collect()
  → ExecutionEngine.Execute(QueryPlan, NivaraExecutionContext)
    → IExecutionStrategy.Execute(QueryPlan, NivaraExecutionContext)
      → QueryExecutor.Execute(QueryPlan)          // Lazy, Eager
      → operation.Execute(columns) in for-loop     // Eager
      → Task.Run / Parallel helpers                // Parallel, Streaming (placeholders)
```

- `ExecutionEngine` routes to one of 4 `IExecutionStrategy` implementations via `ConcurrentDictionary`
- Strategies share ~50 lines of boilerplate (null checks, cancellation, try/catch, progress)
- `QueryExecutor` is the core plan executor — validates, sources data, applies operations sequentially

### Pattern B: DataFrameOperation strategy dispatch

```
DataFrameOperation.Execute(NivaraExecutionContext)
  → switch(context.Strategy):
      Lazy     → ExecuteLazy(context)     → this.Execute()
      Eager    → ExecuteEager(context)    → this.Execute()
      Streaming→ ExecuteStreaming(context)→ this.Execute()
      Parallel → ExecuteParallel(context) → this.Execute()
```

- Lives in `src/Nivara/Helpers/DataFrameOperation.cs`
- All four virtual methods fall through to `this.Execute()` — no differentiation
- Unused in the actual execution path (Pattern A never calls Pattern B)

### Key interfaces

| Interface | File | Purpose |
|-----------|------|---------|
| `IExecutionStrategy` | `ExecutionEngine.cs` | Strategy contract: Execute, ExecuteAsync, ValidatePlan, EstimateExecutionCost |
| `IQueryOperation<T>` | `IQueryInterfaces.cs` | Typed operation: Plan, Strategy, Transform, ExecuteAsync — **unused in executor** |
| `IQueryOperation` | `IQueryInterfaces.cs` | Untyped operation: OperationType, TransformSchema, Execute — **the one actually consumed** |
| `IQuerySource` | `IQueryInterfaces.cs` | Data source: Schema, IsLazy, Execute — **no async methods** |

---

## Findings & Issues

### Critical

| # | Issue | Severity |
|---|-------|----------|
| 1 | Two overlapping execution patterns (A & B) not connected | **Critical** |
| 2 | Parallel strategy is a no-op — all `Execute*Parallel` methods just call `Task.Run(() => operation.Execute(input))`; `ParallelExecutionHelper` utilities are never called | **Critical** |
| 3 | Streaming strategy is a no-op — `chunkSize` calculated but never used; just calls `executor.Execute(plan)` | **Critical** |
| 4 | Async anti-pattern — all `ExecuteAsync` wraps sync code in `Task.Run`; `IQuerySource` has no async methods | **Critical** |

### Moderate

| # | Issue | Severity |
|---|-------|----------|
| 5 | Naming inconsistencies — `EagerExecutionStrategy.ReportProgress` (PascalCase) vs `ParallelExecutionStrategy.reportProgress` / `StreamingExecutionStrategy.reportProgress` (camelCase) | Moderate |
| 6 | Dead API surface — `IQueryOperation<T>.Transform`, `IQueryOperation.Strategy`, `IQueryOperation<T>.ExecuteAsync` all unused; `ExecutionEngine.SetOptimizer()` internal but never called | Moderate |
| 7 | Diagnostics gap — `ExecutionDiagnostics` in `Nivara.Diagnostics` never instantiated/populated by any strategy; ad-hoc `IProgress<ExecutionProgress>` everywhere | Moderate |
| 8 | Redundant schema validation — `QueryExecutor.Execute()` calls `ValidatePlan()` then re-validates after each operation | Moderate |
| 9 | Cost estimate type mismatch — `QueryExecutor.EstimateExecutionCost` returns `int` vs interface `long` | Moderate |

### Minor

| # | Issue | Severity |
|---|-------|----------|
| 10 | Magic string operation types — `"Filter"`, `"Select"`, `"Sort"`, `"GroupBy"`, `"Join"`, `"Concatenation"` as string literals across 7+ files — no constants or enum | Minor |
| 11 | `"Concatenation"` is a consistent misspelling — should be `"Concatenation"` | Minor |
| 12 | No serialization/printing for `QueryPlan` beyond `ToString` — no JSON or structured output | Minor |
| 13 | `ExecutionDiagnostics.MemoryAllocated` can be negative if GC collects between measurements | Minor |

### Test Coverage Gap

| Area | Tests |
|------|-------|
| `EagerExecutionStrategy` | **Zero** |
| `LazyExecutionStrategy` | **Zero** |
| `ParallelExecutionStrategy` | **Zero** |
| `StreamingExecutionStrategy` | **Zero** |
| `ExecutionEngine` | **Zero** |
| `ParallelExecutionHelper` | **Zero** |
| `NivaraExecutionContext` | 8 tests (existing) |
| `ExecutionProgress` | 6 tests (existing) |
| `ExecutionDiagnostics` | 6 tests (existing) |
| `QueryExecutionTests` | 4 integration tests (CSV path only) |

---

## Phase Breakdown

### Phase 1: Unify Execution Patterns (Architectural)

**Goal**: Eliminate Pattern B (`DataFrameOperation` strategy dispatch), make `IExecutionStrategy` the single execution routing mechanism.

**Key changes**:
- Remove `DataFrameOperation.Execute(NivaraExecutionContext)` strategy-switch method and all four virtual strategy methods (`ExecuteLazy`, `ExecuteEager`, `ExecuteStreaming`, `ExecuteParallel`)
- Replace with a single `Execute()` abstract method; operations call `ExecutionEngine` directly when they need strategy-aware execution
- Simplify `DataFrameOperation` to hold a reference to `IExecutionStrategy` or `ExecutionEngine` instead of duplicating dispatch
- Remove unused `IQueryOperation<T>.Transform`, `IQueryOperation<T>.ExecuteAsync`, `IQueryOperation.Strategy`
- Add `IQuerySource.ExecuteAsync(CancellationToken)` for async data sources
- Add characterization tests for all 4 strategy classes before refactoring

**Files involved**: `DataFrameOperation.cs`, `ExecutionEngine.cs`, `IQueryInterfaces.cs`, all 4 strategy files, `QueryExecutor.cs`, `QueryPlan.cs`

**Risks**: Breaking changes to any code that consumes `DataFrameOperation` directly; fragile tests may not exist for the strategies.

---

### Phase 2: Implement Real Parallel Execution

**Goal**: Wire `ParallelExecutionHelper` utilities into `ParallelExecutionStrategy` so operations actually parallelize.

**Key changes**:
- Replace all no-op `Execute*Parallel` methods with real parallel implementations that use `ParallelExecutionHelper.ProcessInParallelAsync` / `ParallelAggregateAsync`
- Replace `GetAwaiter().GetResult()` sync wrapper with separate sync path to avoid deadlocks
- Add parallelism to data source execution when source supports chunked reading
- Use `IQuerySource.ExecuteAsync` (added in Phase 1) for non-blocking parallel execution

**Files involved**: `ParallelExecutionStrategy.cs`, `ParallelExecutionHelper.cs`, `IQueryInterfaces.cs`

---

### Phase 3: Implement Real Streaming Execution

**Goal**: Make `StreamingExecutionStrategy` actually process data in chunks using the memory budget.

**Key changes**:
- Implement chunk-based data pipeline in `StreamingExecutionStrategy.Execute()`
- Use `calculateChunkSize` output to drive row-batch iteration
- Support partial materialization with chunked `NivaraFrame` concatenation
- Fall back to lazy for operations requiring full data (Sort, GroupBy, Join)
- Connect streaming-aware data sources where `IQuerySource` supports `IAsyncEnumerable<ReadOnlyMemory<IColumn>>`

**Files involved**: `StreamingExecutionStrategy.cs`, `NivaraExecutionContext.cs` (may need buffer/chunk config), `IQueryInterfaces.cs`

---

### Phase 4: Add Diagnostics Integration

**Goal**: Hook `ExecutionDiagnostics` into the execution pipeline so all strategies record timings, warnings, and optimizations.

**Key changes**:
- Have `ExecutionEngine` create/own an `ExecutionDiagnostics` instance per execution
- Pass diagnostics to each strategy and record per-operation timings
- Remove ad-hoc `ReportProgress` duplication across all 4 strategies
- Replace with unified progress reporting + diagnostics recording at the `ExecutionEngine` level
- Record kernel selection from `OperationDiagnostics` / `DiagnosticsTracker` as per AGENTS.md

**Files involved**: `ExecutionEngine.cs`, all 4 strategy files, `ExecutionDiagnostics.cs`, `DiagnosticsTracker.cs`, `OperationDiagnostics.cs`

---

### Phase 5: Clean Up Dead Code & Naming

**Goal**: Remove unused API surface, fix naming inconsistencies, replace magic strings with constants.

**Key changes**:
- Remove unused `IQueryOperation<T>.Transform` and `IQueryOperation<T>.ExecuteAsync`
- Remove unused `IQueryOperation.Strategy` property
- Remove unused `ExecutionEngine.SetOptimizer()` or wire it up properly
- Normalize private method naming: all strategies use PascalCase for private methods (consistent with C# conventions)
- Extract operation type constants (e.g., `OperationType.Filter`, `OperationType.Select`)
- Fix `"Concatenation"` → `"Concatenation"` (or extract to constant)
- Change `QueryExecutor.EstimateExecutionCost` return type from `int` to `long` to match interface
- Remove redundant schema validation in `QueryExecutor.Execute()`

**Files involved**: `IQueryInterfaces.cs`, `DataFrameOperation.cs`, `ExecutionEngine.cs`, all strategy files, `QueryExecutor.cs`, `ColumnEliminationRule.cs`, `QueryPlanVisitor.cs`

---

### Phase 6: Comprehensive Test Coverage

**Goal**: Achieve unit test coverage for all execution components.

**Key changes**:
- Unit tests for `EagerExecutionStrategy` — verify immediate execution, cancellation, empty results
- Unit tests for `LazyExecutionStrategy` — verify deferred execution, progress reporting
- Unit tests for `ParallelExecutionStrategy` — verify parallel dispatch, thread safety
- Unit tests for `StreamingExecutionStrategy` — verify chunking, memory budget, fallback
- Unit tests for `ExecutionEngine` — verify strategy registration, routing, error wrapping
- Unit tests for `ParallelExecutionHelper` — verify chunking, parallel processing, aggregation
- Integration tests for the full execution pipeline with real and mock data sources
- Tests for `QueryPlan` construction, `WithOperation`, `ResultSchema` computation

**Files involved**: All test files under `tests/Nivara.Tests/`

---

## Execution Order & Dependencies

```
Phase 1 ───────────────────────────────────────────────┐
  (prerequisite for all subsequent phases)              │
                                                        ▼
Phase 2 ──┐                                  Phase 4 ──┤
  (parallel impl)                            (diagnostics)
           ├── can run in parallel ──────────┤
Phase 3 ──┘                                  Phase 5 ──┤
  (streaming impl)                           (cleanup)  │
                                                        ▼
                                              Phase 6 ──┤
                                                (tests)  │
                                                        ▼
                                                  Done
```

- **Phase 1 must complete first** — every other phase depends on a unified execution architecture.
- **Phases 2 & 3 can run in parallel** — they touch different strategy files.
- **Phase 4 depends on Phase 1** — diagnostics need the unified routing to hook into.
- **Phase 5 depends on Phase 1** — some dead code removal (e.g., `IQueryOperation.Strategy`) is part of the unification.
- **Phase 6 is ongoing** — characterization tests in Phase 1, then expand coverage in parallel with later phases.

---

## Key Design Decisions (for Phase 1)

1. **Pattern A survives, Pattern B is eliminated.** `IExecutionStrategy` is the single strategy contract. `DataFrameOperation` no longer dispatches by strategy.

2. **Strategy references in operations.** Strategy selection is the `ExecutionEngine`'s responsibility, not the operation's. Operations accept an `IExecutionStrategy` or `ExecutionEngine` reference if they need strategy-aware behavior.

3. **Common boilerplate moves to `ExecutionEngine`** or a base `ExecutionStrategyBase` class. Null checks, cancellation checks, try/catch/wrap with `QueryExecutionException`, and progress reporting should live in one place, not duplicated 4 times.

4. **Async is optional but supported.** `IQuerySource.ExecuteAsync` added for data sources that can stream. `IExecutionStrategy.ExecuteAsync` remains but the default implementation calls sync if not overridden.

5. **Characterization tests first.** Before any refactoring, write tests that capture the current behavior of each strategy. This protects against regressions.
