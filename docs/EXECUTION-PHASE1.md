# Execution Engine — Phase 1: Unify Execution Patterns

## Purpose

Eliminate the duplicated strategy dispatch in `DataFrameOperation` (Pattern B), making `IExecutionStrategy` / `ExecutionEngine` the single execution routing mechanism. This is the prerequisite for all subsequent phases.

## Context

The codebase has **two overlapping execution patterns**:

- **Pattern A**: `ExecutionEngine` → `IExecutionStrategy` → `QueryExecutor` → operation pipeline. This is the real path used at runtime.
- **Pattern B**: `DataFrameOperation.Execute(NivaraExecutionContext)` switches on `context.Strategy` and dispatches to virtual `ExecuteLazy`/`ExecuteEager`/`ExecuteStreaming`/`ExecuteParallel` — all four just call `this.Execute()`.

Pattern B is **dead code**: no operation in the codebase extends `DataFrameOperation`. All 10+ operations (`FilterOperation`, `SelectOperation`, `SortOperation`, etc.) implement `IQueryOperation` directly.

## Suggested Execution Order

1. **Task 1**: Write characterization tests for existing behavior (protects against regressions)
2. **Task 2**: Extract shared strategy boilerplate into `ExecutionEngine` or base class
3. **Task 3**: Remove Pattern B strategy dispatch from `DataFrameOperation`
4. **Task 4**: Remove dead API surface from `IQueryInterfaces`
5. **Task 5**: Add `IQuerySource.ExecuteAsync` for async data sources
6. **Task 6**: Normalize naming conventions across strategies

## Coordination Notes

- **Task 1 is a decision gate**: No refactoring in tasks 2–6 should begin until characterization tests pass.
- **Tasks 2–6 can run sequentially** since they touch overlapping files.
- **File conflicts** are high: `DataFrameOperation.cs`, `ExecutionEngine.cs`, all 4 strategy files, and `IQueryInterfaces.cs` are touched by multiple tasks. Use a single branch/agent for Phase 1.
- `DataFrameOperation` is confirmed dead (no subclasses, no external references) — safe to modify aggressively.

---

## Task 1: Write Characterization Tests for All Four Strategy Classes

### Priority

High

### Goal

Capture the current behavior of `EagerExecutionStrategy`, `LazyExecutionStrategy`, `ParallelExecutionStrategy`, and `StreamingExecutionStrategy` in automated tests before any refactoring.

### Why this exists

Zero tests exist for any strategy class. Without characterization tests, refactoring carries high regression risk. These tests will be expanded in Phase 6 but need to exist now.

### Scope

- Write tests for `EagerExecutionStrategy`:
  - Executes operations immediately in sequence
  - Progress reporting fires correctly
  - Cancellation stops execution mid-pipeline
  - Throws on null plan/context
  - Throws on empty result columns
  - `ValidatePlan` delegates to `QueryExecutor`
  - `EstimateExecutionCost` returns expected costs
- Write tests for `LazyExecutionStrategy`:
  - Delegates to `QueryExecutor.Execute`
  - Progress reporting fires
  - Cancellation works
  - `ExecuteAsync` wraps sync on background thread
  - `ValidatePlan` delegates correctly
- Write tests for `ParallelExecutionStrategy`:
  - Execute calls `GetAwaiter().GetResult()` (capture this bad behavior for later fix)
  - Parallel dispatch methods exist but are no-ops (capture current state)
  - `ValidatePlan` validates both plan and parallelism config
  - Degenerate configs are rejected
- Write tests for `StreamingExecutionStrategy`:
  - Falls through to `executor.Execute(plan)` (same as lazy)
  - `isSuitableForStreaming` correctly identifies streamable/non-streamable ops
  - Falls back to lazy for non-streamable plans
  - Chunk size calculation respects memory budget bounds
- Write tests for `ExecutionEngine`:
  - Routes to correct strategy based on `context.Strategy`
  - Applies optimizer if available
  - Wraps non-`QueryExecutionException` in `QueryExecutionException`
  - `ValidatePlan` and `EstimateExecutionCost` delegate to strategy
  - `RegisterStrategy` adds/updates strategies
  - Default strategies are registered at construction

### Constraints

- Tests must use mocking for `IQuerySource`, `IQueryOperation`, `QueryPlan` to avoid file I/O dependencies
- Tests should NOT depend on the real strategy behavior being correct — they capture current behavior exactly

### Suggested implementation path

1. Create a `Strategies` test class (or one per strategy) in `tests/Nivara.Tests/` alongside `ExecutionContextTests.cs`
2. Use `NSubstitute` or manual stubs for `IQueryOperation`, `IQuerySource`, `QueryPlan`
3. Extract shared test helpers (e.g., `CreateTestPlan()`, `CreateTestContext()`)
4. Run tests to confirm they pass with current implementation

### Acceptance criteria

- Minimum 5–7 tests per strategy (20–28 total)
- All tests pass against the current (unrefactored) code
- Test failures after refactoring correctly identify regressions

### Files likely involved

- `tests/Nivara.Tests/EagerExecutionStrategyTests.cs`
- `tests/Nivara.Tests/LazyExecutionStrategyTests.cs`
- `tests/Nivara.Tests/ParallelExecutionStrategyTests.cs`
- `tests/Nivara.Tests/StreamingExecutionStrategyTests.cs`
- `tests/Nivara.Tests/ExecutionEngineTests.cs`
- `tests/Nivara.Tests/TestHelpers/ExecutionTestHelpers.cs` (optional shared helpers)

---

## Task 2: Extract Shared Boilerplate from Strategy Classes

### Priority

High

### Goal

Eliminate the ~50 lines of duplicated boilerplate (null checks, cancellation, try/catch, progress reporting) across all 4 strategy implementations.

### Why this exists

Each strategy duplicates:
```csharp
if (plan == null) throw new ArgumentNullException(nameof(plan));
if (context == null) throw new ArgumentNullException(nameof(context));
context.CancellationToken.ThrowIfCancellationRequested();
// try/catch with QueryExecutionException wrapping
// progress reporting
```

This makes changes risky (4 places to update) and obscures the intent of each strategy.

### Scope

- Create an `abstract` base class `ExecutionStrategyBase` that implements `IExecutionStrategy`
- Move common null validation, cancellation check, and exception wrapping into the base
- Provide a `protected static ReportProgress` helper
- Provide a template method pattern:
  ```csharp
  // Base class
  public NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context)
  {
      ValidateArgs(plan, context);
      context.CancellationToken.ThrowIfCancellationRequested();
      try { return ExecuteCore(plan, context); }
      catch (OperationCanceledException) { throw; }
      catch (Exception ex) when (ex is not QueryExecutionException)
      { throw new QueryExecutionException($"...", ex); }
  }
  protected abstract NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context);
  ```
- Do the same for `ExecuteAsync` with a virtual `ExecuteCoreAsync` that defaults to `Task.Run(() => ExecuteCore(...))`
- Update all 4 strategies to extend `ExecutionStrategyBase` and implement only `ExecuteCore`/`ExecuteCoreAsync`

### Constraints

- Must preserve existing public API surface (strategies are `internal sealed` but implement the public `IExecutionStrategy` interface)
- `ExecutionEngine` should not need changes — it already calls `IExecutionStrategy.Execute(plan, context)`

### Acceptance criteria

- All characterization tests from Task 1 still pass
- Each strategy's `ExecuteCore` contains only its unique logic (no boilerplate)
- `EagerExecutionStrategy.ExecuteCore` is < 30 lines (down from ~60)
- `LazyExecutionStrategy.ExecuteCore` is < 10 lines (down from ~30)

### Files likely involved

- `src/Nivara/Execution/ExecutionStrategyBase.cs` (new file)
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

---

## Task 3: Remove Pattern B Strategy Dispatch from DataFrameOperation

### Priority

High

### Goal

Remove the duplicate strategy dispatch in `DataFrameOperation` and simplify it to a single `Execute()` abstract method. Operations should not make strategy decisions — that's the `ExecutionEngine`'s job.

### Why this exists

`DataFrameOperation` maintains a complete parallel strategy dispatch:
```csharp
public virtual NivaraFrame Execute(NivaraExecutionContext context)
{
    return context.Strategy switch
    {
        Lazy => ExecuteLazy(context),     // → this.Execute()
        Eager => ExecuteEager(context),    // → this.Execute()
        Streaming => ExecuteStreaming(context), // → this.Execute()
        Parallel => ExecuteParallel(context), // → this.Execute()
    };
}
```

All four virtual methods fall through to `this.Execute()`. No subclass exists or overrides them. This is dead code that confuses the architecture.

### Scope

- Remove `Execute(NivaraExecutionContext context)` strategy-switch method
- Remove all four protected virtual strategy methods (`ExecuteLazy`, `ExecuteEager`, `ExecuteStreaming`, `ExecuteParallel`)
- Remove `ValidateExecutionContext` (duplicated in `ExecutionStrategyBase`)
- Remove `ReportProgress` static helper (moved to `ExecutionStrategyBase`)
- Remove `ThrowIfCancellationRequested` static helper (moved to `ExecutionStrategyBase`)
- Simplify `DataFrameOperation` to hold a `QueryPlan` and expose a single `abstract NivaraFrame Execute()` method
- Remove `Strategy` property override (strategy is now the engine's concern, not the operation's)
- Update `TransformOperation<TResult>` to remove its `Strategy` property dependency

### Constraints

- `DataFrameOperation` is `public abstract` with `protected` constructor — keep the class, change its shape
- `TransformOperation<TResult>` is `internal sealed` — keep it working for any existing consumers
- Confirm no external code references `DataFrameOperation` or its removed members (confirmed: none found)

### Acceptance criteria

- `DataFrameOperation` has no strategy-related members (no `Execute(NivaraExecutionContext)`, no `ExecuteLazy`/`ExecuteEager`/`ExecuteStreaming`/`ExecuteParallel`, no `ValidateExecutionContext`, no `Strategy` property)
- `DataFrameOperation` exposes only `QueryPlan Plan`, `string OperationType`, and `NivaraFrame Execute()` (with `IQueryOperation<T>` interface members)
- All characterization tests still pass
- Existing code compiles without errors

### Files likely involved

- `src/Nivara/Helpers/DataFrameOperation.cs`
- `src/Nivara/Helpers/TransformOperation.cs` (if refactored out of DataFrameOperation.cs)
- `src/Nivara/Query/IQueryInterfaces.cs` (if `IQueryOperation<T>.Strategy` is removed)

---

## Task 4: Remove Dead API Surface from IQueryInterfaces

### Priority

High

### Goal

Remove unused interface members from `IQueryOperation<T>` and `IQueryOperation` to eliminate dead code and clarify the contract.

### Why this exists

Three members across the query interfaces are defined but never consumed:

| Member | Defined In | Use |
|--------|-----------|-----|
| `IQueryOperation<T>.Transform<TResult>()` | `IQueryOperation<T>` | Never called |
| `IQueryOperation<T>.ExecuteAsync(CancellationToken)` | `IQueryOperation<T>` | Never called by any strategy or executor |
| `IQueryOperation.Strategy` | `IQueryOperation<T>` (via property) | Never read by `QueryExecutor` or any strategy |

### Scope

- Remove `IQueryOperation<T>.Transform<TResult>(Func<T, TResult>)` from the interface
- Remove `IQueryOperation<T>.ExecuteAsync(CancellationToken)` from the interface
- Remove `IQueryOperation.Strategy` property — strategy selection is the engine's concern
- Update `DataFrameOperation` to remove overridden members
- Update `TransformOperation<TResult>` to remove chaining capability based on `Transform`
- Update `IQueryOperation<T>` to contain only `QueryPlan Plan` (and remove `Strategy`)
- Consider merging `IQueryOperation<T>` and `IQueryOperation` if they converge enough (optional — can defer)

### Constraints

- `IQueryOperation` (non-generic) is the interface used by `QueryPlan.Operations` and `QueryExecutor` — do not change its shape unless necessary
- Check for any reflection-based usage that might reference removed members
- `DataFrameOperation` implements `IQueryOperation<NivaraFrame>` — ensure it still satisfies the trimmed interface

### Acceptance criteria

- `IQueryOperation<T>` has exactly: `QueryPlan Plan`
- `IQueryOperation` has exactly: `OperationType`, `TransformSchema`, `Execute`
- All code compiles
- No runtime reflection breaks
- All characterization tests pass

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Helpers/DataFrameOperation.cs`
- `src/Nivara/Helpers/TransformOperation.cs` (in DataFrameOperation.cs)

---

## Task 5: Add IQuerySource.ExecuteAsync for Async Data Sources

### Priority

Medium

### Goal

Add an asynchronous execution method to `IQuerySource` so that async-capable data sources can be consumed without blocking threads via `Task.Run`.

### Why this exists

`IQuerySource` currently has only `Execute()` (synchronous). All strategies' `ExecuteAsync` implementations wrap the sync call in `Task.Run`, which is an anti-pattern for I/O-bound data sources (network files, cloud storage, databases). Adding async at the source level enables truly non-blocking execution.

### Scope

- Add `Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default)` to `IQuerySource`
- Default implementation calls `Task.Run(() => Execute(), cancellationToken)` for backward compatibility
- Update `ExecutionStrategyBase.ExecuteCoreAsync` to call `plan.Source.ExecuteAsync()` instead of `Task.Run(() => plan.Source.Execute())`
- Update `EagerExecutionStrategy.ExecuteCoreAsync` to use `plan.Source.ExecuteAsync()` for the data source
- Update `QueryExecutor` to expose an async path that calls `plan.Source.ExecuteAsync()`

### Constraints

- Must not break existing `IQuerySource` implementations (everything that compiles today must still compile)
- The default `ExecuteAsync` should work for synchronous sources without modification
- Do not force all sources to implement async — the default `Task.Run` fallback is acceptable

### Acceptance criteria

- `IQuerySource` has `ExecuteAsync` with a default non-breaking implementation
- `ExecutionStrategyBase.ExecuteCoreAsync` uses `Source.ExecuteAsync()` for data source
- Existing sync-only sources compile without changes
- All characterization tests pass

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ExecutionStrategyBase.cs`
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Query/QueryExecutor.cs`

---

## Task 6: Normalize Naming Conventions Across Strategy Classes

### Priority

Medium

### Goal

Fix inconsistent naming across the four strategy implementations to follow C# PascalCase conventions for all methods.

### Why this exists

Current state:

| Method | Strategy | Convention |
|--------|----------|-----------|
| `ReportProgress` | `EagerExecutionStrategy` | PascalCase ✅ |
| `ReportProgress` | `LazyExecutionStrategy` | PascalCase ✅ |
| `reportProgress` | `ParallelExecutionStrategy` | camelCase ❌ |
| `reportProgress` | `StreamingExecutionStrategy` | camelCase ❌ |
| `isSuitableForStreaming` | `StreamingExecutionStrategy` | camelCase ❌ |
| `calculateChunkSize` | `StreamingExecutionStrategy` | camelCase ❌ |
| `isParallelizable` | `ParallelExecutionStrategy` | camelCase ❌ |
| `shouldUseParallelism` | `ParallelExecutionStrategy` | camelCase ❌ |
| `validateParallelismConfiguration` | `ParallelExecutionStrategy` | camelCase ❌ |

This is a minor issue but violates C# conventions and creates cognitive friction.

### Scope

- Rename `reportProgress` → `ReportProgress` in `ParallelExecutionStrategy`
- Rename `reportProgress` → `ReportProgress` in `StreamingExecutionStrategy`
- Rename `isSuitableForStreaming` → `IsSuitableForStreaming` in `StreamingExecutionStrategy`
- Rename `calculateChunkSize` → `CalculateChunkSize` in `StreamingExecutionStrategy`
- Rename `isParallelizable` → `IsParallelizable` in `ParallelExecutionStrategy`
- Rename `shouldUseParallelism` → `ShouldUseParallelism` in `ParallelExecutionStrategy`
- Rename `validateParallelismConfiguration` → `ValidateParallelismConfiguration` in `ParallelExecutionStrategy`
- Note: `EagerExecutionStrategy.ReportProgress` and `LazyExecutionStrategy.ReportProgress` are already PascalCase — keep them

### Constraints

- Methods are `private` or `static` — no external API impact
- Update all call sites within the same file (they're called locally)

### Acceptance criteria

- Every method in all 4 strategy files follows PascalCase
- All characterization tests pass
- Code compiles without warnings

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

---

## Suggested Agent Handout Batches

### Batch A: decision-critical (Task 1)

- Task 1: Characterization tests

This batch MUST complete and pass before any work on Batches B–D begins.

### Batch B: core refactoring (Tasks 2–3)

- Task 2: Extract shared boilerplate into `ExecutionStrategyBase`
- Task 3: Remove Pattern B strategy dispatch from `DataFrameOperation`

These share files with Task 4; do them together.

### Batch C: API cleanup (Tasks 4–5)

- Task 4: Remove dead API from `IQueryInterfaces`
- Task 5: Add `IQuerySource.ExecuteAsync`

Can run after Batch B since they touch `IQueryInterfaces.cs` and `DataFrameOperation.cs`.

### Batch D: polish (Task 6)

- Task 6: Normalize naming conventions

Low risk, can run last.

---

## Final Checklist

- [ ] Every task has a clear owner-sized scope
- [ ] Every task has acceptance criteria
- [ ] Task 1 (characterization tests) is clearly marked as a decision gate — nothing starts until it passes
- [ ] Likely files are listed to reduce agent search time
- [ ] Execution order reflects real dependencies (Task 1 → Tasks 2–3 → Tasks 4–5 → Task 6)
- [ ] `DataFrameOperation` is confirmed dead code — safe for aggressive simplification
