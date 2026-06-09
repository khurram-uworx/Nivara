# Phase 4: Add Diagnostics Integration

## Purpose

Hook `ExecutionDiagnostics` into the execution pipeline so all strategies record timings, warnings, and optimizations. Replace ad-hoc `ReportProgress` duplication across the four strategy classes with unified diagnostics recording at the `ExecutionEngine` level, and bridge kernel-level `OperationDiagnostics`/`DiagnosticsTracker` data into execution-level diagnostics.

## Suggested Execution Order

1. Task 1: Add `ExecutionDiagnostics` property to `NivaraExecutionContext` (API-shape decision gate)
2. Task 2: Have `ExecutionEngine` create/own a per-execution `ExecutionDiagnostics` instance
3. Task 3: Replace `ReportProgress` calls in strategies with `DiagnosticHelper.ExecuteWithDiagnostics` / `DiagnosticScope`
4. Task 4: Bridge `DiagnosticsTracker` kernel-level data into `ExecutionDiagnostics`
5. Task 5: Tests — new tests + update existing strategy tests to verify `ExecutionDiagnostics` recording
6. Task 6: Remove `ReportProgress` from `ExecutionStrategyBase` after all callers migrate

## Coordination Notes

- **Task 1 is a decision gate** — the property name, nullability, and ownership model must be settled before downstream tasks start.
- **Task 2 and Task 3 can be paired** — once the context carries `ExecutionDiagnostics`, wiring it through `ExecutionEngine` and consuming it in strategies is a contiguous change.
- **Task 4 is independent of Tasks 2–3** — it only touches `DiagnosticsTracker`/`ExecutionDiagnostics` bridge code. It can run in parallel with the strategy-wiring work.
- **Task 5 must wait for Tasks 2–4** — tests need the full pipeline wired.
- **Task 6 must be last** — removing `ReportProgress` breaks all callers.
- **Watch for shared-file conflicts**: `ExecutionStrategyBase.cs`, `NivaraExecutionContext.cs`, and `ExecutionEngine.cs` are touched by multiple tasks.

---

## Task 1: Add `ExecutionDiagnostics` property to `NivaraExecutionContext`

### Priority

High

### Goal

Add an `ExecutionDiagnostics? Diagnostics` property to `NivaraExecutionContext` so strategies and helpers can access the active diagnostics instance without requiring additional constructor parameters.

### Why this exists

Currently there is no way for strategies to receive an `ExecutionDiagnostics` instance. The context object is the natural carrier — it already holds `Progress`, `CancellationToken`, `Strategy`, etc. Adding a diagnostics property avoids threading a separate parameter through `IExecutionStrategy.Execute()`.

### Decision (resolved)

- **Property name: `ExecutionDiagnostics`** (chosen — matches the type name and avoids ambiguity with `ColumnDiagnostics`).
- **Nullability**: `null` by default (backward-compatible); set by `ExecutionEngine` before execution.
- **Ownership**: `ExecutionEngine` creates, owns, and optionally retrieves the instance. The context does not own disposal.

### Scope

- Add `public ExecutionDiagnostics? ExecutionDiagnostics { get; set; }` to `NivaraExecutionContext`
- Update `Clone()` to copy the diagnostics reference
- Update `ToString()` to include diagnostics status

### Constraints

- Must not break existing callers that construct `NivaraExecutionContext` directly (the property defaults to `null`)
- The diagnostics instance must outlive the execution so `GenerateReport()` / `GetSummary()` can be called after disposal

### Suggested implementation path

```csharp
// NivaraExecutionContext addition:
public ExecutionDiagnostics? ExecutionDiagnostics { get; set; }
```

### Acceptance criteria

- New property exists on `NivaraExecutionContext` with getter/setter
- `Clone()` copies the property
- Existing tests pass without modification (default `null` is backward-compatible)
- New unit test verifies set/get round-trip and null default

### Files likely involved

- `src/Nivara/Execution/NivaraExecutionContext.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs` (optional: add helper to set diagnostics)

---

## Task 2: Have `ExecutionEngine` create/own a per-execution `ExecutionDiagnostics` instance

### Priority

High

### Goal

Have `ExecutionEngine.Execute()` and `ExecuteAsync()` create an `ExecutionDiagnostics` instance, set it on the context, call `StartExecution()`/`EndExecution()`, and make the instance available post-execution.

### Why this exists

Without engine-level ownership, each strategy would need to create its own diagnostics, which fragments reporting and defeats the purpose of a unified diagnostics pipeline.

### Scope

- In `ExecutionEngine.Execute(QueryPlan, NivaraExecutionContext)`:
  - After null checks, instantiate `var diagnostics = new ExecutionDiagnostics()`
  - Set `context.ExecutionDiagnostics = diagnostics`
  - Set `diagnostics.ExecutionStrategy = context.Strategy`
  - Set `diagnostics.ParallelismDegree = context.MaxDegreeOfParallelism`
  - Call `diagnostics.StartExecution()` before strategy dispatch
  - Call `diagnostics.EndExecution()` after strategy returns (in `finally` block)
  - Re-throw after recording
- Apply the same pattern to `ExecuteAsync(QueryPlan, NivaraExecutionContext)`
- Consider adding a `public ExecutionDiagnostics? LastDiagnostics { get; }` property on `ExecutionEngine` for post-hoc retrieval (or return from `Execute` — requires interface change; property is cleaner)
- Store optimization info: if `optimizer?.Optimize(plan)` returns a different plan than input, record an `OptimizationApplied` entry

### Constraints

- `ExecutionDiagnostics` is not `IDisposable` today — no disposal concern
- Must not double-wrap if caller already set context.ExecutionDiagnostics (check for null before creating)
- The `LastDiagnostics` property must be thread-safe or documented as last-writer-wins

### Suggested implementation path

```csharp
public NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context)
{
    // ... null checks ...

    var diagnostics = context.ExecutionDiagnostics ?? new ExecutionDiagnostics();
    context.ExecutionDiagnostics = diagnostics;
    diagnostics.ExecutionStrategy = context.Strategy;
    diagnostics.ParallelismDegree = context.MaxDegreeOfParallelism;
    diagnostics.StartExecution();
    lastDiagnostics = diagnostics;

    try
    {
        // ... optimizer, strategy dispatch ...
        var result = strategy.Execute(optimizedPlan, context);
        return result;
    }
    finally
    {
        diagnostics.EndExecution();
    }
}
```

### Acceptance criteria

- `ExecutionEngine.Execute` creates an `ExecutionDiagnostics` when context has none
- `StartExecution()` / `EndExecution()` are called bracketing strategy dispatch
- Strategy and parallelism degree are recorded on diagnostics
- Optimizer changes are recorded as `OptimizationApplied` entries
- `LastDiagnostics` property returns the most recent diagnostics
- Unit tests verify diagnostics are populated after `Execute` returns

### Files likely involved

- `src/Nivara/Execution/ExecutionEngine.cs`
- `tests/Nivara.Tests/Execution/ExecutionEngineTests.cs`

---

## Task 3: Replace `ReportProgress` calls in strategies with `DiagnosticHelper.ExecuteWithDiagnostics` / `DiagnosticScope`

### Priority

High

### Goal

Replace all ad-hoc `ReportProgress(context, ...)` calls across the four strategy classes with unified diagnostics recording using `DiagnosticHelper.ExecuteWithDiagnostics<T>` or `DiagnosticScope`. Keep `IProgress<ExecutionProgress>` reporting but unify it with timing/memory tracking.

### Why this exists

All four strategies independently call `ExecutionStrategyBase.ReportProgress()` — duplication of string formatting, no timing or memory tracking, no connection to `ExecutionDiagnostics`. `DiagnosticHelper` already provides the wrapper we need.

### Scope

- **EagerExecutionStrategy** `ExecuteCore`:
  - Wrap source execution in `DiagnosticHelper.ExecuteWithDiagnostics` with operation type `"SourceExecute"`
  - Wrap each `operation.Execute(currentColumns)` in `DiagnosticHelper.ExecuteWithDiagnostics` with operation type `operation.OperationType`
  - Remove direct `ReportProgress` calls; let `ExecuteWithDiagnostics` record timings
  - Use `IProgress<ExecutionProgress>` separately for UI progress (can be kept; just no longer the sole recording mechanism)
- **LazyExecutionStrategy** `ExecuteCore`:
  - Wrap `executor.Execute(plan)` in `DiagnosticHelper.ExecuteWithDiagnostics`
- **ParallelExecutionStrategy** `ExecuteCore`:
  - Wrap source execution and each parallel operation dispatch
  - For parallel sub-operations (`executeSortParallelSync`, `executeFilterParallelSync`, etc.), use `DiagnosticScope` around each chunk since `ExecuteWithDiagnostics` would capture per-chunk timing inside the `Parallel.ForEach` lambda
- **StreamingExecutionStrategy** `ExecuteCore`:
  - Wrap chunk processing loop body in `DiagnosticScope` per chunk
  - Wrap the entire streaming execution in an outer `DiagnosticScope`
- **ExecuteCoreAsync** variants: use `DiagnosticHelper.ExecuteWithDiagnosticsAsync<T>` or `DiagnosticScope`

### Constraints

- `IProgress<ExecutionProgress>` must remain available for external consumers — only remove the internal `ReportProgress` calls if progress is redundant with diagnostics
- Do not break progress-reporting tests in strategy test files (they test `tracker.Reports.Count > 0`)
- `Parallel.ForEach` lambdas cannot use `ref` structs or by-ref captures — `DiagnosticScope` is a class, so it's safe

### Suggested implementation path

```csharp
// EagerExecutionStrategy.ExecuteCore — before:
ReportProgress(context, "Starting eager execution", 0, plan.Operations.Count + 1);
var currentColumns = plan.Source.Execute();
ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);
for (int i = 0; i < plan.Operations.Count; i++)
{
    var operation = plan.Operations[i];
    currentColumns = operation.Execute(currentColumns);
    ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
}

// EagerExecutionStrategy.ExecuteCore — after:
using var overallScope = DiagnosticHelper.CreateScope(context.ExecutionDiagnostics!, "EagerExecution");
var currentColumns = DiagnosticHelper.ExecuteWithDiagnostics(
    context.ExecutionDiagnostics!, "SourceExecute", () => plan.Source.Execute());
for (int i = 0; i < plan.Operations.Count; i++)
{
    var operation = plan.Operations[i];
    var capturedOp = operation;
    currentColumns = DiagnosticHelper.ExecuteWithDiagnostics(
        context.ExecutionDiagnostics!, operation.OperationType,
        () => capturedOp.Execute(currentColumns));
}
```

### Acceptance criteria

- All 4 strategies record operation timings via `DiagnosticHelper` instead of bare `ReportProgress`
- Existing progress-reporting tests continue to pass (the `IProgress<ExecutionProgress>` path is preserved)
- `ExecutionDiagnostics.OperationTimings` is populated after any strategy executes
- Timing granularity is per-operation (not per-chunk for chunked operations — per-chunk is a `DiagnosticScope` refinement)
- No `ReportProgress` calls remain in strategy files

### Files likely involved

- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Execution/ExecutionStrategyBase.cs` (will remove `ReportProgress` later)
- `tests/Nivara.Tests/Execution/EagerExecutionStrategyTests.cs` (update progress tests)
- `tests/Nivara.Tests/Execution/ParallelExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

---

## Task 4: Bridge `DiagnosticsTracker` kernel-level data into `ExecutionDiagnostics`

### Priority

Medium

### Goal

Connect the static `DiagnosticsTracker` (which records kernel-level `OperationDiagnostics` from column operations) to the per-execution `ExecutionDiagnostics` instance, so a complete picture spans both execution-level and kernel-level diagnostics.

### Why this exists

`DiagnosticsTracker` is a static global — it records kernel selection, vectorization decisions, and null-handling info from `NivaraColumn` arithmetic/comparison operations. `ExecutionDiagnostics` is per-execution and records operation timings, warnings, and optimizations. Today there is no bridge between them. For a given query execution, the user should be able to see both levels in one report.

### Scope

- Add a `public void ImportFromDiagnosticsTracker()` method (or constructor flag) to `ExecutionDiagnostics` that:
  - Calls `DiagnosticsTracker.GetRecordedOperations()` to snapshot current operations
  - Records each as an `OptimizationApplied` entry (kernel selection is an optimization decision) or a dedicated `OperationDiagnostics` list
  - Clears `DiagnosticsTracker` after import to avoid double-counting across executions
- Alternatively, add a new `public List<OperationDiagnostics> KernelOperations { get; }` property to `ExecutionDiagnostics` and populate it directly
- In `ExecutionEngine`, after strategy execution completes, call the bridge method to pull kernel-level data into the execution diagnostics
- Add `KernelOperations` to the `GenerateReport()` output (a new section after Operation Breakdown)

### Constraints

- `DiagnosticsTracker` is static and thread-safe — snapshot must be atomic relative to the tracker's lock
- Import should be idempotent: calling `ImportFromDiagnosticsTracker` twice on the same `ExecutionDiagnostics` should not duplicate entries (clear tracker after import)
- Must not break existing tests that rely on `DiagnosticsTracker` isolation (they call `ClearRecordedOperations()` in teardown)

### Suggested implementation path

```csharp
// In ExecutionDiagnostics:
public List<OperationDiagnostics> KernelOperations { get; } = new();

public void ImportFromDiagnosticsTracker()
{
    var recorded = DiagnosticsTracker.GetRecordedOperations();
    KernelOperations.AddRange(recorded);
    DiagnosticsTracker.ClearRecordedOperations();
}

// In ExecutionEngine.Execute, before return:
if (context.ExecutionDiagnostics != null)
    context.ExecutionDiagnostics.ImportFromDiagnosticsTracker();
```

### Acceptance criteria

- `ExecutionDiagnostics.KernelOperations` is populated after an execution that triggered `DiagnosticsTracker`-tracked column operations
- `DiagnosticsTracker` is cleared after import (no double-counting across executions)
- `GenerateReport()` includes kernel operations in its output
- Existing `DiagnosticsTracker` tests continue to pass independently

### Files likely involved

- `src/Nivara/Diagnostics/ExecutionDiagnostics.cs`
- `src/Nivara/Execution/ExecutionEngine.cs`
- `src/Nivara/Diagnostics/OperationDiagnostics.cs` (minimal — may need to make `RecordOperation` accessible or use existing `GetRecordedOperations`)
- `tests/Nivara.Tests/Diagnostics/ExecutionDiagnosticsTests.cs`

---

## Task 5: Tests — verify diagnostics recording through each strategy

### Priority

High

### Goal

Add and update tests ensuring that `ExecutionDiagnostics` is correctly populated through all execution paths: sync/async for each strategy, including error paths and cancellation.

### Why this exists

The existing strategy tests verify `ProgressTracker` reports but have zero coverage of `ExecutionDiagnostics` recording. Without tests, the integration in Tasks 2–4 is untrusted and will regress.

### Scope

- **ExecutionEngine tests** (`ExecutionEngineTests.cs`):
  - Add test: `Execute_WithDiagnostics_PopulatesOperationTimings()` — after calling `Execute`, verify `LastDiagnostics` has timings, strategy, and parallelism
  - Add test: `ExecuteAsync_WithDiagnostics_PopulatesOperationTimings()` — same for async path
  - Add test: `Execute_DiagnosticsRecordOptimization_WhenOptimizerChangesPlan()`
  - Add test: `Execute_DiagnosticsRecordWarning_OnFailure()` — verify warnings recorded on exception

- **EagerExecutionStrategy tests** (`EagerExecutionStrategyTests.cs`):
  - Add test: `Execute_DiagnosticsRecordsPerOperationTiming()` — verify `OperationTimings` has one entry per operation
  - Update existing `Execute_ProgressReporting_FiresCorrectly` to also assert `ExecutionDiagnostics` is populated (requires setting `context.ExecutionDiagnostics`)

- **ParallelExecutionStrategy tests** (`ParallelExecutionStrategyTests.cs`):
  - Add test: `Execute_DiagnosticsRecordsParallelismDegree()` — verify `ParallelismDegree` matches context setting
  - Add test: `Execute_DiagnosticsPerOperationTimingCount_MatchOperationCount()`

- **StreamingExecutionStrategy tests** (`StreamingExecutionStrategyTests.cs`):
  - Add test: `Execute_DiagnosticsRecordsChunkCount()` — verify number of chunks processed is recorded (via scope)
  - Add test: `Execute_DiagnosticsRecording_NonStreamableFallback()` — verify diagnostics still recorded when falling back to Lazy

- **ExecutionDiagnostics unit tests** (`ExecutionDiagnosticsTests.cs`):
  - Add test: `ImportFromDiagnosticsTracker_ImportsAndClears()` — verify bridge behavior
  - Add test: `GenerateReport_IncludesKernelOperations_WhenPresent()` — verify report section

### Constraints

- Tests must not depend on real column arithmetic triggering `DiagnosticsTracker` (use stub operations)
- For bridge tests, enable `DiagnosticsTracker.IsEnabled = true` in test setup, disable and clear in teardown
- Follow the existing test patterns in `ExecutionTestHelpers.cs` (use `StubQuerySource`, `StubQueryOperation`)

### Acceptance criteria

- At least 12 new test methods across all affected test files
- All existing tests continue to pass
- Code coverage for `ExecutionDiagnostics` integration paths reaches >80%

### Files likely involved

- `tests/Nivara.Tests/Execution/ExecutionEngineTests.cs`
- `tests/Nivara.Tests/Execution/EagerExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/LazyExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/ParallelExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Diagnostics/ExecutionDiagnosticsTests.cs`

---

## Task 6: Remove `ReportProgress` from `ExecutionStrategyBase`

### Priority

Low

### Goal

Remove the `ReportProgress` protected static method from `ExecutionStrategyBase` after all strategy callers have migrated to `DiagnosticHelper`-based recording.

### Why this exists

The method exists only as a shared helper for the four strategies. Once all callers are ported (Task 3), keeping it around is dead code that could mislead future strategy implementations.

### Scope

- Remove `protected static void ReportProgress(...)` from `ExecutionStrategyBase`
- Verify no remaining callers via grep across the codebase
- Remove unused `using Nivara.Exceptions` import from `ExecutionStrategyBase.cs` if it becomes unused

### Constraints

- Must be done after Task 3 is fully verified (all `ReportProgress` calls removed from strategies)

### Acceptance criteria

- `ReportProgress` method no longer exists in `ExecutionStrategyBase`
- Build succeeds
- All tests pass

### Files likely involved

- `src/Nivara/Execution/ExecutionStrategyBase.cs`

---

## Suggested Agent Handout Batches

### Batch A: decision-gate + infrastructure

- Task 1: Add `ExecutionDiagnostics` to `NivaraExecutionContext`
- Task 2: `ExecutionEngine` creates/owns per-execution diagnostics

### Batch B: strategy wiring + bridge

- Task 3: Replace `ReportProgress` in all 4 strategies
- Task 4: Bridge `DiagnosticsTracker` into `ExecutionDiagnostics`

### Batch C: tests + cleanup

- Task 5: Add/update tests for diagnostics integration
- Task 6: Remove `ReportProgress` from `ExecutionStrategyBase`

---

## Final Checklist

- [ ] every task has a clear owner-sized scope
- [ ] every task has acceptance criteria
- [ ] decision-gate tasks are clearly marked (Task 1)
- [ ] likely files are listed to reduce agent search time
- [ ] execution order reflects real dependencies (Tasks 2–4 depend on Task 1; Task 5 depends on 2–4; Task 6 depends on 3)
- [ ] `NivaraExecutionContext.Diagnostics` defaults to `null` — backward-compatible
- [ ] `IProgress<ExecutionProgress>` path preserved for external consumers
- [ ] `DiagnosticsTracker` static state managed correctly (enable/disable/clear in tests)
