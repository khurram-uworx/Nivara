# Plan: Wire budget tracking into streaming execution path

**Issue:** [#323](https://github.com/khurram-uworx/Nivara/issues/323)
**Branch:** `khurram/323`

## Problem

`StreamingExecutionStrategy` processes data in chunks but does not track actual memory usage during chunk accumulation. `context.MemoryBudget` only controls chunk size and channel capacity — peak memory can far exceed the budget when chunks accumulate before boundary operations (Sort, GroupBy, Join, etc.).

`StreamingBufferManager` (in `Nivara.Extensions`) is a byte-buffer I/O manager and lives in the wrong assembly to be referenced from core. We need equivalent budget-tracking logic in core.

## Approach

1. Make `NivaraFrame.estimateFrameMemoryUsage()` and `estimateColumnMemoryUsage()` `internal` so the strategy can use them.
2. Create `StreamingBudgetTracker` in core (`Nivara.Execution`) — lightweight class that tracks accumulated frame memory via `NivaraFrame.estimateFrameMemoryUsage()`, enforces a configurable budget multiplier (default 2×), and emits a single `PerformanceWarning` when exceeded.
3. Wire the tracker into all three execution paths in `StreamingExecutionStrategy`:
   - `ExecuteCore` (sync)
   - `executeCoreInternalAsync` (async channel)
   - `StreamChunksAsync` (replaces the rudimentary row-count tracking)
4. Add unit tests for `StreamingBudgetTracker` and integration tests for budget warnings across all paths.

## Files to change

| File | Change |
|------|--------|
| `src/Nivara/NivaraFrame.cs:25,299` | `estimateColumnMemoryUsage`, `estimateFrameMemoryUsage` → `internal` |
| `src/Nivara/Execution/StreamingBudgetTracker.cs` | **New** — budget tracker class |
| `src/Nivara/Execution/StreamingExecutionStrategy.cs` | Wire tracker into all 3 paths |
| `tests/Nivara.Tests/Execution/StreamingBudgetTrackerTests.cs` | **New** — tracker unit tests |
| `tests/Nivara.Tests/Execution/StreamingBudgetDiagnosticTests.cs` | Add sync/async budget warning tests |

## Blast radius

- `StreamingExecutionStrategy` is used by `ExecutionEngine` (registered as `ExecutionStrategy.Streaming`). All callers go through `ExecutionEngine.Execute`/`ExecuteAsync`.
- The tracker is purely additive — it emits warnings via the existing `ExecutionDiagnostics` path. No behavior changes for callers that don't set `ExecutionDiagnostics`.
- `estimateFrameMemoryUsage` visibility change is internal-only (same assembly).

## Planned commits

1. `docs: plan budget tracking in TODO.md`
2. `refactor: expose frame memory estimation as internal`
3. `feat: add StreamingBudgetTracker for chunk accumulation budget enforcement`
4. `feat: wire budget tracker into StreamingExecutionStrategy sync/async paths`
5. `feat: upgrade StreamChunksAsync budget tracking to use StreamingBudgetTracker`
6. `test: add StreamingBudgetTracker unit and integration tests`

## Verification

- `dotnet build Nivara.slnx` — clean build
- `dotnet test` — all existing + new tests pass (ask before running)

## GitHub issues log

- (none yet)
