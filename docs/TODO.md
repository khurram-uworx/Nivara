# Plan: Fix #268 and #269 — sync/async alignment bugs

Branch: `khurram/bugs` (off `main` @ `49d6b40`)

## Problems

### #268 — QueryFrame: `DisposeAsync()` disposes source, sync `Dispose()` does not (inconsistent)

- `src/Nivara/Query/QueryFrame.cs:1034` — `Dispose()` only untracks + sets `disposed`; never releases the underlying `IQuerySource`.
- `DisposeAsync()` (line 1044) releases the source only when it implements `IAsyncDisposable` — and **no `IQuerySource` implementation does** (verified repo-wide), so async also effectively never releases it today.
- The abandoned-resource cleanup action (lines 33-44) *does* call `source?.Dispose()` — so GC-abandonment releases the source but explicit disposal does not. Backwards.
- Intended design was documented in `docs/PHASE4.md:137`: `await asyncDisposable.DisposeAsync()` **alongside existing `source.Dispose()`**; the implementation dropped the sync dispose.
- Real impact: disposing a lazy CSV/Parquet `QueryFrame` without `Collect()` can leak the persistent chunk-reader file handle.

**Fix:** make both paths release the source. `Dispose()` calls `source.Dispose()` (swallow errors, mirroring abandoned-cleanup). `DisposeAsync()` keeps the `is IAsyncDisposable` check for forward-compat and falls back to `source.Dispose()` otherwise.

**Blast radius:** `QueryFrame.Dispose`/`DisposeAsync` callers — fluent chains share one `source`, so disposing any node releases the shared source (same semantics abandoned-cleanup already applies). All source `Dispose()` impls are idempotent (`CsvLazySource` nulls chunk reader, `MemoryQuerySource`/`JsonLazySource`/`ParquetLazySource` guard with `disposed`). Covered by `QueryFrameDisposal_ShouldCleanupLazyQueryResources` (ResourceManagementPropertyTests), `QueryFrame_DisposeAsync_ReleasesResources` + `CollectAsync_EmptyDataFrame_ReturnsFinally` (AsyncStreamingTests).

### #269 — Streaming: sync `ExecuteCore` vs async `ExecuteCoreAsync` diverge on non-streamable plans

- Sync `ExecuteCore` (`src/Nivara/Execution/StreamingExecutionStrategy.cs:92`) uses `!isSuitableForStreaming(plan)` → falls back entirely to Lazy for Sort/GroupBy/Join/Distinct/etc.
- Async `ExecuteCoreAsync` (line 159) only falls back on window expressions; non-streamable ops flow through `PartitionAtNonStreamableOps` flush-concatenate-resume.
- The async partition path is deliberate: PHASE4 resolved decision #1 ("filter before sort on 10GB CSV"), covered by `StreamingStrategy_BoundaryOperation_FlushesAndResumes` (AsyncStreamingTests.cs:438). Sync was never ported.
- Secondary sync gap: `ExecuteCore` never disposes intermediate `chunkFrames` (async does, line 243).

**Fix:** port flush-concatenate-resume into sync `ExecuteCore` so both paths behave identically. Window-expression plans still fall back to Lazy.

**Blast radius:** `StreamingExecutionStrategy.Execute` callers (tests + `ExecutionEngine` when `context.Strategy == Streaming`). `ValidatePlan`/`EstimateExecutionCost`/`StreamChunksAsync` keep using `isSuitableForStreaming` unchanged (StreamChunksAsync's single-frame fallback is its documented contract). Sync "FallsBackToLazy" tests (`Execute_NonStreamableOpInChunkedSource_FallsBackToLazy`, `Execute_SortInChunkedPlan_FallsBackToLazy`, `Execute_JoinInChunkedPlan_FallsBackToLazy`, `Execute_WithNonStreamablePlan_FallsBackToLazy`) need renaming/assertion updates.

## Changes

### Commit 1 — `docs: plan sync/async alignment fixes in TODO.md`
This file.

### Commit 2 — `fix: release IQuerySource from QueryFrame sync Dispose and DisposeAsync fallback (#268)`
`src/Nivara/Query/QueryFrame.cs`:

```csharp
public void Dispose()
{
    if (!disposed)
    {
        NivaraResourceManager.UntrackResource(this);

        try
        {
            source.Dispose();
        }
        catch
        {
            // Ignore disposal errors, mirroring abandoned-resource cleanup
        }

        disposed = true;
    }
}

public async ValueTask DisposeAsync()
{
    if (!disposed)
    {
        NivaraResourceManager.UntrackResource(this);

        if (source is IAsyncDisposable asyncDisposable)
            await asyncDisposable.DisposeAsync().ConfigureAwait(false);
        else
            source.Dispose();

        disposed = true;
    }
}
```

Tests (`tests/Nivara.Tests/Query/AsyncStreamingTests.cs`): add a disposal-recording `IQuerySource` stub + `QueryFrame_Dispose_ReleasesSourceResources` and `QueryFrame_DisposeAsync_ReleasesSourceResources`.

### Commit 3 — `fix: port flush-concatenate-resume partition to sync streaming ExecuteCore (#269)`
`src/Nivara/Execution/StreamingExecutionStrategy.cs`:

- Replace the `!isSuitableForStreaming(plan)` guard with the window-expression-only check (matching async).
- After the `CanReadInChunks` branch, use `PartitionAtNonStreamableOps`/`OperationSegment` (already static): per chunk apply `segments[0].StreamableOps`; after concatenation dispose intermediate chunkFrames when `Count > 1`; run each remaining segment's boundary op + following streamable ops synchronously, disposing prior results (mirror async lines 247-273).
- Keep the empty-source `chunkFrames.Count == 0` → `executor.Execute(plan)` fallback.

Tests (`tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`): rename the "FallsBackToLazy" sync tests to assert prefix-streaming (`source.ChunksRead.Count > 0` + equality with Lazy), add `Execute_FilterThenSortInChunkedSource_StreamsPrefixThenBoundary` parity test.

### Commit 4 — `docs: changelog sync/async alignment fixes (#268, #269)`
CHANGELOG.md entry.

## Verification

- `dotnet build Nivara.slnx` before each commit.
- `dotnet test` requires explicit confirmation (AGENTS.md). Run affected fixtures first (`StreamingExecutionStrategyTests`, `AsyncStreamingTests`, `ResourceManagementPropertyTests`, `ExecutionIntegrationTests`), then the full suite.

## Blast radius (summary)

- **#268:** `QueryFrame` only; shared-source disposal semantics; all source `Dispose()` idempotent.
- **#269:** `StreamingExecutionStrategy.ExecuteCore` only; strategy tests; no auto-routing depends on `ValidatePlan` (caller sets `context.Strategy` directly), so no behavior change in `ExecutionEngine` selection.

## GitHub issues log

- None yet. As each task executes, any deferred work or concern found should be logged via `gh issue create --repo khurram-uworx/Nivara` and recorded here — never held in memory.
