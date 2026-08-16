# Plan: Fix #270 — streaming empty-source fallback re-applies boundary ops

Branch: `khurram/bugs` (off `main` @ `49d6b40`, on top of #268/#269 commits)

## Problem

When a chunk-capable source yields zero chunks (`chunkFrames.Count == 0`), both
`StreamingExecutionStrategy.ExecuteCore` (sync, `src/Nivara/Execution/StreamingExecutionStrategy.cs:136`)
and `ExecuteCoreAsync` (async, line 264) fall back to `executor.Execute(plan)`, which runs the
**full** operation list — including every non-streamable boundary op (Sort/GroupBy/Join/...).
Execution then falls through to the flush-concatenate-resume segment loop (sync line 153 /
async line 281), which re-applies each boundary op on the already-processed result.

**Example:** Filter -> Sort on an empty chunked source: `executor.Execute(plan)` applies
filter + sort; the segment loop then runs the Sort boundary op again on the sorted frame.

**Impact:** double-executes boundary ops and double-reads the source on the empty-source path.
Functionally benign on empty data today (ops on empty stay empty), but it is a latent
correctness smell: any boundary op with side effects or an op that is not idempotent would
misbehave, and it costs a redundant full-plan execution. Reported as #270.

## Fix

Early-return the fallback result from the `chunkFrames.Count == 0` branch, after reporting
the "Streaming execution completed" progress (mirroring the other early-return at the
`!CanReadInChunks` guard). The segment loop then runs only when chunk frames were actually
streamed and concatenated. Apply identically to sync `ExecuteCore` and async
`ExecuteCoreAsync` so the two paths stay aligned (the #269 invariant).

```csharp
if (chunkFrames.Count == 0)
{
    context.Progress?.Report(new ExecutionProgress("No data from chunks, falling back to full execution", 0, 1));
    var fallbackResult = executor.Execute(plan);
    context.Progress?.Report(new ExecutionProgress("Streaming execution completed", 1, 1));
    return fallbackResult;
}
```

The non-empty branch keeps `result = chunkFrames.Count == 1 ? chunkFrames[0] : ConcatenateVertical(chunkFrames)`
and the segment loop unchanged.

## Changes

1. `src/Nivara/Execution/StreamingExecutionStrategy.cs` — early-return in both sync and async empty-source branches.
2. `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs` — regression tests
   `Execute_EmptyChunkedSourceWithBoundaryOp_RunsBoundaryOpOnce` (sync) and
   `ExecuteAsync_EmptyChunkedSourceWithBoundaryOp_RunsBoundaryOpOnce` (async): a
   `StubChunkedQuerySource(rowCount: 0)` with a counting `StubQueryOperation("Sort")`
   `ExecuteFn`; assert the op executed exactly once and the result is a usable empty frame.
   (The default stub `Execute` aliases input columns, but with the fix the segment loop is
   skipped so no disposal aliasing occurs; the count is the observable that pins the bug —
   before the fix it is 2.)
3. `CHANGELOG.md` — entry under [Unreleased] > Fixed.

## Verification

- `dotnet build Nivara.slnx` before each commit.
- `dotnet test` requires explicit confirmation (AGENTS.md). Run affected fixtures
  (`StreamingExecutionStrategyTests`, `AsyncStreamingTests`) first, then the full suite.

## Blast radius

- `StreamingExecutionStrategy.ExecuteCore`/`ExecuteCoreAsync` empty-source path only.
- `StubChunkedQuerySource(rowCount: 0)` is the only test source that yields zero chunks;
  existing empty-source coverage (`Execute_Empty...`, AsyncStreamingTests) exercises the
  same fallback and must remain green.
- No public API change.

## GitHub issues log

- [ ] (this task resolves) #270 — Streaming: empty-source fallback re-applies boundary ops in flush-concatenate-resume
- As each task executes, any deferred work or concern found should be logged via `gh issue create --repo khurram-uworx/Nivara` and recorded here — never held in memory.
