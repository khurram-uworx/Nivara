# Plan: Remove window-expression streaming fallback

## Problem

`StreamingExecutionStrategy` has a blanket fallback: if **any** operation contains a
`WindowExpression` (Rolling, Shift, Rank, etc.), the entire strategy abandons itself
and creates a fresh `LazyExecutionStrategy` to handle the whole plan. This means:

1. Diagnostics and progress tracking are lost — the streaming scope starts but never completes
2. The existing segment partitioning logic (`PartitionAtNonStreamableOps`) is bypassed — it
   already handles non-streamable ops (including Rolling/Cumulative/Shift/Rank, which are in
   `NonStreamableOperations`) as boundaries, but the fallback fires before it runs
3. Leading streamable ops before the window are forced into single-frame materialization
   instead of being processed per-chunk

## Key insight

`NonStreamableOperations` (line 10 of `StreamingExecutionStrategy.cs`) already contains
`Rolling`, `Cumulative`, `Shift`, and `Rank`. `PartitionAtNonStreamableOps` splits the
pipeline at these boundaries into streamable segments. The fallback at lines 109–110 /
211–212 is redundant — it fires *before* the partitioning logic runs.

**Example:** `Filter → Select → Sort → RollingMean → Shift → Select`

Without fallback (correct behavior via partitioning):
- Segment 0: streamable `[Filter, Select]`, boundary `Sort` → chunks stream, then Sort runs on concatenated
- Segment 1: streamable `[]`, boundary `RollingMean` → RollingMean runs on sorted result
- Segment 2: streamable `[Shift, Select]` → Shift+Select run on window result

With fallback (current): everything materializes via LazyExecutionStrategy in one shot.

## Blast radius

| Change | Files affected | Callers / dependents |
|--------|---------------|---------------------|
| Remove window fallback from `ExecuteCore` | `StreamingExecutionStrategy.cs` | `ExecutionEngine.Execute` |
| Remove window fallback from `executeCoreInternalAsync` | `StreamingExecutionStrategy.cs` | `ExecutionEngine.ExecuteAsync` |
| Remove window fallback from `StreamChunksAsync` | `StreamingExecutionStrategy.cs` | `QueryFrame.AsStream`, `NivaraQuery<T>.AsStream` |
| Update streaming window test | `StreamingExecutionStrategyTests.cs` | None (test only) |
| Add new streaming window tests | `StreamingExecutionStrategyTests.cs` | None (test only) |
| Update README | `samples/NivaraIncident/README.md` | None (docs only) |

No public API changes. No changes to window operation correctness — the same kernels
execute on the same data. The only behavioral change is *how* the streaming strategy
orchestrates the pipeline (partitioning vs. fallback).

## Planned commits

1. `docs: plan streaming window fallback removal in TODO.md` — the plan itself
2. `fix: remove window-expression fallback from StreamingExecutionStrategy` — the core change
3. `test: update streaming window tests and add new coverage` — verify new behavior
4. `docs: update README streaming limitation` — reflect current state
5. `docs: remove TODO.md — plan executed` — cleanup

## GitHub issues log

- [x] #307 — Streaming falls back to single-frame for window-heavy queries (CLOSED/COMPLETED — this change removes the remaining fallback)
- [x] #308 — StreamingBufferManager not wired into query execution path (CLOSED/COMPLETED — tracked as new issue #323)
- [ ] #322 — Window overlap buffer for per-chunk streaming (created — future follow-up)
- [ ] #323 — Wire StreamingBufferManager into streaming query execution path (created — future follow-up)
