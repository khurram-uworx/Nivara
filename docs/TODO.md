# Plan: Streaming lead/negative-shift windows via delayed emission (issue #331)

## Problem

Streaming window boundaries that require **lookahead** (`Lead`, `Shift` with negative
periods) are excluded from per-chunk streaming: `StreamingWindowProcessor.hasOnlyStreamableWindows`
rejects them, so any select containing them falls through to full boundary materialization
(tier 3, reported via `ExecutionDiagnostics.StreamMaterializationCount`). Standalone
`ShiftOperation` with `Periods < 0` is likewise rejected in `TryCreate`. Results are correct
today but pay an avoidable O(frame) materialization.

Desired: stream lead windows by holding each chunk's tail rows back (delayed emission,
symmetric to the existing lookback prepend). Emit chunk *i*'s rows only after enough of
chunk *i+1* is read; the final chunk flushes with nulls/fill values at the tail. Memory
bounded by `max(leadPeriods)` rather than frame size.

Scope confirmed with human: **full support** for mixed selects containing lead/negative-shift
alongside rolling/lag **and cumulative** windows (cumulative needs a small per-slot FIFO of
computed-but-unemitted values).

## Design

### StreamingWindowProcessor (src/Nivara/Execution/StreamingWindowProcessor.cs)

New state:

- `int leadDistance` — max lookahead over Lead expressions / negative Shift periods (0 when none).
- `Dictionary<string, IColumn>? pendingRows` — last ≤ `leadDistance` pre-boundary input rows not yet emitted.
- Per-`CarrySlot` pending FIFO (`Queue<object?>`) + `Queue<bool>` null flags of computed-but-unemitted cumulative values.

ProcessChunk becomes delayed-emission:

```
combined   = pendingRows == null ? chunk : Concatenate(pendingRows, chunk)
extended   = overlapBuffer?.PrependToChunk(combined) ?? combined      // unchanged lookback
result     = boundaryOp.Execute(extended)
final      = overlapBuffer != null ? TrimFirstN(result, overlapSize) : result
E          = max(0, combined.Length - leadDistance)                   // emit prefix only

per column of final:
    emitted[col] = final[col].Slice(0, E)                             // drop last P rows (premature boundaries)

carry slots: compute over FRESH chunk only (never replay held rows);
    state advances over all fresh values; values append to slot FIFO;
    emitted cumulative column = drain FIFO oldest-first to fill E rows
    (when FIFO short — first rounds with L < P — pad by deferring: emit what is available
     once combined length exceeds P; counts always reconcile because
     available = previouslyHeld + fresh = L and E = max(0, L - P))

overlapBuffer.UpdateFromChunk(combined)                               // context includes held rows
pendingRows = last min(P, combined.Length) rows of combined           // zero-copy Slice
return emitted (empty columns via ColumnFilterHelper.CreateEmptyColumn when E == 0)
```

New `Flush()`:

- No-op when `pendingRows == null` (returns null).
- Otherwise run boundary op over `overlap prepend(pendingRows)` + trim; tail rows get true
  boundary nulls/fill naturally since nothing follows. Drain cumulative FIFOs into those columns.
- When `leadDistance == 0` the whole mechanism is inert: ProcessChunk emits everything as today.

Gates:

- `isStreamableNode`: `Lead => true`; `Shift => true` for any periods.
- `TryCreate`: standalone branch accepts `shift.Periods < 0` with `leadDistance = -Periods`, overlap 0.

### Lead-distance helper

Static traversal mirroring `WindowOverlapBuffer.getMaxOverlapFromExpression` — walks ColumnExpression
nodes collecting max `|Periods|` for `WindowFunctionKind.Lead` and negative-period `Shift`.
Placement: private static in StreamingWindowProcessor or a sibling internal static class near
WindowOverlapBuffer.

### StreamingExecutionStrategy call sites (3)

After each read loop where `windowProcessor != null`: call `Flush()`, wrap non-empty results in a
frame, feed budget tracker/diagnostics, append to `chunkFrames` (sync path ~line 173, async producer
before `channel.Writer.TryComplete()` ~line 367) or yield (StreamChunksAsync windowProcessor branch
~line 624, skipping zero-row flush frames).

Cadence note: with lead present, AsStream frames lag one chunk plus a final flush frame.

### Docs

- docs/STREAMING.md: move lead/negative-shift from tier 3 to tier 1; document bounded memory
  (`max(leadPeriods)`), delayed-emission cadence, partitioned lead stays tier 2.
- WindowOverlapBuffer.cs remarks (`Lead => 0` comment) reference delay mechanism.
- CHANGELOG entry.

## Blast radius

- **Changed**: `StreamingWindowProcessor` (internal), `StreamingExecutionStrategy` (internal),
  possibly `WindowOverlapBuffer` remarks/docs.
- **Downstream callers**: only the three call sites inside StreamingExecutionStrategy;
  public API unchanged (QueryFrame.Lead/Shift, AsStream cadence documented behavior change
  for lead-containing plans only).
- **Tests covering**: tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs
  (streaming-vs-lazy property pattern with `chunkEquivalenceBudgets`),
  tests/Nivara.Tests/Query/* (eager semantics untouched).
- **Risk**: medium — cross-chunk state machine grows a second buffer; mitigated by property
  tests vs lazy strategy across chunk sizes and the existing 1948-test suite.

## Steps (one logical unit each)

1. [x] Write plan to docs/TODO.md → commit
2. [x] StreamingWindowProcessor: leadDistance helper, pending buffer, delayed-emission ProcessChunk, Flush(), gate changes → build → commit
3. [x] StreamingExecutionStrategy: Flush() integration at 3 call sites → build → commit
4. [x] Tests: streaming-vs-lazy lead properties, mixed selects, tiny chunks, single-row source, diagnostics, StreamChunksAsync cadence → ask → run targeted tests → fix if needed → commit
5. [x] Docs: STREAMING.md tiers + CHANGELOG + code remarks → commit
6. [x] Review TODO.md, remove it, offer push + PR

## Verification

- `dotnet build Nivara.slnx` after each code step.
- Targeted `dotnet test --filter StreamingExecutionStrategyTests` (ask first per AGENTS.md),
  then full suite before final review.

## Deferred-work policy

As each step executes: if deferred work or a concern outside this plan is found, create a
GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number
below — don't rely on memory or wait until the plan finishes (compaction can lose it).

## GitHub issues log

- [ ] #331 — Streaming: support lead/negative-shift windows via delayed emission (the issue being implemented)
