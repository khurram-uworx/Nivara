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

## Design (as built)

The first cut used two cross-chunk buffers (lookback overlap + pending lookahead rows).
Branch testing exposed row duplication between them — both cover the tail of the
previous chunk — so the shipped design uses **one sliding context run** instead:

```
contextSize = max(rolling lookback, lag periods) + max(lead periods)

run        = last min(contextSize, seen) input rows ++ fresh chunk      // contiguous
result     = boundaryOp.Execute(run)                                    // no trimming
finalEnd   = min(seen + chunkLen - leadDistance, runEnd)
emitted    = result[emittedCount - runStart .. finalEnd - runStart]     // global range

carry slots: cumulative recomputed over FRESH chunk only, seeded with state;
    values append to a per-slot FIFO; drain exactly emitCount oldest values.
    (leadDistance == 0 keeps the direct kernel column — no boxing.)

tail       = last contextSize input rows of run
lastRun    = result retained for Flush()
```

Flush(): slices the premature-boundary rows of the last run — their null/fill tails are
exact once no further data exists — and drains remaining FIFO values. No re-execution.

Gates:

- `isStreamableNode`: `Lead => true`; `Shift => true` for any periods.
- `TryCreate`: standalone branch accepts `shift.Periods < 0` with
  `leadDistance = -Periods`, overlap 0.

### StreamingExecutionStrategy call sites (3)

After each read loop where `windowProcessor != null`: call `Flush()`, wrap non-empty results in a
frame, feed budget tracker/diagnostics, append to `chunkFrames` (sync path, async producer
before `channel.Writer.TryComplete()`) or yield (`StreamChunksAsync` windowProcessor branch,
skipping zero-row flush frames). Zero-row delayed-emission prefixes are suppressed.

Cadence note: with lead present, AsStream frames lag one chunk plus a final flush frame.

### Docs

- docs/STREAMING.md: lead/negative-shift moved from tier 3 to tier 1; bounded memory
  (`max(rolling lookback, lag periods) + max(lead periods)`); delayed-emission cadence;
  partitioned lead stays tier 2. DONE.
- CHANGELOG [Unreleased] entry. DONE.
- WindowOverlapBuffer.cs remarks reference delay mechanism; instance buffering API
  removed (superseded), class reduced to static overlap-size determination. DONE.

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
