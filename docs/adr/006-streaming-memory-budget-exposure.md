# ADR-006: Exposing the Streaming Memory Budget Through an `AsStream` Context Overload

**Status:** Accepted  
**Date:** 2026-10-10

## Context

`StreamingExecutionStrategy` derives a chunk size from
`NivaraExecutionContext.MemoryBudget` when no explicit `ChunkSize` is set
(`StreamingExecutionStrategy.CalculateChunkSize`), and sizes the bounded
producer/consumer channel from the same budget
(`StreamingExecutionStrategy.CalculateChannelCapacity`). Until now no public
surface could set that budget for a chunked stream: `QueryFrame.AsStream` built
its own context internally and the strategy type is `internal`, so the 1 GB
constructor default was the only budget in play. `docs/STREAMING.md`
§"Memory budget → chunk size" was written in terms of an input a caller could not
provide (tracked as #514).

Three directions were considered:

1. A context-taking `AsStream` overload (smallest surface).
2. Making `StreamingExecutionStrategy` public and adding a streaming member to
   `IExecutionStrategy` (larger, but makes the strategy substitutable).
3. A `long? memoryBudget` parameter on `AsStream` (narrow; does not generalise to
   `Progress` / `ExecutionDiagnostics`).

## Decision

**Adopt option 1: a context-taking `AsStream` overload. Keep
`StreamingExecutionStrategy` internal.**

`QueryFrame` now exposes:

```text
public IAsyncEnumerable<NivaraFrame> AsStream(int chunkSize = 10000, CancellationToken ct = default)
public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, CancellationToken ct = default)
public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, int chunkSize, CancellationToken ct = default)
```

1. **The legacy `int` overload is unchanged.** A bare `AsStream()` yields the
   documented 10,000-row default. The budget-derived chunk size is opt-in: pass a
   context and leave `NivaraExecutionContext.ChunkSize` null.
2. **An explicit chunk size always wins** — via `AsStream(..., chunkSize)` or a
   populated `NivaraExecutionContext.ChunkSize` — over the budget-derived value.
3. **The context is cloned before execution.** The caller's instance is not
   mutated, so one context can be shared across concurrent streams without a race
   on `Strategy`, `ChunkSize`, `CancellationToken`, or `ExecutionDiagnostics`.
4. **The redundant `AsStream(int chunkSize, long memoryBudget)` overload is
   removed.** Forcing `ChunkSize` made its budget inert for sizing, so it implied a
   ceiling it could not deliver.

## Consequences

**Positive:**

- `NivaraExecutionContext.MemoryBudget` reaches `CalculateChunkSize` on the public
  chunked path, so the derivation documented in `docs/STREAMING.md` is reachable
  and the docs match the code.
- `Progress` and `ExecutionDiagnostics` travel with the same overload, closing the
  gap that option 3 would have left open.
- No public type is promoted, so the execution-strategy abstraction stays private
  and free to change.

**Negative / limits:**

- On the `StreamChunksAsync` path (what `AsStream` calls), the budget bounds only
  the derived chunk size and the advisory `StreamingBudgetTracker`; the bounded
  channel — and therefore `CalculateChannelCapacity` — belongs to the
  single-frame `ExecuteCoreAsync` path. `docs/STREAMING.md` §"AC3 resolution" is
  corrected to say so; backpressure on `AsStream` is the pull-based
  `IAsyncEnumerable` semantics.
- A caller wanting the derivation must pass a context rather than a scalar; the
  older "just pass `memoryBudget`" shorthand is deliberately not offered.

## Addendum (2026-10-10, #556)

The `CalculateChannelCapacity` half of #514's acceptance criterion is resolved by documentation and
a test, not by a code change. The chunked `AsStream` path enumerates the source through a pull-based
async iterator, so at most one chunk frame is in flight between consumer pulls — a bound strictly
tighter than the channel's `[2, 16]`. Adding a channel there would loosen that bound from one frame
to up to `capacity` frames and invert the pull-based backpressure without improving enforcement, and
it would apply to only the per-chunk branches (the partition-streamer and materialization branches
buffer the whole dataset by design). The chunked path's one-frame bound is stated in
`docs/STREAMING.md` §"Channel capacity (async pipeline)" and pinned by
`StreamingBackpressureTests.StreamChunksAsync_PullBased_ProducerNeverRunsAheadOfConsumer`.
