# Phase 4.5 — Streamix Bridge

**Issue:** [#171](https://github.com/khurram-uworx/Nivara/issues/171)
**Branch:** `khurram/171`
**Design:** `docs/PHASE45.md`

## Problem

Nivara has async streaming (`QueryFrame.AsStream` → `IAsyncEnumerable<NivaraFrame>`), but no
bridge to Streamix's `IFlux<T>` ecosystem. Users can't apply Streamix operators (Retry,
ScopedAsync, BufferByTime, backpressure, observability) to Nivara queries.

## Plan

### Step 1 — Create `NivaraFlux` bridge class

**File:** `src/Nivara.Extensions/Streamix/NivaraFlux.cs`
**Namespace:** `Nivara.Streamix`
**Blast radius:** new file only — no existing code modified

Four static methods:

1. `QueryFrame.ToFlux(chunkSize, backpressureMode, channelCapacity, name)` → `IFlux<NivaraFrame>`
   - `Flux.From(queryFrame.AsStream(chunkSize, ct), name)` as base
   - Optional `.PipeThroughChannel(channelCapacity, backpressureMode)` when capacity > 0

2. `NivaraFrame.ToFlux(name)` → `IFlux<NivaraFrame>`
   - `Flux.From(new[] { frame }, name)` — single-element stream

3. `QueryFrame.ToFluxRows(chunkSize, ct)` → `IFlux<NivaraRow>`
   - Async enumerable: iterate chunks via `AsStream`, yield `NivaraRow` per row
   - `NivaraRow` internal ctor accessible via InternalsVisibleTo

4. `IFlux<NivaraFrame>.ToNivaraFrameAsync(ct)` → `Task<NivaraFrame>`
   - `stream.ToListAsync(ct)` → `NivaraFrameExtensions.ConcatenateVertical(list)`

### Step 2 — Create bridge tests

**File:** `tests/Nivara.Tests/Streamix/StreamixBridgeTests.cs`
**Blast radius:** new test file only

Tests:
- `ToFlux_MatchesCollectResults` — chunk-level round-trip
- `ToFlux_WithName_PreservesName` — diagnostics name
- `SingleFrame_ToFlux_RoundTrips` — one-shot frame
- `ToFluxRows_MatchesFrameRowCount` — row-level bridge
- `ToNivaraFrameAsync_MatchesCollectAsync` — reverse terminal
- `Backpressure_FailMode_ThrowsOnFullChannel` — synthetic backpressure
- `Cancellation_PropagatesCleanly` — OperationCanceledException flow

### Step 3 — Update PHASE45.md status

Change `Status: planning` → `Status: implemented`.

## Commit Plan

1. `docs: plan Streamix bridge in TODO.md`
2. `feat: add NivaraFlux bridge — QueryFrame.ToFlux, NivaraFrame.ToFlux, ToFluxRows, ToNivaraFrameAsync`
3. `test: Streamix bridge round-trip and backpressure tests`
4. `docs: update PHASE45.md status to implemented`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (no deferred issues yet)
