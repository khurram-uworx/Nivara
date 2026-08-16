# Plan: streaming docs disposal contract (#278) + streaming cancellation race (#280)

Branch: `khurram/issues` (base `main`).

## Problems

### #278 — `STREAMING.md` documents chunk-frame disposal the pipeline never performs (docs, medium)

`StreamChunksAsync` (`src/Nivara/Execution/StreamingExecutionStrategy.cs:336`) `yield return`s
each `NivaraFrame` to the consumer and never disposes it; `await foreach` only disposes the
enumerator. `docs/STREAMING.md` line 117 claims "Chunk frames are disposed by the pipeline
after the consumer moves past them" and the sample comment (line 101) says the chunk is
disposed by the loop. Users following the doc leak chunk frames. This is a doc fix only — the
pipeline cannot own the frames because the consumer needs them.

### #280 — Streaming cancellation race: `Complete()` on completed channel masks OCE (medium)

In `executeCoreInternalAsync` (`src/Nivara/Execution/StreamingExecutionStrategy.cs:198`),
consumer-side cancellation produces `QueryExecutionException("Async Streaming execution
failed: The channel has been closed.")` (inner `ChannelClosedException`) instead of a clean
`OperationCanceledException`. Three related defects in one catch path:

1. `channel.Writer.Complete()` in the consumer catch (line 270) throws `ChannelClosedException`
   when the producer's `finally` (lines 251-254) already completed the channel — masks the OCE.
2. The producer task is never observed on the fault path (unobserved task fault).
3. In-flight and channel-buffered chunk frames leak on cancellation (GC-only; violates Phase 4
   step 5 "intermediate chunk frames are disposed on cancellation").

## Proposed changes

### Step 1 — docs fix (#278): `docs/STREAMING.md`

- Replace the sample comment at line 101 with a consumer-owned `try/finally chunk.Dispose()`
  pattern.
- Replace the "Chunk frames are disposed by the pipeline" bullet (line 117) with the
  consumer-ownership contract.
- Note consumer ownership on the single-frame fallback sample too.
- Add a "caller owns each yielded frame" note to the `StreamChunksAsync` XML doc remark
  (complements the doc; API contract unchanged).

### Step 2 — code fix (#280): `src/Nivara/Execution/StreamingExecutionStrategy.cs`

Producer lambda (lines 226-255): track the in-flight `chunkFrame` in a local, null it after a
successful `WriteAsync` (ownership transfers to the consumer), dispose it in the producer's
own `finally`, and switch the producer `finally` to `channel.Writer.TryComplete()`.

```csharp
var producer = Task.Run(async () =>
{
    NivaraFrame? inFlight = null;
    try
    {
        await foreach (var chunkData in plan.Source.ToAsyncEnumerable(chunkSize, context.CancellationToken)
            .ConfigureAwait(false))
        {
            context.CancellationToken.ThrowIfCancellationRequested();

            using var chunkScope = diag != null ? DiagnosticHelper.CreateScope(diag, $"Chunk_{chunkIndex}") : null;

            var processedData = await executeOperationsOnDataAsync(
                chunkData, segments[0].StreamableOps, context.CancellationToken).ConfigureAwait(false);

            if (chunkScope != null)
                chunkScope.SetRowCount(processedData.Values.FirstOrDefault()?.Length ?? 0);

            var chunkFrame = NivaraFrame.Create(processedData);
            inFlight = chunkFrame;
            await channel.Writer.WriteAsync(chunkFrame, context.CancellationToken).ConfigureAwait(false);
            inFlight = null;

            chunkIndex++;
            var totalWork = totalChunks > 0 ? totalChunks : chunkIndex;
            context.Progress?.Report(new ExecutionProgress($"Processing chunk {chunkIndex}", chunkIndex, totalWork));
        }
    }
    finally
    {
        inFlight?.Dispose();
        channel.Writer.TryComplete();
    }
}, context.CancellationToken);
```

Consumer catch (lines 267-272): dispose drained frames, complete the channel first
(`TryComplete()`), then drain + dispose any channel-buffered frames, await the producer
(swallowing its fault so the consumer's own OCE stays primary), then rethrow.

```csharp
catch
{
    foreach (var f in chunkFrames) f.Dispose();
    channel.Writer.TryComplete();
    while (channel.Reader.TryRead(out var buffered))
        buffered.Dispose();
    try { await producer.ConfigureAwait(false); } catch { /* producer fault is secondary; preserve consumer exception */ }
    throw;
}
```

Rationale (grounded in MS Learn `System.Threading.Channels` producer/consumer docs):

- `ChannelWriter.Complete()` throws `ChannelClosedException` when the channel is already
  completed; `TryComplete()` returns `false` instead (no-throw).
- Completing the channel *before* draining guarantees no new writes can land; buffered items
  remain readable via `TryRead`, and a pending `WriteAsync` faults with `ChannelClosedException`
  whose frame is the producer's `inFlight` (disposed in the producer's `finally`).
- The producer runs under `context.CancellationToken` (already cancelled in this path), so
  `await producer` in the catch completes promptly — no deadlock.
- Consumer observes producer faults in the normal path via the existing `await producer`.

### Step 3 — regression test: `tests/Nivara.Tests/Query/AsyncStreamingTests.cs`

Add a chunk-capable source helper (modeled on `PerfChunkedSource` in
`tests/Nivara.PerformanceTests/Program.cs`) that cancels its CTS after N chunks, plus:

```csharp
[Test]
public async Task StreamingStrategy_CancellationMidStream_ThrowsOperationCanceledException()
{
    var source = new CancellationChunkSource(totalRows: 200_000, chunkSize: 10_000, cancelAfterChunks: 3);
    var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());
    var engine = new ExecutionEngine();
    using var cts = new CancellationTokenSource();
    var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
    {
        CancellationToken = cts.Token,
        ChunkSize = 10_000,
    };
    source.CancelWhenChunkCountReaches(cts, 3);

    Assert.ThrowsAsync<OperationCanceledException>(async () => await engine.ExecuteAsync(plan, context));

    Assert.That(source.ChunksRead, Is.GreaterThan(0), "cancellation must fire mid-stream, not pre-cancelled");
    Assert.That(source.ChunksRead, Is.LessThan(20), "run must not complete the full source");
}
```

This pins AC2: clean OCE (not wrapped in `QueryExecutionException`) with prompt unwind; the
`await producer` inside the fix means the engine returns without an unobserved producer task.

### Step 4 — perf scenario note + docs: `tests/Nivara.PerformanceTests/Program.cs` and `README.md`

`CreateStreamingCancellationScenario` (Program.cs:816) and `tests/Nivara.PerformanceTests/
README.md:31` say the scenario is "currently failing on issue #280". Update both comments to
reflect the fix (goes green).

### Step 5 — CHANGELOG

Add under `[Unreleased]`:
- `### Fixed`: #280 cancellation-race fix (clean OCE, observed producer, no frame leaks).
- `### Changed` (or Fixed): #278 STREAMING.md disposal-contract correction.

## Verification

- `dotnet build Nivara.slnx` (requires human confirmation — see AGENTS.md).
- `dotnet test` on `tests/Nivara.Tests` — ask human before running; run targeted
  `AsyncStreamingTests` filter first (`--filter FullyQualifiedName~AsyncStreamingTests`).
- Perf harness `tests/Nivara.PerformanceTests` `Streaming cancel mid-stream` scenario goes
  green (long-running; ask before running).

## Planned commits

1. `docs: correct chunk-frame ownership contract in STREAMING.md (#278)`
2. `fix: clean OCE and no frame leaks on streaming cancellation (#280)` (+ regression test in same or following commit)
3. `test: pin streaming mid-stream cancellation to clean OCE (#280)`
4. `chore: update perf-scenario notes referencing #280`
5. `docs: record #278/#280 fixes in CHANGELOG`

## Blast radius

- `src/Nivara/Execution/StreamingExecutionStrategy.cs` — `executeCoreInternalAsync` producer/
  consumer channel section; private method. Downstream: `ExecutionEngine` async path,
  `QueryFrame.AsStream` (`StreamChunksAsync` untouched), `CollectAsync`. Covered by
  `AsyncStreamingTests`, `StreamingExecutionStrategyTests`, `ExecutionIntegrationTests`,
  `ExecutionEdgeCaseTests`, `JsonStreamingTests`, perf `RunStreamingCancellationScenarios`.
- `docs/STREAMING.md` — doc only.
- `tests/Nivara.Tests/Query/AsyncStreamingTests.cs` — new test + helper source (test-only).
- `tests/Nivara.PerformanceTests/Program.cs` + `README.md` — comment updates.
- `CHANGELOG.md` — doc only.
- No public API changes; behavior change is strictly the exception surfaced on cancellation
  (OCE instead of `QueryExecutionException`) and frame disposal on the cancelled path.

## GitHub issues log

- [ ] #278 — STREAMING.md documents chunk-frame disposal the pipeline never performs (target of this plan)
- [ ] #280 — Streaming cancellation race: Complete() on completed channel masks OCE with QueryExecutionException (target of this plan)
- [ ] #281 — AC3 backlog: streaming memory budget / backpressure never verified; StreamingBufferManager not wired (existing, out of scope)
- [ ] #279 — Disposing a QueryFrame from NivaraFrame.AsQueryFrame() destroys source frame columns (already fixed on main; not in scope)

As each task executes, if deferred work or a concern is found, create a GitHub issue
immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number here —
do not rely on memory or wait until the end of the plan.
