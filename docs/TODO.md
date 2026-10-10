# Plan — #556: the bounded channel belongs to the materializing path only

**Branch:** `khurram/556` (off `khurram/514`)
**Issue:** [#556 — AsStream path has no bounded channel: MemoryBudget cannot cap in-flight rows via StreamChunksAsync](https://github.com/khurram-uworx/Nivara/issues/556)

## Problem

#514 exposed the streaming memory budget through `QueryFrame.AsStream(NivaraExecutionContext)`
and it reaches `StreamingExecutionStrategy.CalculateChunkSize`. #514's acceptance criterion also
named `CalculateChannelCapacity`, and #556 was filed to close that half by adding a bounded
producer/consumer channel to `StreamChunksAsync`.

**The filed prescription is wrong.** On the chunked `AsStream` path the source is enumerated by a
pull-based async iterator: `IQuerySource.ToAsyncEnumerable` (`src/Nivara/Query/IQueryInterfaces.cs:47`)
is a plain `for` loop that calls `ReadChunkAsync` once per `MoveNextAsync`, and `StreamChunksAsync`
(`src/Nivara/Execution/StreamingExecutionStrategy.cs:501`) yields each processed chunk back to the
consumer. Nothing runs ahead. Therefore the chunked path already holds **at most one chunk frame in
flight** between consumer pulls — a bound strictly *tighter* than any bounded channel
(`CalculateChannelCapacity` clamps to `[2, 16]`).

Adding the channel would *loosen* that bound (1 frame → up to 16), invert the documented pull-based
backpressure (`docs/STREAMING.md:146`, `docs/ACCELERATION.md:224`), and apply to only a subset of
branches — the tier-2 partition streamer and tier-3 materialization buffer the whole dataset by
design. The honest resolution is to state the real, tighter bound and prove it, not to add a channel.

The issue message itself confirms this (no acceptance criteria beyond the direction), and the human
chose "document + test the real bound".

## Decisions (confirmed by the human)

1. **Direction:** do not add a channel. Document that the bounded channel / `CalculateChannelCapacity`
   governs only the materializing path, state the chunked path's one-frame in-flight bound, and add a
   regression test proving the producer never runs ahead of the consumer.
2. **Branch base:** `khurram/556` off `khurram/514` (#514 completed, `docs/TODO.md` removed at `8281959d`).

## Proposed changes

### 1. Test — `tests/Nivara.Tests/Execution/StreamingBackpressureTests.cs`

Add one deterministic, timing-free test beside the existing materializing-path probe
(`CreateBoundChannel_UnderLoad_ProducerBlocksAndNeverExceedsCapacity`, `:72`):

- `StreamChunksAsync_PullBased_ProducerNeverRunsAheadOfConsumer`
- Reuse the existing `StubChunkedQuerySource` (`ExecutionTestHelpers.cs:136`) and
  `StubQueryOperation(OperationType.Filter)`; a pure-streamable plan over a chunk-capable source
  takes the per-chunk branch at `StreamingExecutionStrategy.cs:536`.
- `ChunkSize = 1`; in the `await foreach` body assert `source.ChunksRead.Count == consumed`
  **before** the next pull — the source must have been asked for exactly the chunk just delivered.
  C# async iterators run only to the next `yield return` per `MoveNextAsync`, so this is exact, not
  a stress probe.
- No wall-clock/allocation assertions → no `[Category("Performance")]` (the test is O(8)).

### 2. Docs

- `docs/STREAMING.md`: scope the §"Channel capacity" claim (`:137`) to the materializing path; add
  the positive statement (after `:150`) that the chunked path holds one frame in flight — tighter
  than the channel's `[2, 16]` — which is why no channel is used; rewrite the cancellation bullet
  (`:254-255`) so it no longer claims a channel exists on the `AsStream` path.
- `docs/adr/006-streaming-memory-budget-exposure.md`: dated addendum under "Negative / limits"
  recording that the channel half of #514's AC was resolved as documentation + test, with the
  1-frame-vs-`[2,16]` reasoning. (Addendum avoids an index change that `AdrIndexTests` enforces.)
- `docs/ACCELERATION.md:224-227`: append a pointer to the new test.
- `CHANGELOG.md`: one bullet under Unreleased `### Fixed` (`:83`).

No gated `csharp` fence is touched, so ADR-005 is not in play. No source change.

## Blast radius

- **Public API:** none. `src/Nivara` and `src/Nivara.Extensions` are unchanged.
- **Call graph:** `CreateBoundChannel` → only `executeCoreInternalAsync` (materializing) + the test;
  `CalculateChannelCapacity` → only `CreateBoundChannel` + tests; `StreamChunksAsync` → only tests /
  `QueryFrame.asStreamCore`. Confirmed via the code-memory `RelationshipRecord` join.
- **Tests:** one added test in `StreamingBackpressureTests`; existing streaming suite unchanged.
- **Docs:** `STREAMING.md`, `ADR-006`, `ACCELERATION.md`, `CHANGELOG.md`.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` after the test lands.
2. Targeted run: `dotnet test -c Release --filter "FullyQualifiedName~StreamingBackpressureTests"`
   (allowed autonomously).
3. Full suite only on request, always with `--filter "Category!=Performance"`.
4. Docs are prose-only; no `DocSnippetCompilerTests` impact.

## Planned commits

1. `docs: plan 556 AsStream in-flight bound in TODO.md`
2. `test(streaming): prove the AsStream pull path never runs ahead of the consumer`
3. `docs(streaming): scope the bounded channel to the materializing path; state the AsStream one-frame bound`
4. `docs: remove TODO.md — plan executed` (only after G2)

## GitHub issues log

- [x] #556 — resolved here: no channel added; the chunked path's one-frame pull bound is documented
      and pinned by a test.
- [ ] #514 — the knob this completes (#514 itself is complete; this closes the `CalculateChannelCapacity`
      half of its AC).

## Probe / harness lifecycle

Not applicable — no performance measurement or kernel probe. The new test asserts an ordering
invariant, not a timing or allocation measurement.
