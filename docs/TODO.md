# Plan: Fix #334 — Backpressure canary fails on bridge path (test-side timing bug)

## Problem

Issue #334 reports `Backpressure_FailMode_BridgePath_PropagatesException`
failing "deterministically" — `BackpressureException` never reaches the
consumer through `Flux.From(IAsyncEnumerable).PipeThroughChannel(1, Fail)`.

## Root cause (investigated)

**Not a Streamix bug and not a Nivara library bug — the test itself cannot
generate backpressure on Windows.**

Evidence:

- All involved types (`Flux`, `PipeThroughChannel`, `ChannelBackpressureMode`,
  `BackpressureException`) live in Streamix; zero hits in the Nivara symbol
  index.
- Streamix v1.2.3 already contains the fix for this exact swallow class
  (`7dd7870` "surface PipeThroughChannel scope faults via MoveNextAsync",
  PRs #155/#156): `ScopeHelper.ReadAllSupervisedAsync` re-checks
  `scope.IsFaulted` at the top of the read loop. Published to NuGet
  2026-08-19T20:37 UTC, i.e. built from the fixed tag.
- Nivara references Streamix 1.2.3 since `a91ed37` ("nugets"), an ancestor of
  both commits where the failure was reported (`186f2d7`, `6b81fa9`).
- The failing canary's producer paces with `await Task.Delay(1)`
  (`SlowAsyncFrames`) while the consumer delays 10 ms. On Windows (~15.6 ms
  default timer granularity) both quantize to roughly the same cadence, so the
  producer never gets ≥2 items ahead of the capacity-1 channel over 100 frames
  → Fail mode never fires → no exception → red.
- The sibling canary passes because its producer uses `await Task.Yield()`
  (floods). Linux CI has fine-grained timers → producer genuinely ~10× faster →
  overflow → green CI/CD.

## Proposed change

File: `tests/Nivara.Tests/Streamix/StreamixBridgeTests.cs` (test-only; no
production code touched).

1. `SlowAsyncFrames` (:304): replace `await Task.Delay(1);` with
   `await Task.Yield();` — flood producer vs slow consumer gives structural
   overflow guarantee (mirrors the passing sibling and Streamix's own
   regression test `PipeThroughChannel_Fail_ThrowsWhenBoundaryIsFull_FromEnumerable`,
   Streamix FluxTests.cs:348).
2. Both backpressure canaries (:249, :277): switch from try/catch +
   null-assert to `Assert.ThrowsAsync<BackpressureException>` (Streamix style)
   so a clean completion fails loudly with the right exception type.
3. One brief comment explaining why the producer must not pace
   (non-obvious design decision guarding against regression of this exact bug).

## Blast radius

- Symbols: `SlowAsyncFrames`,
  `Backpressure_FailMode_BridgePath_PropagatesException`,
  `Backpressure_FailMode_AsyncEnumerablePath_PropagatesException`.
- Downstream callers: none (test helpers only).
- Risk: minimal; no public contract change → no CHANGELOG entry needed.

## Verification

1. `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj` (no ask required).
2. Ask human, then run the two canaries repeatedly for stability:
   `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj --no-build --filter "FullyQualifiedName~Backpressure_FailMode"` (loop ×10).
3. Confirm sibling + cancellation tests in the fixture still pass
   (`FullyQualifiedName~StreamixBridgeTests`).
4. Comment root cause on issue #334 (via `--body-file` temp file per AGENTS.md).

## Planned commits

1. `docs: plan #334 backpressure canary fix in TODO.md`
2. `test: make Streamix backpressure canaries deterministic (#334)`
3. `docs: remove TODO.md — #334 plan executed` (after verification)

## GitHub issues log

- [ ] #334 — existing issue; comment root cause after verification, close when
      PR merges (human-confirmed).
