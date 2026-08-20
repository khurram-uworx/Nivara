# Issue #315 — Flux.From(IEnumerable<T>) swallows BackpressureException

## Problem

GitHub issue #315 reported that `Flux.From(IEnumerable<T>).PipeThroughChannel(capacity, ChannelBackpressureMode.Fail)` silently swallows `BackpressureException`. The root cause was identified as a Streamix-side bug: `AsyncEnumerable.FromEnumerable` swallows exceptions from `PipeThroughChannel`'s finally block.

**Key finding:** Nivara's bridge (`NivaraFlux.ToFlux`) uses `Flux.From(IAsyncEnumerable<T>)` — the safe path that propagates exceptions correctly. The problematic `IEnumerable<T>` overload is never used in Nivara code. After updating NuGets solution-wide (Streamix now at 1.2.2), the upstream bug appears fixed.

## Plan

1. Add regression canary tests to `StreamixBridgeTests.cs`:
   - `Backpressure_FailMode_BridgePath_PropagatesException` — validates BackpressureException propagates through the actual `NivaraFlux.ToFlux` bridge path.
   - `Backpressure_FailMode_AsyncEnumerablePath_PropagatesException` — directly tests `Flux.From(IAsyncEnumerable<T>).PipeThroughChannel()` (the exact Streamix bug path from #315).
2. Run the Streamix bridge test suite to verify all tests pass.
3. Close issue #315 on GitHub with a note about the regression canary tests.

## Blast radius

- `tests/Nivara.Tests/Streamix/StreamixBridgeTests.cs` — two new tests added. No production code changes.

## GitHub issues log

- [ ] #315 — Flux.From(IEnumerable<T>) swallows BackpressureException (original issue, to be closed)
