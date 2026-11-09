# Plan: issue #1060 — Metrics zero-copy regression pin (allocation test + harness mode)

## Problem

`966d3f9` made a null-free `double` pair read through `TryGetSpan`, so regression no longer
rents two buffers it only copies into. Nothing measures that. Timing is the wrong tool here;
allocation is the right one, but only below the pooling threshold, because a pooled rent is
invisible once warm.

## Proposed changes

1. `tests/Nivara.Tests/Metrics/MetricsAllocationTests.cs` (`[Category("Performance")]`):
   - null-free `double` pair vs `float`/`float` pair, both length 512 (below
     `RegressionMetrics.PoolThreshold = 1024`).
   - `GC.GetAllocatedBytesForCurrentThread()` pattern (warmup + GC settle + best-of-5).
   - assert `allocations(double pair) + 512 * 8 <= allocations(float pair)`.
   - comment explaining why 512 is the informative length.
2. `tests/Nivara.PerformanceTests/MetricsProbe.cs` with a `Run()` method, wired as
   `--metrics`, printing:
   - rows/second for `RegressionSummaryFrame`, `ConfusionMatrixFrame`, `PerClassMetricsFrame`
     at n = 1e4/1e5/1e6 and k = 10/100/1000;
   - a class-count curve (fixed n, varying k) making the O(k²) warning in `docs/METRICS.md`
     concrete.
   - documented in `tests/Nivara.PerformanceTests/README.md` as a human-run mode.

## Blast radius

- Touches `src/Nivara/Metrics/RegressionMetrics.cs` only in the red-verification step
  (temporary removal of the in-place arm, then restore); no permanent source change.
- New test file under `tests/Nivara.Tests/Metrics` (Performance category, filtered out).
- New harness file under `tests/Nivara.PerformanceTests` plus a one-line dispatch addition
  in `Program.cs` and a README entry.

## Verification

- Force the allocation test red against the copy-only implementation, then green.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- `dotnet test` filtered suite (`Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`)
  unaffected.
- Targeted Release run of `MetricsAllocationTests`.

## Planned commits

1. docs: plan #1060 in TODO.md
2. test(Metrics): add allocation pin for zero-copy regression path
3. feat(perf-harness): add --metrics mode and document it
4. Red verification evidence note (no separate commit unless needed)
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none captured yet)
