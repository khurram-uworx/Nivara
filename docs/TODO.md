# Phase 4 — Restore RollingMax & PercentRank in Incident Analyses

## Problem

The NivaraIncident sample README claims the analyses exercise `RollingMax` and `PercentRank`,
but the actual code uses typed LINQ GroupBy + manual ranking loops instead. The APIs exist and
are well-tested — the sample just doesn't use them. This means:

- The benchmark doesn't exercise window+rank ops over the dataset (only Analysis A uses RollingMean/Shift)
- The README pipeline diagrams are stale
- The Nivara-vs-Polars comparison misses these operation classes

## Changes

### 1. Restore RollingMax in `AnalyzeSaturationOrdering` (Analysis.cs)

Add `RollingMax("QueueDepth", "PeakQueueDepth", windowSize, WindowSpec)` to the QueryFrame
pipeline *before* Collect, partitioned by Service. This tracks peak queue depth per service
over a sliding window, making the saturation ordering observable.

- Window size: 10 (10 consecutive instance snapshots per service)
- Collect the result, then use typed LINQ GroupBy to aggregate `g.Max(r => r.PeakQueueDepth)`
- Add `PeakQueueDepth` (int) to `InstanceRow`
- Keep existing P50/P95/P99/StdDev aggregation

### 2. Restore PercentRank in `AnalyzeRegionalPartitioning` (Analysis.cs)

Add `PercentRank("DurationPercentRank", [SortKey("DurationMs", Ascending)], "Region")` to the
QueryFrame pipeline *before* Collect. This exercises the partitioned rank kernel over the full
dataset.

- Add `DurationPercentRank` (double) to `RequestRow`
- After Collect, typed LINQ GroupBy aggregates `g.Max(r => r.DurationPercentRank)` per region
- Keep existing ErrorRate, P50Duration, P95Duration aggregation
- Keep manual `ErrorRank` LINQ loop (post-aggregation ranking of aggregated ErrorRate is not
  expressible via `ColumnExpressions.Quantile` — that's a row-level window, not a group aggregate)

### 3. Mirror in Polars benchmark (benchmark.py)

- `analysis_saturation`: add `rolling_max` over `QueueDepth` partitioned by `Service`
- `analysis_regional`: add per-Region percent-rank of `DurationMs`

### 4. Update README pipeline diagrams

Fix the stale diagrams in `samples/NivaraIncident/README.md` to match the restored code.

## Blast radius

- `samples/Nivara.Samples/Incident/Analysis.cs` — row types + two analysis methods
- `samples/NivaraIncident/Python/benchmark.py` — two Polars functions
- `samples/NivaraIncident/README.md` — pipeline diagrams
- `tests/Nivara.Tests/Incident/AnalysisTests.cs` — existing tests should still pass; add
  PeakQueueDepth/DurationPercentRank column assertions

## Verification

- `dotnet build Nivara.slnx` clean
- Existing AnalysisTests pass (ask human before running)
- `AnalyzeSaturationOrdering` returns rows with PeakQueueDepth column
- `AnalyzeRegionalPartitioning` returns rows with DurationPercentRank column

## Commit plan

1. `feat: restore RollingMax in SaturationAnalysis and PercentRank in RegionalAnalysis`
2. `feat: mirror RollingMax/PercentRank in Polars benchmark`
3. `docs: fix stale pipeline diagrams in README`

## GitHub issues log

(none yet)
