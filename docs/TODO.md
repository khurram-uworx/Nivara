# Plan: Nivara vs Polars Processing Benchmark

## Problem

NivaraInference has a clear PyTorch vs Nivara timing comparison. NivaraIncident has no
performance comparison against an established DataFrame library. We need an apples-to-apples
benchmark: same Parquet data, same analytical operations, timed on both Nivara (C#) and
Polars (Python).

## Approach

1. Add `--records <N>` to `DatasetGenerator` so we can generate exactly 1M records (currently
   only `--scale` which is int × 10M).
2. Add `--benchmark` switch to the `analyze` CLI command — times each of the 5 analyses
   individually and prints a summary table.
3. Create `Python/benchmark.py` — reads the same Parquet files, runs equivalent Polars
   operations, prints matching timing table.
4. Update README with benchmark instructions and results section.

## Scope

- Scenario A only (database degradation).
- Idiomatic Polars (join_asof for deployment correlation, native rolling/groupby/quantile).
- 1M records first; scale to 10M if difference is negligible.

## Files to change

| File | Action |
|------|--------|
| `samples/Nivara.Samples/Incident/DatasetGenerator.cs` | Add `Generate(path, scenarioId, int totalRecords)` overload |
| `samples/NivaraIncident/NivaraIncident.Cli/Program.cs` | Add `--records` flag + `--benchmark` on `analyze` |
| `samples/NivaraIncident/Python/benchmark.py` | New — Polars benchmark script |
| `samples/NivaraIncident/README.md` | Add benchmark section |

## Blast radius

- `DatasetGenerator.Generate` is called from: `Program.cs` (generate command), `IncidentLabBenchmark.cs` (perf tests).
  The new overload is additive; existing callers are unaffected.
- `Program.cs` changes are local to the CLI entry point.
- `benchmark.py` is standalone, no impact on existing code.

## Verification

1. `dotnet build samples/NivaraIncident/NivaraIncident.Cli` — compiles
2. `dotnet run ... generate --records 1000000 --dataset ./data/benchmark-1m --scenario A`
3. `dotnet run ... analyze --benchmark --dataset ./data/benchmark-1m` — prints timings
4. `python benchmark.py --dataset ./data/benchmark-1m` — prints timings
5. Both sides report same row counts per analysis

## Planned commits

1. `docs: plan Nivara vs Polars benchmark in TODO.md`
2. `feat(incident): add --records overload to DatasetGenerator`
3. `feat(incident): add --benchmark switch and --records flag to CLI`
4. `feat(incident): add Polars benchmark script`
5. `docs(incident): add benchmark section to README`

## GitHub issues log

- (none yet)
