# Plan: Issues #289 & #290 — Incident sample test coverage

## Problem
The Incident Lab CLI was implemented in Phase 3a but is missing:
- #289: DatasetGenerator determinism/heavy tests (in Nivara.PerformanceTests console app)
- #290: Analysis pipeline integration tests (in Nivara.Tests with NUnit)

## Branch: `khurram/issues`

---

## Issue #289: DatasetGenerator tests (console app scenarios)

**File:** `tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` (extend existing)

Keep Nivara.PerformanceTests as a console app. Add a new method `RunDatasetGeneratorTests(string[] args)` invoked via `--dataset-test` switch in Program.cs.

Tests run as console assertions (throw on failure, print results):

1. `Determinism` — generate same scenario twice to two temp dirs, byte-compare CSV files
2. `RowCount` — generate at scale=1, load parquet via ScanAsQueryFrame + Collect, assert 10M rows
3. `FieldRanges` — load small dataset, assert StatusCode ∈ [200–503], DurationMs > 0, all 10 regions present
4. `ParquetRowGroups` — with rowGroupSize=1000, verify multiple row groups via Parquet.Net API
5. `CsvVariant` — load CSV via Csv.ScanAsQueryFrame, assert columns and row count

---

## Issue #290: Analysis pipeline integration tests (NUnit)

**File:** `tests/Nivara.Tests/Incident/AnalysisTests.cs` (new)

`[OneTimeSetUp]`: generate small dataset (scale=0.001 → ~10K rows) for all 4 scenarios.
`[OneTimeTearDown]`: delete temp dir.

Tests:
1. Non-empty results — Collect each analysis × scenario, assert RowCount > 0
2. Determinism — run same analysis twice, compare RowCount
3. Diagnostics — assert GetExecutionDiagnostics() not null, RowsRead > 0, TotalExecutionTime > 0
4. Scenario A — degradation ordering: orders → checkout → payments → gateway
5. Scenario B — deployment correlation: deploy at min 17 precedes error spike
6. Scenario C — saturation ordering: matches expected recovery order
7. Scenario D — regional partitioning: ap-south-1 is affected region
8. Parquet ↔ CSV convergence — same analysis from both sources, compare row counts
9. Replay convergence — materialize frame, re-analyze, compare results

---

## Execution order
1. Add `--dataset-test` switch to Program.cs + `RunDatasetGeneratorTests` in IncidentLabBenchmark.cs (#289)
2. Create AnalysisTests.cs (#290)
3. Build both projects
4. Run new tests (with human confirmation)

## Blast radius
- `IncidentLabBenchmark.cs` — adding new method + Program.cs switch; existing benchmark unchanged
- `Nivara.Tests` — new test file only; no existing tests modified
- No changes to production code

## GitHub issues log
- [ ] #289 — Incident sample: DatasetGenerator tests in Nivara.Performance
- [ ] #290 — Incident sample: Analysis pipeline integration tests
