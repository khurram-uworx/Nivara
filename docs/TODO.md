# Plan: Issues #289 & #290 — Incident sample test coverage

## Problem
The Incident Lab CLI was implemented in Phase 3a but is missing:
- #289: DatasetGenerator determinism/heavy tests (should live in Nivara.PerformanceTests)
- #290: Analysis pipeline integration tests (should live in Nivara.Tests)

## Branch: `khurram/issues`

---

## Issue #289: DatasetGenerator tests

**File:** `tests/Nivara.PerformanceTests/Incident/DatasetGeneratorTests.cs`

### Step 1: Add NUnit to Nivara.PerformanceTests.csproj
- Add `IsTestProject`, `IsPackable=false`
- Add NUnit 4.6.1, NUnit3TestAdapter 6.2.0, Microsoft.NET.Test.Sdk 18.9.0, coverlet 10.0.1
- Keep `<OutputType>Exe</OutputType>` so benchmark runner still works

### Step 2: Create DatasetGeneratorTests.cs
- `[OneTimeSetUp]`: generate small dataset (scale=0.001 → ~10K rows) in temp dir
- `[OneTimeTearDown]`: delete temp dir

Tests:
1. `Generate_SameSeed_ProducesIdenticalOutput` — byte-compare CSV files from two generations
2. `Generate_Scale1_ProducesExpectedRowCount` — load parquet+CSV, assert 10M rows
3. `Generate_ProducesValidFieldRanges` — StatusCode ∈ [200–503], DurationMs > 0, all 10 regions present
4. `Generate_Parquet_MultipleRowGroups` — verify row groups via Parquet.Net API
5. `Generate_Csv_LoadableByCsvScanAsQueryFrame` — load CSV via ScanAsQueryFrame, assert columns/rows

---

## Issue #290: Analysis pipeline integration tests

**File:** `tests/Nivara.Tests/Incident/AnalysisTests.cs`

### Step 3: Create AnalysisTests.cs
- `[OneTimeSetUp]`: generate small dataset (scale=0.001) for all 4 scenarios
- `[OneTimeTearDown]`: delete temp dir

Tests:
1. Non-empty results — Collect each analysis method × scenario, assert RowCount > 0
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
1. Modify .csproj → add NUnit
2. Create DatasetGeneratorTests.cs (#289)
3. Create AnalysisTests.cs (#290)
4. Build both projects
5. Run new tests (with human confirmation)

## Blast radius
- `Nivara.PerformanceTests.csproj` — adding NUnit packages; existing Program.cs benchmark runner unchanged
- `Nivara.Tests` — new test file only; no existing tests modified
- No changes to production code

## GitHub issues log
- [ ] #289 — Incident sample: DatasetGenerator tests in Nivara.Performance
- [ ] #290 — Incident sample: Analysis pipeline integration tests
