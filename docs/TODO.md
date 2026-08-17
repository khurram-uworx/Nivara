# Plan: Issues #291 + #293 — Incident Sample Tests & Analysis Depth

## Problem

1. **Issue #291**: `Ingestion.cs` (LoadParquet, LoadCsv, StreamChunks) has zero dedicated tests.
2. **Issue #293**: `Analysis.cs` methods B/C/D use shallow filter+sort+select chains instead of
   the rich composition patterns the Incident Lab was designed to surface (join, quantile,
   stddev, partitioned rank, lead/lag recovery deltas).

## Blast radius

- `tests/Nivara.Tests/Incident/IngestionTests.cs` (new file)
- `samples/Nivara.Samples/Incident/Analysis.cs` (return-type changes on B/C/D)
- `tests/Nivara.Tests/Incident/AnalysisTests.cs` (adapt to new return types + new depth tests)
- `samples/NivaraIncident.Cli/Program.cs` (caller of Analysis methods — must adapt to NivaraFrame returns)

## Planned changes

### Step 1 — Ingestion wrapper tests (issue #291)

Create `tests/Nivara.Tests/Incident/IngestionTests.cs` with:

| Test | Asserts |
|------|---------|
| `LoadParquet_ReturnsCorrectRowCount` | RowCount == 10_000 |
| `LoadCsv_ReturnsIdenticalData` | Row + column parity with Parquet |
| `StreamChunks_YieldsExpectedChunkCount` | ~100 chunks (rowGroupSize=100, 10K rows) |
| `StreamChunks_DisposesResources` | Clean completion, no exceptions |
| `StreamChunks_CancellationStopsStream` | Cancelled after 3 chunks → exactly 3 |

Dataset: generate with `rowGroupSize: 100` so streaming produces multiple chunks.
Uses same `[OneTimeSetUp]` pattern as `AnalysisTests.cs`.

### Step 2 — Rewrite Analysis B (issue #293)

`AnalyzeDeploymentCorrelation` → return `NivaraFrame`

1. Load + Filter + Sort requests via QueryFrame → Collect()
2. Load deployments via NivaraParquetReader.ReadParquet()
3. For each request, find most recent deployment for same service (as-of join logic)
4. Compute: Service, Endpoint, Timestamp, StatusCode, Region, DeploymentVersion, TimeSinceDeploy
5. Categorize StatusCode into error buckets (server_error / client_error / success)

### Step 3 — Rewrite Analysis C (issue #293)

`AnalyzeSaturationOrdering` → return `NivaraFrame`

1. Load + Filter + Sort instances via QueryFrame → Collect()
2. Typed LINQ: GroupBy Service
3. Per service: Quantile(0.50), Quantile(0.95), Quantile(0.99) of QueueDepth
4. StdDev of QueueDepth per service
5. Lead of QueueDepth to compute recovery deltas
6. Return: Service, P50QueueDepth, P95QueueDepth, P99QueueDepth, StdDevQueueDepth, InstanceCount

### Step 4 — Rewrite Analysis D (issue #293)

`AnalyzeRegionalPartitioning` → return `NivaraFrame`

1. Load + Filter requests via QueryFrame → Collect()
2. Typed LINQ: GroupBy Region
3. Per region: Count, error count (StatusCode >= 500), error rate
4. Rank regions by error rate
5. Quantile of DurationMs per region
6. Return: Region, TotalRequests, ErrorCount, ErrorRate, ErrorRank, P50Duration, P95Duration

### Step 5 — Update AnalysisTests for new return types

- B/C/D tests: change from `using var qf = ... Collect()` to direct `var frame = ...`
- Add depth-specific tests:
  - B: verify DeploymentVersion column exists, TimeSinceDeploy > 0
  - C: verify P50/P95/P99 columns exist and are ordered P50 <= P95 <= P99
  - D: verify Region column, ErrorRate > 0 for incident periods
- Add determinism tests for B, C, D

### Step 6 — Update CLI caller

`samples/NivaraIncident.Cli/Program.cs` calls Analysis methods. If it expects `QueryFrame`
returns from B/C/D, adapt to `NivaraFrame`.

### Step 7 — Build verify

`dotnet build Nivara.slnx` clean (0 warnings).

## GitHub issues log

- [ ] #291 — Incident sample: Ingestion wrapper tests
- [ ] #293 — Incident sample: Analysis.cs depth gap vs plan spec
