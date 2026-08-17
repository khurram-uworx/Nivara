# Phase 3a — CLI: The Incident Lab (Milestone 1)

**Branch:** `khurram/incident-3`
**Plan source:** `docs/PHASE3A.md`

---

## Overview

Implement the Incident Lab CLI: deterministic dataset generator, ingestion wrappers, analysis queries, formatted CLI output, and tests. Architecture: single library folder (`samples/Nivara.Samples/Incident/`) + thin executable (`samples/NivaraIncident.Cli/`).

## Blast radius

- **New files only** in `samples/Nivara.Samples/Incident/`, `samples/NivaraIncident.Cli/`, `tests/Nivara.Tests/Incident/`
- **Modified:** `Nivara.slnx` (add CLI project), `samples/Nivara.Samples/Nivara.Samples.csproj` (add Parquet ref if missing), `samples/NivaraIncident/README.md` (update)
- **No changes to core `src/Nivara/` or `src/Nivara.Extensions/`** — this is a sample exercise only
- **Existing tests:** unaffected (new test files only)

## API surface used

| API | Source | Purpose |
|-----|--------|---------|
| `Parquet.ScanAsQueryFrame(path)` | `Nivara.Extensions` | Lazy Parquet source |
| `Csv.ScanAsQueryFrame(path)` | `Nivara.Extensions` | Lazy CSV source |
| `QueryFrame.Filter/Select/Sort/GroupBy` | `Nivara.Query` | Expression-level query ops |
| `ColumnExpressions.RollingMean/RollingMax/RollingMin/Shift/Lead/Rank/PercentRank/DenseRank` | `Nivara.Expressions` | Window & rank expressions |
| `ColumnExpressions.Col/Lit/Not` | `Nivara.Expressions` | Column refs, literals, negation |
| `NivaraQuery<T>.GroupBy().Select()` with `Grouping.Sum/Count/Quantile/StdDev/Median` | `Nivara.Linq` | Typed grouped aggregation |
| `QueryFrame.AsStream(chunkSize, ct)` | `Nivara.Query` | Streaming chunks |
| `QueryFrame.CollectAsync()` | `Nivara.Query` | Async eager execution |
| `QueryFrame.LastExecutionDiagnostics` | `Nivara.Query` | Execution diagnostics |
| `NivaraFrame.ToParquet(path, options)` | `Nivara.Extensions` | Parquet write |
| `ParquetWriteOptions.With(rowGroupSize:)` | `Nivara.Extensions` | Small row groups |
| `NivaraColumn<T>.Create(values)` | `Nivara` | Column creation |
| `NivaraFrame.Create(namedColumns)` | `Nivara` | Frame creation |

---

## Sub-steps (commit plan)

### Step 1 — Project scaffolding + telemetry schema
**Commit:** `feat(incident): add project scaffolding and telemetry schema`

**Files:**
- `samples/NivaraIncident.Cli/NivaraIncident.Cli.csproj` (new)
- `samples/NivaraIncident.Cli/Program.cs` (new, skeleton)
- `samples/Nivara.Samples/Incident/Schema.cs` (new)
- `Nivara.slnx` (add CLI project to samples folder)

**Schema.cs:** Four `sealed record` types: `RequestTelemetry`, `DeploymentEvent`, `ServiceDependency`, `InstanceState`.

**CLI skeleton:** `args[0]` = mode (`generate`/`analyze`/`replay`), `--flag value` pairs. Switch statement, no logic yet.

**Validation:** `dotnet build Nivara.slnx` clean (0 warnings).

---

### Step 2 — Deterministic dataset generator + incident scenarios
**Commit:** `feat(incident): add deterministic dataset generator and incident scenarios`

**Files:**
- `samples/Nivara.Samples/Incident/Scenarios.cs` (new)
- `samples/Nivara.Samples/Incident/DatasetGenerator.cs` (new)

**Scenarios.cs:** `IncidentScenario` class + four static scenarios (A–D) with timeline, degradation events, affected services.

**DatasetGenerator.cs:**
- Seeded `Random(seed)` for determinism
- Generate 10M+ `RequestTelemetry` records (50+ endpoints, 10 regions, 100+ instances)
- Generate `DeploymentEvent`, `ServiceDependency`, `InstanceState` records
- Baseline latency model per service/endpoint (normal distribution via Box-Muller)
- During incident: shift distribution for affected services
- Write Parquet with `ParquetWriteOptions.With(rowGroupSize: 10_000)`
- Also write CSV variant
- Use `Span<T>` for bulk value generation where possible

**Validation:** `dotnet build Nivara.slnx` clean.

---

### Step 3 — Ingestion wrappers + replay helpers
**Commit:** `feat(incident): add ingestion wrappers and replay helpers`

**Files:**
- `samples/Nivara.Samples/Incident/Ingestion.cs` (new)

**Ingestion.cs:** Static class with:
- `LoadParquet(path)` → `Parquet.ScanAsQueryFrame(path)`
- `LoadCsv(path)` → `Csv.ScanAsQueryFrame(path)`
- `StreamChunks(path, chunkSize, ct)` → `ScanAsQueryFrame` then `.AsStream(chunkSize, ct)`

**Validation:** `dotnet build Nivara.slnx` clean.

---

### Step 4 — Analysis queries (the core exercise)
**Commit:** `feat(incident): add incident analysis queries`

**Files:**
- `samples/Nivara.Samples/Incident/Analysis.cs` (new)

**Analysis methods (static class):**
- `AnalyzeDegradationOrdering(QueryFrame, scenario)` → scenario A: rolling error-rate, shift deltas, propagation delay
- `AnalyzeDeploymentCorrelation(QueryFrame, scenario)` → scenario B: rank by error delta, correlate with deployments, time-to-impact
- `AnalyzeSaturationOrdering(NivaraQuery<RequestTelemetry>, scenario)` → scenario C: P50/P95/P99, z-scores, queue-depth rolling windows, recovery ordering
- `AnalyzeRegionalPartitioning(NivaraQuery<RequestTelemetry>, scenario)` → scenario D: per-region error rate, latency percentiles, which region degraded first

Each captures `QueryFrame.LastExecutionDiagnostics`.

**API patterns:**
- Rolling windows: `ColumnExpressions.RollingMean(Col("Error"), 60)` on sorted QueryFrame
- Shift deltas: `ColumnExpressions.Shift(rollingCol, 1)` then computed subtraction via expression
- Rank: `ColumnExpressions.PercentRank(orderBy)` or `ColumnExpressions.Rank(orderBy)`
- Grouped aggregation: typed `NivaraQuery<RequestTelemetry>.GroupBy(r => r.Service).Select(g => new { Service = g.Key, P95 = g.Quantile(r => r.DurationMs, 0.95) })`

**Validation:** `dotnet build Nivara.slnx` clean.

---

### Step 5 — CLI entry point + formatted output
**Commit:** `feat(incident): add CLI entry point with formatted output`

**Files:**
- `samples/NivaraIncident.Cli/Program.cs` (complete implementation)

**Three modes:** generate, analyze, replay. Argument parsing, formatted console output per IDEA §"CLI".

**Output format:**
```
NIVARA INCIDENT LAB
Dataset       X.XX GB
Rows          XX,XXX,XXX
Duration      X.XX s
...
TOP IMPACTED SERVICES
...
EXECUTION
X operators / X kernels / XX MB peak memory
```

**Validation:** `dotnet build Nivara.slnx` clean.

---

### Step 6 — Tests
**Commit:** `feat(incident): add incident test suite`

**Files:**
- `tests/Nivara.Tests/Incident/DatasetGeneratorTests.cs` (new)
- `tests/Nivara.Tests/Incident/AnalysisTests.cs` (new)
- `tests/Nivara.Tests/Incident/IngestionTests.cs` (new)

**Test categories:**
1. Determinism: same seed → same dataset → same results
2. Schema correctness: field ranges, all regions present
3. Row counts: ≥ 10M requests, ≥ 50 endpoints, 10 regions, ≥ 100 instances
4. Analysis correctness per scenario (A–D)
5. Convergence: Parquet == CSV == replay results
6. Streaming: expected chunk count, disposal, cancellation
7. Diagnostics populated after analysis

**Validation:** `dotnet build Nivara.slnx` clean.

---

### Step 7 — Wire, build, end-to-end + README update
**Commit:** `docs(incident): update README with CLI usage and gap status`

**Files:**
- `samples/NivaraIncident/README.md` (update with CLI usage, resolved gaps, performance)

**Final validation:** `dotnet build Nivara.slnx` clean, CLI end-to-end for all 4 scenarios.

---

## GitHub issues log

- (none yet — will be created during execution if gaps are found)
