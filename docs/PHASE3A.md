# Phase 3a — CLI: The Incident Lab (Milestone 1)

**Status:** PLANNING
**Scope:** `samples/Nivara.Samples/Incident/` (library code) + `samples/NivaraIncident.Cli/` (thin executable)
**Depends on:** Phase 1 (core gap-fills 1.1–1.5) ✅, Phase 2 (ADR-001 cleanup + SIMD) ✅
**Followed by:** Phase 4 (performance assessment), then Phase 3b (Web UI, follow-up)
**Parent plan:** `samples/Incident-PLAN.md`
**Related:** `samples/NivaraIncident/IDEA.md` (product spec), `samples/NivaraIncident/README.md` (gap inventory)

---

## Architecture decision

The IDEA's 7-project layout is replaced by a single library folder + one thin executable:

- **`samples/Nivara.Samples/Incident/`** — schema, generator, scenarios, ingestion, analysis.
  Lives in the existing `Nivara.Samples` class library (already references `Nivara` +
  `Nivara.Extensions` + `Parquet.Net`). No new dependencies.
- **`samples/NivaraIncident.Cli/`** — `OutputType=Exe`, project-references `Nivara.Samples`.
  Arg parsing + formatted output only. No third-party CLI library.
- **`tests/Nivara.Tests/Incident/`** — NUnit tests in the existing test project.

Rationale: the guiding principle says complexity should be in data and workload, not infra
ceremony. A 7-project split for a sample violates that.

---

## Prerequisites (Phase 2 handoff)

These are available from the Phase 2 branch (`khurram/incident-phase2`):

| Capability | API | Used by |
|------------|-----|---------|
| Quantile / Median | `NivaraSeries<T>.Quantile(q)` / `.Median()` | 3a.4 analysis (latency P50/P95/P99) |
| StdDev / Variance | `NivaraSeries<T>.StdDev()` / `.Variance()` | 3a.4 analysis (z-scores) |
| Execution diagnostics | `QueryFrame.LastExecutionDiagnostics` / `.GetExecutionDiagnostics()` | 3a.5 CLI output |
| Parquet chunked read | `Parquet.ScanAsQueryFrame`, row-group-aligned chunks, single reader | 3a.2 generator, 3a.3 ingestion |
| Streamed row materialization | `NivaraQuery<T>.ToObjectsAsync` | 3a.5 CLI `--stream` output |
| Rank family | `ColumnExpressions.Rank/DenseRank/PercentRank/RowNumber` | 3a.4 analysis |
| Rolling / Shift / Lead | `ColumnExpressions.Rolling*/Shift/Lead` | 3a.4 analysis |
| Streaming | `QueryFrame.AsStream(chunkSize, ct)`, `CollectAsync`, `IAsyncDisposable` | 3a.3 replay |

**Stale anchor caveat:** line numbers from Phase 1/2 handoff notes may have drifted — always
re-grep before editing.

---

## Sub-steps

### 3a.1 — Project scaffolding + telemetry schema

**Files:**
- `samples/NivaraIncident.Cli/NivaraIncident.Cli.csproj` (new)
- `samples/NivaraIncident.Cli/Program.cs` (new, skeleton)
- `samples/Nivara.Samples/Incident/Schema.cs` (new)
- `Nivara.slnx` (add `NivaraIncident.Cli.csproj` to samples folder)

**Schema.cs** defines the telemetry record types modeled on the IDEA §"Example production
environment":

```csharp
namespace Nivara.Samples.Incident;

public sealed record RequestTelemetry(
    DateTimeOffset Timestamp,
    string Service,        // "gateway", "orders", "payments", etc.
    string Endpoint,       // "/api/v1/checkout", "/api/v1/payments/process", etc.
    double DurationMs,
    int StatusCode,        // 200, 429, 500, 503, etc.
    string Region,         // "us-east-1", "eu-west-1", "ap-south-1", etc.
    string TraceId,        // hex correlation id
    bool IsRetry);

public sealed record DeploymentEvent(
    DateTimeOffset Timestamp,
    string Service,
    string Version,        // "v4.21"
    string Region);

public sealed record ServiceDependency(
    string Parent,         // upstream service
    string Child);         // downstream service

public sealed record InstanceState(
    DateTimeOffset Timestamp,
    string Service,
    string InstanceId,
    string Region,
    int ActiveRequests,
    int QueueDepth);
```

Use `sealed record` types. The schema should be deliberately simple — no inheritance
hierarchies or polymorphic dispatch. The complexity is in the data volume and analytical
workload, not the type system.

**CLI skeleton (Program.cs):**
Follow the `NivaraChat` convention: `args[0]` = mode, remaining args parsed in a `for` loop
for `--flag value` pairs. Modes: `generate`, `analyze`, `replay`. No third-party CLI library.

```csharp
// Skeleton — full logic added in 3a.5
switch (args[0])
{
    case "generate": GenerateDataset(datasetPath, scenario, scale); break;
    case "analyze": AnalyzeDataset(datasetPath, scenario, stream); break;
    case "replay": ReplayDataset(datasetPath, scenario, chunkSize); break;
    default: PrintUsage(); break;
}
```

**NivaraIncident.Cli.csproj:**
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\Nivara.Samples\Nivara.Samples.csproj" />
    <ProjectReference Include="..\..\src\Nivara.Extensions\Nivara.Extensions.csproj" />
  </ItemGroup>
</Project>
```

**Validation:** `dotnet build Nivara.slnx` clean (0 warnings).

---

### 3a.2 — Deterministic dataset generator + incident scenarios

**Files:**
- `samples/Nivara.Samples/Incident/Scenarios.cs` (new)
- `samples/Nivara.Samples/Incident/DatasetGenerator.cs` (new)

**Scenarios.cs** — four static scenario configs defining the incident timeline, degradation
points, propagation chains, and affected services. Each scenario is a plain data object:

```csharp
public sealed class IncidentScenario
{
    public string Id { get; init; }            // "A", "B", "C", "D"
    public string Name { get; init; }          // "Database degradation"
    public DateTimeOffset IncidentStart { get; init; }
    public DateTimeOffset IncidentEnd { get; init; }
    public IReadOnlyList<ServiceEvent> Events { get; init; }  // degradation signals
    public IReadOnlyList<string> AffectedServices { get; init; }
}
```

Scenarios (from the IDEA):
- **A — Database degradation:** PostgreSQL latency → orders → checkout → payment timeout →
  retry storm. Timeline: degradation starts T+0, propagates through dependency chain over
  5–10 minutes.
- **B — Bad deploy:** orders-api v4.21 deployed → exception rate ↑ → failures ↑.
  Deployment event precedes degradation by 2–3 minutes.
- **C — Traffic spike:** traffic ×8 → queue depth ↑ → latency ↑ → timeouts ↑.
  Affects multiple services simultaneously; some recover earlier.
- **D — Regional failure:** us-east/eu-west healthy, ap-south degraded. Tests partitioning
  and per-region analysis.

**DatasetGenerator.cs:**
- Deterministic seeded RNG (`new Random(seed)`) — no randomness in scenarios, reproducible
  output.
- Generate 10M+ `RequestTelemetry` records across 50+ endpoints, 10 regions, 100+ instances.
- Generate `DeploymentEvent` records (dozens, aligned with scenario B timeline).
- Generate `ServiceDependency` edges (the dependency graph between services).
- Generate `InstanceState` records (queue depth, active requests per instance).
- Write to Parquet with small row groups: `ParquetWriteOptions.With(rowGroupSize: 10_000)`
  (exercises Phase 1.4's row-group-aligned chunking contract).
- Also write a CSV variant for streaming tests.
- Use `Span<T>` and `TensorPrimitives` for bulk value generation where types allow (e.g.
  vectorize `DurationMs` generation for the baseline latency distribution).

**Baseline latency model (before incident):**
Each service/endpoint has a configurable baseline latency (e.g. normal distribution around
a mean with a small stddev). During an incident, affected services shift their distribution
(increased mean, increased variance). This produces realistic-looking telemetry without
actual randomness — the seeded RNG gives the same numbers every time.

**Validation:**
- Same seed → same output (determinism test).
- Row counts match expected scale (10M+ requests).
- Parquet file has multiple row groups (small row groups exercised).
- CSV variant is loadable by `Csv.ScanAsQueryFrame`.

---

### 3a.3 — Ingestion wrappers + replay helpers

**Files:**
- `samples/Nivara.Samples/Incident/Ingestion.cs` (new)

Thin wrappers over existing Nivara APIs. The goal is to provide a clean incident-specific
surface while proving that the core APIs work end-to-end:

```csharp
public static class Ingestion
{
    public static QueryFrame LoadParquet(string path);
    public static QueryFrame LoadCsv(string path);
    public static IAsyncEnumerable<NivaraFrame> StreamChunks(
        string path, int chunkSize, CancellationToken ct = default);
}
```

- `LoadParquet` → `Parquet.ScanAsQueryFrame(path)` (exercises 1.4's single-reader path).
- `LoadCsv` → `Csv.ScanAsQueryFrame(path)`.
- `StreamChunks` → `Parquet.ScanAsQueryFrame(path)` then `.AsStream(chunkSize, ct)`.
  Uses `await foreach` with `IAsyncDisposable` on the frame.
- Measure and report: peak memory, chunk count, backpressure behavior (exercise gap 6).

**Design-test:** historical Parquet path (`LoadParquet` → analyze) vs replay path
(`StreamChunks` → analyze per chunk) should converge on the same Nivara analytical queries.
If they require radically different APIs, that is the core feedback the sample exists to
produce. Record findings.

**Validation:**
- Parquet load returns correct row count.
- CSV load returns identical data (round-trip test).
- Streaming yields multiple chunks for a dataset with small row groups.
- `await using` disposes resources correctly.

---

### 3a.4 — Analysis queries (the core exercise)

**Files:**
- `samples/Nivara.Samples/Incident/Analysis.cs` (new)

This is the largest and most important sub-step. It implements, as plain Nivara query code,
the four incident analyses from the IDEA. Every number here must be computed by Nivara — no
LINQ-to-objects fallback.

#### A: Degradation ordering

- Rolling error-rate windows per service (e.g. 1-minute rolling mean of `StatusCode >= 500`).
- `Shift` deltas: error rate delta = current − previous interval.
- First service whose delta crossed a threshold.
- Propagation delay = time between successive service crossings in the dependency chain.
- Retry amplification = retry volume / initial error count.

Nivara APIs: `ColumnExpressions.RollingMean`, `ColumnExpressions.Shift`, `Where`, `GroupBy`,
`OrderBy`, `Select`.

#### B: Deployment correlation

- Rank services by error-rate delta (`PercentRank` / `Rank`).
- Correlate with `Deployment` events: join service error timeline with deployment timeline.
- Time-to-customer-impact: delta between deployment timestamp and first error spike.
- Affected endpoint / error-type breakdown: `GroupBy` endpoint × `StatusCode`, aggregate counts.
- group→aggregate→rank→filter: exercise this plan pattern (README gap 8) and record findings.

Nivara APIs: `ColumnExpressions.Rank`, `ColumnExpressions.PercentRank`, `GroupBy`, `Join`
(if available), `Select`, `Where`, aggregation (`Sum`, `Count`).

#### C: Saturation ordering

- Per-service latency P50/P95/P99: `GroupBy("Service")` → `Quantile(0.50/0.95/0.99)` on
  `DurationMs` (exercises Phase 1.1).
- Z-scores: `StdDev` per service (exercises Phase 1.2), then `(value − mean) / stddev`.
- Queue-depth rolling windows: `RollingMax` on `QueueDepth` from `InstanceState`.
- Recovery ordering: `Lead` deltas to find when each service returned to baseline.

Nivara APIs: `Quantile`, `StdDev`, `RollingMax`, `Lead`, `GroupBy`, `OrderBy`.

#### D: Regional partitioning

- `GroupBy("Region")` + per-region rank/rolling/percentile analysis.
- Per-region error rate, latency percentiles, deployment correlation.
- Identify which region degraded first and worst.

Nivara APIs: `GroupBy`, per-group rolling/rank/quantile.

#### Top-k + computed ordering

- `TopKDescending` (already on `NivaraSeries<T>`) for most-impacted services by error-rate Δ.
- Computed ordering: `Select` a weighted score expression, `Sort` by it.

#### Execution diagnostics capture

Each analysis captures `QueryFrame.LastExecutionDiagnostics` after execution. The CLI
displays this as the "Execution" summary (operators, kernels, peak memory, elapsed,
rows read/returned).

**Validation:**
- Each analysis returns non-empty results for its scenario.
- Results are deterministic (same dataset → same output).
- Diagnostics are populated (rows read > 0, elapsed > 0).
- **Design-test (gap 8):** document whether group→aggregate→rank→filter is cleanly
  expressible. If awkward, record the gap with code evidence and escalate to core.

---

### 3a.5 — CLI entry point + formatted output

**Files:**
- `samples/NivaraIncident.Cli/Program.cs` (complete implementation)

Full CLI with three modes, following the repo convention (`args[0]` = mode, `--flag` pairs):

```
NivaraIncident.Cli generate
    --dataset <path>       Output path (default: ./data/incident-lab)
    --scenario <A|B|C|D>  Incident scenario (default: A)
    --scale <N>            Row count multiplier (default: 1, for 10M+)

NivaraIncident.Cli analyze
    --dataset <path>       Path to Parquet/CSV dataset
    --scenario <A|B|C|D>  Which incident to analyze (default: A)
    --stream               Use streaming path instead of eager load
    --chunk-size <N>       Chunk size for streaming (default: 100000)

NivaraIncident.Cli replay
    --dataset <path>       Path to Parquet dataset
    --scenario <A|B|C|D>  Which incident to replay
    --chunk-size <N>       Chunk size for replay stream (default: 100000)
```

**Output format** (from the IDEA §"CLI"):

```
NIVARA INCIDENT LAB

Dataset       4.2 GB
Rows          48,921,332
Duration      2.14 s
Streamed      48,921,332 (3 chunks)
Backpressure  4 chunks in flight (bounded channel)

TOP IMPACTED SERVICES

payments      +418% errors
orders        +172%
checkout       +91%

TOP CORRELATED EVENT

14:17:32 deployment orders-api/4.21

EXECUTION

5 operators
3 fused kernels
0 per-row boxing
412 MB peak memory
```

The analysis output varies by scenario. Each scenario prints its specific answers
(degradation ordering, deployment correlation, saturation ordering, or regional breakdown)
plus the execution diagnostics.

**Validation:**
- `dotnet run -- generate --dataset ./data/incident-lab` produces Parquet + CSV.
- `dotnet run -- analyze --dataset ./data/incident-lab` prints results + diagnostics.
- `dotnet run -- analyze --dataset ./data/incident-lab --stream` uses streaming path.
- `dotnet run -- replay --dataset ./data/incident-lab --chunk-size 50000` streams chunks.
- No crashes, no unhandled exceptions, clean exit.

---

### 3a.6 — Tests + Polars cross-validation

**Files:**
- `tests/Nivara.Tests/Incident/DatasetGeneratorTests.cs` (new)
- `tests/Nivara.Tests/Incident/AnalysisTests.cs` (new)
- `tests/Nivara.Tests/Incident/IngestionTests.cs` (new)
- `samples/NivaraIncident/Python/gen_reference.py` (extend)

**Test categories:**

1. **Determinism:** same seed → same dataset → same analysis results.
2. **Schema correctness:** generated records have expected field ranges (StatusCode 200–503,
   DurationMs > 0, all regions present, etc.).
3. **Row counts:** total requests ≥ 10M, endpoints ≥ 50, regions = 10, instances ≥ 100.
4. **Parquet structure:** multiple row groups (small row groups), correct column types.
5. **Analysis correctness per scenario:**
   - A: first degraded service matches expected propagation chain.
   - B: deployment precedes error spike, time-to-impact within expected range.
   - C: saturation ordering matches expected recovery order.
   - D: regional partitioning shows expected affected/unaffected regions.
6. **Convergence:** Parquet analysis results == CSV analysis results == replay analysis results
   for the same dataset and scenario (the core convergence claim).
7. **Streaming:** `StreamChunks` yields expected chunk count; `await using` disposes cleanly;
   cancellation stops the stream.
8. **Diagnostics:** `LastExecutionDiagnostics` is populated after each analysis.

**Polars cross-validation:**
Extend `samples/NivaraIncident/Python/gen_reference.py` to emit fixtures for:
- Latency percentiles (P50/P95/P99) per service → compare with Nivara's `Quantile`.
- Error-rate rolling windows per service → compare with Nivara's `RollingMean`.
- Rank/PercentRank per service by error delta → compare with Nivara's `Rank`/`PercentRank`.
- StdDev per service → compare with Nivara's `StdDev`.

Write fixtures to `samples/data/polars-incident/manifest.json`.

**Naming:** `Method_Scenario_ExpectedBehavior` (repo convention).

**Validation:** all new tests pass; no regressions in existing test suite.

---

### 3a.7 — Wire, build, end-to-end validation

- Add `NivaraIncident.Cli.csproj` to `Nivara.slnx` (`samples/` folder).
- `dotnet build Nivara.slnx` clean (0 warnings).
- Run CLI end-to-end: generate → analyze → replay for all 4 scenarios.
- Verify streaming path produces same results as eager path.
- Record any core gaps discovered during implementation:
  - Awkward API patterns → escalate as GitHub issues.
  - Performance surprises → record in README performance section.
  - Missing capabilities → escalate with code evidence.
- Update `samples/NivaraIncident/README.md`:
  - Mark resolved gaps with commit/issue references.
  - Document any new gaps found during 3a.
  - Add CLI usage section.
  - Add initial performance numbers (real, not illustrative).

---

## Core gaps to watch for (design-test)

These are the known open items that 3a is expected to surface or validate:

| Gap | What to watch | Where it surfaces |
|-----|---------------|-------------------|
| **Gap 5:** Window semantics across chunk boundaries | Rolling/rank queries defeat true streaming (`AsStream` falls back to single frame) | 3a.3 streaming + 3a.4 analysis |
| **Gap 6:** Memory budget is advisory, not hard | Peak memory may exceed budget during large dataset streaming | 3a.3 streaming measurement |
| **Gap 8:** group→aggregate→rank→filter ergonomics | May require awkward multi-step construction instead of fluent API | 3a.4 analysis (especially B and D) |
| **ColumnExpressions.Quantile** (issue #277) | `Quantile` as a window/expression node for group→aggregate plans | 3a.4 scenario C (per-service percentiles) |

Record every gap found with code evidence. Escalate to core if the workaround is ugly.

---

## Definition of done (Phase 3a)

- [ ] `dotnet build Nivara.slnx` clean (0 warnings).
- [ ] CLI can generate a dataset for all 4 scenarios.
- [ ] CLI can analyze all 4 scenarios (eager + streaming paths).
- [ ] CLI can replay streamed data with chunk progress output.
- [ ] Analysis results are deterministic and converge across Parquet/CSV/replay.
- [ ] Polars cross-validation fixtures pass.
- [ ] Execution diagnostics visible in CLI output.
- [ ] All new tests pass; no regressions in existing suite.
- [ ] `samples/NivaraIncident/README.md` updated with CLI usage, gap status, and performance.
- [ ] Any core gaps found are escalated as GitHub issues with code evidence.

---

## Execution notes

- **Ask before running `dotnet test`** (repo rule); verify with `dotnet build Nivara.slnx` first.
- Use `dotnet build Nivara.slnx` after each sub-step.
- Keep each sub-step as a separate commit.
- When launching sub agents, include: *"Use the code-memory MCP to learn symbols/relationships
  and the microsoft-learn MCP for official API documentation where relevant."*
- Follow repo code style (`.editorconfig`): sealed by default, no comments, no Hungarian
  notation, omit braces from single-line `if`/`else` bodies.
- Existing test baseline: keep it green; ask before running full test suite.
- **After 3a completes:** proceed to Phase 4 (performance assessment, `docs/PHASE4.md`).
  Phase 3b (Web UI) is deferred as a follow-up after Phase 4.
