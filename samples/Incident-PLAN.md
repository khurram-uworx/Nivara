# NivaraIncident — Implementation Plan

**Status:** Phase 1 core gap-fills (1.1–1.5) are **complete**:
1.1 ✅, 1.2 ✅, 1.3 ✅, 1.4 ✅, 1.5 ✅ (see the 1.6 completion marker below). Phase 2+ is
**deferred** to a follow-up (issue #284) — see the "Phase 1 → Phase 2+ handoff notes" section.
**Scope:** `samples/NivaraIncident/` reference application + the core-library improvements it
drives (`src/Nivara`, `src/Nivara.Extensions`)
**Inputs:** `samples/NivaraIncident/IDEA.md` (product spec), `samples/NivaraIncident/README.md`
(audited gap inventory), `docs/adr/001-004`, `docs/PHASE4.md`, `docs/STREAMING.md`

---

## Guiding principles (from the maintainer)

These rules govern every decision in this plan and every future edit of this sample:

1. **The examples are a forcing function for the core library.** The goal of implementing each
   example is to *find gaps, fix the issues while implementing, assess performance, and improve
   things where needed*. This makes the core library better and showcases what is possible. When
   the sample requires an awkward workaround, first ask whether Nivara itself is missing a
   capability or has an incorrect abstraction — fix the core, then build the sample on the fixed
   API. Do not hide the problem inside the sample.
2. **AutoDiff is a non-null domain (ADR-001).** While implementing, clean up code branches that
   deal with nulls wherever an opportunity appears, and implement any missing null checks at the
   domain **boundaries** (entry points). The interior of the AutoDiff domain must stay null-free.
3. **Nivara is about Tensors and Vectors.** Embrace SIMD / Numerics intrinsics
   (`TensorPrimitives`, `Vector<T>`, spans) wherever an opportunity appears. Vectorized kernels
   are the default; scalar fallbacks are the exception, and they are documented.
4. **Use the MCPs.** These capabilities are new:
   - **microsoft-learn MCP** — for official Microsoft/.NET documentation (e.g. `TensorPrimitives`
     overloads, `IAsyncEnumerable`/channel patterns, `Tensor<T>`, source generators, ASP.NET Core
     streaming responses).
   - **code-memory MCP** — for learning code symbols and their relationships before editing
     (`sql_query` over `SymbolRecord`/`ChunkRecord`/`RelationshipRecord`, `find_related_code`,
     `get_edit_context`, `impact_analysis`).
   When launching sub agents, **always instruct them to use both MCPs where they make sense.**

---

## Phase 0 — Preparation (already complete)

- ✅ Reviewed `samples/README.md`, `samples/NivaraIncident/IDEA.md`, ADR-001..004.
- ✅ Audited the current core via sub agents (using both MCPs) against every IDEA.md capability.
- ✅ Produced `samples/NivaraIncident/README.md` with the verified gap inventory.
- ✅ Confirmed rank/rolling/window/streaming/ADR-004/ADR-001 work is **already shipped** on `main`
  (see README "Already fixed" table).

---

## Phase 1 — Core gap-fills (fix the library first)

The sample cannot be built honestly until these analytical operations exist. Deliver each with
unit tests, and (where parity exists) Polars/NumPy cross-validation fixtures.

### 1.1 Percentile / quantile / median aggregation
- **Gap:** README gap 1. No public `Quantile`/`Percentile`/`Median` anywhere in core.
- **Work:**
  - `QuantileAggregation` and `MedianAggregation` in `src/Nivara/Operations/AggregationFunction.cs`
    (follow the `SumAggregation`/`MeanAggregation` pattern incl. null handling).
  - `NivaraSeries<T>.Quantile(double q)` / `.Median()` public methods.
  - `ColumnExpressions.Quantile(...)` expression node so group→aggregate→rank plans can carry it.
  - Numeric-type dispatch consistent with the existing 17-type aggregation domain
    (`TypeCompatibilityValidator.GetNumericTypes()`), with widening rules where needed.
- **Validation:** extend `samples/NivaraIncident/Python/gen_reference.py` to emit
  `quantile`/`median` fixtures into `samples/data/polars-window/manifest.json` (or a sibling
  `polars-quantile/` manifest); new `tests/Nivara.Tests` cross-validation tests.
- **MCP guidance for sub agent:** code-memory (`find_related_code` on `SumAggregation`,
  `MeanAggregation`, `AggregationFunction`) + microsoft-learn (`.NET 10` quantile/percentile
  intrinsics — is there a `TensorPrimitives` or LINQ helper, or must it be a partitioned
  selection algorithm?).

### 1.2 StdDev / Variance aggregation
- **Gap:** README gap 2. `TensorPrimitives.StdDev` exists but is only used by an internal
  `TryNormalize*` helper.
- **Work:** `StdDevAggregation` / `VarianceAggregation` in `AggregationFunction.cs`,
  `NivaraSeries<T>.StdDev()/.Variance()` (population + sample overloads), expression node.
  Anomaly detection (z-scores over latency/error-rate) needs this.
- **Validation:** NumPy parity fixture (`np.std` population/sample); unit tests for null masking.

### 1.3 Public execution diagnostics on the query path
- **Gap:** README gap 3. `ExecutionEngine.LastDiagnostics` is on an internal class;
  `QueryFrame.Collect()` discards diagnostics; `NivaraExecutionContext` is internal; rows
  read/returned/materialized columns are not instrumented at all.
- **Work (preferred — product-visible, per README):** add public diagnostics accessors on
  `QueryFrame` (e.g. `GetExecutionDiagnostics()`) and wire a default
  `ExecutionDiagnostics` into the engine when the caller requests it; add `RowsRead` /
  `RowsReturned` / materialized-column counters to `OperationDiagnostics`/`ExecutionDiagnostics`
  and populate them in the execution strategies (Lazy/Eager/Parallel/Streaming).
- **Alternative (sample-only):** add `NivaraIncident` to `InternalsVisibleTo` and drive the
  internal `ToQueryPlan()` + `NivaraExecutionContext` + `ExecutionEngine` route (the pattern
  proven in `ExpressionEvaluatorTypedFastPathTests.cs`). Decide based on whether the diagnostics
  UI is core surface or sample-only; product direction says make it public.
- **MCP guidance:** code-memory `impact_analysis` on `ExecutionEngine.LastDiagnostics` before
  changing the public surface (blast radius across tests/extensions).

### 1.4 Parquet chunk streaming: single reader, metadata parsed once
- **Gap:** README gap 4. `ParquetLazySource` opens a new `FileStream`+`ParquetReader` per chunk
  (re-parses thrift metadata every chunk); `Execute()` uses sync-over-async; `ReadParquetStreaming`
  is a whole-file stub.
- **Work:**
  - Reuse one `ParquetReader` (and its metadata) across `ReadChunk`/`ReadChunkAsync` calls;
    seek row groups by `rowGroupIndex` instead of reopening.
  - Replace `.GetAwaiter().GetResult()` in `ParquetDataSource.Execute()` with a proper async path
    (safe in CLI/server, but the IDEA flags UI SynchronizationContext; make it robust).
  - Honor `chunkSize` as a hint below row-group granularity only if feasible; otherwise document
    row-group alignment as the contract (row-group-aligned chunks are the honest replay model).
  - Exercise `ParquetWriteOptions.With(rowGroupSize: ...)` in the dataset generator so replay
    timing is meaningful.
- **MCP guidance:** microsoft-learn for Parquet.Net chunked/row-group read patterns; code-memory
  `find_related_code` on `ParquetDataSource`, `NivaraParquetReader` before editing.

### 1.5 Streamed row materialization (`ToObjectsAsync`)
- **Gap:** README gap 7. `ToListAsync` materializes the whole frame first.
- **Work:** add `IAsyncEnumerable<T> ToObjectsAsync(...)` on `NivaraQuery<T>` (per-chunk row
  projection) so the CLI can print "Streamed N rows (M chunks)" with constant memory. If the core
  change is large, ship a sample-local chunk projection first and record the core gap as an
  issue (per the escalation rule).
- **MCP guidance:** microsoft-learn for `IAsyncEnumerable<T>` producer patterns + `Channel<T>`.

### 1.6 Phase 1 completion marker
Run this when 1.1–1.5 are all shipped (this is the explicit "Phase 1 done"
step; tick each item and commit):
- [x] `dotnet build Nivara.slnx` clean (0 warnings/errors).
- [x] Full test suite green — full `dotnet test` run;
      targeted 1.4 (`ParquetStreamingTests`) and 1.5 (`NivaraQueryToObjectsAsyncTests`)
      suites are green (15 passed) plus the modified `ParquetLazySource_ScanQuery_PersonTypedRows`.
- [x] `samples/NivaraIncident/README.md` gap inventory updated: gaps 1, 2, 3, 4, 7 → *resolved*
      with commit/issue references; gap 8 stays *open* (Phase 3 design-test, escalated there).
- [x] `docs/TODO.md` removed as executed (commit: `docs: remove TODO.md — plan executed`).
- [x] This file's Status header updated to reflect the completed Phase 1 and the deferred Phase 2+.
- [x] Phase 2+ deferred work remains tracked only in issue #284 (never re-expanded in memory).

---

## Phase 1 → Phase 2+ handoff notes

Facts the Phase 2+ agents/teams need from the Phase 1 branch (`khurram/incident`):

- **Scope as executed (maintainer decision 2026-08-16):** this branch ships Phase 1 only
  (1.1–1.5). Phase 2 (2.1–2.2), Phase 3 sample, Phase 4 bench, and web UI (3.5) are tracked in
  issue #284 and must build on top of these fixed APIs.
- **1.1:** Quantile/Median shipped via the aggregation classes (`AggregationFunction.cs`) and
  `NivaraSeries<T>.Quantile/Median`. The `ColumnExpressions.Quantile` expression node did
  **not** ship — deferred to issue #277. Group→aggregate→rank plans (Phase 3, README gap 8) may
  need it; resume from #277.
- **1.3:** execution diagnostics are now public: `QueryFrame.LastExecutionDiagnostics` and
  `QueryFrame.GetExecutionDiagnostics()` (incl. `RowsRead` / `RowsReturned` /
  `MaterializedColumns`). The plan's "alternative sample-only route" (`InternalsVisibleTo`) is
  obsolete — use the public surface for the Phase 3.4 CLI summary and 3.5 query-plan view.
- **1.4:** row-group-aligned chunks are the contract (`chunkSize` below row-group granularity is
  only a hint). One `ParquetReader` is reused for the source lifetime with metadata parsed once,
  guarded by a `SemaphoreSlim` — Parquet.Net readers are **not** thread-safe and
  `ParallelExecutionStrategy` issues concurrent `ReadChunkAsync` calls, so the guard is required.
  `Execute`/`ReadChunk` are true sync paths; `ExecuteAsync`/`ReadChunkAsync` are async.
  `ReadParquetStreaming` now yields one frame per row group. Phase 3.1 generator should write
  small row groups (`ParquetWriteOptions.With(rowGroupSize: ...)`) and 3.2 replay should consume
  chunked `QueryFrame.AsStream`.
- **1.5:** `NivaraQuery<T>.ToObjectsAsync` gives constant-memory per-chunk row projection — use it
  for the Phase 3.4 CLI streamed output.
- **Phase 2 anchors:** the line numbers in 2.1/2.2 (e.g. `ReverseGradOperations.cs:2423-2441`)
  were recorded before the ADR-001/span-ification refactors and may be stale — re-grep fresh
  before editing, and re-run the ADR-001 null-audit sweep before and after (the domain interior
  must stay null-free; boundary checks stay).
- **Test baseline:** 3028 tests green after 1.3 (0 failures); keep it that way, and ask the
  human before running `dotnet test`.

---

## Phase 2 — AutoDiff ADR-001 cleanup + SIMD (small, high-value, low-risk)

> **Status: DEFERRED** to a follow-up (issue #284). Do these while the sample is being built;
> each is a verified, isolated improvement.

### 2.1 Dead branch removal inside the non-null domain
- `Gather` (`ReverseGradOperations.cs:2423-2441`): drop the unreachable `TryGetSpan` `else`
  indexer fallback.
- `BroadcastGradient` (`GradOperationKernels.cs:241-258`): drop the `TryGetSpan` + `ArrayPool`
  fallback; keep the single `Array.Fill` path.
- Re-run the ADR-001 audit (grep for `HasNulls|NullMask|nullMask|IsNull(|WithoutNulls|TryGetNullMask`
  in `src/Nivara/AutoDiff/`) before and after; the domain interior must stay null-free and the
  boundary checks (constructors, `TensorDataset`, `AsSpan`) must remain.

### 2.2 `Pow` routed through the shared SIMD kernel
- `ReverseGradOperations.Pow` (`:1637-1655`) is the inconsistent outlier — scalar `Math.Pow`
  loop while forward-mode and `GradOperationKernels` use `TensorPrimitives.Pow`. Route reverse
  `Pow` through `ApplyPow`/`ApplyPowGradient`; add a PyTorch-parity regression test (NivaraTorch
  fixtures cover Pow — regenerate/extend).

### 2.3 Verify optimizer SIMD coverage (no code needed if complete)
- Float/double/Half each dispatch to dedicated SIMD kernels in Adam (`Adam.cs:83/97/111`) and
  AdamW (`AdamW.cs:83/85/97`); the trailing scalar loop is a defensive fallback for other
  `IFloatingPointIeee754<T>` types (none in net10). Confirm with a quick kernel-selection check;
  no new `double` kernel is required (earlier audit claim was incorrect).
- **Validation:** existing optimizer tests + a `double` training parity test.

### 2.4 Secondary SIMD candidates (if time permits)
- RMSNorm grad chain (`GradOperationKernels.cs:226-227`).
- `BroadcastMultiply`/`BroadcastAdd` per-channel-run `TensorPrimitives`
  (`ReverseGradOperations.cs:2651-2683, 2723-2749`; `ForwardGradOperations.cs:1357-1454`).
- SGD momentum step (`SGD.cs:20-39`).

---

## Phase 3 — The Incident Lab sample itself

> **Status: DEFERRED** to a follow-up (issue #284).

Layout (IDEA §"Architecture"): `IncidentLab.Core` / `IncidentLab.Analysis` / `IncidentLab.Ingestion`
/ `IncidentLab.App` / `IncidentLab.Cli` / `IncidentLab.Web` / `IncidentLab.Tests`, or a slimmer
project set if the modular split is overkill — decide during execution (the IDEA prefers a modular
monolith; the complexity must be in the data and workload, not infra ceremony). Add projects to
`Nivara.slnx`.

### 3.1 Dataset generator + incident scenarios (`Core`)
- Deterministic seeded RNG (`Random` with fixed seed, no randomness in scenarios).
- Schema modeled on the IDEA telemetry records: `Timestamp`, `Service`, `Endpoint`, `DurationMs`,
  `StatusCode`, `Region`, `TraceId`, plus `Dependency` edges and `Deployment` events.
- 10M+ requests, 50+ endpoints, 10 regions, 100+ instances, configurable scale.
- Scenarios A (db degradation), B (bad deploy), C (traffic spike), D (regional failure).
- Write Parquet with small row groups (exercises 1.4) + CSV variant for streaming tests.
- Use `TensorPrimitives`/spans for bulk value generation (vectorize where types allow).

### 3.2 Ingestion + replay stream (`Ingestion`)
- `Parquet.ScanAsQueryFrame` / `Csv.ScanAsQueryFrame` entry points.
- Replay: `IAsyncEnumerable<Chunk>` driven by `QueryFrame.AsStream(chunkSize, ct)` with
  `await using`; measure backpressure (bounded channel in flight), cancellation end-to-end,
  peak memory vs budget (exercise gap 6 and report honestly).
- **Design-test:** historical Parquet path vs replay path should converge on the same Nivara
  analytical queries. If they require radically different APIs, that is the core feedback the
  sample exists to produce.

### 3.3 Analysis queries (`Analysis`) — the "library API" surface
Implement, as plain Nivara query code, the incident answers:
- **A:** degradation ordering — rolling error-rate windows per service, `Shift` deltas, first
  service whose delta crossed a threshold; propagation delay = time between successive service
  crossings; retry amplification = retry volume / initial error count.
- **B:** deployment correlation — rank services by error-rate delta (`PercentRank`/`Rank`),
  correlate with `Deployment` events; time-to-customer-impact; affected endpoint/error-type
  breakdown (group→aggregate→rank→filter).
- **C:** saturation ordering — per-service latency P50/P95/P99 (needs 1.1), z-scores (needs 1.2),
  queue-depth rolling windows, recovery ordering via `Lead` deltas.
- **D:** regional partitioning — GroupBy region + per-region rank/rolling/percentile analysis.
- Top-k impacted services: error-rate Δ + `TopKDescending` (already on `NivaraSeries<T>`).
- Computed ordering: sort by calculated score (`Select` a score expression, `Sort` by it).
- **Design-test:** group→aggregate→rank→filter plan representability (README gap 8) — record
  findings and escalate to core (e.g. `Aggregate`/`Having`/`Where` alias) with evidence.

### 3.4 CLI (`Cli`)
- `dotnet run -- incident generate|analyze|replay <dataset>`; `--stream` mode using 3.2.
- Output: top impacted services (rank table), correlated event, execution summary
  (operators, fused kernels, peak memory, elapsed, rows read/returned — from 1.3).

### 3.5 Web UI (`Web`) — Milestone 2
- ASP.NET Core minimal API + static client (Server-Sent Events or `IAsyncEnumerable<Chunk>`
  response for live replay).
- Views: Timeline, Services, Endpoints, Regions, Dependencies, Errors, Deployments, Query Plan.
- Query-plan view renders `ExplainPlan()` / `GetDiagnosticInfo(mode)` / new public diagnostics
  (1.3) as the "Logical plan → Physical kernels → Diagnostics" visual.

### 3.6 Tests
- NUnit project; name `Method_Scenario_ExpectedBehavior`.
- Cross-validation fixtures (Polars for rank/rolling/quantile/stddev; numpy where applicable).
- Parity test: replay analysis == offline analysis on the same snapshot (the convergence claim).
- Streaming property tests over chunk sizes (following `StreamingExecutionStrategyTests` patterns).

---

## Phase 4 — Performance assessment

> **Status: DEFERRED** to a follow-up (issue #284).

Produce a small benchmark/report (CLI `--bench` or a `Nivara.PerformanceTests`-style harness):

1. End-to-end analyze of the full dataset: elapsed, rows read/returned, peak memory vs budget.
2. Streaming vs eager: memory curve for a window-heavy query; does `AsStream` stay chunked or
   fall back to single-frame (gap 5)? Measure and decide whether cross-chunk window computation
   is a core improvement (file an issue with evidence if so).
3. Kernel-selection visibility: what % of kernels run vectorized (fused/SIMD) vs scalar.
4. AutoDiff impact (2.x): before/after for `Pow` SIMD and RMSNorm-grad on a training microbenchmark.

Record results in the sample README's Performance section. Escalate any core limitation found
as a GitHub issue referencing the sample.

---

## Definition of done

> **On `khurram/incident` the DoD is scoped to Phase 1 only:** the Phase 1 items below, plus the
> 1.6 completion marker. The full DoD (replay/CLI/Web UI convergence) applies when Phase 2+ land.

- All README gap items marked **open** are either fixed in core, worked around in the sample with
  an escalation issue recorded, or explicitly accepted with evidence.
- `dotnet build Nivara.slnx` passes; all existing tests pass (1948+ baseline grows); new tests
  cover every core change (quantile/median/stddev, diagnostics, Parquet reader, `Pow` SIMD).
- The CLI can generate a dataset, analyze it, and replay it streamed; the Web UI shows the
  timeline + query plan (or is explicitly deferred to Milestone 2).
- Replay and offline analysis converge on the same Nivara queries (the core validation).
- Execution diagnostics are visible through a public surface (or the escalation issue is open).
- `samples/NivaraIncident/README.md` is updated: gaps move from *open* to *resolved* with
  file/issue references; performance numbers are real, not illustrative.
- No regressions in ADR-001 boundary enforcement (audit sweep clean in the domain interior).

## Execution notes for the next session

- **Ask before running `dotnet test`** (repo rule); verify with a targeted build first.
- Start with Phase 1.1/1.2 (blocking for the sample), then 1.3 and 2.x (small, isolated), then the
  sample milestones. Keep each change unit small and reviewable.
- Use `dotnet build Nivara.slnx` after each project change.
- When launching sub agents, include: *"Use the code-memory MCP to learn symbols/relationships and
  the microsoft-learn MCP for official API documentation where relevant."*
