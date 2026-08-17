# NivaraIncident — Implementation Plan

**Status:** Phase 1 core gap-fills (1.1–1.5) are **complete**:
1.1 ✅, 1.2 ✅, 1.3 ✅, 1.4 ✅, 1.5 ✅ (see the 1.6 completion marker below). Phase 2
(2.1–2.4) is **complete** on `khurram/incident-phase2` — see the "Phase 2 → Phase 3 handoff
notes" section. Phase 3a (CLI) is **complete** on `khurram/incident-3` — see the
"Phase 3a → Phase 3b handoff notes" section. Phase 3b and Phase 4 are next.
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

## Phase 1 → Phase 2+ handoff notes (SUPERSEDED)

> Phase 2 is now complete on `khurram/incident-phase2`. The notes below are kept for historical
> reference; the authoritative handoff for Phase 3 is the "Phase 2 → Phase 3 handoff notes"
> section above.

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

> **Status: COMPLETE** on `khurram/incident-phase2` (513 targeted tests green, 0 failures).
> 2.1 ✅, 2.2 ✅, 2.3 ✅, 2.4 ✅.

### 2.1 Dead branch removal inside the non-null domain ✅
- `Gather` (`ReverseGradOperations.cs`): dropped the unreachable `TryGetSpan` `else`
  indexer fallback.
- `BroadcastGradient` (`GradOperationKernels.cs`): dropped the `TryGetSpan` + `ArrayPool`
  fallback; kept the single `Array.Fill` path.
- ADR-001 audit clean: 7 boundary matches only, zero interior matches.

### 2.2 `Pow` routed through the shared SIMD kernel ✅
- `ReverseGradOperations.Pow` now routes through `GradOperationKernels.ApplyPow`/`ApplyPowGradient`
  (TensorPrimitives.Pow SIMD) instead of scalar `Math.Pow`. Forward-time `aArr` copy allocation removed.
- NivaraTorch parity fixture (`pow`): forward + backward vs PyTorch `gen_reference.py`.
- Hand-computed gradient tests: integer exponent (2.0) and fractional (0.5).

### 2.3 Verify optimizer SIMD coverage ✅
- Confirmed float/double/Half dispatch to dedicated SIMD kernels in Adam and AdamW.
- New `Adam_TrainingLoop_FloatAndDouble_ProduceEquivalentTrajectories` test validates double path.

### 2.4 Secondary SIMD candidates ✅
- **RMSNorm grad chain** (`GradOperationKernels.cs`): element loop replaced with
  `TensorPrimitives.Multiply` + `TensorPrimitives.MultiplyAdd` (SIMD).
- **Broadcast per-channel-run SIMD** (reverse + forward `BroadcastMultiply`/`BroadcastAdd`):
  element-by-element loops replaced with per-run `TensorPrimitives.Multiply`/`Add`/`Dot`/`Sum`.
  Scale/bias gradients use SIMD Dot/Sum per channel run.
- **SGD momentum step** (`SGD.cs`): element loops in `stepNoMomentumInPlace` and
  `stepWithMomentumInPlace` replaced with `TensorPrimitives.Multiply`/`Add`/`Subtract`/`MultiplyAdd`
  (ArrayPool temp buffer for intermediates).

---

## Phase 2 → Phase 3 handoff notes

Facts the Phase 3 agents/teams need from the Phase 2 branch (`khurram/incident-phase2`):

- **2.1:** ADR-001 audit remains clean (7 boundary matches only). Dead nullable fallbacks removed
  from `Gather` and `BroadcastGradient`; domain interior is fully null-free.
- **2.2:** Reverse-mode `Pow` now routes through the shared `GradOperationKernels.ApplyPow`/`ApplyPowGradient`
  (SIMD via `TensorPrimitives.Pow`). Any Phase 4 AutoDiff microbenchmark (Phase 4 item 4) should
  use Pow to show the SIMD impact; NivaraTorch `pow` fixture (forward + backward) exists for
  regression. The scalar `Math.Pow` path and the forward-time `aArr` copy allocation are eliminated.
- **2.3:** Optimizer float/double/Half SIMD kernel routing confirmed — no new kernel work needed
  for Phase 4 training microbenchmarks. The `Adam_TrainingLoop_FloatAndDouble_ProduceEquivalentTrajectories`
  test validates cross-type training parity.
- **2.4:** RMSNorm grad, Broadcast per-channel runs, and SGD momentum are all SIMD-accelerated.
  Phase 4 kernel-selection visibility (% vectorized) should reflect these. The broadcast ops
  now use per-run `TensorPrimitives.Multiply`/`Add`/`Dot`/`Sum` instead of element-by-element
  loops; SGD uses `TensorPrimitives.Multiply`/`Add`/`Subtract`/`MultiplyAdd` with an
  `ArrayPool`-rented temp buffer.
- **Stale-anchor caveat:** line numbers in Phase 2 specs drifted before execution — always
  re-grep fresh before editing any AutoDiff file (code evolves between phases).
- **ADR-001 audit remains clean** after 2.1; run the same grep before/after any future
  AutoDiff interior changes.

---

## Phase 3a → Phase 3b handoff notes

Facts the Phase 3b (Web UI) and Phase 4 (benchmarks) agents need from Phase 3a (`khurram/incident-3`):

### What was built

| File | Purpose |
|------|---------|
| `samples/Nivara.Samples/Incident/Schema.cs` | 4 sealed record types: `RequestTelemetry`, `DeploymentEvent`, `ServiceDependency`, `InstanceState` |
| `samples/Nivara.Samples/Incident/Scenarios.cs` | 4 deterministic scenarios (A–D) with event timelines and affected services |
| `samples/Nivara.Samples/Incident/DatasetGenerator.cs` | 10M+ record generator using seeded RNG + Box-Muller latency; outputs Parquet + CSV |
| `samples/Nivara.Samples/Incident/Ingestion.cs` | `LoadParquet`, `LoadCsv`, `StreamChunks` wrappers |
| `samples/Nivara.Samples/Incident/Analysis.cs` | 5 analysis methods + typed LINQ group-by example |
| `samples/NivaraIncident.Cli/Program.cs` | CLI: `generate`, `analyze` (with `--stream`), `replay` commands |

### Key API findings (Gap 8 design-test)

1. **`NivaraFrameExtensions.GroupBy(frame, keys, aggregations)` is a trap.** It validates
   columns, builds `GroupByOperation` **without aggregations**, and returns only the grouped keys.
   Its own comment says "simplified implementation". Do **not** use it for real grouped aggregation.

2. **Typed LINQ works for group→aggregate.** `frame.Query<T>().GroupBy(r => r.Prop).Select(g => ...)` with
   `Count()`, `Sum()`, `Average()` on the grouped query produces correct aggregation results.
   Requires `class, new()` constraint on the row type — positional records must include a
   parameterless constructor.

3. **Manual aggregation after `Collect()` is the other fallback.** Collect the filtered frame, then
   aggregate in a dictionary loop using typed `GetColumn<T>().GetValue(i)`. Works but is not
   fused.

4. **`ColumnExpression` has no `&`/`&&` operator.** Compound filters must be chained as
   `.Filter(A).Filter(B)` instead of `.Filter(A & B)`.

5. **`ColumnExpression` has no ternary operator.** No `cond ? litA : litB` — use separate columns
   or compute the derived column after `Collect()`.

6. **`QueryFrame.AsStream` requires `IAsyncEnumerable` iteration.** Use `await foreach`, not
   `foreach`. The CLI entry point must be `async Task`.

7. **No `using`/`IAsyncDisposable` on `QueryFrame` in the analysis methods.** The caller
   disposes. `QueryFrame` implements `IAsyncDisposable`.

### Test strategy

- **`tests/Nivara.Tests/Incident/ScenarioTests.cs`** — fast, no data generation. Tests scenario
  properties, determinism, case-insensitive lookup, boundary invariants.
- **`tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs`** — full dataset generation + all 5
  analyses for all 4 scenarios. Runs as a console app, not in CI.
- Ingestion/analysis integration tests were moved out of `Nivara.Tests` to keep CI fast (dataset
  generation takes seconds). Re-add them in `Nivara.Tests/Incident/` if generation becomes
  fast enough (e.g., scale=0 with tiny data).

### What Phase 3b (Web UI) should build on

- Reuse `Nivara.Samples/Incident/` directly — no changes needed to the library code.
- Add `samples/NivaraIncident.Web/` project referencing `Nivara.Samples`.
- The 5 analysis methods in `Analysis.cs` are the data source for the dashboard.
- `QueryFrame.GetExecutionDiagnostics()` (public since Phase 1.3) is available for the
  diagnostics/query-plan view.
- `QueryFrame.ExplainPlan()` is available for the query-plan panel.
- `QueryFrame.AsStream()` enables SSE streaming for live-replay views.

### What Phase 4 (benchmarks) should measure

- End-to-end `analyze` elapsed + row counts for each scenario at scale 1 and scale 10.
- Streaming (`--stream`) vs eager: memory curve; does `AsStream` stay chunked or fall back?
  (Gap 5 measurement.)
- Dataset generator throughput: rows/second and MB/second at different scales.
- Group-by performance: typed LINQ path vs manual aggregation path.
- All numbers go into `samples/NivaraIncident/README.md` Performance section.

### Commits

| Commit | What |
|--------|------|
| `36c0e67` | feat(incident): add project scaffolding and telemetry schema |
| `af66d00` | feat(incident): add deterministic dataset generator and incident scenarios |
| `e829eae` | feat(incident): add ingestion wrappers and replay helpers |
| `ee1cab0` | docs: plan Phase 3a Incident Lab CLI in TODO.md |
| `fb2af76` | feat(incident): add analysis queries for the Incident Lab |
| `fbebf87` | feat(incident): implement CLI entry point and update README |

---

## Phase 3 — The Incident Lab sample itself

> **Status: Phase 3a COMPLETE** on `khurram/incident-3`. Phase 3b (Web UI) is deferred after
> Phase 4 (benchmarks). See "Phase 3a → Phase 3b handoff notes" below.

### Architecture decision (maintainer, 2026-08-17)

The IDEA's 7-project layout (`IncidentLab.Core` / `.Analysis` / `.Ingestion` / `.App` /
`.Cli` / `.Web` / `.Tests`) is overkill for a reference sample. The guiding principle says
*complexity should be in the data and workload, not infra ceremony*. After review:

1. **All analytical code lives in `samples/Nivara.Samples/Incident/`** (folder in the existing
   class library). This includes schema, generator, scenarios, ingestion, and analysis queries.
   `Nivara.Samples` already references `Nivara` and `Nivara.Extensions` — no new dependencies
   needed.
2. **One thin CLI executable:** `samples/NivaraIncident.Cli/` — `OutputType=Exe`, project-references
   `Nivara.Samples`. The CLI does arg parsing and formatted output; all logic is in the library.
3. **Tests in the existing test project:** `tests/Nivara.Tests/Incident/` — no new test project.
4. **No third-party CLI library** — follows repo convention (raw `args[]`, `switch` on `args[0]`,
   hand-rolled `--flag` parsing; see `NivaraChat/Program.cs`, `NivaraVAE/Program.cs`).
5. **Phase 3a (CLI) and Phase 3b (Web) are separate phases.** Web adds one more project
   (`NivaraIncident.Web`) that references the same `Nivara.Samples/Incident/` code.

**Resulting project count: 1 new exe project for 3a, +1 more for 3b (2 total).** Not 7.

### Resulting layout

```
samples/
├── Nivara.Samples/Incident/          # shared library code (class library, no new .csproj)
│   ├── Schema.cs
│   ├── Scenarios.cs
│   ├── DatasetGenerator.cs
│   ├── Ingestion.cs
│   └── Analysis.cs
│
├── NivaraIncident.Cli/               # Phase 3a — thin CLI executable
│   ├── NivaraIncident.Cli.csproj
│   └── Program.cs
│
├── NivaraIncident.Web/               # Phase 3b — ASP.NET Core (deferred)
│   ├── NivaraIncident.Web.csproj
│   └── Program.cs
│
└── NivaraIncident/                   # existing (IDEA.md, Python/, README.md)
```

### 3a — CLI (Milestone 1) — ✅ COMPLETE

| Sub-step | What | Status |
|----------|------|--------|
| 3a.1 | Project scaffolding + telemetry schema | ✅ `NivaraIncident.Cli.csproj`, `Schema.cs` |
| 3a.2 | Deterministic dataset generator + incident scenarios (A/B/C/D) | ✅ `DatasetGenerator.cs`, `Scenarios.cs` |
| 3a.3 | Ingestion wrappers (Parquet/CSV scan, replay stream) | ✅ `Ingestion.cs` |
| 3a.4 | Analysis queries — the Nivara analytical pipeline (the core exercise) | ✅ `Analysis.cs` |
| 3a.5 | CLI entry point with formatted output | ✅ `Program.cs` (generate/analyze/replay) |
| 3a.6 | Tests + Polars cross-validation | ✅ Fast tests in `Nivara.Tests/Incident/`, perf bench in `Nivara.PerformanceTests` |
| 3a.7 | Wire, build, end-to-end validation | ✅ 3063 tests passing, build clean |

### 3b — Web UI (Milestone 2) — see `docs/PHASE3B.md`

**Deferred as a follow-up after Phase 4.** Adds ASP.NET Core minimal API, SSE streaming,
and 8 dashboard views. Every number computed by Nivara. Not a prerequisite for benchmarking.

---

## Phase 4 — Performance assessment

> **Status: FOLLOWS 3a** — execution begins after Phase 3a (CLI) is complete.
> Produces benchmark report + core gap evidence before Phase 3b (Web UI) starts.
> Detailed execution plan: `docs/PHASE4.md`.

Produce a benchmark report with real numbers (see `docs/PHASE4.md` for detailed sub-steps):

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

> **Execution order:** Phase 3a ✅ → Phase 4 (benchmarks, next) → Phase 3b (Web UI, follow-up).
> The full DoD (replay/CLI/Web UI convergence) applies when all three land.

- All README gap items marked **open** are either fixed in core, worked around in the sample with
  an escalation issue recorded, or explicitly accepted with evidence.
- `dotnet build Nivara.slnx` passes; all existing tests pass (3063 baseline grows); new tests
  cover every core change (quantile/median/stddev, diagnostics, Parquet reader, `Pow` SIMD).
- ✅ The CLI can generate a dataset, analyze it, and replay it streamed.
- Replay and offline analysis converge on the same Nivara queries (the core validation).
- Execution diagnostics are visible through a public surface (or the escalation issue is open).
- `samples/NivaraIncident/README.md` is updated: gaps move from *open* to *resolved* with
  file/issue references; performance numbers are real, not illustrative.
- No regressions in ADR-001 boundary enforcement (audit sweep clean in the domain interior).

## Execution notes for the next session

- **Ask before running `dotnet test`** (repo rule); verify with a targeted build first.
- Execution order: Phase 3a ✅ → Phase 4 (benchmarks, next) → Phase 3b (Web UI, follow-up).
- Phase 4: run `Nivara.PerformanceTests/IncidentLabBenchmark.cs` at scale 1 and 10; measure
  streaming vs eager; record numbers in README.
- Phase 3b: add `NivaraIncident.Web/` project; reuse `Analysis.cs` methods as data source.
- Keep each change unit small and reviewable.
- Use `dotnet build Nivara.slnx` after each project change.
- When launching sub agents, include: *"Use the code-memory MCP to learn symbols/relationships and
  the microsoft-learn MCP for official API documentation where relevant."*
