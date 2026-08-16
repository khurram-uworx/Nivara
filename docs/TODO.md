# TODO — Incident Lab core gap-fills, Phase 1 + Phase 2 (branch `khurram/incident`)

Full spec: `samples/Incident-PLAN.md`; gap inventory: `samples/NivaraIncident/README.md`;
product spec: `samples/NivaraIncident/IDEA.md`.

**Scope (maintainer decision 2026-08-16):** only Phase 1 (core gap-fills 1.1–1.5) and Phase 2
(AutoDiff 2.1–2.2) ship on this branch. Phase 3 (sample projects), Phase 4 (bench), README
DoD, and web UI (3.5) are **deferred** to a follow-up (tracked in the issues log). After 2.2,
`docs/TODO.md` is removed as executed.

**Rule of engagement (from maintainer):** the sample is a forcing function for the core
library. When the sample needs a workaround, fix the core first, then build the sample on
the fixed API. AutoDiff stays a non-null domain (ADR-001). Embrace Tensors/Vectors
(`TensorPrimitives`, spans, SIMD) everywhere. Use the code-memory and microsoft-learn MCPs.

**Verification:** `dotnet build Nivara.slnx` after each project change; **ask the human
before running `dotnet test`**. Existing baseline is 1948+ tests and must stay green unless
a contract deliberately changes.

---

## Blast radius

| Change | Affected files | Downstream callers / tests |
|--------|----------------|----------------------------|
| 1.1 Quantile/Median agg | `src/Nivara/Operations/AggregationFunction.cs`, `src/Nivara/NivaraSeries.cs`, `src/Nivara/Expressions/ColumnExpression.cs`, `src/Nivara/Helpers/ColumnFactory.cs`, `TypeCompatibilityValidator` | group-by aggregation tests, window/rank tests; Polars fixtures `samples/data/polars-window/` |
| 1.2 StdDev/Variance agg | same aggregation surface | same; `TensorsHelper.TryNormalize*` stays untouched |
| 1.3 Public execution diagnostics | `src/Nivara/Execution/ExecutionEngine.cs`, `ExecutionDiagnostics`, `OperationDiagnostics`, `QueryFrame.cs`, all 4 strategy files | every query-path test; Extensions I/O tests |
| 1.4 Parquet chunk streaming | `src/Nivara.Extensions/IO/ParquetDataSource.cs`, `NivaraParquetReader.cs` | Parquet read tests, `ReadParquetStreaming` callers |
| 1.5 `ToObjectsAsync` | `src/Nivara/Linq/NivaraQuery.cs` | LINQ tests, `AsStream`/`CollectAsync` tests |
| 2.1 Dead branch removal | `ReverseGradOperations.cs` (Gather), `GradOperationKernels.cs` (BroadcastGradient) | AutoDiff tests, InferenceGraphTests (ADR-001 boundary guards) |
| 2.2 `Pow` SIMD | `ReverseGradOperations.cs` (Pow) | NivaraTorch Pow fixtures |
| ~3.x Sample projects | `samples/NivaraIncident/IncidentLab.*`, `Nivara.slnx` | **DEFERRED** — tracked in issues log |
| ~4 Bench report | sample CLI `--bench` | **DEFERRED** — tracked in issues log |

---

## Planned commit list

Done:
1. `docs: plan Incident Lab implementation in TODO.md` — this file.
2. `feat(core): Quantile/Median aggregations + NivaraSeries.Quantile/Median` (1.1)
3. `feat(core): StdDev/Variance aggregations + NivaraSeries.StdDev/Variance` (1.2)
4. `Merge remote-tracking branch 'origin/main' into khurram/incident` — sync (2026-08-16)

Remaining:
5. `feat(core): public execution diagnostics on QueryFrame + row counters` (1.3)
6. `fix(extensions): Parquet chunk streaming with reused reader + async Execute` (1.4)
7. `feat(core): NivaraQuery.ToObjectsAsync streamed row projection` (1.5)
8. `refactor(autodiff): remove dead non-null branches (Gather, BroadcastGradient)` (2.1)
9. `perf(autodiff): route Pow through SIMD GradOperationKernels` (2.2)
10. `docs: remove TODO.md — plan executed`

Deferred (tracked in issues log): Phase 3 sample projects, Phase 4 bench, README DoD,
web UI (3.5).

---

## Execution order

Phase 1 core gap-fills in order 1.1 → 1.2 → 1.3 → 1.4 → 1.5, then Phase 2 (2.1 → 2.2).
No Phase 3/4 work on this branch.

MCP reminders for sub agents: use code-memory (`find_related_code`, `sql_query`,
`impact_analysis`) before editing core symbols; use microsoft-learn for official .NET 10
API facts (`TensorPrimitives` overloads, quantile/percentile intrinsics, `IAsyncEnumerable`
patterns).

---

## GitHub issues log

- [ ] [#277](https://github.com/khurram-uworx/Nivara/issues/277) —
      `ColumnExpressions.Quantile` expression-node support for quantile/median aggregations
      (deferred from 1.1; typed LINQ + series + aggregation-class paths shipped instead).
- [ ] [#284](https://github.com/khurram-uworx/Nivara/issues/284) —
      Defer Incident Lab sample/bench (Phases 3–4), web UI (3.5), README DoD to a follow-up
      (created while narrowing scope of `khurram/incident` to Phase 1 + Phase 2).
- [ ] (create issues at discovery time via
      `gh issue create --repo khurram-uworx/Nivara` and record the number here; never hold
      deferred work in memory)
