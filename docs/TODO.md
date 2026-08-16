# TODO — Incident Lab implementation (branch `khurram/incident`)

Full spec: `samples/Incident-PLAN.md`; gap inventory: `samples/NivaraIncident/README.md`;
product spec: `samples/NivaraIncident/IDEA.md`.

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
| 3.x Sample projects | `samples/NivaraIncident/IncidentLab.*`, `Nivara.slnx` | new tests only |
| 4 Bench report | sample CLI `--bench` | none |

---

## Planned commit list

1. `docs: plan Incident Lab implementation in TODO.md` — this file.
2. `feat(core): Quantile/Median aggregations + NivaraSeries.Quantile/Median` (1.1)
3. `feat(core): StdDev/Variance aggregations + NivaraSeries.StdDev/Variance` (1.2)
4. `feat(core): public execution diagnostics on QueryFrame + row counters` (1.3)
5. `fix(extensions): Parquet chunk streaming with reused reader + async Execute` (1.4)
6. `feat(core): NivaraQuery.ToObjectsAsync streamed row projection` (1.5)
7. `refactor(autodiff): remove dead non-null branches (Gather, BroadcastGradient)` (2.1)
8. `perf(autodiff): route Pow through SIMD GradOperationKernels` (2.2)
9. `feat(sample): IncidentLab Core — dataset generator + scenarios` (3.1)
10. `feat(sample): IncidentLab Ingestion — replay stream` (3.2)
11. `feat(sample): IncidentLab Analysis — incident queries` (3.3)
12. `feat(sample): IncidentLab CLI` (3.4)
13. `feat(sample): IncidentLab Tests — cross-validation + parity` (3.6)
14. `perf(sample): IncidentLab --bench report` (Phase 4)
15. `docs(sample): update NivaraIncident README — gaps resolved + performance` (DoD)
16. `docs: remove TODO.md — plan executed`

Web UI (3.5) is Milestone 2 — decide during execution whether it ships on this branch or is
explicitly deferred with an escalation issue.

---

## Execution order

Phase 1 core gap-fills first (1.1/1.2 block the sample), then 1.3–1.5, then Phase 2
(2.x small/isolated), then Phase 3 sample, then Phase 4 benchmark, then README/DoD.

MCP reminders for sub agents: use code-memory (`find_related_code`, `sql_query`,
`impact_analysis`) before editing core symbols; use microsoft-learn for official .NET 10
API facts (`TensorPrimitives` overloads, quantile/percentile intrinsics, `IAsyncEnumerable`
patterns).

---

## GitHub issues log

- [ ] (none yet — create issues at discovery time via
      `gh issue create --repo khurram-uworx/Nivara` and record the number here; never hold
      deferred work in memory)
