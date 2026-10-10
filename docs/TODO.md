# Plan — #564: IncidentLabBenchmark still carries a fourth copy of the dataset generator

Branch: `khurram/564` (off `main`; `#563` merged as PR #573).

> As each task executes, if you find deferred work or a concern (known limitation,
> follow-up, refactor) that is outside this plan, create a tracked issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
> **GitHub issues log** below. Do not rely on memory and do not wait until the plan
> finishes — compaction during execution loses items.

## Problem

`tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` carries `GenerateSmallWithRowGroupSize`
(lines 199-260), a fourth private reimplementation of
`samples/Nivara.Samples/Incident/DatasetGenerator.cs`. Its only caller is
`TestParquetRowGroups` (line 142), which asserts only `parquetReader.RowGroupCount > 1`.

It is a partial reimplementation of the canonical generator:

- 5,000 request rows with the remainder distributed across the first 20 minutes (5,000 rows,
  no trailing defaults after `#563`);
- only 5 services and 4 regions (`gateway, catalog, orders, checkout, payments` /
  `us-east-1, us-west-2, eu-west-1, ap-south-1`) against the canonical 8 services and
  10-region per-service endpoint tables;
- fixed `statusCode = 500` at a 10% rate; no `affectedRegion` and no per-event latency ramp;
- writes only `requests.parquet` and `requests.csv`; the canonical generator also writes
  `deployments`, `dependencies` and `instances`.

`#563` fixed the trailing-default defect in this copy (`078d91d6`, "tighten the incident
field-range gate and its duplicate generator"), so the copy is now a correct-but-divergent
duplicate. The fix was deliberately in place; `#564` removes the copy entirely.

## Decision (chosen with the human)

Fold the row-group fixture into `DatasetGenerator.GenerateFromRecordCount` rather than keep a
corrected private copy, and record the `--dataset-test` runtime before and after because the
canonical generator writes three extra files.

## Proposed changes

### 1. `tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs`

- Line 142 — replace `GenerateSmallWithRowGroupSize(tempDir, "A", rowGroupSize: 1000)` with
  `DatasetGenerator.GenerateFromRecordCount(tempDir, "A", 5_000, rowGroupSize: 1000)`.
- Delete `GenerateSmallWithRowGroupSize` (lines 199-260).
- Reword the progress line at 141 so it no longer implies a requests-only write.

Row-group arithmetic: `totalRecords = 5_000`, `rowGroupSize = 1_000`,
`rowGroupCount = ceil(5_000 / 1_000) = 5` (`NivaraParquetWriter.cs:440`), so
`RowGroupCount > 1` holds.

### 2. `tests/Nivara.PerformanceTests/README.md`

- Record the before/after `--dataset-test` runtime on the existing `--dataset-test`
  bullet (precedent: `--cpu-gemm` "Runs in ~7 s").

## Blast radius

| File / symbol | Change | Depends on it |
|---|---|---|
| `IncidentLabBenchmark.GenerateSmallWithRowGroupSize` | deleted | only `TestParquetRowGroups` |
| `IncidentLabBenchmark.TestParquetRowGroups` | calls the canonical generator | manual `--dataset-test` mode |
| `IncidentLabBenchmark` (`Scenarios` use) | last use of `Scenarios` removed | none — `DatasetGenerator` still imported and used |

No production code changes. No CI surface: CI never invokes `Nivara.PerformanceTests`
(`.github/` has no `--dataset-test` reference).

## Verification

- `dotnet build Nivara.slnx`.
- Manual harness (heavy; Release only, ask first):
  `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --dataset-test`
  — expect all five `PASS` lines, with `ParquetRowGroups` printing `5 row groups OK`.
- No new timing assertions, so no `[Category("Performance")]` needed.

## Planned commits

1. `refactor(perf): fold the incident row-group fixture into DatasetGenerator` —
   `IncidentLabBenchmark.cs`.
2. `docs(perf): record the --dataset-test runtime for the row-group fixture` — `README.md`.

## Probe / harness lifecycle

No new probe or harness. The existing `--dataset-test` mode is the verification surface; no temp
artifact is created.

## GitHub issues log

- None created during this work. `#564` is the tracked issue.

## G1 grounding

- **microsoft-learn** — `Random(Int32)` is documented as initializing "using the specified seed
  value" (learn.microsoft.com/dotnet/api/system.random.-ctor). Both the deleted helper and the
  canonical `GenerateFromRecordCount` seed `new Random(42)`, so swapping them preserves
  determinism in kind. `TestParquetRowGroups` compares no bytes (it counts row groups), so no
  golden fixture depends on the swap. No Tensor/Vector or new API surface is touched — the change
  is deleting a helper and calling an existing one.
- **code-memory** — `RelationshipRecord` holds no edge for `GenerateSmallWithRowGroupSize`
  (private static, same-file caller not captured as a symbol relationship), so the single caller
  `TestParquetRowGroups` was confirmed by grep. `get_symbol_history` shows the file's last change
  is `078d91d6` ("tighten the incident field-range gate and its duplicate generator") — the `#563`
  fix that corrected this copy in place and that `#564` now supersedes.
- **Blast radius (confirmed)** — `GenerateSmallWithRowGroupSize` is referenced only at line 142
  of `IncidentLabBenchmark.cs`; the file is invoked only by `--dataset-test`
  (`Program.cs:45`), which CI never runs (`.github/` has no reference). No production code path.

### Decisions / red flags

- **Resolved:** the earlier plan (`docs/TODO.md` on `khurram/563`) said to fix the duplicate in
  place; `#563` did exactly that and merged (#573). `#564` therefore deletes a
  correct-but-divergent copy rather than a defective one. No churn remains.
- **Open, to raise at measurement time:** the issue asks for a `--dataset-test` runtime before and
  after. The "before" must be captured on pre-change code, and one full run generates ~40M rows
  across the other four legs, so the ~34k extra rows this change adds to `TestParquetRowGroups`
  are expected to fall inside run-to-run noise. The human will be asked before either heavy run.
