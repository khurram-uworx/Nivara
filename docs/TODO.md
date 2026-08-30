# TODO plan — Issue #352: record `Row.Where nullable-element GetValue` perf baseline for #349

Branch: `khurram/352` (off `main`)

## Problem

`RunRowWhereScenarios` in `tests/Nivara.PerformanceTests/Program.cs` registers
`Row.Where nullable-element GetValue 100k`. It gates the issue #349 fix
(nullable-element `ColumnFilterHelper`/`FilterByMask` no longer boxes per
element). The row is a NEW baseline: `--compare` prints NEW (ungated) until a
baseline is recorded. Issue #352 asks for that baseline measurement.

Two additional facts found during investigation:
- **Baseline file is stale on runtime.** `baseline-v140.json` was recorded on
  .NET 10.0.11 (2026-08-20/21); the 2026-08-22 `net11` retarget moved all
  projects to net11.0, so the harness now runs on the 11.0.0-preview.7 runtime.
  The existing baseline's header no longer matches the build it gates.
- **README Results table is stale too.** Machine line says ".NET 10.0.11";
  the table does not list the new scenario.

## Plan

User decisions: full refresh of `baseline-v140.json`; full README Results table
refresh; harness run happens here during implementation.

### Edits

1. Run the harness (long — ask before starting):
   `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <tmp> --runs 3`
   Full fresh measurement on net11, includes `Row.Where nullable-element
   GetValue 100k` (positioned between `RankKernel RowNumber` and `GroupBy`).
   Record the B/op — expected to drop materially vs the ~24 B/row pre-fix
   boxing residual.
2. Replace `tests/Nivara.PerformanceTests/baseline-v140.json` wholesale with
   the harness JSON (same serializer format/ordering; header auto-records the
   new runtime/timestamp).
3. Update `tests/Nivara.PerformanceTests/README.md` Results table:
   - machine/runtime line → fresh runtime + date 2026-08-30;
   - shift Current→Prev, new numbers in Current, recompute Ratio/Δ%;
   - add `Row.Where` row (Prev/Ratio/Δ% blank) between `RankKernel` and `GroupBy`;
   - note that Prev↔Current spans a runtime change (indicative-only ratios).

### Blast radius

`tests/Nivara.PerformanceTests/baseline-v140.json` +
`tests/Nivara.PerformanceTests/README.md` only. No library behavior change.

## Verification

- `dotnet run -- ... --compare baseline-v140.json --runs 3` → Gate PASS
  (optional; another full run — ask before running).
- JSON sanity: `Row.Where` row present, in registration order, `runs` = 3.

## Planned commits

1. `docs: plan issue #352 perf baseline record in TODO.md`
2. `perf: record fresh net11 baseline incl. Row.Where nullable-element (issue #352)`
   — `baseline-v140.json` replaced with fresh harness output.
3. `docs: refresh perf Results table to net11 (issue #352)` — `README.md`.
4. `docs: remove TODO.md — issue #352 plan executed`

## GitHub issues log

- [ ] #352 — record the `Row.Where nullable-element GetValue 100k` perf harness
  baseline (gate for #349); being executed on this branch.
- [ ] #349 — the fix (nullable-element typed ColumnFilterHelper kernels),
  merged to main via PR #353; #352 records its gate baseline.
- [ ] #354 — `Frame Slice` and `AutoDiff RMSNorm fwd+bwd` gate-flaky on net11
  preview (ops/s noise with byte-identical B/op); created after a follow-up
  `--compare` run failed those rows during #352 execution.