# Benchmark README refresh task template

Use this template when creating a release benchmark-table-refresh GitHub issue (the 1506-like task). It keeps the refresh scope explicit, auditable, and deterministic.

## How to use

1. Replace placeholders: `<X.Y.Z>`, `<YYYY-MM-DD>`, `<machine>`, `<owner>`, `<issue-url>`.
2. Confirm `release/v<X.Y.Z>/` contains the benchmark artefacts and `status.txt` (or record what is missing).
3. Create the issue: `gh issue create --repo khurram-uworx/Nivara --title "[release v<X.Y.Z>] Refresh benchmark README tables from release/v<X.Y.Z> artefacts" --body-file <temp-body>.md`.
4. Link the issue from the release PR. Close it only when every inventory table is refreshed from committed artefacts or explicitly carried/blocked with reason.

## Issue body template

```md
## Goal

Refresh benchmark README tables once using committed artefacts from `release/v<X.Y.Z>/`. This is a data refresh, not a benchmark rerun on a non-comparable machine.

## Rules

- Source of truth: committed files under `release/v<X.Y.Z>/` only.
- No fabricated numbers. No cross-machine/cross-session ratio recomputation.
- If data is unavailable, **skip that table/row**, leave existing values, add an explicit carried/not-refreshed reason with date/cause, and record it here.
- Do not defer refresh as the normal path; incomplete refresh means release prep incomplete unless an explicit maintainer waiver is recorded.
- Do not run benchmarks on a different/older/non-comparable machine. If captures are missing, run them later on the original benchmark machine and update this issue.

## Artefacts

- [ ] `release/v<X.Y.Z>/status.txt`
- [ ] `release/v<X.Y.Z>/nivara-perf-v<X.Y.Z>.json`
- [ ] `release/v<X.Y.Z>/nivaraincident-v<X.Y.Z>.txt`
- [ ] `release/v<X.Y.Z>/nivaraincident-1m-nivara-v<X.Y.Z>.txt` (if 1M dataset available)
- [ ] `release/v<X.Y.Z>/nivaraincident-1m-polars-v<X.Y.Z>.txt` (if Python/Polars + 1M dataset available)
- [ ] `release/v<X.Y.Z>/nivarainference-cpu-v<X.Y.Z>.txt`
- [ ] `release/v<X.Y.Z>/nivarainference-silk-v<X.Y.Z>.txt`
- [ ] `release/v<X.Y.Z>/nivarainference-blas-v<X.Y.Z>.txt`
- [ ] `release/v<X.Y.Z>/nivarafinetuning-v<X.Y.Z>.txt`
- [ ] PyTorch refs present as needed (`pytorch-*-v<X.Y.Z>.txt`)
- [ ] Any missing artefact listed below with owner/reason

## Table inventory

| Area | File | Table/section | Source artefact(s) | Status | Notes |
|---|---|---|---|---|---|
| Core perf | `tests/Nivara.PerformanceTests/README.md` | `## Results` | `nivara-perf-v<X.Y.Z>.json` | pending | Roll Current->Prev, set Current from vX.Y.Z, recompute Δ%, update B/op/gen0, recording line |
| Incident | `samples/NivaraIncident/README.md` | Nivara-only analysis timings | `nivaraincident-v<X.Y.Z>.txt` | pending | Roll Current->Prev, update medians/Ratio/Δ%/recording |
| Incident | `samples/NivaraIncident/README.md` | Nivara vs Polars | Polars capture + Nivara same-basis capture | pending | Skip/carry if no same-basis Polars artefact; no cross-basis ratios |
| Inference | `samples/NivaraInference/README.md` | OpenBLAS settings/end-to-end + main perf table | `nivarainference-*-v<X.Y.Z>.txt`, `pytorch-*-v<X.Y.Z>.txt` | pending | Refresh available rows; keep `(carried)` with reason for missing captures; update footnote to vX.Y.Z artefacts |
| FineTuning | `samples/NivaraFineTuning/BERT-TUNING.md` | Main benchmark table | `nivarafinetuning-v<X.Y.Z>.txt` | pending | Refresh measured ms/batch and slowdowns; update recording line |
| FineTuning | `samples/NivaraFineTuning/BERT-TUNING.md` | Estimated full-run table | derived from refreshed main table | pending | Recompute estimates from refreshed measurements; keep disclaimer |
| FineTuning | `samples/NivaraFineTuning/BERT-TUNING.md` | Paired CPU vs Silk #1003 B=2/4/8 | dedicated paired capture if present | pending | Skip/carry with reason if not in artefact; no fabricated matrix |

## Missing data / carry-forward

| Table/row | Reason | Artefact missing | To be captured on | Owner |
|---|---|---|---|---|
| <table/row> | <why unavailable> | <file/command> | original benchmark machine | <owner> |

## Verification

- [ ] Every refreshed number traces to a `release/v<X.Y.Z>/` artefact.
- [ ] Carried/skipped rows have explicit reason/date.
- [ ] Δ%, ratios, slowdowns, derived estimates recomputed.
- [ ] No cross-machine/cross-session ratios.
- [ ] `git diff --check` clean.
- [ ] Docs coverage/pins checked if docs/release files affect gates.
- [ ] PR links this issue and includes artefact paths.

## PR test plan

- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (if doc/code formatting scope requires)
- [ ] Targeted doc/gate tests as needed
- [ ] Manual markdown/table rendering sanity check
``` 

## Notes

- Release retrospective/review belongs in `release/v<X.Y.Z>/RELEASE-REVIEW.md`, not `docs/`.
- If `release/run-benchmarks.cmd` is missing a capture needed by a table, fix the runner in the same release-prep cycle and note it in the issue/PR.