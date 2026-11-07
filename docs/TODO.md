# Plan: refresh incident/inference/fine-tuning README benchmark tables (#991)

## Problem
Issue #991 asks that the `## Release Benchmark` README tables for
`samples/NivaraIncident/README.md`, `samples/NivaraInference/README.md`, and
`samples/NivaraFineTuning/README.md` be repopulated from captured 1.5
`%TEMP%` evidence. Each sample README's benchmark table should gain a 2026-11-07
recording line and a refreshed `Current` column backed by captured output. The
documented technique is "hidden-tag": wrap each results table with
`<!--hidden-start-->`/`<!--hidden-end-->`, regenerate rows from captured output
with a Python helper in the sample's `Python/` folder, review, then remove the
markers before committing.

## Captured evidence (%TEMP%, source-of-truth)
- `nivara-incident-quick.txt`: scenario A quick run, 6,666,665 rows;
  Degradation Ordering 9291.3 ms, Deployment Correlation 14456.5 ms,
  Saturation Ordering 91.8 ms, Regional Partitioning 14725.1 ms,
  Grouped Aggregation 7049.2 ms.
- `ni-mobilenet.txt`: 463.5 ms avg; `ni-resnet18.txt`: 123.3 ms avg;
  `ni-minilm.txt`: 78.7 ms; `ni-distilbert.txt`: 200.6 ms;
  `ni-distilbert-sst.txt`: 179.4 ms; `ni-modernbert.txt`: 1457.5 ms (seq128),
  2084.6 ms (seq256); `ni-smollm-f32.txt`: 814 ms median (32 tokens);
  `ni-smollm-bf16.txt`: 1041 ms median (32 tokens).
- `nivara-finetuning.txt`: PyTorch 377.4 ms/batch; Nivara CPU ~1.2 s/batch;
  Nivara Silk steady-state 251.7 ms/batch.

## Proposed changes

1. Incident README: shift old Nivara-only timings to Prev, new timings to
   Current, add Delta%; refresh recording line to 2026-11-07. Keep Polars table
   unchanged (no 1M rerun captured); annotate accordingly.
2. Inference README: refresh CPU rows; add Silk column only where captured
   evidence exists, else `—` rather than invent numbers.
3. Fine-tuning README: refresh performance table (PyTorch 377.4 ms/batch,
   CPU ~1.2 s/batch, Silk 251.7 ms/batch), ratios and extrapolations.
4. Update helpers in `samples/*/Python/regen_perf_table.py` to use the
   canonical `<!--hidden-start-->`/`<!--hidden-end-->` markers.
5. Remove markers and `.bak` files before commit.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (no code change, ensures docs still gate)
- `git diff` review for unintended changes; confirm no citation-count pins move.

## Planned commits
1. docs: plan #991 in TODO.md
2. docs(NivaraIncident): refresh release benchmark table (#991)
3. docs(NivaraInference): refresh release benchmark table (#991)
4. docs(NivaraFineTuning): refresh release benchmark table (#991)
5. chore: update perf-table regen helpers to canonical markers (#991)
6. docs: remove TODO.md — plan executed (#991)

## Blast radius
Documentation-only change in `samples/*/README.md` and helpers in
`samples/*/Python/`. No source/`src/` edits, no tests, no pinned doc counts
moved (no citation added/removed). Risk: low. Snippet gate unaffected.

## GitHub issues log
(none yet)
"