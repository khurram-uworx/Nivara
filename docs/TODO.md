# Plan: fix Core perf table Qwen full-forward sub-1 ops/s display (#989)

## Problem

`regen_perf_table.py` formats `Current` with `f"{round(r['opsPerSec']):,}"`. For
`Qwen full fwd [64 tok]` and `[256 tok]` the fresh throughputs are `0.39` and
`0.24` ops/s, which round to `0` and make the table show

- `Qwen full fwd [64 tok] | 0.6 | 0 | −100.0%`
- `Qwen full fwd [256 tok] | 0.2 | 0 | −100.0%`

The `−100.0%` is a display artifact, not a regression. The delta is also made
destructive because it is recomputed from the rounded `0`.

## Proposed changes

1. In each `samples/*/Python/regen_perf_table.py`, format `Current` with one
   decimal when `abs(opsPerSec) < 1`, otherwise keep the integer form.
2. Compute the delta from the raw `r['opsPerSec']`, not from the rounded /
   formatted cell.
3. Update the two core rows in `tests/Nivara.PerformanceTests/README.md` so the
   `Current` cell shows `0.4` and `0.2`, with deltas computed from the raw
   `0.39` / `0.24` values.
4. Apply the same formatting fix to all three sample helper copies
   (Inference/Incident/FineTuning), which are byte-identical duplicates, to
   prevent drift.

## Blast radius

- Documentation-only plus three Python helper scripts.
- No source (`src/`) or tests changed; no citation/coverage pins moved.
- `DocsCoverageTests` and snippet gate unaffected.

## Verification

- Review `git diff` for scope.
- Synthetic temp README + JSON smoke test of the helper's sub-1 branch.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (if
  build is clean) for hygiene.

## GitHub issues log

(none created during this work)
