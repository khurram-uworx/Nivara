# Plan: Issue #1385 — docs(samples): make --silk/--blas/CPU reporting consistent and transparent in NivaraIncident

## Problem

`NivaraIncident` already supports `--silk` (OpenCL) for grouped aggregation and
`--blas` (OpenBLAS) for the online AutoDiff path, but the three execution options
(CPU / Silk / BLAS) are not reported consistently across the sample README's
benchmark tables, and the measured non-wins are split across sections with
different scales, machine lines, and pass conventions.

## Proposed changes

Docs-only change to `samples/NivaraIncident/README.md`. No runtime/default
behavior change.

1. Add a reporting convention block to the Performance section (1 warmup + 5
   timed passes, median, named machine line; opt-in Silk/BLAS, managed default).
2. Restructure the Silk availability table into uniform columns:
   `Kernel / capability | Wired into --silk? | Scale | CPU (ms) | Silk (ms) | Ratio | Verdict | Notes`,
   one row per measured scale, with a machine line and pass convention, keeping
   the explicit non-wins (rolling mean/max/min, multi-stat, grouped agg).
3. Update the BLAS table to the SimdProbe layout:
   `Shape | M | K | N | managed (ms) | blas (ms) | ratio`, plus machine line
   and pass convention, keeping the small-batch caveat.
4. Add a consolidated CPU / Silk / BLAS comparison where applicable
   (grouped aggregation: CPU vs Silk, BLAS n/a; online AutoDiff learning:
   CPU vs BLAS, Silk n/a), marking unmeasured cells with reason.
5. Align the two existing timing tables (Nivara-only analysis timings and
   Nivara vs Polars) so their machine/pass-convention lines are explicit.

## Verification

- No C# changes; `IncidentSurfaceTests` and docs-coverage pins unaffected.
- Re-read rendered tables for column alignment.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
  (unaffected, but confirm clean).

## Planned commits

1. `docs: plan #1385 in TODO.md`
2. `docs(samples): make --silk/--blas/CPU reporting consistent in NivaraIncident README`
3. `docs: remove TODO.md — plan executed`

## Blast radius

- `samples/NivaraIncident/README.md` only.
- No source, tests, or pins touched.

## GitHub issues log

- (none yet)
