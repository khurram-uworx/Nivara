# Plan: issue #960 — Silk (OpenCL) column in NivaraFineTuning benchmark table

## Problem

`NivaraFineTuning` has three training paths (CPU AutoDiff, Silk/OpenCL, PyTorch), but the
README "Performance benchmarks" table compares only CPU Nivara vs PyTorch. There is no Silk
column and no reproducible Silk timing. #963 (per-batch timing in `RunTrainSilk`) has landed
on main, so per-batch `Batch X/Y [time] - loss` output and steady-state ms/batch are now
available for Silk.

## Proposed changes

1. `samples/NivaraFineTuning/benchmark_timing.cmd` — add a third section running Nivara
   with `--silk` (tee'd into `benchmark_results.txt`) so one session yields all three
   columns (PyTorch CPU, Nivara CPU, Nivara Silk).
2. `samples/NivaraFineTuning/README.md` — Performance benchmarks:
   - Add `Nivara (Silk/OpenCL)` column to the comparison table (ms/batch).
   - Add honest caveats: iGPU path expected competitive only at larger batch sizes /
     decode-style workloads; small-batch (B=2..4) fine-tuning may not beat the CPU SIMD
     path. No win claimed without measurement.
   - Add scaling-study note (B=2/4/8 commands) and AC-power recording line (machine,
     date, AC-power note, driver version).
   - Update "Re-running" with the Silk command.
3. `docs/SILK.md` — mark #960 (and #963) status in the follow-up table (optional).

No kernel code or trainer logic changes. Measurement/docs + harness only.

## Verification

- `dotnet build samples/NivaraFineTuning -c Release` succeeds.
- Smoke run: `dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --silk --max-examples 1 --epochs 1 --batch-size 2` prints per-batch line (requires OpenCL + data).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- DocsCoverage pins unaffected (no new docs files / citations).

## Planned commits

1. docs: plan #960 in TODO.md
2. feat(samples): add Silk run to benchmark_timing.cmd
3. docs(samples): add Silk column + caveats to NivaraFineTuning README
4. docs(silk): update follow-up table status

## Blast radius

- `benchmark_timing.cmd`: harness only, no runtime code.
- README.md / SILK.md: documentation only.
- No public API or test changes.

## GitHub issues log

- #960 — bench: add Silk column to NivaraFineTuning benchmark table (this plan)
- #963 — feat(samples): per-batch timing in RunTrainSilk (landed on main; prerequisite)

