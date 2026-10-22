# Plan: issue #798 — native Silk-only GEMM baseline/comparison harness (no ILGPU leg)

## Problem

The only Silk timing A/B path (`--silk-kernels`) aborts when the ILGPU runtime is
unavailable, so measurement/perf tuning cannot run on Silk-only machines. Need a
native Silk-only GEMM baseline/comparison entry that reuses existing perf-harness
infrastructure and reports timings + parity against a CPU reference.

## Proposed changes

1. New `tests/Nivara.PerformanceTests/SilkGemmProbe.cs` — `SilkGemmProbe.Run(args)`:
   Silk `gemm` kernel over `GemmBenchmark.s_shapes`, parity vs ascending-K CPU
   reference (mul+add), timings via `GemmBenchmark.TimeBest`.
2. `Program.cs` — add `--silk-gemm` flag (aliases `--silk-gemm-baseline`,
   `--silk-gemm-compare`), dispatch to `SilkGemmProbe.Run`.
3. README.md — document the flag.

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-gemm`
- DocsCoveragePins unaffected (no docs/*.md count change beyond README edits? README is under tests/, not docs/).

## Planned commits

1. docs: plan #798 in TODO.md
2. feat(perf): add Silk-only GEMM baseline probe
3. docs: document --silk-gemm flag
4. docs: remove TODO.md — plan executed

## Blast radius

- tests/Nivara.PerformanceTests only. No src changes. No docs/*.md count change.

## GitHub issues log

- (none yet)
