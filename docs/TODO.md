# Plan — #1466 Silk GEMV efficiency pass for decode

## Problem

Hot decode GEMV paths (`rms_norm_gemv`, `rms_norm_gemv_bf16`, `gemm_gemv_add`, `gemm_gemv_add_bf16`, decomposed MLP projections) use scalar loops with per-element BF16 widening and no local-memory staging of normed `x` / subgroup reductions. This likely leaves bandwidth and ALU utilization on the table. Baseline (synthetic, Intel iGPU, BF16, 64 prompt / 32 generated):

- SmolLM-135M: 11.80 tok/s steady-state, MLP 50.752 ms/token (55%)
- Qwen2.5-0.5B: 6.29 tok/s steady-state, MLP 113.193 ms/token (66%)

## Proposed changes

1. Vectorize weight loads where safe (`vload4`/packed ushort paths) with scalar tails; preserve accumulation order/correctness constraints.
2. Stage normed `x` in local memory or registers where beneficial.
3. Evaluate subgroup reductions where OpenCL subgroup extensions available; guard behind capability checks.
4. Confirm BF16 widening/coalescing in cheapest form.
5. Measure achieved GB/s vs device copy/bandwidth peak and steady-state tok/s before/after.

## Constraints / correctness

- Bit-identical where accumulation order preserved; tolerance only with explicit justification.
- No regression to LoRA/hasBias paths and BF16/F32 variants.
- Docs updated with any new profiling/measurement guidance if behavior changes.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — kernel bodies (4 GEMV kernels).
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — dispatch (likely unchanged signatures).
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs`, `SilkQwenGpuRunner.cs` — MlpBlock decode path (no-LoRA).
- Tests: `SilkRmsNormGemvTests`, `SilkGemmGemvAddTests`, `SilkMlpBlockDecompositionTests` (F32 only; BF16 parity tests missing — must add).
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — measurement A/B.

## Verification steps

- Build Release.
- Targeted Gpu tests: SilkRmsNormGemv*, SilkGemmGemvAdd*, SilkMlpBlockDecomposition* (and new BF16 ones).
- Synthetic SmolLM/Qwen profile (human-approved) for before/after.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. docs: plan #1466 in TODO.md
2. docs: add BF16 parity tests (red first) for RmsNormGemvBf16 and GemmGemvAddBf16
3. perf(silk): vectorize GEMV weight loads (vload4) with scalar tails, preserve accumulation order
4. perf(silk): stage normed x in local memory (only if measured benefit)
5. docs: record before/after measurements
6. Remove TODO.md

## GitHub issues log

(none yet)
