# Plan — #1466 Silk GEMV efficiency pass for decode

## Problem

Hot decode GEMV paths (`rms_norm_gemv`, `rms_norm_gemv_bf16`, `gemm_gemv_add`, `gemm_gemv_add_bf16`, decomposed MLP projections) use scalar loops with per-element BF16 widening and no local-memory staging of normed `x` / subgroup reductions. This likely leaves bandwidth and ALU utilization on the table. Baseline (synthetic, Intel iGPU, BF16, 64 prompt / 32 generated):

- SmolLM-135M: 11.80 tok/s steady-state, MLP 50.752 ms/token (55%)
- Qwen2.5-0.5B: 6.29 tok/s steady-state, MLP 113.193 ms/token (66%)

## Proposed changes

1. Vectorize weight loads where safe (`vload4`/packed ushort paths) with scalar tails; preserve accumulation order/correctness constraints. (DONE: all 4 kernels.)
2. Stage normed `x` in local memory or registers where beneficial. (DEFERRED -> #1471.)
3. Evaluate subgroup reductions where OpenCL subgroup extensions available; guard behind capability checks. (DEFERRED -> #1472.)
4. Confirm BF16 widening/coalescing in cheapest form. (DONE: packed ushort4 load + per-component BF16_TO_F32.)
5. Measure achieved GB/s vs device copy/bandwidth peak and steady-state tok/s before/after. (DEFERRED -> #1473.)

## Also in this branch

- Added BF16 parity tests: `SilkRmsNormGemvBf16Tests`, `SilkGemmGemvAddBf16Tests` (bit-identical vs C# reference with BF16 widening, including tail widths and hasBias/no-bias). Previously only F32 parity tests existed.

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
2. perf(silk): vectorize GEMV weight loads (vload4) with scalar tails, preserve accumulation order
3. test(gpu): add BF16 parity tests for RmsNormGemvBf16 and GemmGemvAddBf16
4. docs: record deferred work as follow-up issues (#1471, #1472, #1473)
5. Remove TODO.md

## GitHub issues log

- [ ] #1471 — Silk: stage normed-x in local memory for decode GEMV (deferred from #1466)
- [ ] #1472 — Silk: evaluate subgroup reductions for decode GEMV (deferred from #1466)
- [ ] #1473 — Silk: add achieved GB/s measurement guidance for decode GEMV (deferred from #1466)
