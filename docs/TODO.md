# Plan: Perf — vectorized B-tile load in BF16 GEMM path (#790)

## Problem
`gemm_row4_bf16` (Silk/OpenCL) still stages its B tile element-by-element and converts each
BF16 element with `BF16_TO_F32`, while #785 (PR #789) already vectorized the f32 Row4 GEMM's
B load with `float4`/`vload4` + scalar fallback, yielding 9.7%–36.1% faster GEMM cells.
Apply the same treatment to the BF16 path, keeping the widening bit-exact and the accumulation
order unchanged. Adopt only if a before/after measurement shows a gain.

## Proposed changes
- `src/Nivara.Extensions/Silk/SilkKernels.cs`: in `GEMM_ROW4_ACCUMULATE_BF16` (defined 3×
  identically), replace the per-element B staging loop with a guarded `ushort4 vload4` load
  plus scalar fallback:
  - vector path only when `bRow < aCols`, `outCol0 + 3 < bCols`, and `bBase % 4 == 0`
    (8-byte aligned, since `b` is `ushort*`).
  - widen each of `bv.s0..s3` with the existing `BF16_TO_F32` bitcast.
- No ILGPU side: `samples/Nivara.Samples/Gpu/GemmKernels.cs` has no BF16 GEMM kernel.
- Add a temporary `--bf16-gemm` probe to `tests/Nivara.PerformanceTests` to take before/after
  measurements on MiniLM/DistilBERT-shaped GEMMs.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- `SilkQwenBf16ParityTests` (and SmolLM BF16 parity if present).
- `--silk-kernels` gate (f32 path unchanged).
- Before/after probe medians.

## Planned commits
1. docs: plan #790 in TODO.md
2. perf(silk): vectorized ushort4 B-tile load in BF16 GEMM macro
3. test(perf): add --bf16-gemm probe for before/after measurement (if retained)
4. docs: remove TODO.md — plan executed

## Blast radius
- `src/Nivara.Extensions/Silk/SilkKernels.cs` (macro used by `gemm_row4_bf16` kernel string).
- Host callers: `SilkLauncher.GemmBf16`, `SilkQwenGpuRunner`, `SilkSmolLMGpuRunner` (unchanged call sites).
- Tests: `SilkQwenBf16ParityTests`.

## GitHub issues log
- (none yet)
