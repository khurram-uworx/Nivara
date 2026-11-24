# Plan: Silk K-split decode GEMV subgroup reduction (#1486)

## Problem

Issue #1486: The subgroup variants of `rms_norm_gemv` / `gemm_gemv_add` (and BF16) only
restructure the RMS-norm sum-of-squares reduction (or are no-ops for the add kernel). The
contraction `a[k]*b[k]` is still done per-lane over the full K. A "true K-split" splits K
across the subgroup lanes and reduces partial dot products via `sub_group_reduce_add`, with
the existing per-thread-full-K kernels as fallback. Then measure (synthetic + SmolLM-135M +
Qwen2.5-0.5B) and decide merge/close.

## Proposed changes

- `src/Nivara.Extensions/Silk/SilkKernels.cs`:
  - `rms_norm_gemv_subgroup` (F32 + BF16): keep SSQ reduction; split contraction over k
    across subgroup lanes (`k = sgLocal; k += sgSize`), reduce partials via
    `sub_group_reduce_add`. Geometry changes to one subgroup per 4 output columns.
  - `gemm_gemv_add_subgroup` (F32 + BF16): split contraction over k across subgroup lanes,
    reduce via `sub_group_reduce_add`; one subgroup per 4 output columns.
  - Add `GemvKSplitGlobalCols(cols)` helper (ceil(cols/4)*32 threads).
- `src/Nivara.Extensions/Silk/SilkLauncher.cs`:
  - Subgroup launchers use `GemvKSplitGlobalCols` for thread counts (local size 32).
  - Capability gating + fallback unchanged (`SupportsSubgroups`/`HasKernel`).
- Tests:
  - `SilkRmsNormGemvSubgroupTests`: tolerance widened to 1e-4 (strided partial sums),
    doc comment updated for K-split.
  - `SilkGemmGemvAddSubgroupTests`: renamed to `..._WithinTolerance`, tolerance 1e-4,
    doc comment updated. Fallback bit-identical path covered by
    `SilkSubgroupForceFallbackTests`.

## Verification steps

- Build Extensions + Tests (Release): PASS.
- Targeted GPU tests: `SilkRmsNormGemvSubgroupTests`, `SilkGemmGemvAddSubgroupTests`,
  `SilkSubgroupForceFallbackTests`, `SilkGemvGeometryTests`, BF16 tests: PASS (15 tests).
- Perf gate / inference benchmarks to be run for measurement (see below).

## Measurement

- Synthetic A/B via perf gate (`tests/Nivara.PerformanceTests`) with both paths
  (`NIVARA_SILK_SUBGROUPS`).
- SmolLM-135M and Qwen2.5-0.5B checkpoint benchmarks (`--silk`, both paths via
  `NIVARA_SILK_SUBGROUPS`) for tok/s + decode ms/token.
- Decision recorded after measurement: promote if gains measurable, else close.

## Blast radius

- Kernel source text in `SilkKernels.Source` (compiled into OpenCL program); subgroup
  kernel names unchanged, so `SubgroupKernelNames` / `SilkRuntime` build path unchanged.
- Launcher geometry (thread counts) for subgroup variants changed; base kernels unaffected.
- Tests updated for tolerance; fallback bit-identical path unchanged.

## GitHub issues log

- #1486 — Silk: K-split decode GEMV subgroup reduction (implement + measure + decide)
- (none new created yet)
