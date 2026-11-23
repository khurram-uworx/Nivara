# Plan: Silk subgroup reductions for decode GEMV (#1472)

## Problem

Follow-up to #1466. Subgroup reductions (`cl_khr_subgroups` / `cl_intel_subgroups`)
were deferred because (a) SilkRuntime has no capability detection or guarded kernel
selection, (b) subgroup intrinsics change reduction structure and risk bit-identity,
and (c) benefit is unmeasured.

Current decode GEMV kernels (`rms_norm_gemv`, `rms_norm_gemv_bf16`, `gemm_gemv_add`,
`gemm_gemv_add_bf16`) accumulate per-thread over the full K range; the RMS-norm
sum-of-squares is redundantly computed by every lane in a work-group.

## Proposed changes

1. **Capability detection** in `SilkRuntime`: query `CL_DEVICE_EXTENSIONS`, expose
   `SupportsSubgroups` (true if "subgroup" appears in the extension string). Pass
   `-D SILK_SUBGROUPS=1` to `clBuildProgram` when supported.
2. **Guarded subgroup variants** in `SilkKernels.Source` (compiled only under
   `SILK_SUBGROUPS`):
   - `rms_norm_gemv_subgroup` (+ `_bf16`): norm (ssq) computed via strided partial
     sums + `sub_group_reduce_add`, with the same staged/fallback structure and
     per-lane dot accumulation as the base kernel.
   - `gemm_gemv_add_subgroup` (+ `_bf16`): provided for API symmetry; gemm has no
     cross-lane reduction, so the body is identical to the base kernel but is gated
     by a subgroup intrinsic reference so it only compiles where supported.
3. **Runtime registration**: `SilkRuntime` creates subgroup-variant kernels only when
   supported; `HasKernel(name)` helper.
4. **Launcher methods** (`SilkLauncher`): `RmsNormGemvSubgroup*`, `GemmGemvAddSubgroup*`,
   dispatching the subgroup kernel and falling back to the base kernel when unsupported.
5. **Runners** (`SilkSmolLMGpuRunner`, `SilkQwenGpuRunner`): route decode GEMV
   through the subgroup launcher methods (which fall back automatically).
6. **Parity policy**: base kernel remains bit-identical to today's reference. Subgroup
   path reduces ssq in strided order → scale may differ by a few ULP; tests use a
   tolerance (`1e-5` relative) with justification. gemm subgroup path is identical →
   bit-identical.
7. **Parity tests**: new Gpu-category tests comparing subgroup vs base on capable
   devices, plus capability-property smoke test.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkRuntime.cs` — constructor, `BuildProgram`, new property.
- `src/Nivara.Extensions/Silk/SilkKernels.cs` — Source string (new kernels under macro).
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — new launcher methods.
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs`, `SilkQwenGpuRunner.cs` — routing.
- `tests/Nivara.Tests/Gpu/` — new parity tests.
- `docs/SILK.md` — capability note (optional).

Existing kernel tests and catalog test unaffected (base names unchanged).

## Verification

- `dotnet build Nivara.slnx` (Release).
- Targeted Gpu tests (require OpenCL GPU): new subgroup parity tests.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. docs: plan #1472 in TODO.md
2. SilkRuntime: subgroup capability detection + build option
3. SilkKernels: guarded subgroup variant kernels
4. SilkLauncher: subgroup launcher methods with fallback
5. Runners: route decode GEMV through subgroup launchers
6. Tests: subgroup parity tests
7. docs: SILK.md capability note (if applicable)

## GitHub issues log

- (none yet)
