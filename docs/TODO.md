# Plan: Issue #804 — collapse duplicate GEMM_ROW4_ACCUMULATE_BF16 macro definitions

## Problem
`GEMM_ROW4_ACCUMULATE_BF16` is defined three times identically in
`src/Nivara.Extensions/Silk/SilkKernels.cs` (OpenCL C source inside the
`SilkKernels.Source` raw string). This is a maintenance/drift hazard.

## Proposed changes
Single file: `src/Nivara.Extensions/Silk/SilkKernels.cs`.
Remove the second and third duplicate `#define GEMM_ROW4_ACCUMULATE_BF16`
blocks (currently lines 531–646), keeping one definition and one blank line
before `inline uint qkv_dest(...)`. Macro text stays byte-identical.

## Blast radius
- `SilkKernels.Source` (single field) — OpenCL translation unit compiled by
  `SilkRuntime` via `clCreateProgramWithSource`.
- Downstream: all Silk GEMM paths using `gemm_row4_bf16` (the only consumer).
- Tests: `SilkKernelsCatalogTests` (device-free) and `--silk-kernels` gate
  (GPU, human-run).

## Verification
1. Build `Nivara.Extensions` (Release).
2. `dotnet test ... --filter "FullyQualifiedName~SilkKernelsCatalogTests"`.
3. `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` (human-run).

## Planned commits
1. `docs: plan issue #804 in TODO.md`
2. `refactor(silk): collapse duplicate GEMM_ROW4_ACCUMULATE_BF16 macro definitions`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log
- (none yet)