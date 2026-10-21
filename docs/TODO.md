# Plan: Issue #784 — increase GEMM register blocking (1×4 → 2×4 / 4×4) as measured variants

## Problem

GEMM family is ~62% of encoder time. Current production kernel uses a 1×4 register block
(one output row × 4 columns per thread). Issue asks to try 2×4 / 4×4 to raise arithmetic
intensity. Must preserve byte-identity (accumulation order per (row,col) ascending in K) and
apply identically to ILGPU and Silk, incl. grid math. Shared-memory budget (64 KB) must hold.

## Approach (measured variants, NOT promoted to default in this pass)

Following the #440 methodology: implement as additional geometries, gate byte-identity,
benchmark; promote only if a win is measured.

## Proposed changes

1. ILGPU: `samples/Nivara.Samples/Gpu/GemmKernels.cs` — add
   `TiledGemmKernelReg2x4K16` (BR=2,BC=4,KT=16) and `TiledGemmKernelReg4x4K16` (BR=4,BC=4,KT=16).
   Single wide `Allocate2D`, row-striping for A, ascending-K `acc += a*b`. No fused epilogues.
2. Harness: `samples/Nivara.Samples/Gpu/GpuBuffers.cs` — add `GemmGeometries` entries.
3. Silk: `src/Nivara.Extensions/Silk/SilkKernels.cs` — constants + kernel-name constants + OpenCL
   kernel definitions porting the ILGPU bodies; grid helpers.
4. Silk launcher: `src/Nivara.Extensions/Silk/SilkLauncher.cs` — grid helpers.
5. Parity gate: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — add measured cells (byte-identity).
6. Tests: `SharedMemoryAllocationTests` floor bump; `SilkKernelsCatalogTests` coverage.

## Blast radius

- ILGPU: new methods only; no change to Row4 production kernels or callers.
- Silk: new constants/kernels/names; `KernelNames` array grows; catalog test.
- Tests: floor count + catalog.
- No production call-site or runner changes (variants measured, not routed).

## Verification

- Build `Nivara.slnx`.
- Targeted tests: SilkKernelsCatalogTests, SharedMemoryAllocationTests.
- `--silk-kernels` parity gate (human-run, GPU) for byte-identity.
- `--silk-encoder-profile` + benchmark (human-run) for improvement.

## Planned commits

1. docs: plan #784 in TODO.md
2. feat(gpu): add TiledGemmKernelReg2x4K16 and Reg4x4K16 (ILGPU)
3. feat(gpu): register 2x4/4x4 geometries in GpuBuffers
4. feat(silk): port 2x4/4x4 GEMM kernels to OpenCL
5. feat(silk): grid helpers for 2x4/4x4
6. test: parity gate cells + catalog/shared-memory floor
7. docs: ILGPU.md/SILK.md notes
8. docs: remove TODO.md

## GitHub issues log

- (none yet)
