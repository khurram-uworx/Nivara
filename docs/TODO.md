# Plan: #470 — re-measure the #440 tile family on one shared-addressing form

## Problem

The 2026-09-29 #440 experiment concluded every wider tile loses to Row4. That conclusion is conditioned: Row4 migrated to a single wide `Allocate2D` (#468) and got ~20% faster, while the four #440 geometries stayed on the 1-D `Allocate<float>` form, so the recorded ratios mix addressing forms. Re-measure on one form before settling.

## Proposed changes

Port `TiledGemmKernelReg2x2K16`, `Reg2x2K32`, `Reg4x2K32`, `Reg1x8K16` in `samples/Nivara.Samples/Gpu/GemmKernels.cs` to the single-wide `Allocate2D<float, Stride2D.DenseX>` form (Index2D(KT, KT+TileCols), DenseX(KT+TileCols)), A in cols [0,KT), B in cols [KT,KT+TileCols), staging row-strips of KT rows. Preserve strictly-ascending-K accumulation so the fingerprint stays bit-identical. Update comments/banner.

## Verification

- SharedMemoryAllocationTests green (one allocation per method).
- Targeted GPU tests.
- Human runs `--gemm` on AC power; fingerprint gate (171 cells) must pass or be deliberately re-recorded.
- Update docs/README/AGENTS per outcome.

## Planned commits

1. docs: plan #470 in TODO.md
2. refactor(gpu): port Reg2x2K32 to wide Allocate2D
3. refactor(gpu): port Reg2x2K16, Reg4x2K32, Reg1x8K16 to wide Allocate2D
4. docs: update ILGPU/README/AGENTS addressing-form notes

## Blast radius

- `samples/Nivara.Samples/Gpu/GemmKernels.cs` (4 methods + banner)
- Docs: `docs/ILGPU.md`, `tests/Nivara.PerformanceTests/README.md`, `AGENTS.md`
- No public API change; no Silk changes.

## GitHub issues log

- (none yet)
