# Plan: Issue #805 — vectorized A-tile staging in candidate GEMM kernels (reg2x4/reg4x4)

## Problem

Issue #791 vectorized A-tile staging (`vload4`/`Float4`, 4 columns per thread) in the
production Row4 macros and ILGPU Row4 kernels. The #784 candidate kernels
`gemm_reg2x4` and `gemm_reg4x4` still use scalar A staging and would benefit from the
same treatment for consistency.

## Scope

- Apply the A-tile `vload4`/`Float4` staging pattern from #791 to `gemm_reg2x4` and
  `gemm_reg4x4` (both Silk OpenCL C and ILGPU C#).
- Re-run the `--gemm` gate and confirm byte-identity + no perf regression.

Out of scope: the other #784 candidate kernels (Reg2x2K16/K32, Reg4x2K32, Reg1x8K16)
and any runner routing/promotion.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — `gemm_reg2x4`, `gemm_reg4x4` A-staging
  blocks only. Shared-memory layout, B-staging, barriers, accumulation, and output stores
  unchanged → contraction order (byte-identity) preserved.
- `samples/Nivara.Samples/Gpu/GemmKernels.cs` — `TiledGemmKernelReg2x4K16`,
  `TiledGemmKernelReg4x4K16` A-staging blocks only.
- Tests: `SilkKernelsCatalogTests` (kernel-name/source consistency, unaffected since no
  new kernel name), Gpu shared-memory allocation tests, `--gemm` perf gate (ILGPU).

## Proposed changes

### Silk (OpenCL C) — both kernels share identical A-staging text

Replace scalar staging:
```
for (uint r = 0; r < BR; ++r) {
    uint srcRow = outRowBase + x * BR + r;
    uint slot = x * BR + r - s * KT;
    for (uint kk = 0; kk < KT; kk += TILE_K) {
        uint srcCol = k0 + kk + y;
        smem[slot + (kk + y) * (KT + TileCols)] =
            (srcRow < aRows && srcCol < aCols) ? a[srcRow * aCols + srcCol] : 0.0f;
    }
}
```
with y-guarded 4-wide staging (y < 4u), using vload4 with alignment/bounds fallback,
writing `smem[slot + (kk + y*4 + aw) * (KT + TileCols)]`.

### ILGPU (C#) — same shape, anchored to BR=2 (reg2x4) and BR=4 (reg4x4) to avoid
touching the other candidate kernels

Replace scalar staging with y < 4 guard and `a.Cast<ILGPU.Util.Float4>()[aBase/4]`,
4-wide fallback loop.

## Verification steps

1. `dotnet build Nivara.slnx`
2. Targeted tests: `SilkKernelsCatalogTests` + Gpu shared-memory allocation tests.
3. `--gemm` gate (human-run): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --gemm`
   — byte-identity for Reg2x4K16/Reg4x4K16, no perf regression vs baseline.

## Planned commits

1. `docs: plan #805 in TODO.md`
2. `perf(silk): vectorized A-tile staging in gemm_reg2x4/reg4x4`
3. `perf(gpu): vectorized A-tile staging in ILGPU Reg2x4/Reg4x4 GEMM kernels`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)
