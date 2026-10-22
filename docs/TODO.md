# Plan: Issue #792 — give the #784 candidate GEMM kernels their own tile constant

## Problem
`src/Nivara.Extensions/Silk/SilkKernels.cs` defines a global `#define TILE 16` used by the production
Row4 macro. The #784 candidate kernels (`gemm_reg2x4`, `gemm_reg4x4`) declare a local
`const uint ... TILE = 16;`, which the preprocessor expands via the global macro, so the whole
OpenCL program failed to compile. PR #789 worked around this with `#undef TILE` immediately before
`gemm_reg2x4`. That is a global macro being switched off halfway through the source: any kernel
added after that point can no longer use the shared `TILE` macro, and the failure mode is a compile
error in the whole program rather than a local one.

## Proposed changes
- Delete the `#undef TILE` line.
- In the two candidate-kernel blocks (`gemm_reg2x4`, `gemm_reg4x4`), rename the local constant
  `TILE` to `TILE_K` everywhere it appears as a standalone token (declaration plus derived uses:
  `TileRows = TILE * BR`, `TileCols = TILE * BC`, `smem[TILE * (KT + TileCols)]`, `kk += TILE`,
  `cc += TILE`). `TileRows`/`TileCols` are distinct identifiers and stay as-is.
- Leave the global `#define TILE 16` and all production Row4 macro uses untouched.

This is a pure spelling change: the candidate constant is always 16, identical to the macro value,
so accumulation order, tiling geometry, and kernel selection are unchanged.

## Blast radius
- Single file: `src/Nivara.Extensions/Silk/SilkKernels.cs`, candidate-kernel region only.
- Downstream: `SilkKernels.Source` string consumed by `SilkRuntime.BuildProgram` (compile-time) and
  the `--silk-kernels` gate. No C# API or public symbol changes.
- Tests: `SilkKernelsCatalogTests` (device-free, checks `__kernel void <name>(` presence) and the
  `--silk-kernels` gate (compiles + runs). No pin/count changes expected.

## Verification
1. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
2. `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
3. `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` (human-run; needs OpenCL GPU)

## Planned commits
1. `docs: plan #792 in TODO.md`
2. `refactor(silk): give candidate GEMM kernels their own TILE_K constant (drop #undef TILE)`

## GitHub issues log
- [x] #792 — cleanup: give the #784 candidate GEMM kernels their own tile constant (drop #undef TILE) (working on this branch)
