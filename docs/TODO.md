# Plan: #796 — A/B manual unroll vs #pragma unroll in GEMM tile K-loop (Silk)

## Problem
#786 manually unrolled the 16-iteration K-accumulation loop in the Row4 GEMM macros so
the ILGPU/OpenCL accumulation order stays textually identical for the parity gate. The
compiler-directive variant (`#pragma unroll`) was never tried. This plan A/Bs the two
forms on the Silk backend only, keeps one as canonical, and documents why.

## Scope (Silk only)
- Target file: `src/Nivara.Extensions/Silk/SilkKernels.cs` (`SilkKernels.Source` string).
- Macros in scope:
  - `GEMM_ROW4_ACCUMULATE` (f32) ~ lines 415-459
  - `GEMM_ROW4_ACCUMULATE_BF16` (three identical copies) ~ lines 467-615
- Out of scope: `rms_norm_lm_head` inline loop (#795), vectorized A-tile staging (#791).

## Proposed changes
1. Add `GEMM_ROW4_ACCUMULATE_PRAGMA` (f32) macro identical to the manual one except the
   16 accumulation blocks become one `for (uint k = 0; k < TILE; ++k)` loop guarded by
   `_Pragma("unroll 16")` (macro-safe form of `#pragma unroll`), preserving
   `a0 += av * smem[...]` ascending-K order.
2. Add perf-only kernel `gemm_row4_pragma` using the new macro (bias-free store, same as
   `gemm_row4`), plus `GemmPragma` constant and a `KernelNames` entry so the
   device-free catalog test covers it.
3. Add a Silk-native A/B probe in `tests/Nivara.PerformanceTests` measuring both
   `gemm_row4` and `gemm_row4_pragma` on identical inputs/shapes: byte-identity of
   outputs (parity gate) and interleaved-round timing ratio.
4. Decide canonical form based on parity + timing; remove the non-canonical macro/kernel
   and update `docs/SILK.md` §11.3 with the decision and measured numbers.

## Verification steps
- C# build Release succeeds.
- `SilkKernelsCatalogTests` passes.
- OpenCL compile of the variant kernel on the target GPU (runtime, human-run).
- Silk-native A/B probe: 100% byte-identity and timing ratio reported.
- Targeted tests for touched area.

## Planned commits
1. docs: plan #796 in TODO.md
2. feat(silk): add GEMM_ROW4_ACCUMULATE_PRAGMA macro + gemm_row4_pragma kernel
3. test(perf): add Silk A/B probe for manual vs pragma unroll
4. docs(silk): record canonical-form decision and measured numbers (after decision)
5. chore(silk): remove non-canonical macro/kernel (after decision)
6. docs: remove TODO.md

## Blast radius
- Files: `src/Nivara.Extensions/Silk/SilkKernels.cs`, `tests/Nivara.PerformanceTests/*`,
  `docs/SILK.md`, `tests/Nivara.Tests/Gpu/SilkKernelsCatalogTests.cs` (unchanged, but
  catalog coverage depends on `KernelNames` staying consistent).
- Symbols: `SilkKernels.Source`, `SilkKernels.KernelNames`, `GEMM_ROW4_ACCUMULATE`,
  `GEMM_ROW4_ACCUMULATE_BF16`, new `GemmPragma`.
- Downstream: any runner that compiles `SilkKernels.Source` picks up the new kernel;
  perf-only so no production routing changes. Cleanup step removes it again.

## GitHub issues log
- (none yet)
