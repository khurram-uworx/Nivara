# Plan: decode-shaped gemm_row4_add for remaining aRows=1 GEMMs (#681)

## Problem

After #670, the norm-fused decode projections (Q/K/V, gate/up, tied head) no longer
idle 15 of 16 row lanes. Two decode GEMMs are not norm-fused and still launch the
16-row tile geometry for a single useful row:

- `o_proj`: `gemm_row4_add(context, Wo, x, x, 1, qDim, hiddenDim)`
- `down`: `gemm_row4_add(mlp, Wdown, x, x, 1, intermediateDim, hiddenDim)`

Both are `aRows == 1`, so `GemmGlobalRows(1) == 16` over a `[16,16]` local size leaves
15 row lanes idle while they still run the full K-loop against zeroed A tiles (barriers
prevent an early return). At Qwen 2.5-0.5B / 24 layers these are ~124 M MACs/token
(down 105 M, o_proj 19 M) — the majority of the decode MACs left at 6% utilization.

## Proposed changes

The decode-shaped variant already exists as `gemm_gemv_add` (added in #672): lanes own
output columns, one accumulator per output element over ascending K, residual added at
the store. `#681` completes the acceptance criteria that #672 left open:

1. **Exact-equality pin.** New device test `SilkGemmGemvAddTests` asserts
   `gemm_gemv_add` reproduces `gemm_row4_add` (aRows = 1) bit-for-bit on identical
   inputs, including tail widths (aCols not a multiple of `GemmTileSize`, bCols
   straddling a 4-lane boundary).
2. **Gate cell.** Add a `MeasureCpuReference` cell for `GemmGemvAdd` in
   `MeasureSmolLMKernels` (CPU reference = ascending-K `mul+add` + residual), so the
   decode shape is measured alongside the other SmolLM-family kernels.
3. **Byte-identity decision.** `gemm_gemv_add` is Silk-only (no ILGPU counterpart), so
   it is **not** added to the ILGPU byte-identity list in `docs/SILK.md` /
   `docs/ILGPU.md` §10.2. Its correctness is pinned by (a) the Silk-to-Silk exact
   equality above and (b) the CPU-reference gate cell. Document this decision.

No change to prefill paths (they stay on the tile kernels).

## Blast radius

- New test file: `tests/Nivara.Tests/Gpu/SilkGemmGemvAddTests.cs`.
- Gate: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` (`MeasureSmolLMKernels`).
- Docs: `docs/SILK.md` (Silk-only kernel note).
- No production code changes: `gemm_gemv_add` and its wiring already exist (#672).

## Verification

- `dotnet build Nivara.slnx -c Release`
- Targeted: `SilkGemmGemvAddTests`, `SilkQwenParityTests`, `SilkQwenKvCacheTests`,
  `SilkDecodeAttentionParityTests`, `SilkKernelsCatalogTests`.
- Docs suite if `docs/SILK.md` changes: `FullyQualifiedName~Nivara.Tests.Docs`.
- Per-token decode win: human-run per `docs/QWEN.md` protocol; re-measure and update the
  figure, never assume.

## Planned commits

1. docs: plan #681 in TODO.md
2. test(gpu): pin gemm_gemv_add against gemm_row4_add bit-for-bit (#681)
3. test(perf): add MeasureCpuReference cell for GemmGemvAdd (#681)
4. docs(silk): record gemm_gemv_add as Silk-only, outside ILGPU byte-identity (#681)
5. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)