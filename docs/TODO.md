# Plan: Silk BF16 decode GEMV GB/s measurement (#1479)

Follow-up to #1473. Extend the `--silk-kernels` perf gate to report achieved GB/s for
the BF16 decode GEMV shapes (`rms_norm_gemv_bf16`, `gemm_gemv_add_bf16`, and optionally
`mlp_block_bf16`) using the same logical bytes-moved model but with BF16 weight bytes
(`sizeof(ushort)`), so the production decode path (SmolLM/Qwen default to BF16 weight
transport) gets reproducible GB/s numbers comparable to the F32 ones from #1473.

## Problem

`SilkKernelsGate.MeasureDecodeShapeAb` (#1473) only covers the F32 `rms_norm_gemv` /
`rms_norm_lm_head` A/B. SmolLM and Qwen default to BF16 weight transport on the Silk
path (`rms_norm_gemv_bf16`, `gemm_gemv_add_bf16`, `mlp_block_bf16`), so the real
production decode path has no GB/s numbers today. Document the BF16 weight-byte
convention in `docs/SILK.md` §14a.

## Agreed scope

- (a) Land the BF16 GB/s measurement in the `--silk-kernels` gate + docs.
- (b) Do NOT promote `MlpBlockBf16` into inference runners in this PR. The existing
  P1/#1453 note flags its single-WG serialization risk on iGPUs; promotion needs
  same-device iGPU decode evidence beyond raw GB/s. Capture as a follow-up if the
  numbers warrant it.

## Proposed changes

### 1. `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`

Add a BF16 sibling measurement method, e.g. `MeasureDecodeShapeAbBf16(silk, buffers)`,
called from `Run()` next to `MeasureDecodeShapeAb` and `MeasureCopyBandwidth`.

Shapes: reuse the same `(hidden, width, label)` table as #1473
(`SilkKernelsGate.cs:1349-1361`).

A. `rms_norm_gemv_bf16` (single-slot, no bias) across all shapes:
- Allocate `float[hidden]` x/gamma, `ushort[hidden*width]` weights (BF16), `float[width]` out.
- Upload via `silk.Upload` (ushort overload).
- Time `SilkLauncher.RmsNormGemvBf16(silk, sx, sg, [new GemvSlot(sw, 0, sout, width)], hasBias: false, aRows: 1, aCols: hidden, eps)`.
- Bytes = `(hidden + hidden + (long)hidden*width + width)` with weights in BF16:
  `bytes = hidden*sizeof(float) + hidden*sizeof(float) + (long)hidden*width*sizeof(ushort) + width*sizeof(float)`.
- Report median ms + GB/s, interleaved warmup/round discipline (5 warmups, 10 rounds).

B. `gemm_gemv_add_bf16` across the down/O-proj shapes (gate/up intermediate also
   covered by reusing width=intermediate):
- Allocate `float[aCols]` a, `ushort[aCols*bCols]` b (BF16), `float[bCols]` residual, `float[bCols]` out.
- Time `SilkLauncher.GemmGemvAddBf16(silk, sa, sb, sout, sres, aCols, bCols)`.
- Bytes = `aCols*sizeof(float) + (long)aCols*bCols*sizeof(ushort) + bCols*sizeof(float) /*out*/ + bCols*sizeof(float) /*residual*/`.

C. `mlp_block_bf16` (optional, with caveat): single-WG fused launch
   `SilkLauncher.MlpBlockBf16(silk, sx, sg, wg, wu, wd, outBuf, residual, hidden, intermediate, eps)`.
   Report GB/s with bytes = x(F32)+gamma(F32)+3*hidden*intermediate BF16 weights + out(F32)+residual(F32),
   but mark clearly as single-WG fused and compare against the decomposed BF16 path.

Reuse `Time`, `Median`, existing buffer tracking (`buffers.Add`).

### 2. `docs/SILK.md` §14a

Add a "BF16 decode GEMV GB/s measurement" subsection documenting:
- the BF16 weight-byte convention,
- which kernels/variants measured,
- the byte formulas,
- caveats (single-WG MlpBlock, subgroup variants used in Qwen runner).

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` (add method + call site).
- `docs/SILK.md` (doc subsection).
- No production code changes. No runner changes (out of scope per agreed scope).
- DocsCoveragePins: adding a fenced csharp block? Only if docs gain a code fence. The §14a addition is prose + tables (no new fenced csharp blocks expected), so no pin bump likely. Re-verify after edit.

## Verification

- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- GPU run (human-run): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` — capture numbers.
- Docs snippet gate: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"` (only if fenced blocks added).

## Planned commits

1. `docs: plan Silk BF16 decode GEMV GB/s measurement in TODO.md` (this file)
2. `perf: report BF16 decode GEMV GB/s in Silk kernels gate (#1479)`
3. `docs: document BF16 decode GEMV GB/s measurement in SILK.md §14a (#1479)`
4. (if needed) `docs: revise DocsCoveragePins for #1479`

## GitHub issues log

- [ ] #1479 — Silk: BF16 decode GEMV GB/s measurement (this issue; being implemented)
- (potential follow-up) promote MlpBlockBf16 into inference runners pending iGPU decode evidence — create if numbers warrant.

## Grounding notes

- #1473 methodology (docs/SILK.md §14a) is the template: logical bytes-moved, interleaved warmup/rounds, copy-bandwidth probe ceiling.
- BF16 kernels already have parity tests (`SilkRmsNormGemvBf16Tests`, `SilkGemmGemvAddBf16Tests`) and are used in SmolLM/Qwen runners, so no correctness risk from measuring them.
- `MlpBlockBf16` is single-WG (SilkLauncher.cs:1091-1094); keep optional/caveated.
