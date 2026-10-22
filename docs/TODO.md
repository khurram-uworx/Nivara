# Plan — issue #807: unaligned float4 A-tile staging

## Problem
A-tile vectorization (#791) vectors four A elements via `vload4` / `Float4` cast, guarded by
`(aBase % 4 == 0)`, and falls back to scalar otherwise. The OpenCL `vload4` is the
unaligned vector-load primitive and only requires element (`float`, 4-byte) alignment, which
`a + aBase` always has as a `float*`. The `(aBase % 4 == 0)` guard is therefore overly
conservative — it demands `float4` (16-byte) alignment and denies vectorization whenever a
row stride or hidden dimension is not a multiple of 4.

## Proposed changes
- `src/Nivara.Extensions/Silk/SilkKernels.cs`: remove only the `(aBase % 4u) == 0u`
  conjunct in the four A-staging conditions. Keep the range guards
  (`srcRow < aRows`, `aColBase + 3 < aCols`). The 4-byte alignment of `a + aBase` is
  inherent (float array), so vload4 stays defined.
  Sites: `GEMM_ROW4_ACCUMULATE` (two macro copies: f32 + BF16), `gemm_reg2x4`,
  `gemm_reg4x4`.
- `samples/Nivara.Samples/Gpu/GemmKernels.cs`: **no change.** ILGPU's `Cast<Float4>()` is an
  unchecked alignment assertion and ILGPU exposes no safe unaligned-vector (or byte-view)
  load in device code; a scalar-lane replacement preserves nothing worth the risk.
- `docs/SILK.md`: note that the OpenCL vector loads require float (not float4) alignment,
  so no `(base % 4)` guard is needed.

## Blast radius
- Silk Row4 macro drives the production BERT-family kernels
  (`gemm_row4`, `gemm_row4_bias`, `gemm_row4_qkv`, `gemm_row4_relu`) and the BF16
  variant. Accumulation order, smem slot mapping, `FP_CONTRACT OFF` are untouched, so
  the byte-identity claim should hold.
- `gemm_reg2x4`/`gemm_reg4x4` are measured-candidate kernels (not routed).
- Tests: `--silk-kernels` gate (50 cells, byte-identity for pure-multiply cells) and
  the Silk unit tests (catalog/parity) in `tests/Nivara.Tests`.

## Verification plan
1. Build `Nivara.slnx -c Release`.
2. `--silk-kernels` — expected Gate PASS, all cells (including byte-identity ones).
3. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
4. Targeted Silk tests in `Nivara.Tests` (catalog/parity).

## Planned commits
1. `docs: plan #807 in TODO.md`
2. `perf(silk): drop overly conservative A-tile vload4 alignment guard`
3. `docs: note OpenCL vloadn float alignment in SILK.md`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log
(none yet)
