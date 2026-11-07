# Plan: perf(silk) — LN/fc GEMM backward fusion probe (#1010)

## Problem

`SilkDistilBertTrainer` backward chain spends ~763 ms in `LnFcGemmBackward` at B=8. That stage
launches, per layer: LayerNorm backward + two `col_sum` passes for LayerNorm γ/β, FC2 `GemmTn` +
`col_sum` (bias) + `GemmNt`, GELU backward, FC1 `GemmTn` + `col_sum` (bias) + `GemmNt`, a residual
`Add`, LayerNorm backward, output projection `GemmTn` + `col_sum` + `GemmNt`, QKV `GemmTn` +
`col_sum` + `GemmNt`, and another residual `Add`. Many of these passes re-read or re-write the same
gradient buffers and could be folded into GEMM epilogues, the way the forward `gemm_row4_bias` /
`gemm_row4_gelu` family already does.

## Proposed changes

1. **Residual epilogue on the NT backward GEMM** (`gemm_row4_nt_residual`):

   - Reuse `GEMM_NT_ACCUMULATE` verbatim (ascending-K `mul + add`, `FP_CONTRACT OFF`), only the
     store changes: each output cell adds `residual[outRow * bCols + col]` before storing.
   - Threader through `SilkKernels` constant `GemmNtResidual` and `SilkLauncher.GemmNtResidual(...)`.
   - Replace in `SilkDistilBertTrainer`:
     - `GemmNt(dInterm, W1, dTmp2)` + `Add(dTmp, dTmp2, dTmp)` → `GemmNtResidual(dInterm, W1, dTmp2, dTmp)` storing into `dTmp`.
     - `GemmNt(dQkv, Wqkv, dTmp2)` + `Add(dCur, dTmp2, dCur)` → `GemmNtResidual(dQkv, Wqkv, dTmp2, dCur)` storing into `dCur`.
     The destination buffer doubles as the residual operand; registers are loaded before stores.

2. **Bias-gradient fusion: documented as not viable under byte-identity parity**

   `col_sum` is one work-item per output column, ascending over rows. Fusing it into
   `GEMM_ROW4_TN` would require either atomic accumulation across the tile’s multiple output-row
   work-items (ordering non-deterministic, breaks exact replay) or a second reduction launch (no
   bandwidth win). Keep `col_sum` separate and state that in docs/SILK.md.

3. **Cross-check candidate preserves byte-identity parity**

   Residual add is a plain `acc + residual`, identical to the current `GEMM_NT` store followed by
   the standalone `add` kernel, so a bit-identical replay holds.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs`: source string + `KernelNames` + `GemmNtResidual`.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs`: dispatch.
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`: two call-site pairs in `BackwardChain`.
- `tests/Nivara.Tests/Gpu/SilkBackwardKernelTests.cs`: new bit-identity test.
- `docs/SILK.md`: #1010 outcome section + follow-up table.
- Parity coverage: `SilkDistilBertTrainerTests`, `SilkBackwardKernelTests`, `SilkKernelsCatalogTests`.

## Verification

- `dotnet build Nivara.slnx` (targeted build of Extensions + tests)
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkBackwardKernelTests|SilkDistilBertTrainerTests|SilkKernelsCatalogTests"` (ask before running; GPU-gated, ignored where no OpenCL device)
- Human-run on Intel Graphics:
  - `SilkDistilBertTrainingProfileProbe` before/after medians at B=4 and B=8.
  - `SilkDistilBertTrainerTests` and `SilkBackwardKernelTests` green.

## Planned commits

1. `docs: plan #1010 in TODO.md`
2. `feat(silk): add gemm_row4_nt_residual fused backward kernel`
3. `perf(silk): use residual-fused NT GEMM in DistilBERT backward`
4. `test(silk): pin gemm_row4_nt_residual against nt + add`
5. `docs(silk): record #1010 measurement / conclusion`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #1010 — perf(silk): probe LN/fc GEMM backward fusion (bias+residual epilogues) (in progress)

Grounding note (G1): kernel design grounded in existing `gemm_row4_nt` macro + OpenCL source and
docs/ILGPU.md §10.2 exactness policy. If GPU measurement shows <20% gain we keep the kernel if
parity-neutral and document the outcome rather than reverting; the issue explicitly allows a
documented conclusion.