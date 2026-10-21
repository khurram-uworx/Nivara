# Plan: Silk elementary/activation/norm Torch parity (issue #774)

## Problem

Phase A added `SilkVisionTorchParityTests` (conv/bn/pool/relu) but the elementary/activation/norm/GEMM Silk kernels have no Torch-fixture parity test class. Issue #774 asks for `SilkElementaryTorchParityTests` under `tests/Nivara.Tests/Gpu/` exercising `silu`, `geglu`, `layernorm`/`layer_norm_residual`, `add`, `add_bias`, `gemm`/`gemm_bias` (against `linear_*`/`matmul_*`), `rms_norm` (and `rms_norm_lm_head` where fixtures exist).

## Proposed changes

New file `tests/Nivara.Tests/Gpu/SilkElementaryTorchParityTests.cs` mirroring `SilkVisionTorchParityTests`:

- SiLU: `Silu_1D/4D_MatchesPyTorch` (`silu_1d/4d_input.bin`, `_output.bin`), tolerance.
- GeGLU: construct 2x-wide input from `gelu_exact_1d_input.bin` (first half gelu input, second half ones) -> reproduces `gelu_exact_1d_output.bin`.
- LayerNorm 2D/3D: `layernorm_2d/3d_*` fixtures with gamma/beta, eps 1e-5, tolerance.
- LayerNormResidual: zero partner against 2D LayerNorm fixture.
- Add: deterministic arrays, bit-exact `SingleToInt32Bits`.
- AddBias: `add_bias_*` fixtures, bit-exact.
- Gemm: `matmul_*` and transposed `linear_*_weight.bin` (Gemm) bit-exact.
- GemmBias: `linear_*` fixtures via `SilkLauncher.GemmEpilogue(GemmBias,...)` bit-exact.
- RMSNorm: `rmsnorm_2d/3d` (unit gamma) and `rmsnorm_module_2d` (shipped gamma), tolerance.

No production code changes expected unless a kernel bug surfaces during the run.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- targeted test run for the new class in Release (GPU-gated; Assert.Ignore on no-GPU)
- documentation snippet gate if any docs changed (none planned)

## Planned commits

1. `docs: plan Silk elementary Torch parity in TODO.md`
2. `test: add SilkElementaryTorchParityTests`
3. `docs: remove TODO.md — plan executed`

## Blast radius

- Tests only. Touches `tests/Nivara.Tests/Gpu/SilkElementaryTorchParityTests.cs`.
- Uses internal `Nivara.Silk.SilkLauncher` (InternalsVisibleTo `Nivara.Tests`).
- No runtime behavior change; risk is latent Silk kernel discrepancy (handled if run surfaces it).

## GitHub issues log

- None yet.