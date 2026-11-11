# Plan: #1149 activation-level BF16 training kernels (GeluBackward/LayerNormBackward/ColSum)

## Problem

The narrow-precision training lane keeps weights BF16 on device while activations and
gradients stay F32. Three backward kernels remain F32-only because they operate over
activations: `gelu_backward`, `layer_norm_backward`, `col_sum`. BF16-storage variants
(halving activation traffic, accumulation still F32) are needed before activation-level
BF16 training can be wired (#1148 depends on this).

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkKernels.cs`
   - Add constants `GeluBackwardBf16Act`, `LayerNormBackwardBf16Act`, `ColSumBf16Act`.
   - Register all three in `KernelNames`.
   - Add OpenCL kernel definitions `gelu_backward_bf16_act`, `layer_norm_backward_bf16_act`,
     `col_sum_bf16_act` to `Source` (BF16 storage in, F32 accumulation, BF16 store where output
     is an activation gradient).
2. `src/Nivara.Extensions/Silk/SilkLauncher.cs`
   - Add typed launchers matching the F32 signatures.
3. `tests/Nivara.Tests/Gpu/SilkBackwardKernelTests.cs`
   - Add `Category("Gpu")` parity tests: seeded F32 inputs -> F32 reference; convert inputs to
     BF16, run BF16 variant, widen, assert `maxAbs <= 1e-2 * (1+|reference|)`.
4. `docs/SILK.md`
   - Update narrow-precision section: variants are opt-in, default lane stays F32 activations,
     accumulation F32; record measured cost (human-run).

## Blast radius

- `SilkKernels.Source` / `KernelNames` (catalog test enforces definition registration).
- `SilkLauncher` (new methods only; no signature changes to existing).
- No trainer/runner wiring in this issue (opt-in only).
- Docs only narrative change.

## Verification

- `dotnet build Nivara.slnx`
- Targeted GPU tests on a host with OpenCL.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## GitHub issues log

- (none yet)
