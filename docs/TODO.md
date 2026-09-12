# TODO — Issue #411: TensorPrimitives.Multiply for RMSNorm forward gamma multiply

## Problem

`src/Nivara/AutoDiff/Nn/RMSNorm.cs` `RMSNorm<T>.ForwardInference` (~lines 148-153)
multiplies the normalized rows by the per-dimension gamma with a scalar loop, even
though the surrounding RMS norm kernel is `TensorPrimitives`-backed
(`RMSNormKernel<T>.PerRowRMSNormForwardKernel` uses `TensorPrimitives.Dot` /
`Multiply` / `MultiplyAdd`). Qwen decode takes `ForwardInference` once per layer, so
this is the inferencing hot path. The change is consistency/cleanliness, not a
wall-clock win (~43k FLOP/token for Qwen sizes). See issue #411.

## Proposed changes

1. **`src/Nivara/AutoDiff/Nn/RMSNorm.cs`**
   - Add `using System.Numerics.Tensors;`.
   - `ForwardInference` gamma step: replace the inner scalar J-loop with a per-row
     `TensorPrimitives.Multiply`:

     ```csharp
     for (int i = 0; i < rows; i++)
     {
         int baseIdx = i * cols;
         TensorPrimitives.Multiply(y.AsSpan(baseIdx, cols), gamma, y.AsSpan(baseIdx, cols));
     }
     ```

     Note: the issue's suggested `TensorPrimitives.Multiply(y, gamma, y)` is only
     shape-valid when `rows == 1` (`y` is `[rows, cols]`, `gamma` is `[cols]`
     broadcast across rows), so the per-row slicing loop stays with only the inner
     loop vectorized. In-place (destination aliasing the first source) is supported.
   - Training-forward gamma step (same scalar pattern, ~lines 78-83): identical
     replacement operating on `outputData`. Chosen scope: "Inference + training
     forward". Backward grad loops (gradNorm multiply, gamma-grad reduction) stay
     scalar — explicitly out of scope per the issue.

## Verification

- No new tests: existing coverage already pins both edited paths.
  - `tests/Nivara.Tests/AutoDiff/RMSNormTests.cs` — `Forward_AppliesGammaPerDimension`
    and `Forward_MatchesScalarReference_WithUnitGamma` route through
    `ForwardInference` (`requiresGrad: false` inputs).
  - `tests/Nivara.Tests/NivaraTorch/RMSNormModuleTests.cs` —
    `RMSNormModule_2D_AffineGamma_MatchesPyTorch` (fixture-backed) covers the
    training-forward path + grads.
- Ask human before running `dotnet test`; run filtered set
  (`RMSNormTests`, `RMSNormModuleTests`, `NormalizationTests`), then optionally the
  full suite.

## Planned commits

1. `docs: plan issue #411 (RMSNorm TensorPrimitives gamma multiply) in TODO.md`
2. `perf: use TensorPrimitives.Multiply for RMSNorm forward gamma multiply`

## Blast radius

- Single file touched: `src/Nivara/AutoDiff/Nn/RMSNorm.cs`.
- `ForwardInference` is called only from `RMSNorm<T>.Forward` when
  `!input.RequiresGrad` (inference default). Downstream: Qwen decode / transformer
  inference in samples (`samples/Nivara.Samples`), `TransformerBlock` +
  `PerRowRMSNorm` (separate `ReverseGradOperations` path, not affected).
- Training-forward loop change is behavior-identical (elementwise multiply by a
  per-dimension vector), exercised by the `RMSNormModule_2D_AffineGamma_MatchesPyTorch`
  fixture test which asserts output + input grad + gamma grad numerically.
- No public API change; no nullable/CLR-domain interaction.

## GitHub issues log

- [x] #418 — Vectorize RMSNorm backward grad loops with TensorPrimitives (deferred from #411 scope; created while working on #411).