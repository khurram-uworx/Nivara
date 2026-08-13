# Plan: `Loss<T>.Reduce` scalar-division allocation (issue #207)

Branch: `khurram/207` (off `main`, commit `8b17f27`).

## Problem

`Loss<T>.Reduce` (`src/Nivara/AutoDiff/Nn/Functional/Loss.cs:44-47`) — the
`Reduction.Mean` path (default, PyTorch parity) allocates a full 1-element
`ReverseGradTensor<T>` via `GradientUtils.Full(1, count)` just to divide a scalar
sum by a scalar count, then routes it through the tensor-tensor `Divide` VJP whose
divisor branch is dead weight (the divisor is a leaf constant with `requiresGrad: false`).

```csharp
case Reduction.Mean:
    int count = divisor ?? elementwiseLoss.Length;
    var scale = GradientUtils.Full(1, T.CreateChecked(count));
    return ReverseGradOperations.Divide(ReverseGradOperations.Sum(elementwiseLoss), scale);
```

This runs on every loss computation (every batch, every step). `Reduce` is called by
all five losses: `MSELoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`
(`divisor == null`), and `CrossEntropyLoss` (passes `batchSize` as `divisor`).

## Change

1. Add a scalar-division VJP `ReverseGradOperations.DivideScalar<T>(ReverseGradTensor<T> a, T scalar)`
   in `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` (Element-wise region, after `Divide`):
   - null-check `a`; `if (scalar == T.Zero) throw new DivideByZeroException(...)` (mirrors existing `Divide` guard).
   - forward: `TensorPrimitives.Divide(a.AsSpan(), scalar, resultArr)` → `ResultTensor(resultArr, a, GradientUtils.ShouldTrackGrad(a))`.
     (Scalar overload confirmed in BCL: `TensorPrimitives.Divide<T>(ReadOnlySpan<T> x, T y, Span<T> destination)` — `destination[i] = x[i] / y`.)
   - if `GradientUtils.ShouldTrackGrad(a)`: `OpNode<T>("DivideScalar", [a], ...)` whose backward scales the
     gradient by `1/scalar` (`TensorPrimitives.Divide(gSpan, scalar, gradArr)`) then `AccumulateGradient`.
   - Same inference-default gating as every other op (no graph node outside `Grad()`).

2. Add `public ReverseGradTensor<T> DivideByScalar(T scalar)` on
   `src/Nivara/AutoDiff/ReverseGradTensor.cs` (next to the operator overloads) delegating to
   `ReverseGradOperations.DivideScalar(this, scalar)` — the API requested by the issue.

3. Update `Loss<T>.Reduce` Mean case:
   ```csharp
   case Reduction.Mean:
       int count = divisor ?? elementwiseLoss.Length;
       return ReverseGradOperations.DivideScalar(
           ReverseGradOperations.Sum(elementwiseLoss), T.CreateChecked(count));
   ```

Numerics unchanged (`sum / count`); the 1-element divisor tensor allocation and the
`Divide` op's b-gradient machinery are eliminated. Graph shape goes from `Sum`+`Divide`
to `Sum`+`DivideScalar` (still 2 nodes — the divide itself must stay a node for the
`1/count` gradient scaling; only the *divisor tensor* is removed).

4. Tests (`tests/Nivara.Tests/AutoDiff/`):
   - `GradOperationsTests.cs`: `DivideScalar_*` suite mirroring the existing `Divide`
     tests — forward values, backward `1/scalar` scaling, `DivideByZeroException` on
     zero scalar, parity with `Divide(a, Full(1, count))`, plus the
     `ReverseGradTensor.DivideByScalar` wrapper.
   - `LossTests.cs`: assert the Mean path builds exactly `Sum` + `DivideScalar`
     (`GetGraphInfo` → `TotalNodes == 2`, `OperationCounts` contains `DivideScalar`,
     no `Full` divisor tensor). Existing value/gradient Mean tests stay green and
     lock numeric parity; NivaraTorch fixtures are numeric-only and unaffected.

5. Docs: add a `DivideScalar` row to the reverse-mode op table in `docs/AUTODIFF.md`
   (near the existing `Divide` row ~line 300).

## Blast radius

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — additive (`DivideScalar`); no existing symbol changes.
- `src/Nivara/AutoDiff/ReverseGradTensor.cs` — additive public method; no existing symbol changes.
- `src/Nivara/AutoDiff/Nn/Functional/Loss.cs` — `Reduce` Mean branch rewritten (behavior/numerics unchanged).
- Downstream callers of `Loss<T>.Reduce`: `MSELoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`,
  `CrossEntropyLoss` — none change, all inherit the fix.
- Tests covering the Mean path: `LossTests.cs` (MSELoss/L1Loss/BCELoss/BCEWithLogitsLoss/CrossEntropyLoss
  Mean value + backward), NivaraTorch fixture suite, training loop tests (`TrainingLoop`, `DataParallelTrainer`).
- Docs: `docs/AUTODIFF.md` (op table, `Reduction` section reference), `docs/REVIEW-2026-08-12.md`
  item 11 is the origin of this issue — will be resolved by the fix.

## Verification

1. `dotnet build Nivara.slnx` (after each code change).
2. Targeted tests (ask human first per AGENTS.md):
   `dotnet test tests/Nivara.Tests --filter "FullyQualifiedName~AutoDiff"` — LossTests, GradOperationsTests, InferenceGraphTests, NivaraTorch.
3. Review `docs/TODO.md`; if everything is complete, remove it and commit.

## Planned commits

1. `docs: plan issue #207 — Loss<T>.Reduce scalar-division fix` (this file)
2. `Add ReverseGradOperations.DivideScalar scalar-division VJP`
3. `Add ReverseGradTensor.DivideByScalar public wrapper`
4. `Use DivideScalar in Loss<T>.Reduce Mean path`
5. `Document DivideScalar in AUTODIFF.md op table`
6. `Add DivideScalar and Mean-graph tests`
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet — log new issues created during execution here)
