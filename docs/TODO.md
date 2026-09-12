# Plan: Vectorize RMSNorm backward grad loops with TensorPrimitives (#418)

Branch: `khurram/418` (off `khurram/413`). Issue: https://github.com/khurram-uworx/Nivara/issues/418

## Problem

Follow-up to #411: `RMSNorm<T>` backward closure (`OpNode` in `src/Nivara/AutoDiff/Nn/RMSNorm.cs`)
still uses scalar loops for two steps — dL/dy grad multiply (lines 104-109) and gamma-grad
accumulation (lines 121-126) — even though the forward gamma multiply was vectorized with
`TensorPrimitives.Multiply` in #411. The forward path is now fully `TensorPrimitives`-backed;
leaving the backward scalar is asymmetric and inconsistent.

## Key correctness caveat

The issue's literal suggestion — in-place `TensorPrimitives.Multiply(..., gradOutData, ...)` — is
**unsafe in the current ordering** because `gradOutData` is consumed twice inside the closure:

1. Loop 1 (lines 104-109): `gradNorm[...] = gradOutData[...] * savedGamma[j]` (reads raw gradOutData)
2. Loop 2 (lines 121-126): `gradWeightData[j] += yData[...] * gradOutData[...]` (reads raw gradOutData)

Overwriting `gradOutData` in place before loop 2 would scale the gamma-grad by gamma again.

**Resolution — Option B (reorder + in-place):** compute the gamma-grad first (while `gradOutData`
is still raw), then multiply `gradOutData` in place and reuse it as `gradNorm` for the input-grad
backward kernel. This literally fulfills the "in-place, identical to #411" wording *and*
eliminates the `gradNorm` allocation. The reorder is provably safe — both gradient computations
are independent (different target tensors, pure functions of `savedInput`).

## Proposed changes

### 1. Add module-based perf harness scenario (commit 1)

**File:** `tests/Nivara.PerformanceTests/Program.cs`

Add to `RunAutoDiffSimdScenarios()` after the functional RMSNorm rows (line ~479):

```csharp
Run("AutoDiff RMSNormModule fwd+bwd 1M x float", 5, 20,
    () =>
    {
        const int rows = 256, cols = 4096;
        var rms = new RMSNorm<float>(cols, eps: 1e-5f);
        var inputColumn = NivaraColumn<float>.Create(Fill(new float[rows * cols]));
        var ones = Fill(new float[rows * cols]);
        return () =>
        {
            using (GradientUtils.Grad())
            {
                var input = new ReverseGradTensor<float>(inputColumn, requiresGrad: true);
                input.Reshape(rows, cols);
                var output = rms.Forward(input);
                var gradient = new ReverseGradTensor<float>(NivaraColumn<float>.Create(ones), requiresGrad: false);
                gradient.Reshape(rows, cols);
                output.Backward(gradient);
            }
        };
    });
```

The existing `AutoDiff RMSNorm fwd+bwd 1M x float` row exercises the **functional op**
(`ReverseGradOperations.RMSNorm`) — not the module closure that #418 changes. This new row
covers the actual code being changed (both backward loops).

256x4096 = 1,048,576 ≈ 1M elements. Construction pattern mirrors the `Linear forward+backward`
row (module in setup, fresh input per timed iteration).

### 2. Capture before baseline (commit 1 continued)

```bash
dotnet run --project tests/Nivara.PerformanceTests -c Release \
    -- --only "RMSNorm" --json rmsnorm-before.json --runs 3
```

`--only RMSNorm` covers all three rows (functional fwd+bwd, scalar baseline, new module row).
Record before numbers in this file.

### 3. Implement vectorization (commit 2)

**File:** `src/Nivara/AutoDiff/Nn/RMSNorm.cs`

Replace the backward closure body (lines 97-128). Option B — reorder so gamma-grad runs first,
then in-place gamma multiply reuses `gradOutData` as `gradNorm`:

```csharp
var gradFn = new OpNode<T>("RMSNorm", [input], (typedGradOutput) =>
{
    var gradOutData = new T[typedGradOutput.Length];
    typedGradOutput.CopyTo(gradOutData, default(T)!);

    // dL/dgamma[j] = sum_i y[i,j] * dL/dOut[i,j]
    // Runs first: consumes raw gradOutData before the in-place gamma multiply below.
    var gradWeightData = new T[normalizedShape];
    var yData = new T[gradOutData.Length];
    RMSNormKernel<T>.PerRowRMSNormForwardKernel(
        savedInput, yData, savedRows, savedNormShape, savedEps);
    var rowProduct = new T[normalizedShape];
    for (int i = 0; i < rows; i++)
    {
        int baseIdx = i * normalizedShape;
        TensorPrimitives.Multiply(yData.AsSpan(baseIdx, normalizedShape), gradOutData.AsSpan(baseIdx, normalizedShape), rowProduct);
        TensorPrimitives.Add(rowProduct, gradWeightData, gradWeightData);
    }
    ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.CreateFromOwnedArray(gradWeightData));

    // dL/dy = dL/dOut * gamma (since out = y * gamma) — in-place, aliasing identical to #411
    for (int i = 0; i < rows; i++)
    {
        int baseIdx = i * normalizedShape;
        TensorPrimitives.Multiply(gradOutData.AsSpan(baseIdx, normalizedShape), savedGamma, gradOutData.AsSpan(baseIdx, normalizedShape));
    }

    var gradInputData = new T[gradOutData.Length];
    RMSNormKernel<T>.PerRowRMSNormBackwardKernel(
        savedInput, gradOutData, gradInputData, savedRows, savedNormShape, savedEps);
    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.CreateFromOwnedArray(gradInputData));
});
```

**Numerical-equivalence argument:**
- Loop 1: elementwise `Multiply` — per-element IEEE multiply, bit-identical to scalar.
- Loop 2: `Multiply` per row into `rowProduct` (bit-identical products), then `Add(rowProduct, gradWeightData, gradWeightData)` per row — same accumulation order as scalar (row 0, 1, ...), bit-identical sums.
- `MultiplyAdd` deliberately avoided — its FMA fusion changes rounding, violating "no numerical behavior change".
- In-place `Add(src, dst, dst)` is an established repo pattern (`SGD.cs:121`, `ReverseGradOperations.cs:2427/2459`).

### 4. Capture after baseline + compare (commit 2 continued)

```bash
dotnet run --project tests/Nivara.PerformanceTests -c Release \
    -- --only "RMSNorm" --compare rmsnorm-before.json --runs 3
```

### 5. Record results

Update this file's Results section with before/after numbers.

## Blast radius

- **Changed:** `src/Nivara/AutoDiff/Nn/RMSNorm.cs` (single backward closure body)
- **No public API change** — internal implementation detail only
- **Downstream callers:** `RMSNorm<T>.Forward` backward path (module only; functional op in
  `ReverseGradOperations` is separate and unaffected)
- **Tests (module backward closure — the real coverage):**
  - `tests/Nivara.Tests/NivaraTorch/RMSNormModuleTests.cs` — `RMSNormModule_2D_AffineGamma_MatchesPyTorch`
    (asserts output, input grad, gamma grad vs PyTorch fixtures) — acceptance-criteria test
  - `tests/Nivara.Tests/AutoDiff/RMSNormTests.cs` — `Backward_AccumulatesGradientsOnInputAndWeight`
- **Tests (controls — untouched by this change, kept green as regression checks):**
  - `TransformerBlock_RmsNorm_MatchesPyTorch` — TransformerBlock uses its OWN private non-affine
    `PerRowRMSNorm` helper (no gamma multiply), NOT the `RMSNorm<T>` module (verified in
    `src/Nivara/AutoDiff/Nn/TransformerBlock.cs:196-231`)
  - Functional op tests (`ForwardGradOperationsTests`, `ForwardParityTests`) — separate
    `ReverseGradOperations.PerRowRMSNorm` → `GradOperationKernels.ApplyRMSNorm` path
- **Perf harness:** new scenario row + existing functional RMSNorm row as control (must stay flat)

## Planned commits

1. `perf: add RMSNormModule fwd+bwd scenario and capture baseline` — harness row + before JSON
2. `perf: vectorize RMSNorm backward grad loops with TensorPrimitives (#418)` — code change + after numbers

## Results

### Before (scalar backward, current code)

_Captured during commit 1._

### After (TensorPrimitives backward)

_Captured during commit 2._

## GitHub issues log

- (none discovered during planning)
