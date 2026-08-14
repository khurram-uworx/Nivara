# Plan: Forward-mode JVP parity for ForwardGradOperations (#236)

## Problem

`ForwardGradOperations` (24 public JVP ops) is missing 16 ops that exist in
`ReverseGradOperations` (40 public ops), so forward-mode graphs cannot compute
JVPs through attention, slicing/concatenation, RMSNorm, gather, embedding-bag,
or broadcasting. ADR-003's consistency obligation (any primitive op ships with
forward-mode symmetry) is violated for these ops.

## Approach

Implement the 16 missing forward JVP ops for full parity, backed by a new
shared internal helper class that consolidates the column-level kernels currently
duplicated as private statics in `ForwardGradOperations` and
`ReverseGradOperations` (per AGENTS.md rule 8). Add unit tests, reverse-parity +
finite-difference cross-validation tests, and update the forward-mode docs.

## Changes

### 1. New shared helper class — `src/Nivara/AutoDiff/Operations/GradOperationKernels.cs`

`internal static class GradOperationKernels<T>` (same namespace). Hosts kernels
moved out of the two ops classes (which then call these instead of private copies):

- `ApplyDropout(input, keepMask, scale)` — used by both
- `ApplyDropoutGradient(input, gradOutput, keepMask, scale)` — reverse; identical
  math to forward-mode tangent
- `ApplyKlElementWise(mean, logVar)` — both
- `ApplyKlMeanGradient(mean, gradOutput)` — reverse
- `ApplyKlLogVarGradient(logVar, gradOutput)` — reverse; identical to forward tangent
- `ApplySampleNormalForward(mean, logVar, epsilon)` — both
- `ApplySampleNormalLogVarGradient(logVar, gradOutput, epsilon)` — reverse/forward
- `ApplyPow(input, exponent)` — both
- `ApplyPowGradient(input, gradOutput, exponent)` — reverse; identical to forward tangent
- `ApplyRMSNorm(input, eps)` — both
- `ApplyRMSNormGradient(input, gradOutput, eps)` — reverse; symmetric Jacobian ⇒
  reused as forward JVP
- `BroadcastGradient(scalarGrad, targetLength)` — reverse only

Delete the per-file private duplicates from both ops classes.

### 2. New forward JVP ops — `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs`

Signatures mirror `ReverseGradOperations` exactly. Tangent omitted (null) when
no input requires it; absent tangents contribute zero / skip terms.

- Group A (selection/shape): `Slice(a, start, length)`,
  `Concat(tensors, axis = 0)`, `TransposeAxes(a, axis1, axis2)`,
  `Gather(source, indices, axis = 0)`, `SparseEmbeddingBag(weight, indices,
  paddingIndex = -1)`, `MatMulTransposedB(a, b)` — JVP = same transform applied
  to the tangent (`t_a @ b^T + a @ t_b^T` via `GradKernels.MatMulTransposedB`).
- Group B (elementwise): `GeluExact(a)` via `GradKernels.GeluExact`/`GeluExactGradient`,
  `Pow(a, exponent)` via shared `ApplyPow`/`ApplyPowGradient`.
- Group C (reduction/broadcast): `MeanPool(a, poolSize, embedDim)`,
  `RMSNorm(a, eps = 1e-5)` via `ApplyRMSNormGradient`,
  `PerRowRMSNorm(a, rows, cols, eps = 1e-5)` via
  `RMSNormKernel.PerRowRMSNormBackwardKernel`,
  `BroadcastMultiply(input, scale)`, `BroadcastAdd(input, bias)`.
- Group D (attention): `MultiHeadAttention(query, key, value, numHeads, scale,
  mask = null)`, `BatchedMultiHeadAttention(...)` — per-head softmax JVP:
  `t_scores = scale*(MatMulTransposedB(t_Qh, K_h) + MatMulTransposedB(Q_h, t_Kh))`;
  `SoftmaxBackwardRows(P, t_scores, qLen, kvLen)` (in-place softmax JVP);
  `t_out_h = MatMul(t_P, V_h) + MatMul(P, t_Vh)`; scatter back. Mask is a
  non-differentiable constant (reverse never accumulates to it).

Result: forward-mode public ops 24 → 40 = full reverse-mode parity.

### 3. Tests

- `tests/Nivara.Tests/AutoDiff/ForwardGradOperationsTests.cs` — unit tests per op:
  primal values, output shape, tangent propagation (single-input and both-input
  cases), argument-validation errors (dim mismatches, out-of-range slice/indices,
  rank checks, non-divisible pool length).
- `tests/Nivara.Tests/AutoDiff/ForwardParityTests.cs` — cross-validation:
  - reverse-mode parity (seed tangent = ones, compare vs `Sum(op).Backward()`)
    where `J·1 = J^T·1` holds: Pow, GeluExact, RMSNorm, PerRowRMSNorm, Slice,
    Concat, TransposeAxes, Gather (unique), AddBias (seed on a), broadcast ops
    (seed on input), MatMulTransposedB (symmetric B + seed on A).
  - finite-difference JVP (central diff `(f(x+εv)-f(x-εv))/(2ε)`) for the rest:
    MultiHeadAttention, BatchedMultiHeadAttention, MeanPool, SparseEmbeddingBag,
    AddBias (seed on bias), broadcast ops (seed on scale/bias), MatMulTransposedB
    (general), Gather (duplicates). Small test-local numeric-diff helper.

### 4. Docs

- `docs/AUTODIFF.md` §Forward-Mode: extend the JVP table with the 16 ops,
  replace the "All 21 operations..." sentence, state full reverse-mode parity.

## Blast radius

- `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` — +16 ops, −5 private helpers.
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — −12 private helpers
  (moved to shared class). Public API unchanged.
- `src/Nivara/AutoDiff/Operations/GradOperationKernels.cs` — new internal class.
- `tests/Nivara.Tests/AutoDiff/ForwardGradOperationsTests.cs`,
  `ForwardParityTests.cs` — extended.
- `docs/AUTODIFF.md` — updated.
- Downstream: `src/Nivara/AutoDiff/Nn/*` call `ReverseGradOperations` public API
  only — unaffected by the internal refactor. No public API change, so
  NivaraTorch/sample callers are unaffected.

## Verification

- `dotnet build Nivara.slnx`
- Targeted tests: `ForwardGradOperationsTests`, `ForwardParityTests`
  (ask before running `dotnet test` per AGENTS.md).

## Planned commits

1. `docs: plan forward-mode JVP parity (#236) in TODO.md`
2. `Add shared GradOperationKernels; consolidate forward/reverse op helpers`
3. `Add forward JVP ops: Slice, Concat, TransposeAxes, Gather, SparseEmbeddingBag, MatMulTransposedB`
4. `Add forward JVP ops: GeluExact, Pow, MeanPool, RMSNorm, PerRowRMSNorm, BroadcastMultiply, BroadcastAdd`
5. `Add forward JVP ops: MultiHeadAttention, BatchedMultiHeadAttention`
6. `Add unit tests for forward JVP parity ops`
7. `Add parity and finite-difference cross-validation tests for forward JVP`
8. `Update forward-mode JVP table in AUTODIFF.md`
9. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #236 — forward-mode AD gap: 16 missing JVP ops (this plan implements them)
