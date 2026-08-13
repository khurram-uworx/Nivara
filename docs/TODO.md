# Plan: `ForwardGradOperations.DivideScalar` forward-mode parity (issue #224)

Branch: `khurram/207` (continues the #207 work — PR will cover both #207 and #224).

## Problem

Issue #207 added `ReverseGradOperations.DivideScalar` (reverse-mode scalar-division VJP)
to `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`. The forward-mode engine,
`ForwardGradOperations` (which "mirrors ReverseGradOperations" per the class doc and
`docs/AUTODIFF.md` line 87), has no scalar-divide counterpart — only the tensor-tensor
`Divide(a, b)`. Forward-mode users must wrap the scalar in a 1-element `ForwardGradTensor`,
mirroring the allocation problem #207 fixed in reverse mode.

## Change

1. Add `ForwardGradOperations.DivideScalar<T>(ForwardGradTensor<T> a, T scalar)`
   in `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` (Element-wise region,
   immediately after `Divide`):
   - null-check `a`; `if (scalar == T.Zero) throw new DivideByZeroException(...)`
     (mirrors the existing forward `Divide` guard).
   - primal: `TensorPrimitives.Divide(aSpan, scalar, primalArr)` → `NivaraColumn<T>.CreateFromOwnedArray`.
   - tangent (JVP): if `a.RequiresTangent`, `t_out = t_a / scalar` via
     `TensorPrimitives.Divide(aTanSpan, scalar, tanArr)`; otherwise `null`.
   - shape: `new ForwardGradTensor<T>(primal, tangent, PropagateShape(a))`.

   No op changes to `ForwardGradTensor`/existing ops. `ForwardGradTensor` keeps its
   operator-only API surface (no new wrapper — the issue scoped this to the ops method).

2. Tests (`tests/Nivara.Tests/AutoDiff/`):
   - `ForwardGradOperationsTests.cs`: `DivideScalar_Simple_ComputesCorrectValuesAndTangents`
     (primal `a/scalar`, tangent `t_a/scalar`) and `DivideScalar_ByZero_ThrowsException`.
   - `ForwardParityTests.cs`: `DivideScalar_ForwardTangent_EqualsBackwardGradient` —
     compares `ForwardGradOperations.DivideScalar(fa, c)` JVP against
     `Sum(ReverseGradOperations.DivideScalar(ra, c)).Backward()` gradient, the exact
     reverse-mode counterpart added in #207.

3. Docs: update `docs/AUTODIFF.md`:
   - forward-mode architecture line (line 88): `Add, Subtract, Multiply, Divide, Clip, LeakyRelu`
     → add `DivideScalar`.
   - forward-mode op table (~line 417): add a `DivideScalar(a, scalar)` row with
     `t_out = t_a / scalar`.

## Blast radius

- `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` — additive; no existing symbol changes.
- `ForwardGradTensor<T>` — untouched.
- Downstream: no production caller uses forward-mode scalar division today; forward-mode
  ops are used only via tests (`ForwardGradOperationsTests`, `ForwardParityTests`,
  `ForwardGradTensor` operators).
- Docs: `docs/AUTODIFF.md`.

## Verification

1. `dotnet build Nivara.slnx` (after each code change).
2. Targeted tests (ask human first per AGENTS.md):
   `dotnet test tests/Nivara.Tests --filter "FullyQualifiedName~AutoDiff"`.
3. Review `docs/TODO.md`; if complete, remove it and commit.

## Planned commits

1. `docs: plan issue #224 — ForwardGradOperations.DivideScalar parity` (this file)
2. `Add ForwardGradOperations.DivideScalar scalar-division JVP`
3. `Add forward-mode DivideScalar tests and parity test`
4. `Document forward-mode DivideScalar in AUTODIFF.md`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #224 — ForwardGradOperations has no DivideScalar for forward-mode parity (this plan)
