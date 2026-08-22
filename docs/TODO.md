# TODO — Issue #137: BFloat16 AutoDiff kernels (net11 migration)

**Issue:** https://github.com/khurram-uworx/Nivara/issues/137 (`enhancement`, `future`)
**Branch:** `khurram/137`

## Problem

Issue #137 deferred BFloat16-typed AutoDiff work until the repo targeted .NET 11.
The net11 migration has landed (all projects target `net11.0`; Tensors
`11.0.0-preview.7`). On .NET 11, `System.Numerics.BFloat16` implements
`IBinaryFloatingPointIeee754<BFloat16>` (verified via Microsoft Learn), so it
natively satisfies the AutoDiff constraint `IFloatingPointIeee754<T>` — every
generic op already compiles against it. What remains is admitting it at the
runtime gates, testing it, and documenting the decision.

## Blast radius

- **Changed files (core):**
  - `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` — add BFloat16 to `IsSupportedType`/`GetSupportedTypes`
  - `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` — BFloat16 arm in `ToReverseGradTensorsAuto`
- **Optional tuned arms (defer unless needed):** `RMSNormKernel.cs`, `Adam.cs`, `AdamW.cs` Half-style paths
- **Tests:** new `tests/Nivara.Tests/AutoDiff/BFloat16Tests.cs`; existing AutoDiff tests unaffected (no behavior change for float/double/Half)
- **Docs:** `docs/TENSORS.md`, `docs/AUTODIFF.md`, `AGENTS.md`, `CHANGELOG.md`, `ExpressionTypeInferer` XML comment
- **Downstream callers of the gates:** `TypeSafetyTests`, frame→tensor conversion paths. No public contract removed; purely additive.

## Plan / commit list

1. `docs: plan BFloat16 AutoDiff kernel support in TODO.md` ← this commit
2. **Gates:** admit BFloat16 in `TypeValidator` (+ `ToReverseGradTensorsAuto` arm, doc-comment updates). Verify build.
3. **Tests:** `BFloat16Tests.cs`:
   - TypeValidator accepts BFloat16; `GetSupportedTypes` contains it
   - forward/backward parity vs inline float references (bf16 ≈ 8-bit mantissa → rel tol ~1e-2): elementwise ops, Softmax/Mean, MatMul + Linear module
   - optimizer smoke: SGD + Adam step inside `using GradientUtils.Grad()`; loss decreases, grads finite
   - inference-default guard: no graph nodes outside `Grad()`
   - frame boundary: `NivaraFrame` BFloat16 column → `ToReverseGradTensorsAuto`
4. **Loader check:** verify `SafeTensorsLoader.ConvertBF16<BFloat16>` round-trips losslessly (BF16→F32→BF16 is exact) — expected zero code change; note in docs.
5. **Docs:** TENSORS.md type-support note, AUTODIFF.md type section, AGENTS.md known-issue entry, CHANGELOG, stale `ExpressionTypeInferer` XML comment wording.
6. Review branch vs issue acceptance criteria; remove TODO.md when all items done.

## Decisions

- BFloat16 **joins** the `IFloatingPointIeee754<T>` runtime-admitted set (it satisfies the interface natively at net11).
- Tuned RMSNorm/Adam/AdamW BF16 arms **deferred** behind a benchmark follow-up issue (generic double fallback is correct; no HW BF16 SIMD in TensorPrimitives yet).
- Out of scope (separate issues if ever wanted): fused-expression engine `IsGenericMathType`, DataFrame numeric promotion rules, Parquet/Arrow BF16 interop.

## Verification

- Build: `dotnet build Nivara.slnx`
- Tests: ask before running `dotnet test` (per AGENTS.md); target `Nivara.Tests` AutoDiff category first.

## GitHub issues log

- [ ] #137 — parent issue: BFloat16 AutoDiff kernels (this plan executes it)
- *(create follow-ups here as they are discovered during execution)*
