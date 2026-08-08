# Plan: Issue #164 — char operand pairs excluded from numeric promotion

## Problem

`NumericPromoter.GetPromotedType(char, char)` returns `null` (issue #164). C#
binary numeric promotion (§12.4.7.3) treats `char` as a *small integral* type and
promotes a same-type pair to `int`. The `(char, char) → int` case is already in
`NumericPromoterTests.cs:39` and currently fails.

Root cause: `char` is missing from `TypeCompatibilityValidator.GetNumericTypes()`,
the numeric-domain guard at the top of `NumericPromoter.GetPromotedType`
(`NumericPromoter.cs:22-24`). Char was deliberately deferred from issue #152
(commit `8a6f5c8`); `IsSmallIntegralType` already lists it (`NumericPromoter.cs:85`).

While fixing #164 we also correct a pre-existing C# spec deviation in the same
promoter: C# rule 7 says `uint + byte/ushort/char` promotes to `uint`, but the
current rule-6 branch (`NumericPromoter.cs:69-71`) returns `long` for every `uint`
pair. This is exactly the class of bug #164 is about and is fixed in the same
method, so it ships here.

## Decisions (confirmed with user)

1. Add `typeof(char)` to `GetNumericTypes()` (char enters the numeric domain).
2. Fix the `uint` rule-6/7 deviation in the same issue (no separate follow-up).
3. No plan doc, no comment on the GitHub issue — this `docs/TODO.md` is the plan.

## Proposed changes

### 1. `src/Nivara/Helpers/TypeCompatibilityValidator.cs`

`GetNumericTypes()` (lines 229-238): add `typeof(char)` → 16 → 17 numeric types.

### 2. `src/Nivara/Helpers/NumericPromoter.cs`

Rule 6/7 (lines 69-71): `uint` promotes to `uint` with `byte`, `ushort`, `char`
(C# rule 7); otherwise to `long` (rule 6, signed integral). Keep the existing
comment accurate and cite both rules.

```csharp
// C# rule 7: uint + byte/ushort/char promotes to uint.
if (left == typeof(uint) || right == typeof(uint))
{
    var other = left == typeof(uint) ? right : left;
    return other == typeof(byte) || other == typeof(ushort) || other == typeof(char)
        ? typeof(uint)
        : typeof(long);
}
```

No change needed in the `IsSmallIntegralType` branch — `char` is already listed.

## Blast radius

- `GetNumericTypes()` consumers: `NumericPromoter`, `AreArithmeticCompatible`,
  `AreComparisonCompatible`, `AreTypesCompatible`, `SupportsComparison` (no-op for
  char), `GetCompatibleTypes`, `ValidateAllNumeric`, `GetComparisonSupportedTypes`,
  `NivaraFrame.validateColumnTypeForAddition` (`WithColumn(char)` becomes accepted).
  `ValidateOperationSupport` has no src callers.
- Expression engine needs no kernel changes: `FusedExpressionEvaluator` compiles
  the promoted pair via `Expression.Convert(char → int)` (`ConvertTo`, line 520)
  and the generic node-tree kernel never sees `char` as a result type.
- `NivaraFrameExtensions.ExcludedNumericTypes` (normalization) uses a separate
  INumber-based check that explicitly excludes `char` — unaffected.
- ADR-001/002/003 constrain AutoDiff, not type compatibility — no constraint.
- Only one existing test breaks: `Property_OperationSupport_ValidatesCorrectly`
  asserts the exact 16-type list (`TypeCompatibilityValidatorTests.cs:259-267`).

## Verification

- `dotnet build Nivara.slnx` after each change unit.
- `dotnet test` only with human confirmation.

## Planned commits

1. `docs: plan issue #164 char numeric promotion in TODO.md`
2. `feat: add char to numeric domain (GetNumericTypes)`
3. `fix: promote uint+byte/ushort/char to uint per C# rule 7`
4. `test: cover char promotion pairs and uint rule-7 in NumericPromoterTests`
5. `test: update GetNumericTypes expectation to 17 types`
6. `test: add char fused-evaluator inference/evaluation tests`

## GitHub issues log

- [x] #168 — NivaraColumn<T> arithmetic kernels do not support char element type (created while working on #164: char is now validator-numeric but direct column arithmetic still throws)
