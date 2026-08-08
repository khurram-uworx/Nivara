# TODO: Typed promoted path for mixed-type numerics in ExpressionEvaluator

## Problem

`ExpressionEvaluator` only uses its typed fast path when both operands have the
**same** element type (int/long/float/double for binary; wider set for
comparison). Any mixed numeric pair (`double + int`, `x["A"] + 1` where `A` is
double, `x["A"] > 5`) falls back to a **boxed** implementation that does
per-row `GetValue` + `Convert.ToDouble` + boxing and produces
`NivaraColumn<object?>` results.

This is both slow and type-losing. It also makes mixed-type results diverge
from the library's existing C#-operator semantics (the typed same-type path
already does integer division for integrals via `TensorPrimitives.Divide<T>`).

## Decision (confirmed)

- **Semantics:** C# binary numeric promotion (C# spec §12.4.7.3) — the common
  promoted type is both the result type and the type the operator runs in.
  Integer division stays integer for integral results; floating division stays
  floating. This matches `Math`/`MathF` (no promotion; per-type) and the
  community convention.
- **Two pairs C# rejects (binding-time errors) resolve to `double`**
  (documented extension; matches current boxed `Convert.ToDouble` behavior):
  - `ulong` + `sbyte|short|int|long`
  - `decimal` + `float|double`
- **No boxing:** promoted operands computed via generic `INumber<TResult>`
  (`TResult.CreateChecked`) into a typed `NivaraColumn<TResult>`.
- Non-numeric and non-promotable pairs (Guid, string, etc.) stay on the boxed
  fallback, unchanged.

## Changes

### Task 1 — `src/Nivara/Helpers/NumericPromoter.cs` (new)

`internal static class NumericPromoter` with `Type? GetPromotedType(Type left, Type right)`
over `TypeCompatibilityValidator.GetNumericTypes()`:

```
same type          → same
non-numeric        → null
decimal + fp       → double        (C#: error; extension)
decimal + integral → decimal
either double      → double
either float       → float
ulong + signed     → double        (C#: error; extension)
ulong + unsigned   → ulong
long + uint        → long
long + other signed → long
uint + signed      → long          (C#: uint + int → long)
small integral pairs → int
```

### Task 2 — `src/Nivara/Helpers/ExpressionEvaluator.cs`

- `TryEvaluateTypedBinary`: keep same-type switch; when element types differ,
  compute `NumericPromoter.GetPromotedType`; if non-null dispatch to
  `TryBinaryPromoted<TLeft,TRight,TResult>` via cached `MakeGenericMethod`
  (pattern already used at `NivaraColumn.cs:2441`). `TargetInvocationException`
  wrapping `InvalidOperationException` → return null (boxed fallback).
- New `TryBinaryPromoted<TLeft,TRight,TResult>` (`where TLeft/TRight/TResult : struct, INumber<T>`):
  reuse `NivaraColumn<T>.Zip<T2,TResult>` (NivaraColumn.cs:2482 — already does
  null-OR propagation + pooled `CreateFromSpans`) with
  `static (a,b) => TResult.CreateChecked(a) OP TResult.CreateChecked(b)` for
  Add/Subtract/Multiply/Divide; `null` for And/Or.
- `TryEvaluateTypedComparison`: same structure; new
  `TryComparisonPromoted<TLeft,TRight,TResult>` using `Zip` with promoted
  `==/!=/</>...` operators.
- Add `using System.Reflection;` and a static
  `ConcurrentDictionary<(Type,Type,Type),MethodInfo>` cache.
- Callers already increment `typedPathEvaluationCount`; no caller changes.

### Task 3 — tests

- `tests/Nivara.Tests/Query/ExpressionEvaluatorTests.cs`
  - Rewrite `Evaluate_MixedTypeNumeric_FallsBackToBoxedPath` →
    `Evaluate_MixedTypeNumericBinary_UsesTypedPromotedPath`: typed==1,
    boxed==0, `ElementType == double`, values == `(double)a + (double)b`,
    nulls OR'd.
  - Add: int+long → long; scalar mix `Col("A") + 1` (double col) → double;
    comparison mix `Col("A") > 1` (double col) → typed bool w/ null OR;
    `decimal` + int → decimal; `byte` + int → int.
  - Keep Guid boxed tests (boxed-path preservation coverage).
- `tests/Nivara.Tests/Helpers/NumericPromoterTests.cs` (new): promotion table
  tests. `InternalsVisibleTo` already configured.

### Task 4 — docs

- `CHANGELOG.md` entry under current unreleased section.

## Verification

1. `dotnet build Nivara.slnx` (no confirm needed).
2. Ask human before running `dotnet test tests/Nivara.Tests`.

## Planned commits (one per logical unit)

1. `docs: plan typed promoted path for mixed numerics in TODO.md`
2. `feat: add NumericPromoter with C# binary-numeric-promotion table`
3. `feat: route mixed numeric binary/comparison operands through typed promoted kernels`
4. `test: cover promoted typed path and promotion table`
5. `docs: update CHANGELOG for mixed-type typed promotion`
6. `docs: remove TODO.md — plan executed`

## Follow-ups (not in scope)

- Same-type non-switch numerics (`byte/byte`, `uint/uint`, `decimal/decimal`)
  still box → `double`; could route through promotion (C# says `int` for small
  integral pairs) in a later pass.
- `%` (modulo) operator not currently exposed by expression evaluator.
