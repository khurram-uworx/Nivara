# Plan: Issue #169 — Sum/Mean aggregation for full numeric domain

## Problem

`SumAggregation` and `MeanAggregation` in `src/Nivara/Operations/AggregationFunction.cs`
only support a subset of the numeric domain:

- `Apply` switch (lines 221–231) handles only `int/byte/short/long/float/double/decimal`.
- `GetResultType` (lines 190–200) defaults unknown numerics to `inputType`.
- `ValidateInputType` uses `IsNumericType()` (13 types incl. `bool`, **excludes `Half`**).

Consequences:
- `uint/ushort/sbyte/ulong/char/bool` pass validation then throw `ArgumentException` in `Apply`.
- `Half/nint/nuint/Int128/UInt128` fail validation entirely ("Sum aggregation requires numeric type").
- Group-by Min/Max already support `uint/ushort/sbyte/ulong`; Sum/Mean lag behind.

Root-cause constraint: `System.Half` does **not** implement `IConvertible`
(dropped deliberately, dotnet/runtime PR #37630), so the existing
`SumVectorized<TResult>` widening via `Convert.ChangeType` (lines 261–269) throws
`InvalidCastException` for Half. The fix must use cast-based widening.

## Decisions (confirmed with human)

1. `bool` Sum/Mean → implemented: Sum(bool) = count of `true` (as `long`), Mean(bool) = proportion (as `double`).
2. Scope includes `nint/nuint/Int128/UInt128` now (full 17-type `GetNumericTypes()` domain + `bool`).
3. Plan lives in `docs/TODO.md` on branch `khurram/169`.

## Result-type mapping (`SumAggregation.GetResultType`)

| Input | Result |
|---|---|
| byte, sbyte, short, ushort, int, uint, char, bool | `long` |
| long | `long` |
| ulong | `ulong` |
| nint | `Int128` |
| nuint | `UInt128` |
| Int128 | `Int128` |
| UInt128 | `UInt128` |
| float, Half | `double` |
| double | `double` |
| decimal | `decimal` |

Mean stays `double` for all (unchanged behavior).

## Implementation

### 1. `src/Nivara/Operations/AggregationFunction.cs`

- **Validation**: `SumAggregation.ValidateInputType` and `MeanAggregation.ValidateInputType`
  switch from `inputType.IsNumericType()` to accepting
  `TypeCompatibilityValidator.GetNumericTypes()` (17 types) **+ `typeof(bool)`**.
  Both current helpers already unwrap `Nullable<T>`, so nullable columns keep working.
  Reuse pattern:
  ```csharp
  protected override void ValidateInputType(Type inputType)
  {
      var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
      var supported = TypeCompatibilityValidator.GetNumericTypes().Append(typeof(bool));
      if (!supported.Contains(underlying))
          throw new ArgumentException($"Sum aggregation requires numeric type, got {inputType.Name}");
  }
  ```
  (Message keeps the existing key phrase "Sum aggregation requires numeric type" so tests pass.)

- **`GetResultType`**: replace the `_ => inputType` default with explicit mapping for all 18 types
  (keyed on `underlyingType`). No default fall-through — throw `ArgumentException` if unknown.

- **`Apply`**: add arms per input type routing to the generic sum helper:
  - int/byte/short/long/uint/ushort/sbyte/char → `SumVectorized<T, long>`
  - bool → `SumVectorizedBool<long>`
  - ulong → `SumVectorized<ulong, ulong>`
  - nint → `SumVectorized<nint, Int128>`
  - nuint → `SumVectorized<nuint, UInt128>`
  - Int128 → `SumVectorized<Int128, Int128>`
  - UInt128 → `SumVectorized<UInt128, UInt128>`
  - float/Half → `SumVectorized<float, double>` / `SumVectorized<Half, double>`
  - double → `SumVectorized<double, double>`
  - decimal → `SumScalarDecimal` (unchanged)

- **Widening helper**: replace `Convert.ChangeType` with typed `CreateChecked`:
  ```csharp
  static object SumVectorized<TSource, TResult>(List<object> validValues)
      where TSource : INumberBase<TSource>
      where TResult : unmanaged, INumber<TResult>
  {
      var widened = new TResult[validValues.Count];
      for (int i = 0; i < validValues.Count; i++)
          widened[i] = TResult.CreateChecked((TSource)validValues[i]);
      return TensorPrimitives.Sum(widened.AsSpan());
  }

  static object SumVectorizedBool<TResult>(List<object> validValues)
      where TResult : unmanaged, INumber<TResult>
  {
      var widened = new TResult[validValues.Count];
      for (int i = 0; i < validValues.Count; i++)
          widened[i] = TResult.CreateChecked((bool)validValues[i] ? 1 : 0);
      return TensorPrimitives.Sum(widened.AsSpan());
  }
  ```
  Unboxing `(TSource)validValues[i]` is safe — `ExtractValidValues` boxes values of exactly
  the column element type (same pattern as `MinVectorized`/`MaxVectorized`).
  `TensorPrimitives.Sum<T>` (`T : INumber<T>`) is already proven for ulong/Int128/UInt128/
  nint/nuint in `NivaraSeries.cs:84–93`.

- **`GetZeroValue`**: add explicit `0UL`, `Int128.Zero`, `UInt128.Zero` arms
  (drop reliance on the `Activator.CreateInstance` fallback for these).

- **`CreateColumnFromValues`**: add `typeof(ulong)`, `typeof(Int128)`, `typeof(UInt128)`
  arms so `ApplyToGroups` returns typed columns instead of `NivaraColumn<object>`.

### 2. Tests — `tests/Nivara.Tests/Operations/AggregationFunctionTests.cs`

Extend `SumAggregationTests`, `MeanAggregationTests`, and `ApplyToGroupsTests`:

- Sum + GetResultType for each new input type: uint, ushort, sbyte, ulong, Half, char, bool,
  nint, nuint, Int128, UInt128 (incl. nullable variants for a few).
- Mean for the same types (verifies `Convert.ToDouble` on Int128/UInt128/nint/nuint sums —
  all implement `IConvertible`).
- Empty-group zero for ulong/Int128/UInt128 (`GetZeroValue`).
- `ApplyToGroups` typed result columns for ulong/Int128/UInt128.
- Keep existing tests green (contract unchanged except previously-throwing types now work).

### 3. Docs

- `CHANGELOG.md`: entry for the Sum/Mean numeric-domain fix.
- `AGENTS.md`: mark the #169 known-issues note resolved; correct the inaccurate
  "divideByCount covers all 13 IsNumericType" claim (it covers 12 — bool missing).

## Blast radius

- **Files changed**: `src/Nivara/Operations/AggregationFunction.cs`,
  `tests/Nivara.Tests/Operations/AggregationFunctionTests.cs`, `CHANGELOG.md`, `AGENTS.md`.
- **Direct consumers**: `GroupByOperation.TransformSchema` (uses `GetResultType`,
  GroupByOperation.cs:280) and `ApplyToGroups`/`CreateColumnFromValues` result columns.
  Also `NivaraSeries` Sum/Average are unaffected (same-type `T`, separate code path).
- **Behavioral change**: previously-throwing types (`uint/ushort/sbyte/ulong/char/bool`) and
  previously-rejected types (`Half/nint/nuint/Int128/UInt128`) now aggregate successfully.
  No existing passing behavior regresses. Column types produced by group-by Sum on those inputs
  change from `NivaraColumn<object>` (or exception) to typed columns — this is the fix.
- **Tests covering this area**: `AggregationFunctionTests.cs` (504 lines),
  plus any frame-level group-by tests that use Sum/Mean.

## Verification steps

1. `dotnet build Nivara.slnx` — after each code change.
2. `dotnet test` (ask human before running) — full suite.
3. Grep for other `Convert.ChangeType` in aggregation paths to confirm no stragglers.

## Planned commit list

1. `docs: plan issue #169 Sum/Mean numeric domain fix in TODO.md`
2. `Fix Sum/Mean aggregation for full numeric domain`
3. `test: cover Sum/Mean for new numeric types`
4. `docs: mark #169 resolved, correct divideByCount note`

## GitHub issues log

- [ ] #169 — Sum/Mean aggregation missing uint/ushort/sbyte/ulong/Half/char support
      (this plan — tracked via gh, no new issues expected yet)
