# Plan: Issue #168 — NivaraColumn<T> arithmetic kernels do not support char element type

## Problem

`NivaraColumn<char>` arithmetic throws `InvalidOperationException`. After #164 added
char to the numeric domain, `AreArithmeticCompatible(char, char)` and `WithColumn(char)`
succeed while direct column arithmetic still fails — inconsistent.

Root causes:
1. `TypeExtensions.IsNumericType()` does not include `char` → `NivaraColumn<char>`
   arithmetic validation throws (NivaraColumn.cs:358).
2. `ColumnStorageFactory.IsVectorizable()` does not include `char` → would throw even
   if numeric (NivaraColumn.cs:360).
3. No `typeof(char)` branches in the 14 tensor-primitive dispatch helpers in
   `NivaraColumn.cs` (add/multiply/subtract/divide × {column,scalar} and
   equals/greaterThan/lessThan × {column,scalar}).
4. `NivaraSeries.sumTensorPrimitive` (NivaraSeries.cs:69-96) and `divideByCount`
   (NivaraSeries.cs:101-176) do not dispatch `char` → `Sum<char>/Average<char>` throw.

## Verified feasibility

A scratch probe confirmed `TensorPrimitives<char>` works at runtime for
Add/Multiply/Subtract/Divide/Sum/Min/Max (wraparound semantics, exactly like ushort).
`char` satisfies `INumber<char>`, so `NumericTensorKernels<char>` compiles and runs.

## Decisions (confirmed with user)

- Scope: **char only**, mirroring ushort support. The pre-existing uint/ushort/sbyte/
  ulong group-by `SumAggregation`/`MeanAggregation` gap is tracked as a separate
  follow-up issue (created during planning: #169).
- No change to `NivaraFrameExtensions.ExcludedNumericTypes` (normalization) — char
  stays excluded there (its check is INumber-based and out of scope).

## Proposed changes

### 1. `src/Nivara/Extensions/TypeExtensions.cs`

`IsNumericType()`: add `underlying == typeof(char)`.

### 2. `src/Nivara/Storage/ColumnStorageFactory.cs`

`IsVectorizable(Type)`: add `type == typeof(char)`.

### 3. `src/Nivara/NivaraColumn.cs`

Add a `typeof(char)` dispatch branch (calling `NumericTensorKernels<char>`) to each of
the 14 helpers: `multiplyTensorPrimitive` (scalar + column), `addTensorPrimitive`
(column), `subtractTensorPrimitive` (column), `divideTensorPrimitive` (column +
scalar), `equalsTensorPrimitive` (scalar + column), `greaterThanTensorPrimitive`
(scalar + column), `lessThanTensorPrimitive` (scalar + column).

### 4. `src/Nivara/NivaraSeries.cs`

`sumTensorPrimitive`: add `if (type == typeof(char)) return reinterpretBack(NumericTensorKernels<char>.Sum(SpanReinterpret.ReadOnly<T, char>(values)));`
`divideByCount`: add a char branch mirroring ushort (cast sum, integer divide, cast back).

### 5. Tests

- `NivaraColumnTests.cs` (or a focused new fixture): char Add/Multiply/Subtract/Divide
  (column+column and scalar), operators, comparisons — mirroring existing ushort/byte tests.
- `NivaraSeriesAggregateTests.cs`: `Sum<char>`/`Average<char>` wrap-around results.
- `TypeExtensions`/`ColumnStorageFactory`: assert `IsNumericType(char)` and
  `IsVectorizable<char>()` are true (via existing fixtures if present).
- Existing test risk: `Sum_AllIntegerPrimitives_ReturnsCorrectSum` and others enumerate
  the integer primitives explicitly — char is not asserted as non-numeric anywhere, so
  no existing test should break.

## Blast radius

- `IsNumericType()` consumers now accept char: `NivaraColumn.validateTypeSupportsOperation`
  (arithmetic + comparison), `NivaraSeries.Average` guard, `SumAggregation`/`MeanAggregation`
  `ValidateInputType`, `AggregationFunction.IsComparableType`, `TypeExtensions.IsComparableType`.
  The group-by aggregation switches remain incomplete for char (see #169) — this means
  group-by Sum/Mean on char will pass validation then throw the existing ArgumentException,
  identical to today's uint/ushort/sbyte/ulong behavior. Documented, out of scope.
- `IsVectorizable<char>()` now true: `NivaraColumn.IsVectorizable`, `OperationDiagnostics.IsVectorizable`,
  `KernelSelector.DetermineKernelType` all report vectorizable for char. `ColumnStorageTests`
  asserts Half stays non-vectorizable — Half untouched.
- ADR-001/002/003 constrain AutoDiff, not column kernels — no constraint.

## Verification

- `dotnet build Nivara.slnx` after each change unit.
- `dotnet test` only with human confirmation (focused fixtures first).

## Planned commits

1. `docs: plan issue #168 char column arithmetic kernels in TODO.md`
2. `feat: add char to IsNumericType and IsVectorizable`
3. `feat: dispatch char through NumericTensorKernels in NivaraColumn arithmetic`
4. `feat: support char in NivaraSeries Sum/Average dispatch`
5. `test: cover char column arithmetic and series Sum/Average`
6. `docs: log issue #169 (group-by aggregation gaps for uint/ushort/sbyte/ulong/Half)`
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [x] #169 — SumAggregation/MeanAggregation Apply switches miss uint/ushort/sbyte/ulong/Half (and char) (created while planning #168)
