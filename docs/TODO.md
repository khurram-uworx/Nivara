# TODO: NivaraColumn<T> arithmetic generic-math collapse (#157)

Branch: `khurram/157` (base `main`)

## Problem

`NivaraColumn<T>` arithmetic kernels dispatch only the 11 vectorizable primitives.
`NivaraColumn<Half>.Add(...)` / `Multiply(...)` / etc. throw
`InvalidOperationException` ("Arithmetic operations are not supported for
non-vectorizable type Half") from `validateTypeSupportsOperation` — there is no
`INumber<T>` scalar fallback path. `decimal` passes validation but hits the
kernel dispatch throw. `Half`, `nint`, `nuint`, `Int128`, `UInt128` are rejected
outright because `TypeExtensions.IsNumericType()` does not recognize them.

Phase 2 acceptance criterion "Half/BFloat16 columns execute through the generic
path" is met for the fused evaluator, typed comparisons, and `NivaraSeries`
aggregates — but not for `NivaraColumn<T>` arithmetic.

## Approach (choice A)

Reuse the established `NumericTensorKernels<T>` typed dispatch (the same
pattern `NivaraSeries.sumTensorPrimitive` already uses for these types) and add
the six extended numeric types (`decimal`, `Half`, `nint`, `nuint`, `Int128`,
`UInt128`) to the `NivaraColumn<T>` kernel helpers. Verified against .NET 10
runtime source (`dotnet/runtime` release/10.0, `TensorPrimitives.Add.cs` +
`netcore/Common/TensorPrimitives.IBinaryOperator.cs`):

- `TensorPrimitives.Add/Subtract/Multiply/Divide<T>` **work at runtime** for all
  six types. `decimal`/`Int128`/`UInt128` run through the operator-based
  `SoftwareFallback`; `Half` uses the dedicated Half→float→Half path; `nint`/
  `nuint` get real SIMD. No `NotSupportedException` for any of them.
- `NumericTensorKernels<T>` is `INumber<T>`-constrained; all six types satisfy
  the constraint (already instantiated today for `NivaraSeries.Sum`).

## Changes

### 1. `src/Nivara/Extensions/TypeExtensions.cs`
Extend `IsNumericType()` with `Half`, `nint`, `nuint`, `Int128`, `UInt128`.
Result = the `GetNumericTypes()` domain + `bool`.

Blast radius: only other callers are `TypeExtensions.IsComparableType()` and
`AggregationFunction.IsComparableType()`. Both already accept these types via
`IComparable`, so no behavioral regression. No test references `IsNumericType`
directly.

### 2. `src/Nivara/NivaraColumn.cs`
- `validateTypeSupportsOperation("arithmetic")`: remove the
  `!ColumnStorageFactory.IsVectorizable<T>()` throw. Non-numeric types
  (`string`/`Guid`/`DateTime`) still throw the clear `InvalidOperationException`
  ("Only numeric types ... support arithmetic operations").
- Add `decimal`/`Half`/`nint`/`nuint`/`Int128`/`UInt128` dispatch branches to
  all six `*TensorPrimitive` helpers (scalar `multiply` + scalar `divide`,
  column `multiply`/`add`/`subtract`/`divide`), mirroring `NivaraSeries`:
  ```csharp
  if (type == typeof(decimal)) { NumericTensorKernels<decimal>.Add(reinterpretReadOnly<decimal>(x), reinterpretReadOnly<decimal>(y), reinterpretWritable<decimal>(destination)); return; }
  ```
  (per-helper op name varies).
- Update trailing `NotSupportedException` message to "typed kernel dispatch"
  for consistency with `NivaraSeries` (still reached by `bool`).

Blast radius: `NivaraColumn<T>` arithmetic is used by `NivaraSeries`,
`NivaraFrame` column ops, query evaluators, and AutoDiff interop
(`NivaraAutoGradExtensions`). The change only ADDS support for types that
previously threw, so existing vectorizable-type behavior is byte-for-byte
unchanged. `determineKernelType()` still reports `KernelType.Scalar` for the six
types — diagnostics stay accurate.

### 3. Tests — `tests/Nivara.Tests/NivaraColumnTests.cs`
New region "Extended numeric domain arithmetic" (`[Test]`, no `[TestCase]`):
- Values: for each of `Half`/`decimal`/`nint`/`nuint`/`Int128`/`UInt128`, verify
  scalar `Multiply`/`Divide` and column `Add`/`Subtract`/`Multiply`/`Divide`
  against plain-arithmetic expected values.
- Null propagation: `CreateFromNullable` columns → arithmetic preserves null
  positions (mirrors `NullMaskMaintenance_ArithmeticOperations_PreservesNullPositions`).
- Existing `ArithmeticOperations_OnNonVectorizableTypes_ShouldThrowWithClearErrors`
  (string/Guid/DateTime) stays green unchanged.

### 4. Docs
- `docs/plan/POLARS-ROADMAP.md` lines ~67, ~74, ~76: drop the #157
  "not yet collapsed" caveat.
- `CHANGELOG.md`: add entry.

## Planned commits

1. `docs: plan #157 column arithmetic generic-math collapse in TODO.md`
2. `feat: extend IsNumericType with Half/nint/nuint/Int128/UInt128 and relax NivaraColumn arithmetic validation`
3. `feat: dispatch extended numeric types through NumericTensorKernels in NivaraColumn arithmetic`
4. `test: cover extended-domain NivaraColumn arithmetic and null propagation`
5. `docs: mark #157 delivered in POLARS-ROADMAP and CHANGELOG`
6. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests` (ask human before running)

## GitHub issues log

- No new issues expected. If deferred work surfaces during execution, create the
  issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record
  its number here — don't rely on memory.
