# Plan: Issue #346 — unify nullable column representation

## Problem
`NivaraColumn<T>` accepts `T = Nullable<U>` and stores it as a value-nullable column (`int?[]`, no mask), whereas `CreateFromNullable` stores mask-based `NivaraColumn<U>`. Dual representation causes latent bugs. Unify so nullable inputs funnel through mask-extracting storage.

## Proposed changes
- `ColumnStorageFactory.Create<T>(ReadOnlySpan<T>)`: detect nulls when T is Nullable<U>.
- `ColumnStorage<T>` constructor: detect nulls for nullable value types.
- `NivaraColumn<T>.createStorage`: route nullable T through null-detecting branch.
- `NivaraColumn<T>.IsNull`: mask-only.
- `NivaraColumn<T>.ElementType`: return underlying type for nullable T.
- Update guards in AggregationFunction/QuantileKernel/NivaraRow to use `column is IColumn<T?>`.
- Update tests asserting `typeof(int?)` to `typeof(int)`.

## Verification
- dotnet build Nivara.slnx
- targeted tests for changed area

## GitHub issues log
- (none yet)