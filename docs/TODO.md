# Plan: Fix unsafe `(IColumn<T>)column` casts in row-materialization (issue #344)

## Problem

`NivaraColumn<T?>` (a nullable-**element** column) implements `IColumn<T?>`, **not** `IColumn<T>`.
Direct `(IColumn<T>)column` casts therefore fail when the underlying type `T` is requested but the
column is a nullable-element column. This is distinct from the mask-based `NivaraColumn<T>` produced
by `NivaraColumn.CreateFromNullable<T>(T?[])`, which *does* implement `IColumn<T>` and works today.

The aggregation family already fixed this exact pattern (see
`AggregationFunction.ExtractValidTyped<T>`, `QuantileKernel.TypedQuantile<T>`): when the column's
element type is the nullable variant of `T`, read through `IColumn<T?>` using `HasValue` /
`GetValueOrDefault()`.

Row materialization still has the unsafe pattern:
- `src/Nivara/NivaraRow.cs:70` — `GetValue<T>` cast
- `src/Nivara/NivaraRow.cs:88` — `TryGetValue<T>` cast
- `src/Nivara.Extensions/Streamix/NivaraFlux.cs:196` — `ReadColumnFast<T>` cast
- `src/Nivara.Extensions/Streamix/NivaraFlux.cs:213` — `ReadColumnFastRef<T>` cast

## Current behavior (verified)

- `NivaraColumn.CreateFromNullable<int>(int?[])` → mask-based `NivaraColumn<int>`, `ElementType == int`,
  implements `IColumn<int>` → `GetValue<int>` works (what existing `NivaraRowTests` exercise).
- `NivaraColumn<int?>.Create(int?[])` → `ElementType == int?`, implements `IColumn<int?>`, nulls stored
  inside values (`NivaraColumn.IsNull`, `NivaraColumn.cs:1331`).
- `NivaraRow.GetValue<int>` on a `NivaraColumn<int?>` currently throws `ColumnTypeMismatchException`
  (`ElementType (int?) != typeof(int)`). `GetValue<int?>` on the same column already works.
- `NivaraFlux.ReadColumnFast<T>`/`ReadColumnFastRef<T>` are reached only for non-nullable primitives /
  reference types from `RowsToFrame`'s `ElementType` switch. A `NivaraColumn<int?>` routes to
  `ReadColumnBoxed` → `ColumnFactory.Create` → a mask-based `NivaraColumn<int>` (see
  `ColumnFactory.cs:40-58`). So the NivaraFlux typed casts are **latent/defensive** risk, not active
  throws today.

## Changes

1. **`NivaraRow.GetValue<T>`** — before the mismatch throw, when
   `Nullable.GetUnderlyingType(column.ElementType) == typeof(T) && column is not IColumn<T>` (i.e. the
   column is the nullable-element variant of the requested underlying type), read through
   `IColumn<T?>` via a small cached `MakeGenericMethod` helper and return `GetValueOrDefault()`.
   Reference-type requesters keep the plain `IColumn<T>` cast.

2. **`NivaraRow.TryGetValue<T>`** — mirror the same guard: treat the nullable-element match as a hit,
   return `true` with the unwrapped underlying value; mismatch still returns `false`.

3. **`NivaraFlux.ReadColumnFast<T>`** (`where T : struct`) — add a defensive `IColumn<T?>` branch at the
   read site so a nullable-element column is never mis-cast if routed here. Behavior-preserving; no
   `RowsToFrame` dispatch change.

4. **`NivaraFlux.ReadColumnFastRef<T>`** (`where T : class`) — leave unchanged: nullable-element applies
   only to value types, and reference columns implement `IColumn<T>`. Note in the review that it is
   safe by construction.

### Generic-constraint rationale (NivaraRow)

`GetValue<T>`/`TryGetValue<T>` are unconstrained (must keep `GetValue<string>` working). With an
unconstrained `T`, `T?` collapses to `T` — so `IColumn<T?>` == `IColumn<T>` and `typeof(T?)` ==
`typeof(T)`. Therefore the nullable-element distinction must come from `column.ElementType`
(`Nullable.GetUnderlyingType(column.ElementType) == typeof(T)`), and reading the concrete
`IColumn<int?>` requires constructing `Nullable<int>` as a type argument — only possible through a
`where TValue : struct` helper invoked via `MakeGenericMethod(typeof(T))`. Reusing the codebase's
established `ColumnFactory` pattern (cached `MakeGenericMethod` + `ConcurrentDictionary`) keeps this
consistent and fast. It is a rare cold path; the common `IColumn<T>` cast remains reflection-free.

## Blast radius

- `NivaraRow.GetValue<T>` / `TryGetValue<T>`: public API on `NivaraRow`. Behavior for existing
  mask-based `NivaraColumn<T>` and reference columns unchanged; only the previously-throwing
  nullable-element case is newly supported. Existing tests (`NivaraRowTests`) must stay green.
- `NivaraFlux.ReadColumnFast<T>`: internal helper; the added branch is unreachable for current
  `RowsToFrame` dispatch (defensive), so no behavior change.
- Tests: `NivaraRowTests.cs` (new cases) and a NivaraFlux round-trip test.

## Verification

- `dotnet build Nivara.slnx` (this repo uses `.slnx`).
- NUnit: extended `NivaraRowTests` + NivaraFlux round-trip test. Ask before `dotnet test`
  (AGENTS.md).

## Planned commits

1. `docs: plan fix for issue #344 (nullable-element row casts) in TODO.md`
2. `fix NivaraRow nullable-element GetValue<T>/TryGetValue<T> casts` (+ tests)
3. `harden NivaraFlux ReadColumnFast<T> against nullable-element columns` (+ round-trip test)

## GitHub issues log

- (none yet — this plan directly resolves #344)
