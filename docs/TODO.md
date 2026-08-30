# Plan: issue #347 — Reflective `GetValue<T>` nullable-element unwrap in hot loops

Branch: `khurram/347` (off `main`).

## Problem

`NivaraRow.GetValue<T>` / `TryGetValue<T>` read nullable-element columns (`NivaraColumn<T?>`,
which implements `IColumn<T?>` rather than `IColumn<T>`) through a cached
`MakeGenericMethod` + `MethodInfo.Invoke` helper (`NullableColumnReader`, `NivaraRow.cs:151-170`).

Per call in that path (e.g. inside a `Where()` predicate over a large nullable-element column):

```csharp
var reader = cache.GetOrAdd(typeof(T), static t => readCore.MakeGenericMethod(t));
return (T)reader.Invoke(null, new object[] { column, rowIndex })!;
```

allocates and boxes on **every element**:

1. an `object[] { column, rowIndex }` argument array,
2. a boxed `int` for `rowIndex`,
3. a boxed `T` for the `TValue` return value (unboxed to `T`).

**Measured baseline (scratch probe, Release):** the `Invoke` path allocates **88 B per read**;
a cached closed generic delegate (`Delegate.CreateDelegate`) reads identically at
**0.0000 B per read**. MS Learn guidance agrees: *"Delegates perform better than late-bound
calls"* and `MethodInfo.Invoke` always boxes value-type arguments/results.

## Key finding

The issue's literal suggestion — a public `GetValue` overload constrained to
`where T : struct` — is **not compilable**: generic constraints are not part of a method
signature, so a type cannot host both `GetValue<T>(string)` and the constraint-only variant
(verified: `error CS0111`). An extension-method variant would also not help: instance methods
always win overload resolution, so existing `row.GetValue<int>(...)` callsites would still bind
to the unconstrained method. Fix must therefore remove the boxing/reflection **inside** the
existing unconstrained methods (keeping the public API identical).

## Changes

### 1. `src/Nivara/NivaraRow.cs` — `GetValue<T>` / `TryGetValue<T>` / `NullableColumnReader`

- Reorder dispatch to probe `column is IColumn<T>` **first** (direct, allocation-free path for
  ordinary columns; also removes the per-row `Nullable.GetUnderlyingType` type inspection from
  the common non-nullable hot path).
- Nullable-element underlying-type reads go through `NullableColumnReader.Read<T>` gated on
  `typeof(T).IsValueType && Nullable.GetUnderlyingType(column.ElementType) == typeof(T)`.
- `NullableColumnReader` caches a closed generic **delegate** (`Func<IColumn, int, T>`, built
  once per `T` via `MakeGenericMethod` + `CreateDelegate`) instead of a `MethodInfo`, and
  invokes it directly: zero boxing / zero allocation after first use per `T`. `ReadCore<TValue>`
  (`where TValue : struct`) body is unchanged (`GetValueOrDefault()` semantics preserved).

Behavior preservation (verified branch-by-branch against existing `NivaraRowTests`): direct
`IColumn<T>` reads, `GetValue<int?>` on `NivaraColumn<int?>`, nullable-element reads as
underlying `T`, mismatch throws, `TryGetValue` false paths — all identical.

### 2. `tests/Nivara.Tests/NivaraRowTests.cs` — allocation regression guard

Add a guard in the `Tensors/WindowAllocationTests.cs` style (`MeasureOnce`/`MeasureBestOf`):
`Where` over a nullable-element `NivaraColumn<int?>` vs a mask-based `NivaraColumn<int>` (same
frame shape, ~10 000 rows) must allocate comparably (`allocNullable <= allocOrdinary + margin`).
Old path would allocate ~88 B × rows ≈ 880 KB → guard fails only on a real regression.

The existing 20 `NivaraRowTests` must keep passing unchanged.

### 3. (Optional) `tests/Nivara.PerformanceTests/Program.cs` — scenario

Add a `Run("Row.Where NullableElement GetValue", …)` scenario so the win shows in `--compare`
bytes/op output. Optional; correctness + allocation-guard tests are primary.

## Blast radius

- **`NivaraRow`** (readonly struct, public): fields/ctor unchanged; only private dispatch inside
  two public generic methods + one private nested static class. Public API surface: identical.
- **Downstream callers of `GetValue<T>`/`TryGetValue<T>`**: `NivaraFrameExtensions.Where`
  (predicate callbacks), `NivaraFlux.RowsToFrame` (row materialization), user predicates/tests.
  All use the same method signatures; no caller changes needed.
- **Tests covering the change**: `tests/Nivara.Tests/NivaraRowTests.cs` (20 tests, all must keep
  passing), plus the new allocation guard.
- **Out of scope**: `NivaraRow.this[string]` (boxed `object?` API, by design);
  `NivaraFlux.ReadColumnBoxed`/`ReadColumnFast*` (nullable routing already handled in #344);
  other cached `MakeGenericMethod`+`Invoke` sites (`ColumnFactory`, `GroupKeyReaders`,
  `JoinOperation`, `NumericKernelDispatcher`, `FusedExpressionEvaluator`) — invoked once per
  operation, not per element, so not hot-loop bottlenecks.

## Verification

1. `dotnet build Nivara.slnx`
2. `dotnet test` on `tests/Nivara.Tests` (ask before running — AGENTS.md)
3. Allocation guard test confirms 0 B/row steady-state nullable-element reads.

## Planned commits

1. `docs: plan issue #347 — nullable GetValue reflection fix in TODO.md`
2. `feat: allocation-free nullable-element GetValue/TryGetValue reads (issue #347)` —
   `src/Nivara/NivaraRow.cs`
3. `test: guard nullable-element Where() allocation parity (issue #347)` —
   `tests/Nivara.Tests/NivaraRowTests.cs`
4. `docs: remove TODO.md — issue #347 plan executed` (then offer push + PR)

## GitHub issues log

- [ ] #347 — Reflective `GetValue<T>` nullable-element unwrap can bottleneck hot loops
  (this plan)