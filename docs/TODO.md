# TODO: Column creation dynamic dispatch — cover less common CLR types (#158)

Branch: `khurram/158` (base `main`)

## Problem

Dynamic column-creation dispatch (query operation result columns, constant
columns, join coalesce/gather, window-op results) uses fixed
`Type t when t == typeof(...)` switches that cover only a subset of the CLR
types the engine otherwise supports. The extended domain (`Half`, `nint`/
`nuint`, `Int128`/`UInt128`, `sbyte`/`ushort`/`uint`/`char`, and comparable
non-numerics `DateOnly`/`TimeOnly`/`DateTimeOffset`) falls through to
`NivaraColumn<object>` or throws `NotSupportedException`, while
`MultiColumnComparer`, `TypeCompatibilityValidator.GetNumericTypes()`,
`NumericTensorKernels<T>`, and `NivaraSeries` arithmetic already cover the full
domain. Known-issue follow-up listed in `AGENTS.md`; Phase 1/2 follow-up from
`docs/plan/POLARS-ROADMAP.md`.

Concrete gaps:

| Site | Behaviour today |
|---|---|
| `AggregationFunction.CreateColumnFromValues` (`Operations/AggregationFunction.cs:64`) | missing `Half`, `nint`, `nuint`, `sbyte`, `ushort`, `uint`, `char`, `DateOnly`, `TimeOnly`, `DateTimeOffset`, `Guid`, `TimeSpan` → `NivaraColumn<object>` |
| `GroupByOperation.CreateColumnFromValues` (`Operations/GroupByOperation.cs:397`) | same, *plus* `ulong`, `Int128`, `UInt128` → `NivaraColumn<object>` |
| `FusedExpressionEvaluator.CreateConstantColumn` (`Expressions/FusedExpressionEvaluator.cs:396`) | missing `ulong`, `uint`, `ushort`, `sbyte`, `char`, `Half`, `nint`, `nuint`, `Int128`, `UInt128`, `DateOnly`, `TimeOnly`, `DateTimeOffset` → `FillConstantObject` |
| `JoinOperation.CreateCoalescedJoinKeyColumn` (`Operations/JoinOperation.cs:690`) | falls to `NivaraColumn<object>` even though `*Typed<T>` kernels are fully generic + null-safe |
| `JoinOperation.GatherColumn` (`Operations/JoinOperation.cs:829`) | same |
| `WindowFrameExtensions` rolling/cumulative/count/shift (`WindowFrameExtensions.cs:145-191`) | `NotSupportedException` for extended numerics; `adaptNullHandler`/`convertFillValue` use `Convert.ChangeType` (no `IConvertible` for `Half`/`nint`/`nuint`/`Int128`/`UInt128`) |

Bonus latent bug: the existing `CreateColumnFromValues` uses
`values.Cast<T>().ToArray()`, which throws when any value is null (object
fallback masks it by filtering nulls). The new helper is null-safe for value
types via `CreateFromNullable`.

## Approach (choice: cached generic delegate)

One internal `ColumnFactory` (`src/Nivara/Helpers/ColumnFactory.cs`) exposing
`Create(Type elementType, object?[] values)` that unwraps `Nullable<T>`, then
routes through a `ConcurrentDictionary<Type, MethodInfo>` cached
`MakeGenericMethod` to a single null-safe kernel:

```csharp
static IColumn CreateTyped<T>(object?[] values)
{
    if (typeof(T).IsValueType)
    {
        var nullable = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            nullable[i] = values[i] is null ? null : (T)values[i];
        return NivaraColumn<T>.CreateFromNullable(nullable);
    }
    return NivaraColumn<T>.CreateForReferenceType(values.Cast<T>().ToArray());
}
```

Covers **any** element type with no per-type enumeration to maintain; matches
the existing `ColumnFilterHelper` / `FusedExpressionEvaluator.CreateResultColumn`
cached-`MakeGenericMethod` pattern. All call sites delegate to it, eliminating
the duplicated switch logic (AGENTS.md rule 8).

## Changes

### 1. `src/Nivara/Helpers/ColumnFactory.cs` (new)
Internal static class; cached generic-delegate dispatch; null-safe kernel;
`Nullable<T>` unwrap. Fallback for genuinely unknown types stays
`NivaraColumn<object>` (preserves today's behaviour for custom types).

### 2. `src/Nivara/Operations/AggregationFunction.cs`
`CreateColumnFromValues` body → `ColumnFactory.Create(elementType, values)`.
Keep the `protected static` wrapper (subclass contract).

### 3. `src/Nivara/Operations/GroupByOperation.cs`
`CreateColumnFromValues` body → `ColumnFactory.Create(elementType, values)`.
Keep the `internal static` wrapper.

### 4. `src/Nivara/Operations/JoinOperation.cs`
`CreateCoalescedJoinKeyColumn` and `GatherColumn`: replace the enumeration
switch with cached `MakeGenericMethod` dispatch onto the existing
`CreateCoalescedJoinKeyColumnTyped<T>` / `GatherColumnTyped<T>` kernels.
Unwrap `Nullable` before `MakeGenericMethod` (today's `int?` element would hit
`Nullable<Nullable<int>>` and throw). Removes the `NivaraColumn<object>`
fallback entirely.

### 5. `src/Nivara/Expressions/FusedExpressionEvaluator.cs`
`CreateConstantColumn`: replace the value-switch with
`ColumnFactory.Create(value.GetType(), filledArray)` (null and `string`
handled explicitly first). Removes the enumeration.

### 6. `src/Nivara/WindowFrameExtensions.cs`
`CalculateRolling` / `CalculateCumulative` / `CalculateCumulativeCount` /
`CalculateShift`: add arms for the remaining `INumber<T>` numerics
(`byte`, `sbyte`, `ushort`, `uint`, `ulong`, `char`, `nint`, `nuint`,
`Int128`, `UInt128`, `Half`) — window kernels are `where T : struct,
INumber<T>` and these all satisfy it (`CumulativeCount`/`Shift` have no
constraint). Replace `Convert.ChangeType` in `adaptNullHandler<T>` /
`convertFillValue<T>` with typed conversion: direct `(T)` cast for typed
boxed values, `T.TryParse` (invariant) for strings, throw otherwise. Keep the
`string`/`bool` arms where they exist.

### Out of scope
Parquet/Arrow/ML interop and CSV/JSON value conversion — format-specific type
systems; CSV/JSON inference only yields int/double/bool/DateTime/string, so
the extended types are unreachable there. Note in CHANGELOG if desired.

## Blast radius

- `ColumnFactory` — new internal type; no existing callers; `Nivara.Tests`,
  `Nivara.Extensions`, `Nivara.PerformanceTests` have `InternalsVisibleTo`.
- `AggregationFunction.CreateColumnFromValues` — callers: `ApplyToGroups`
  (aggregation result columns). Downstream: group-by aggregate columns.
- `GroupByOperation.CreateColumnFromValues` — callers: `ExtractDistinctKeyValues`
  (distinct key columns).
- `JoinOperation` — `CreateCoalescedJoinKeyColumn` (full-outer coalesced keys)
  and `GatherColumn` (left/right join value gathering). Behaviour change:
  extended-domain types now produce typed `NivaraColumn<T>` instead of
  `NivaraColumn<object>`; `MultiColumnComparer.compareBoxed` still handles any
  residual types, so no downstream break.
- `FusedExpressionEvaluator.CreateConstantColumn` — constant/literal columns
  in fused expression evaluation. Extended-domain literals now produce typed
  columns instead of object.
- `WindowFrameExtensions` — `Rolling*`/`Cumulative*`/`Shift` frame/query APIs
  and `WindowOperations`. Extended numerics no longer throw; string rolling
  still throws (existing test asserts this — stays valid).
- Tests: `AggregationFunctionTests`, `GroupByOperationTests`,
  `JoinOperationTests`, `FusedExpressionEvaluatorTests`,
  `WindowFunctionsFrameTests`, `WindowOperationTests`, new `ColumnFactoryTests`.

## Verification

1. `dotnet build Nivara.slnx` after each step.
2. `dotnet test` on the touched test classes (ask human before running).
3. Confirm no `NivaraColumn<object>` result for any extended type through
   group-by min/max, distinct, join coalesce/gather, fused constants, window ops.

## Planned commits

1. `docs: plan issue #158 in TODO.md`
2. `feat: add ColumnFactory dynamic column-creation dispatch for the extended CLR domain`
3. `refactor: route aggregation and group-by result columns through ColumnFactory`
4. `refactor: dispatch join coalesce/gather through cached generic kernels`
5. `feat: route fused constant columns through ColumnFactory`
6. `feat: extend window-operation dispatch and typed conversions to the extended numeric domain`
7. `test: cover extended-domain dynamic column creation paths`
8. `docs: remove TODO.md — #158 plan executed` (+ CHANGELOG note if kept)

## GitHub issues log

- (none yet — create issues here at discovery time via
  `gh issue create --repo khurram-uworx/Nivara` and record the number)
