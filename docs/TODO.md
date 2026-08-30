# TODO — Issue #343: remove per-element boxing in `StreamingWindowProcessor.AddConstant`

Branch: `khurram/343`. Plan for implementing GitHub issue #343.

## Problem

`StreamingWindowProcessor.AddConstant(IColumn column, long constant)`
(`src/Nivara/Execution/StreamingWindowProcessor.cs:410-417`) builds `new object?[column.Length]`
and reads each element via a boxed `column.GetValue(i)`. It is the last remaining per-element
box on a vectorizable path, reached from `ComputeCarryColumn` when a lookahead (Lead /
negative Shift) window carries a running **cumulative-count** state across chunks
(`slot.IsCount && slot.HasState`).

- `column` there is always the result of `CalculateCumulativeCount` → `NivaraColumn<long>`
  (`Tensors/WindowFunctions.cs:CumulativeCount` returns `NivaraColumn<long>`); `constant` is
  the carried `long` count.
- Result keeps the source element type (`ColumnFactory.Create(column.ElementType, values)`) —
  `long` in practice.

Acceptance (from the issue):
1. No `object?[]` allocation in `AddConstant` for numeric/vectorizable columns.
2. Lookahead (Lead / negative Shift) results identical before/after.

## Proposed changes

### 1. Core fix — `src/Nivara/Execution/StreamingWindowProcessor.cs`

Replace the boxed `AddConstant` with a type-dispatched version mirroring
`CalculateCumulativeCount` / `QuantileKernel.TypedQuantile<T>` (typed `IColumn<T>` indexer +
null mask), keeping a boxed fallback for non-`NivaraColumn<T>` columns:

```csharp
static IColumn AddConstant(IColumn column, long constant)
{
    return column switch
    {
        NivaraColumn<int> c    => addConstant(c, constant),
        NivaraColumn<long> c   => addConstant(c, constant),
        NivaraColumn<float> c  => addConstant(c, constant),
        NivaraColumn<double> c => addConstant(c, constant),
        // ... decimal, byte, sbyte, short, ushort, uint, ulong, char, nint, nuint,
        // Int128, UInt128, Half, BFloat16
        _ => addConstantBoxed(column, constant)
    };
}

static IColumn addConstant<T>(NivaraColumn<T> column, long constant)
    where T : struct, INumber<T>
    => NumericTensorKernels<T>.Add(column.Storage.Data, T.CreateChecked(constant))  // SIMD when no nulls
        // else typed indexer + mask -> NivaraColumn<T>.CreateFromOwnedArray(s)
}
```

Grounded in Microsoft Learn: generic math (`INumber<T>` gives `+` via `IAdditionOperators`,
`T.CreateChecked` is the checked conversion entry point) and `TensorPrimitives.Add` for the
SIMD no-null path.

### 2. Regression tests — `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

Add streaming-vs-lazy equivalence tests that combine **Lead + CumulativeCount** (the exact
targeted path), including a nullable-source variant, mirroring
`Property_StreamingVsLazy_LeadWithCumulativeSum_MatchesLazy` /
`Property_StreamingVsLazy_Lead_NullableSource_MatchesWithMasks`:
- `Property_StreamingVsLazy_LeadWithCumulativeCount_MatchesLazy` (multiple memory budgets).
- `Property_StreamingVsLazy_LeadWithCumulativeCount_NullableSource_MatchesWithMasks`.

## Blast radius

- **Changed:** `StreamingWindowProcessor.AddConstant` (private static, single caller at
  line 395 within the same file). Affects streaming execution of cumulative-count windows
  across chunk boundaries (with or without lead).
- **Downstream callers:** `ComputeCarryColumn` → carry path; exercised by
  `StreamingExecutionStrategy` for `SelectOperation` window expressions / `CumulativeOperation`.
- **Tests covering:** `StreamingExecutionStrategyTests` (delayed-emission + cumulative kinds
  sections), `Property_StreamingVsLazy_AllCumulativeKinds_OnDoubleSource_MatchesLazy`.
- **Not changed:** `PendingValues` queue / `buildDeferred*` boxing → tracked separately as
  issue #356 (delayed emission can still box; out of #343 scope).

## Planned commits

1. `docs: plan #343 AddConstant per-element boxing removal in TODO.md`
2. `fix: remove per-element boxing in StreamingWindowProcessor.AddConstant` (core change,
   build-verified)
3. `test: add streaming lead + cumulative-count regression tests for #343`

## Verification

- `dotnet build Nivara.slnx` after each change.
- Ask before running `dotnet test Nivara.slnx`; target `StreamingExecutionStrategyTests`
  first.
- Confirm acceptance: no `object?[` in the typed path; lookahead results equal lazy.

## GitHub issues log

- [x] #356 — StreamingWindowProcessor delayed-emission path still boxes pending cumulative values (created while planning #343)