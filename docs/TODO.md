# Plan: Eliminate boxing hot paths (Issues #297 + #299)

## Problem Summary

Two performance-critical code paths suffer from per-element boxing/unboxing that dominates query time on large datasets:

1. **#297 — Parquet reader boxing loop**: `CreateNivaraColumn<T>(Array)` unboxes every element via `Array.GetValue(i)` + `Convert.ChangeType` even though Parquet.Net already produced typed `T[]`/`T?[]` arrays. ~60% of total query time on 1M rows.

2. **#299 — Quantile aggregation triple-boxing**: `QuantileAggregation.Apply()` boxes every value via `column.GetValue(index)` into `List<object>`, then unboxes each via a 17-arm `ToDouble` switch. ~15% of total query time.

## Changes

### Step 1: Fix #297 — Parquet reader fast paths

**File:** `src/Nivara.Extensions/IO/NivaraParquetReader.cs`

Modify `CreateNivaraColumn<T>(Array columnData)` (line 499):
- Add fast path: `if (columnData is T?[] nullableArray)` → `NivaraColumn.CreateFromNullable(nullableArray)`
- Add fast path: `if (columnData is T[] typedArray)` → `NivaraColumn<T>.Create(typedArray)`
- Keep existing fallback for widened types (Half/nint etc. where Parquet returns base type)

Modify `CreateStringColumn(Array columnData)` (line 570):
- Add fast path: `if (columnData is string[] stringArray)` → `NivaraColumn<string>.CreateForReferenceType(stringArray)`
- Keep existing fallback

### Step 2: Fix #299 — Quantile typed dispatch

**File:** `src/Nivara/Helpers/QuantileKernel.cs`

Add `ComputeFromColumn(IColumn, IReadOnlyList<int>, double)`:
- Type-switch dispatch on `column.ElementType`
- For each numeric type, extract values via typed `NivaraColumn<T>` indexer (no boxing)
- Convert to `double` inline, sort, compute quantile
- Fallback to existing `ComputeFromBoxed` for unknown types

Add private `TypedQuantile<T>(IColumn, IReadOnlyList<int>, double, Func<T, double>)`:
- Cast `IColumn` to `NivaraColumn<T>`
- Count non-null via `IsNull(index)`, extract via `typed[index]`
- Sort and compute

**File:** `src/Nivara/Operations/AggregationFunction.cs`

Update `QuantileAggregation.Apply()` (line 607):
- Replace `ExtractValidValues` + `ComputeFromBoxed` with `QuantileKernel.ComputeFromColumn(column, groupIndices, q)`

Update `MedianAggregation.Apply()` (line 663):
- Same — call `QuantileKernel.ComputeFromColumn(column, groupIndices, 0.5)`

## Blast Radius

- `NivaraParquetReader.CreateNivaraColumn<T>` — only called from `CreateNivaraColumnFromParquetData`, no external callers
- `NivaraParquetReader.CreateStringColumn` — only called from `CreateNivaraColumnFromParquetData`, no external callers
- `QuantileKernel.ComputeFromColumn` — new internal method, called only from `QuantileAggregation.Apply` and `MedianAggregation.Apply`
- `QuantileAggregation.Apply` — public API, behavior unchanged (same numerical results)
- `MedianAggregation.Apply` — public API, behavior unchanged (same numerical results)

## Test Coverage

Existing tests cover all correctness properties:
- `tests/Nivara.Tests/IO/ParquetWriterTests.cs` — round-trip for all types
- `tests/Nivara.Tests/IO/ParquetExtendedDomainRoundTripTests.cs` — extended domain
- `tests/Nivara.Tests/IO/ParquetStreamingTests.cs` — streaming read paths
- `tests/Nivara.Tests/Operations/AggregationFunctionTests.cs` — quantile/median correctness, nulls, edge cases
- `tests/Nivara.Tests/Query/PolarsQuantileCrossValidationTests.cs` — Polars cross-validation

## Planned Commits

1. `docs: plan boxing elimination in TODO.md`
2. `fix: eliminate boxing in Parquet reader column creation (#297)`
3. `fix: eliminate boxing in Quantile/Median aggregation (#299)`
4. `docs: remove TODO.md — plan executed`

## GitHub Issues Log

- [ ] #297 — Parquet reader boxing loop (pre-existing)
- [ ] #299 — Quantile aggregation triple-boxing (pre-existing)
