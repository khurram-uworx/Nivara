# Plan: Predicate pushdown edge-case fix + window allocation reduction

## Problem

1. **Predicate pushdown crashes when all row groups are skipped** — `ParquetLazySource.Execute()` returns 0 columns when every row group is eliminated by pushdown, causing `LazyExecutionStrategy` to throw `QueryExecutionException("Query execution resulted in no columns")`. Never caught because no integration test exercised the "all row groups skipped" path.

2. **Window function allocation regression** — `RollingSum_NullFreeFastPath_AllocationBound` was failing at 17.5MB vs 16MB threshold. Root cause: `CreateFromSpans(result, resultMask)` copies the caller-owned arrays via `Create(span)` → `ColumnStorageFactory.Create(span)`, adding a redundant 4MB copy for 1M-element columns.

## Changes

### Fix 1: Predicate pushdown empty-result (ParquetDataSource.cs)
- `ToColumnsFromSchema(Schema)` — new helper that creates zero-length `NivaraColumn<T>` instances matching the schema when all row groups are skipped.
- `CreateZeroLengthColumn(Type)` — factory dispatching on column type.
- `Execute()` / `ExecuteAsync()` — when `frames.Count == 0`, return `ToColumnsFromSchema(schema)` instead of empty dictionary.

### Fix 2: Owned-array fast path (NivaraColumn.cs + WindowFunctions.cs + RankKernel.cs)
- `NivaraColumn<T>.CreateFromOwnedArrays(T[] values, bool[] nullMask)` — internal method that takes ownership of caller arrays without copying. Scans mask for nulls, delegates to `ColumnStorage<T>(T[], ReadOnlyMemory<bool>?)`.
- All 13 `CreateFromSpans(result, resultMask)` calls in `WindowFunctions.cs` replaced with `CreateFromOwnedArrays`.
- `RankKernel.cs` return paths updated similarly.

### Test: Predicate pushdown all-row-groups-skipped (RowGroupFilterEvaluatorIntegrationTests.cs)
- `ApplyFilterPredicate_AllRowGroupsSkipped_ReturnsEmptyColumnsWithSchema` — unit-level: writes Parquet, pushes filter that skips all row groups, asserts `Execute()` returns 1 column with length 0.
- `ApplyFilterPredicate_AllRowGroupsSkipped_EndToEndQueryDoesNotThrow` — integration-level: full `ScanAsQueryFrame` → `Filter` → `Collect` pipeline, asserts 0 rows + 2 columns preserved.

## Blast radius

- `ParquetDataSource.cs` — only affects lazy source execution when pushdown eliminates all row groups (previously crashed, now returns empty frame).
- `NivaraColumn<T>.CreateFromOwnedArrays` — internal, only called from `WindowFunctions.cs` and `RankKernel.cs`.
- `WindowFunctions.cs` — all window function return paths; no behavioral change, only allocation reduction.

## Verification

- `dotnet test` — full suite (3154 tests)
- Allocation regression test: `RollingSum_NullFreeFastPath_AllocationBound` — was 17.5MB, now 12.8MB
- New integration tests verify the fix

## Commits

1. `fix: return empty schema-preserving frame when predicate pushdown skips all row groups`
2. `perf: eliminate redundant array copy in window function and rank kernel return paths`
3. `test: cover predicate pushdown all-row-groups-skipped end-to-end path`
