# Plan: Parquet Predicate Pushdown (#298) + Sort Optimization (#300)

## Overview

Two performance issues in the query execution pipeline:
1. **#298**: Parquet reader loads all row groups before any filter — no predicate pushdown
2. **#300**: Sort uses LINQ OrderBy with per-comparison dictionary lookup and interface dispatch

## Branch

`khurram/issues` off `main`

## Blast Radius

| Change | Files affected | Downstream callers | Test coverage |
|--------|---------------|-------------------|---------------|
| `IPredicatePushdownSource` interface | `IQueryInterfaces.cs` | `ParquetLazySource`, `ExecutionStrategyBase` | New tests |
| `RowGroupFilterEvaluator` | New file in Extensions | `ParquetLazySource` | New tests |
| `ParquetLazySource` pushdown | `ParquetDataSource.cs` | All strategies via `IQuerySource` | Existing + new |
| `ExecutionStrategyBase` extraction | `ExecutionStrategyBase.cs` | `Eager/Lazy/Streaming/Parallel` | Existing |
| `SingleColumnComparers` | New file in Operations | `SortOperation.ComputeSortIndices` | New tests |
| `SortOperation` fast paths | `SortOperation.cs` | All strategy sort dispatch | Existing + new |

## Issue #298 — Parquet Predicate Pushdown

### Problem
`ParquetLazySource.Execute()` (line 87-88) reads ALL row groups unconditionally. Parquet.Net 6.1.0 exposes `ParquetRowGroupReader.GetStatistics(DataField)` → `DataColumnStatistics` with `MinValue`/`MaxValue`, but nothing connects filter predicates to this metadata.

### Architecture: Base class extraction + `SkippedRowGroups` property

**Integration**: `ExecutionStrategyBase.Execute` extracts the leading `FilterOperation`, passes it to the source via `IPredicatePushdownSource`, removes the filter from the plan. The source stores `SkippedRowGroups` which is respected by all entry points (`Execute`, `ExecuteAsync`, `ReadChunk`, `ReadChunkAsync`).

### New files

1. **`src/Nivara/Query/IPredicatePushdownSource.cs`**
   - `CanPushdownFilter(ColumnExpression, Schema)` → bool
   - `ApplyFilterPredicate(ColumnExpression, Schema)` — computes and stores skipped row groups

2. **`src/Nivara.Extensions/IO/RowGroupFilterEvaluator.cs`**
   - Static class with evaluation logic
   - `CanEvaluate(ColumnExpression, Schema)` — true for simple comparisons + AND chains
   - `EvaluateRowGroup(ColumnExpression, Func<DataField, DataColumnStatistics?>, Schema)` → bool
   - `CollectPushdownColumns(ColumnExpression)` → HashSet<string>
   - Three-way AND: walk `BinaryExpression(And, left, right)`, evaluate each leaf
   - OR/Not/arithmetic → conservative (no pruning)

3. **`tests/Nivara.Tests/IO/ParquetPredicatePushdownTests.cs`**

### Modified files

4. **`src/Nivara/Query/IQueryInterfaces.cs`** — add `IPredicatePushdownSource`
5. **`src/Nivara/Execution/ExecutionStrategyBase.cs`** — `TryExtractPredicatePushdown` method
6. **`src/Nivara.Extensions/IO/ParquetDataSource.cs`** — implement `IPredicatePushdownSource`, add `skippedRowGroups` field, modify `Execute/ExecuteAsync/ReadChunk/ReadChunkAsync`

## Issue #300 — Sort Optimization

### Problem
`SortOperation.ComputeSortIndices` (line 215-236) uses `indices.OrderBy(i => i, comparer).ToArray()` with `MultiColumnComparer` that does per-comparison dictionary lookup + `Comparer<T>.Default` interface dispatch.

### New files

1. **`src/Nivara/Operations/SingleColumnComparers.cs`**
   - `SingleColumnSpanComparer<T>` struct — for no-nulls fast path, captures `ReadOnlySpan<T>`
   - `SingleColumnComparer<T>` struct — for nullable path, captures `NivaraColumn<T>`
   - `PreCapturedMultiColumnComparer` — captures all column references once, uses `Func<int,int,int>[]` delegates
   - All comparers include `x.CompareTo(y)` tiebreaker for stable sort via `Array.Sort`

### Modified files

2. **`src/Nivara/Operations/SortOperation.cs`** — add fast paths in `ComputeSortIndices`
3. **`tests/Nivara.Tests/Operations/SortOperationTests.cs`** — add fast path tests

## Execution Order

### Step 1: Sort optimization (#300) — self-contained
1. Create `src/Nivara/Operations/SingleColumnComparers.cs`
2. Modify `SortOperation.ComputeSortIndices` with fast paths
3. Add tests
4. Build verify

### Step 2: Predicate pushdown (#298) — cross-project
1. Create `src/Nivara/Query/IPredicatePushdownSource.cs`
2. Create `src/Nivara.Extensions/IO/RowGroupFilterEvaluator.cs`
3. Modify `ParquetLazySource` to implement `IPredicatePushdownSource`
4. Modify `ExecutionStrategyBase` with extraction logic
5. Add tests
6. Build verify

### Step 3: Full test suite
- Ask human before running `dotnet test`

## GitHub issues log

- [ ] Deferred: investigate ConvTranspose2d grouped convolution support
- [ ] Deferred: BatchNorm2d fused kernel path (currently generic per-element)
