# Execution Engine — Phase 2: Implement Real Parallel Execution

## Purpose

Wire `ParallelExecutionHelper` utilities into `ParallelExecutionStrategy` so operations actually execute in parallel, replace the sync-over-async deadlock pattern with a proper sync path, and add chunked source execution.

## Context

### Current state (after Phase 1)

- All 4 strategies extend `ExecutionStrategyBase` with shared boilerplate.
- `IQuerySource.ExecuteAsync(CancellationToken)` exists with default `Task.Run` fallback.
- `ParallelExecutionStrategy.ExecuteCore` calls `ExecuteCoreAsync(...).GetAwaiter().GetResult()` — sync-over-async anti-pattern that can deadlock.
- All 5 `Execute*Parallel` methods are no-ops that just call `Task.Run(() => operation.Execute(input))` — no actual parallelism.
- `ParallelExecutionHelper` already has fully implemented helpers (`ProcessInParallelAsync`, `ParallelAggregateAsync`, `ProcessColumnsInParallelAsync`, `ShouldUseParallelProcessing`, `CalculateOptimalChunkSize`, `GetRecommendedParallelism`, `ValidateParallelConfiguration`) — unit-tested elsewhere.
- `NivaraColumn<T>.Slice(int start, int length)` and `NivaraFrame.Slice(int start, int length)` exist for row-range extraction.
- 110 characterization tests pass across all strategies.

### Findings relevant to Phase 2

| # | Finding | Impact |
|---|---------|--------|
| F1 | `ConcatenationOperation.OperationType` returns `$"Concatenate{direction}"` (e.g., "ConcatenateVertical"), but `IsParallelizable` checks for `"Concatenation"` — never matches | Will prevent parallel dispatch for concatenation; fix during Phase 2 |
| F2 | `IColumn` has no `Slice` method; only `NivaraColumn<T>` has one | Row-chunking requires casting to `NivaraColumn<T>` or working via `NivaraFrame.Slice` |
| F3 | No general "extract rows by index array" method on `IColumn` | Row-subset operations need reflection-based dispatch or `NivaraFrame` round-trip |
| F4 | `SortOperation` is inherently global — requires all rows to compute correct sort order | True parallel sort requires sort-merge (complex); consider parallel column reorder as practical alternative |

## Suggested Execution Order

1. **Task 1**: Fix sync/async deadlock — separate sync and async execution paths
2. **Task 2**: Implement real parallel Filter execution via row-chunked `ProcessInParallelAsync`
3. **Task 3**: Implement parallel Concatenation via `ProcessColumnsInParallelAsync`
4. **Task 4**: Implement parallel GroupBy/Join via parallel hash map building + merge
5. **Task 5**: Add chunked source execution for parallel source reads
6. **Task 6**: Expand characterization tests for real parallel behavior

## Coordination Notes

- **Task 1 is the decision gate**: The shared-kernel architecture (per-chunk `Func` consumed by both sync `Parallel.ForEach` and async `ProcessInParallelAsync`) must be implemented and stable before Tasks 2–5 begin. All 110 existing tests must pass.
- **Tasks 2–5 can run sequentially** since they all modify `ParallelExecutionStrategy.cs`.
- **Task 6 can start in parallel with Task 2** since tests are in a separate file.
- **File conflicts**: `ParallelExecutionStrategy.cs` and `ParallelExecutionHelper.cs` are touched by all tasks. Use a single branch/agent for Phase 2.

---

## Task 1: Separate Sync/Async Execution Paths

### Priority

High

### Goal

Eliminate the `GetAwaiter().GetResult()` sync-over-async anti-pattern in `ParallelExecutionStrategy.ExecuteCore` by providing a truly synchronous parallel execution path.

### Why this exists

Current code:
```csharp
protected override NivaraFrame ExecuteCore(QueryPlan plan, NivaraExecutionContext context)
    => ExecuteCoreAsync(plan, context).GetAwaiter().GetResult();
```

This wraps the entire async pipeline (`Task.Run` + async `ExecuteOperationParallelAsync`) in a blocking call. In environments with `SynchronizationContext` (ASP.NET, WPF, WinForms), this can cause deadlocks. It also means the "sync" path pays the allocation cost of async state machines with no benefit.

### Decision required

**Decision: Shared-kernel architecture** — a single per-chunk `Func` is the shared computation kernel, consumed by both sync (`Parallel.ForEach`) and async (`ProcessInParallelAsync`) dispatch. This avoids duplicating the per-chunk logic across two paths.

### Scope

- Extract a **shared per-chunk kernel** for each operation type — a `Func<int chunkStart, int chunkLength, IReadOnlyDictionary<string, IColumn> sourceColumns, IReadOnlyDictionary<string, IColumn>>` that:
  - Takes a row range (start + length) and the full input columns
  - Creates a row-subset dictionary for that range
  - Calls `operation.Execute(subset)` with the current `IQueryOperation`
  - Returns the operation's result for that chunk
- Implement **sync dispatch** in `ExecuteCore`:
  - Outer loop over operations (sequential, same as today)
  - For parallelizable operations: chunk rows, run the shared kernel per chunk via `Parallel.ForEach` + `ConcurrentBag<IReadOnlyDictionary<string, IColumn>>`, then combine
  - For non-parallelizable operations: call `operation.Execute(input)` directly
  - No async state machines, no `GetAwaiter().GetResult()`
- Implement **async dispatch** in `ExecuteCoreAsync`:
  - Outer loop over operations (sequential, same as today)
  - For parallelizable operations: chunk rows, run the shared kernel per chunk via `ProcessInParallelAsync`, combine
  - For non-parallelizable operations: call `operation.Execute(input)` via `Task.Run`
- Both paths call the **same per-chunk `Func`** — only the dispatch mechanism differs
- Remove `GetAwaiter().GetResult()` entirely

### Constraints

- `ExecuteCore` (sync) and `ExecuteCoreAsync` must produce identical results for the same input
- Must not change the `IExecutionStrategy.Execute` / `ExecuteAsync` public contract
- Null checks, cancellation, and exception wrapping are handled by `ExecutionStrategyBase` — keep them there

### Acceptance criteria

- `ExecuteCore` no longer calls `GetAwaiter().GetResult()` or any async method
- `ExecuteCore` produces the same result as `ExecuteCoreAsync` for identical input
- All 110 existing characterization tests pass
- No deadlock in synchronization-context-present environments

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`

---

## Task 2: Implement Real Parallel Filter via Row-Chunking

### Priority

High

### Goal

Replace the no-op `ExecuteFilterParallel` with a real implementation that splits input rows into chunks, filters each chunk in parallel, and concatenates the results.

### Why this exists

`ExecuteFilterParallel` currently just calls `Task.Run(() => operation.Execute(input))` — no actual parallelism. Filter is the ideal candidate for row-chunked parallelism because:
- Each row's inclusion is independent of every other row
- The filter conditions are pure functions of column values at each row
- Row-chunked parallelism maps directly to `ProcessInParallelAsync`

### Scope

- Implement the **shared per-chunk kernel** for Filter (consumed by both sync and async dispatch from Task 1):
  - Signature: `Func<int, int, IReadOnlyDictionary<string, IColumn>, IReadOnlyDictionary<string, IColumn>>`
  - Takes `chunkStart`, `chunkLength`, and full input columns
  - Creates row-subset columns for `[chunkStart, chunkStart + chunkLength)`:
    - Cast each `IColumn` to `NivaraColumn<T>` (via `ElementType` dispatch) and call `.Slice(chunkStart, chunkLength)`
    - Or convert to `NivaraFrame`, slice, and convert back
  - Calls `operation.Execute(subset)` on the sliced columns
  - Returns the filtered result for that chunk
- Implement **combine logic** for partial filter results:
  - Vertical concatenation of columns (row-stack) using the same logic as `ConcatenationOperation.ExecuteVerticalConcatenation`
- Wire the kernel + combine into:
  - **Sync dispatch** (Task 1): `Parallel.ForEach` over chunk ranges, `ConcurrentBag` to collect, then combine
  - **Async dispatch** (Task 1): `ProcessInParallelAsync` over chunk ranges, then combine
- Add null propagation: ensure null masks from original columns are preserved in each chunk

### Constraints

- Must handle columns of different types (int, string, double, etc.) correctly
- Must preserve null positions in filtered output
- Small datasets (<1000 rows) should fall through to sequential execution (already handled by `ShouldUseParallelProcessing`)
- Must work correctly when filter eliminates all rows (empty result)

### Suggested implementation path

1. Implement `CreateRowSubset(IReadOnlyDictionary<string, IColumn> source, int start, int length)` helper in `ParallelExecutionHelper`
2. Implement `ConcatenateColumnDictionaries(IReadOnlyList<IReadOnlyDictionary<string, IColumn>> chunks)` helper in `ParallelExecutionHelper`
3. Implement the per-chunk Filter kernel using `CreateRowSubset` + `operation.Execute`
4. In `ExecuteFilterParallel`: get `rowCount`, check `ShouldUseParallelProcessing`, calculate chunk size, create chunk ranges, dispatch via `Parallel.ForEach` (sync) and `ProcessInParallelAsync` (async), combine with `ConcatenateColumnDictionaries`
5. Refactor: extract chunk-ranges creation and combine as shared helpers

### Acceptance criteria

- The per-chunk kernel produces correct filtered results for any row range
- Results from parallel execution are identical to sequential `operation.Execute(input)` for any input
- All 110 existing characterization tests pass
- Null positions are correctly propagated in filtered output

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/ParallelExecutionHelper.cs` (may need a `CreateRowSubset` helper or column concatenation helper)
- `src/Nivara/Operations/ConcatenationOperation.cs` (may reuse `ConcatenateColumns` logic)

---

## Task 3: Implement Parallel Concatenation via ProcessColumnsInParallelAsync

### Priority

High

### Goal

Replace the no-op `ExecuteConcatenationParallel` with a real implementation that uses `ProcessColumnsInParallelAsync` for column-level parallelism during concatenation.

### Why this exists

`ExecuteConcatenationParallel` currently just calls `Task.Run(() => operation.Execute(input))`. Concatenation (especially vertical) involves processing each column independently — columns of different types are concatenated via separate typed code paths. This maps naturally to column-level parallelism.

### Scope

- Implement the **shared per-chunk kernel** for Concatenation:
  - For vertical concatenation: each source's columns can be independently concatenated per column type — the kernel takes a single column and appends it to the combined result per column
  - For horizontal concatenation: columns from different sources are combined in parallel per column
  - The kernel is `Func<string, IColumn, IColumn>` for column-level work (individual column concatenation)
- Fix `IsParallelizable` to recognize `ConcatenationOperation.OperationType` (returns `$"Concatenate{direction}"` like "ConcatenateVertical", not "Concatenation"):
  - Update check to `operationType.StartsWith("Concatenate", StringComparison.Ordinal)`
- Wire the kernel into:
  - **Sync dispatch**: `Parallel.ForEach` over columns or sources, `ConcurrentDictionary` to collect
  - **Async dispatch**: `ProcessColumnsInParallelAsync` over columns or sources
- Both paths use the same column-concatenation kernel

### Constraints

- Must handle both `ConcatenationDirection.Vertical` and `ConcatenationDirection.Horizontal`
- Must handle `ConcatenationMismatchHandling.Error` and `ConcatenationMismatchHandling.FillWithNulls`
- Must preserve null positions in concatenated columns

### Acceptance criteria

- `ExecuteConcatenationParallel` actually processes columns in parallel
- Results match sequential `operation.Execute(input)` for all concatenation modes
- `IsParallelizable` correctly identifies concatenation operations
- All 110 existing characterization tests pass

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Operations/ConcatenationOperation.cs` (reference for column concatenation logic)

---

## Task 4: Implement Parallel GroupBy and Join

### Priority

Medium

### Goal

Replace the no-op `ExecuteGroupByParallel` and `ExecuteJoinParallel` with implementations that parallelize the hash map building step (the most computationally intensive part of both operations).

### Why this exists

GroupBy and Join both build hash maps by iterating over every row — this is O(n) with non-trivial per-row work (key construction, hashing, dictionary insertions). On large datasets, this is the bottleneck. Both can be parallelized by:
- Processing row chunks independently
- Building per-chunk hash maps
- Merging into a single hash map

### Scope

**GroupBy**:
- Implement the **shared per-chunk kernel** for GroupBy:
  - Kernel: takes a row range `(int start, int length)` and the full input columns, creates row subsets using `CreateRowSubset`, calls `GroupByOperation.CreateGroupsInternal` on the subset, returns partial `Dictionary<GroupKey, List<int>>`
  - Combine: merge partial hash maps — for matching `GroupKey`, concatenate index lists; for new keys, add directly
  - Distinct key extraction: after merge, use `GroupByOperation.ExtractDistinctKeyValues` on the final merged groups
- Wire into:
  - **Sync dispatch**: `Parallel.ForEach` over chunk ranges, `ConcurrentBag` for partial maps, single-threaded merge
  - **Async dispatch**: `ParallelAggregateAsync` with merge-combiner

**Join**:
- Implement the **shared per-chunk kernel** for Join hash map building:
  - Kernel: takes a row range on the right-side columns, builds `Dictionary<CompositeKey, List<int>>` for that range
  - Combine: merge partial hash maps — for shared `CompositeKey`, concatenate index lists
  - Pass the merged full hash map to the existing `ComputeInnerJoinIndices` / `ComputeLeftJoinIndices` etc.
- Wire into same sync/async dispatch pattern
- Note: `JoinOperation.Execute(input)` ignores its `input` parameter (uses stored `leftColumns`/`rightColumns`) — the parallel entry point should extract right columns from the operation

### Constraints

- GroupKey equality/hashing must be maintained correctly during merge
- CompositeKey null semantics must be preserved (nulls never match, including null-to-null)
- Must handle edge cases: single-row groups, empty input, all-rows-same-key
- Fall back to sequential if `ShouldUseParallelProcessing` returns false

### Acceptance criteria

- `ExecuteGroupByParallel` produces identical groups to sequential `GroupByOperation.Execute`
- `ExecuteJoinParallel` produces identical join results to sequential `JoinOperation.Execute`
- All 110 existing characterization tests pass
- Merge logic correctly combines partial hash maps without data loss

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Operations/GroupByOperation.cs` (reference for `GroupKey`, `GroupedData`).
- `src/Nivara/Operations/JoinOperation.cs` (reference for `CompositeKey`, `JoinIndices`).
- `src/Nivara/Execution/ParallelExecutionHelper.cs` (may need a `MergeDictionaries` helper)

---

## Task 5: Add Chunked Source Execution

### Priority

Medium

### Goal

Enable parallel chunked reading from data sources that support it, using `IQuerySource.ExecuteAsync` (added in Phase 1) for non-blocking parallel execution.

### Why this exists

Currently, `ParallelExecutionStrategy.ExecuteCoreAsync` calls `plan.Source.ExecuteAsync(context.CancellationToken)` once, getting all source data before applying any operations. For large datasets from async-capable sources (network files, databases), parallel source reading could improve throughput.

### Scope

- Add a `CanReadInChunks` property to `IQuerySource` (default `false` for backward compat):
  ```csharp
  bool CanReadInChunks => false;
  ```
- Add a `ReadChunkAsync(int chunkIndex, int chunkSize, CancellationToken)` method to `IQuerySource`:
  ```csharp
  ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
      int chunkIndex, int chunkSize, CancellationToken cancellationToken = default);
  ```
  Default implementation throws `NotSupportedException`.
- In `ParallelExecutionStrategy.ExecuteCoreAsync`:
  - If source supports chunked reading (`CanReadInChunks`), determine total row count and chunk size
  - Execute multiple `ReadChunkAsync` calls in parallel via `ProcessInParallelAsync`
  - Concatenate chunked results (using Task 3's concatenation logic)
  - Apply operations on the combined data
- If source does NOT support chunked reading, fall back to current behavior (`plan.Source.ExecuteAsync()`)

### Constraints

- Must not break existing `IQuerySource` implementations (default `CanReadInChunks` is `false`)
- Chunked source reading is optional — sources that cannot chunk should work unchanged
- Total row count must be discoverable (add `int? EstimatedRowCount` to `IQuerySource` or determine from schema)

### Acceptance criteria

- Sources with `CanReadInChunks == false` continue to work unchanged
- Sources with chunking enabled are read in parallel chunks
- Results from chunked reading are equivalent to non-chunked reading
- All 110 existing characterization tests pass

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/ParallelExecutionHelper.cs`

---

## Task 6: Expand Characterization Tests for Real Parallel Behavior

### Priority

High

### Goal

Add tests that verify `ParallelExecutionStrategy` actually executes operations in parallel, produces correct results, and handles edge cases properly.

### Why this exists

The current characterization tests (13 tests for `ParallelExecutionStrategy`) verify null checks, progress reporting, validation, and error wrapping — but they don't verify that parallel execution actually happens. Without these tests, regressions in parallel behavior go undetected.

### Scope

- Add tests for `ExecuteFilterParallel`:
  - Verify rows are filtered correctly with parallel dispatch
  - Verify results match sequential execution
  - Verify null propagation in filtered output
  - Verify cancellation during parallel filter stops all chunks
- Add tests for `ExecuteConcatenationParallel`:
  - Verify vertical concatenation with parallel column processing
  - Verify horizontal concatenation
  - Verify null-fill and error mismatch modes
- Add tests for `ExecuteGroupByParallel`:
  - Verify groups match sequential grouping
  - Verify merge of partial hash maps (if testable)
- Add tests for `ExecuteJoinParallel`:
  - Verify join results match sequential join
  - Test all join types (Inner, Left, Right, FullOuter)
- Add tests for **sync path**:
  - Verify `Execute` (sync) produces same results as `ExecuteAsync`
  - Verify sync path doesn't deadlock (can run on a mock `SynchronizationContext`)
- Add tests for **chunked source execution** (Task 5):
  - Create a stub source that simulates chunked reading
  - Verify chunks are read and concatenated correctly
- Add tests for **mixed operations**:
  - Plan with Filter + Sort + GroupBy to verify pipeline coordination
- Add performance/verification tests:
  - Use `ConcurrentBag` or counters to verify work is distributed across threads
  - Verify `CalculateOptimalChunkSize` and `GetRecommendedParallelism` are called correctly

### Constraints

- Tests should not depend on timing or assume specific thread counts
- Use `StubQuerySource` and `StubQueryOperation` helpers from `ExecutionTestHelpers.cs`
- For verifying parallel execution, use a `ParallelTrackingOperation` wrapper that records thread IDs
- Do not use `Thread.Sleep` for synchronization in tests — use `ManualResetEvent` or `CountdownEvent` if needed

### Suggested implementation path

1. Add `ParallelTrackingOperation` helper to `ExecutionTestHelpers.cs` that wraps an `IQueryOperation` and records thread IDs / concurrent execution count
2. Write `ExecuteFilterParallel_ProcessesChunksInParallel` test using a filter that matches all rows (identity filter) with a large input dataset
3. Write `ExecuteFilterParallel_ResultsMatchSequential` test that compares parallel vs sequential output
4. Write `ExecuteConcatenationParallel_ProcessesColumnsInParallel` test
5. Write `ExecuteGroupByParallel_ResultsMatchSequential` test
6. Write `ExecuteJoinParallel_ResultsMatchSequential` test
7. Write `Execute_SyncPath_MatchesAsyncPath` test
8. Run all tests to confirm they pass

### Acceptance criteria

- Minimum 10 new tests added (covering all parallel methods + sync path + chunked source)
- All tests pass
- Existing 110 characterization tests still pass

### Files likely involved

- `tests/Nivara.Tests/Execution/ParallelExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs`

---

## Suggested Agent Handout Batches

### Batch A: decision-critical (Task 1 — shared-kernel architecture)

- Task 1: Shared per-chunk `Func` kernel, consumed by `Parallel.ForEach` (sync) and `ProcessInParallelAsync` (async)

This batch MUST complete and pass before any work on Batches B–D begins.

### Batch B: core parallel implementation (Tasks 2–4)

- Task 2: Parallel Filter via row-chunking
- Task 3: Parallel Concatenation via `ProcessColumnsInParallelAsync`
- Task 4: Parallel GroupBy/Join via hash map build/merge

These share files and should be implemented together.

### Batch C: source parallelism (Task 5)

- Task 5: Chunked source execution

Can run in parallel with Batch B since it mainly modifies `IQueryInterfaces.cs` and adds source-level functionality.

### Batch D: tests and verification (Task 6)

- Task 6: Expanded characterization tests

Can begin once Task 1 is complete; can overlap with Batches B–C for test stubs.

---

## Final Checklist

- [ ] Every task has a clear owner-sized scope
- [ ] Every task has acceptance criteria
- [ ] Task 1 (shared-kernel architecture) is marked as decision gate — nothing starts until it passes
- [ ] Shared per-chunk `Func` kernel pattern is used consistently across all 5 parallel operations
- [ ] Likely files are listed to reduce agent search time
- [ ] Execution order reflects real dependencies (Task 1 → Tasks 2–4 → Task 5 | Task 6)
- [ ] F1 (ConcatenationOperation type mismatch) is fixed during Task 3
- [ ] F2–F4 (IColumn Slice gap, row-subset extraction, Sort parallelism) are acknowledged with practical workarounds
