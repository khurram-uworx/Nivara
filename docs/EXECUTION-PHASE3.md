# Phase 3 — Real Streaming Execution

## Purpose

Make `StreamingExecutionStrategy` actually process data in chunks using the memory budget instead of falling through to `QueryExecutor.Execute(plan)` in one shot.

## Context

[EXECUTION-PLAN.md](./EXECUTION-PLAN.md#phase-3-implement-real-streaming-execution) states:

> **Goal**: Make `StreamingExecutionStrategy` actually process data in chunks using the memory budget.

Current state (from code-memory exploration):

| Aspect | Status |
|--------|--------|
| `StreamingExecutionStrategy.ExecuteCore()` | Calls `calculateChunkSize` then discards the result; falls through to `executor.Execute(plan)` — a no-op |
| `StreamingExecutionStrategy.ExecuteCoreAsync()` | Same pattern wrapped in `Task.Run` |
| `IQuerySource` chunking API | Already has `CanReadInChunks`, `EstimatedRowCount`, `ReadChunkAsync` |
| `StubChunkedQuerySource` test helper | Already exists with 2000-row test data |
| Column concatenation | `ConcatenationOperation.ConcatenateColumns` merges same-type columns; `ConcatenateColumnsTyped<T>` handles typed dispatch |
| `NivaraFrame` concatenation | No built-in `Concat`/`Append` — must be added |
| Non-streamable ops | Sort, GroupBy, Join — fall back to `LazyExecutionStrategy` |
| Memory budget | `NivaraExecutionContext.MemoryBudget` (default 1 GB) |

## Suggested Execution Order

1. Task 1: Decision Gate — Streaming data source contract
2. Task 2: Add `NivaraFrame.Concat()` for chunk merge
3. Task 3: Implement chunk-based pipeline in `StreamingExecutionStrategy`
4. Task 4: Per-chunk operation execution
5. Task 5: Test coverage

## Coordination Notes

- **Decision gate**: Task 1 must settle the `IQuerySource` streaming API shape before Task 3 begins.
- **Parallel-safe**: Tasks 2 and 5 can start in parallel — Task 2 has no dependency on the streaming pipeline shape.
- **Shared files**: `NivaraFrame.cs` (Task 2) and `StreamingExecutionStrategy.cs` (Tasks 3–4) will both be touched; avoid merge conflicts by sequencing edits.

---

## Task 1: Decision Gate — Streaming Data Source Contract

### Priority

High

### Goal

Decide whether to extend `IQuerySource` with `IAsyncEnumerable<ReadOnlyMemory<IColumn>>` support or keep the existing `ReadChunkAsync` API.

### Why this exists

The current `IQuerySource` chunking API (`CanReadInChunks`, `ReadChunkAsync`) is synchronous in return type convention and doesn't compose naturally with `await foreach`. The EXECUTION-PLAN.md mentions connecting "streaming-aware data sources where `IQuerySource` supports `IAsyncEnumerable<ReadOnlyMemory<IColumn>>`".

### Decision required

Choose one of:

1. **Keep `ReadChunkAsync`** — Minimal API surface change; streaming strategy iterates chunk indices in a loop. Simple but requires manual index management.
2. **Add `IAsyncEnumerable<ReadOnlyMemory<IColumn>>`** — Add a `ToAsyncEnumerable(int chunkSize, CancellationToken)` default method to `IQuerySource` that wraps `ReadChunkAsync`. Enables `await foreach` consumption. More idiomatic for streaming.
3. **Add both** — `ReadChunkAsync` remains for random-access chunk reads; `ToAsyncEnumerable` is the default implementation using it. Add as a virtual/default interface method to avoid breaking existing implementations.

### Scope

- Evaluate existing `StubChunkedQuerySource` usage in tests
- Decide API shape for the streaming strategy's data consumption loop
- Document the decision

### Constraints

- Must not break existing `IQuerySource` implementations
- Should support cancellation throughout the chunk pipeline
- Must play well with `NivaraExecutionContext.MemoryBudget`

### Acceptance criteria

- Decision document committed (this doc or adjacent doc)
- At most one new method added to `IQuerySource` if option 2 or 3 chosen
- `StubChunkedQuerySource` updated to implement the new API if added

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs` (`StubChunkedQuerySource`)

---

## Task 2: Add NivaraFrame Concatenation for Chunk Merging

### Priority

High

### Goal

Add a `Concat(NivaraFrame other)` method on `NivaraFrame` that vertically concatenates two frames with the same schema.

### Why this exists

The streaming strategy produces one `NivaraFrame` per chunk. These must be merged into a single result frame. No such method exists today.

### Scope

- Add `NivaraFrame.Concat(NivaraFrame other)` — validates schema compatibility, then uses `ConcatenationOperation.ConcatenateColumns` to merge each column pair
- Add `NivaraFrame.Concat(IEnumerable<NivaraFrame> frames)` — merges N frames by accumulating per-column lists
- Use `ConcatenationOperation.ConcatenateColumns` as the underlying column merge primitive (already handles typed dispatch, null preservation)
- Handle empty frames and single-frame passthrough
- Add `ValidateSchemaCompatibility` reuse or inline schema check

### Constraints

- Must preserve null masks via column concatenation
- Must throw `ArgumentException` on schema mismatch (column name, type, or order)
- Must not allocate intermediate arrays per-chunk beyond what `ConcatenateColumns` already does

### Suggested implementation path

1. Open `NivaraFrame.cs` and locate the column dictionary (`columns` field, `IReadOnlyDictionary<string, IColumn>`)
2. For `Concat(NivaraFrame other)`: validate `other` is not null, validate schema compatibility, then for each column in `this`, collect `[thisColumn, otherColumn]` and call `ConcatenationOperation.ConcatenateColumns`
3. Build a new `NivaraFrame` from the concatenated columns
4. For the multi-frame overload, generalize to N frames by collecting all per-column lists first

### Acceptance criteria

- `NivaraFrame.Concat(other)` merges two frames with identical schema correctly
- `NivaraFrame.Concat(new[] { f1, f2, f3 })` merges N frames
- Concatenation preserves row count total (len(A) + len(B))
- Schema mismatch throws `ArgumentException`
- Single frame passthrough (concat with empty) works
- Existing tests pass

### Files likely involved

- `src/Nivara/NivaraFrame.cs`
- `src/Nivara/Operations/ConcatenationOperation.cs` (already has `ConcatenateColumns`)

---

## Task 3: Implement Chunk-Based Data Pipeline in StreamingExecutionStrategy

### Priority

High

### Goal

Make `StreamingExecutionStrategy.ExecuteCore()` actually read data source in chunks, apply operations per-chunk, and merge results.

### Why this exists

The current `ExecuteCore()` calls `calculateChunkSize` but does nothing with its result, then falls through to `executor.Execute(plan)` — exactly what `LazyExecutionStrategy` does.

### Scope

- Wire `calculateChunkSize(context.MemoryBudget)` output into the iteration loop
- Check `plan.Source.CanReadInChunks` before attempting chunk-based execution
- If source does not support chunking, fall through to `executor.Execute(plan)` with a progress report
- If source supports chunking, compute `totalChunks = ceil(EstimatedRowCount / chunkSize)`
- For each chunk index:
  - Read chunk via `plan.Source.ReadChunkAsync(chunkIndex, chunkSize, context.CancellationToken)`
  - Check cancellation token
  - Report progress as `(chunkIndex + 1) / totalChunks`
  - Accumulate chunk frames in a `List<NivaraFrame>`
- After all chunks, merge using `NivaraFrame.Concat()`
- Handle case where `EstimatedRowCount` is null — read until empty chunk returned
- In `ExecuteCoreAsync`, use the async path directly with `await foreach` if available, or sequential `ReadChunkAsync` calls

### Constraints

- Must not change public API of `IExecutionStrategy` or `ExecutionStrategyBase`
- Must respect `context.CancellationToken` between each chunk
- Must respect `context.Progress` reporting per chunk
- Fallback to `LazyExecutionStrategy` for non-streamable plans must remain

### Suggested implementation path

1. In `ExecuteCore`, after `isSuitableForStreaming` check and fallback, branch on `plan.Source.CanReadInChunks`
2. If false: call `executor.Execute(plan)` (current path)
3. If true: compute `chunkSize`, iterate chunks, read via `ReadChunkAsync`, apply operations via `executor.Execute` (or per-chunk operation pipeline — see Task 4), accumulate, merge
4. Add chunk iteration to `ExecuteCoreAsync` similarly but using async read directly

### Acceptance criteria

- `StreamingExecutionStrategy.Execute()` with a chunked source reads data in batches
- `StreamingExecutionStrategy.Execute()` with a non-chunked source falls through to the base executor (no regression)
- Results are identical to non-streamed execution for the same data
- Non-streamable operations (Sort, GroupBy, Join) still fall back to `LazyExecutionStrategy`
- Cancellation during chunk iteration stops execution and propagates
- Progress is reported per chunk

### Files likely involved

- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Query/IQueryInterfaces.cs` (if Task 1 changes the API)
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs` (`StubChunkedQuerySource`)

---

## Task 4: Per-Chunk Operation Execution

### Priority

Medium

### Goal

Apply streamable operations per-chunk individually instead of merging raw chunk data and then executing operations.

### Why this exists

The chunk pipeline in Task 3 may read raw chunks and execute the full plan on each chunk. This is correct but wasteful — operations like Filter and Select (streamable) can be applied per-chunk before merging, reducing peak memory. This is the "streaming" benefit.

### Scope

- For each chunk read, create a per-chunk `QueryPlan` with the chunk as source and only the streamable operations
- Execute the per-chunk plan using `executor.Execute()`
- Collect the per-chunk result frames
- Merge via `NivaraFrame.Concat()`
- Non-streamable operations should trigger fallback to full materialization (Task 3 handles this at plan level already)

### Constraints

- Must not execute non-streamable operations per-chunk
- Must preserve operation ordering within each chunk's plan
- Must keep the same schema transformation path

### Suggested implementation path

1. Extract streamable operations from `plan.Operations` (Filter, Select, Concatenation)
2. For each chunk, build `new QueryPlan(chunkPlan, streamableOps)` where `chunkPlan` wraps the chunk data
3. Execute per-chunk plan, collect frames
4. Concatenate frames

### Acceptance criteria

- Filter operation applied per-chunk before merge produces same result as post-merge filter
- Select operation per-chunk reduces per-chunk memory
- Sort/GroupBy/Join in the operation list force the full-data fallback
- Chained streamable ops (Filter → Select) work per-chunk

### Files likely involved

- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

---

## Task 5: Test Coverage for Streaming Execution

### Priority

High

### Goal

Achieve thorough unit test coverage for all streaming execution paths.

### Why this exists

Current `StreamingExecutionStrategyTests` has 12 tests, all of which exercise the no-op fallthrough path. None test real chunking behavior.

### Scope

- Chunked source read validation:
  - Large dataset (e.g., 10,000 rows) split into correct chunk sizes
  - Partial final chunk (< chunkSize rows)
- Result correctness:
  - Chunked execution == non-chunked execution for same data
  - Schema preservation across chunks
- Non-streamable fallback:
  - Sort, GroupBy, Join each trigger fallback correctly
- Memory budget boundaries:
  - Budget too small produces small chunks
  - Budget at 0 returns false from `ValidatePlan`
- Cancellation:
  - Cancellation between chunks stops execution
  - `OperationCanceledException` propagates (not wrapped)
- Error handling:
  - Source failure in mid-chunk wraps in `QueryExecutionException`
  - Invalid chunk source (e.g., throws) propagates correctly
- Edge cases:
  - Single row dataset
  - Empty dataset (0 rows)
  - `EstimatedRowCount` = null (read-until-empty behavior)
  - Source with `CanReadInChunks = false` falls through
- Null mask preservation in concatenation

### Constraints

- Must use existing `StubChunkedQuerySource` test helper (add missing API if Task 1 changed it)
- Must add a helper to create large chunked sources with configurable row count
- All existing tests must continue to pass

### Suggested implementation path

1. Add a `CreateLargeChunkedSource(int rowCount, int? estimatedRowCount = null)` helper to `ExecutionTestHelpers`
2. Write tests in `StreamingExecutionStrategyTests.cs` following the existing pattern
3. Verify chunk count and chunk size via `StubChunkedQuerySource.ChunksRead`
4. Cross-verify results against `LazyExecutionStrategy` for the same plan

### Acceptance criteria

- At least 15 new tests covering the scenarios above
- All existing streaming strategy tests pass
- Null mask preservation verified in concatenation
- Chunk boundary conditions verified (exact multiple, partial final chunk)
- Merged result row count equals sum of chunk row counts

### Files likely involved

- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs` (`StubChunkedQuerySource`)

---

## Suggested Agent Handout Batches

### Batch A: decision gate (Task 1)

- Task 1 — Streaming data source API shape

### Batch B: framing + pipeline (Tasks 2–4)

- Task 2 — `NivaraFrame.Concat()`
- Task 3 — Chunk-based data pipeline
- Task 4 — Per-chunk operation execution

### Batch C: tests (Task 5)

- Task 5 — Comprehensive streaming tests

---

## Final Checklist

- [x] Task 1 decision documented and `IQuerySource` API finalized (option 3: `ToAsyncEnumerable` default interface method)
- [x] Task 2 `NivaraFrame.Concat()` implemented and tested
- [x] Task 3 chunk iteration pipeline wired into `StreamingExecutionStrategy`
- [x] Task 4 per-chunk operation execution applied (applies all plan operations per-chunk — `isSuitableForStreaming` guarantees only streamable ops reach the loop)
- [x] Task 5 at least 15 new tests covering chunk boundaries, cancellation, fallback, errors, and edge cases (16 new tests, 37 total)
- [x] All existing tests in `StreamingExecutionStrategyTests` pass
- [x] `calculateChunkSize` result is actually used (no dangling discard)
- [x] Fallback to `LazyExecutionStrategy` for non-streamable plans preserved
- [x] Null masks preserved through chunk concatenation
