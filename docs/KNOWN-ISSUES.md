# Known Issues & Immediate Technical Debt

This file documents current bugs, correctness risks, and immediate technical
debt that should be addressed before relying on the affected behavior. It is
distinct from `*-GAPS.md` files that catalog forward-looking work.

---

## Phase 1 Gaps (reviewed against `EXECUTION-PHASE1.md`)

---

## Gap 2: Concrete Type Casting in ParallelExecutionStrategy Breaks LSP

### Priority

High

### Files involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`

### Issue

`ParallelExecutionStrategy.executeOperationParallelSync` and
`executeOperationParallelAsync` cast `IQueryOperation` to concrete types:

```csharp
if (opType == "Sort")
    return executeSortParallelSync((SortOperation)operation, ...);
if (opType == "GroupBy")
    return executeGroupByParallelSync((GroupByOperation)operation, ...);
if (opType == "Join")
    return executeJoinParallelSync((JoinOperation)operation, ...);
if (opType.StartsWith("Concatenate", ...))
    return executeConcatenationParallelSync((ConcatenationOperation)operation, ...);
```

This couples the strategy to concrete operation implementations. The strategy
pattern was designed to work with any `IQueryOperation`. Any custom or
third-party operation will throw `InvalidCastException` even if it is logically
"parallelizable".

### Impact

- Violates Liskov Substitution Principle — the strategy cannot handle arbitrary `IQueryOperation` implementations
- Breaks extensibility: custom filter operations (that implement `IQueryOperation` with `OperationType == "Filter"`) will throw at runtime
- The pre-Phase 1 code used `Task.Run(() => operation.Execute(input))` for all operations without casting — this was correct but less performant

### Suggested fix

Remove per-type parallel specializations and use a generic chunk-and-merge
pattern that works with any `IQueryOperation` via:
```csharp
var subset = ParallelExecutionHelper.CreateRowSubset(input, start, length);
var partial = operation.Execute(subset);
// merge partial results
```

---

## Gap 3: Unused Return Value from calculateChunkSize

### Priority

Medium

### Files involved

- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

### Issue

Both `ExecuteCore` (line 28) and `ExecuteCoreAsync` (line 44) call
`calculateChunkSize(context.MemoryBudget)` but discard the return value.
The chunk size is never used — the strategy falls through to
`executor.Execute(plan)` which processes all data at once.

### Impact

- Dead code — `calculateChunkSize` runs but its result has no effect
- Misleading to readers who expect chunked/streaming behavior
- Streaming is currently identical to lazy execution in practice

### Suggested fix

Either:
- Wire `calculateChunkSize` into actual chunked processing (true streaming), or
- Remove the call until streaming is implemented, or
- Add a tracking issue for implementing true streaming semantics.

---

## Gap 4: Redundant Task.Run Overhead in EagerExecutionStrategy.ExecuteCoreAsync

### Priority

Low

### Files involved

- `src/Nivara/Execution/EagerExecutionStrategy.cs`

### Issue

`EagerExecutionStrategy.ExecuteCoreAsync` wraps each individual operation in
`Task.Run`:

```csharp
currentColumns = await Task.Run(() => operation.Execute(currentColumns), context.CancellationToken);
```

The base class `ExecutionStrategyBase.ExecuteCoreAsync` already wraps the
entire `ExecuteCore` in `Task.Run` when not overridden. Since
`IQueryOperation.Execute` is synchronous, the per-operation `Task.Run` adds
unnecessary thread-pool scheduling overhead without benefit.

### Impact

- Each operation incurs extra thread-pool scheduling overhead
- The base class default of `Task.Run(() => ExecuteCore(...))` would be sufficient

### Suggested fix

Remove the `ExecuteCoreAsync` override from `EagerExecutionStrategy` and rely on
the base class default that wraps the entire sync path in `Task.Run`.

---

## Gap 5: Scope Creap — Chunked Source Interface Members Added Without Plan Coverage

### Priority

Low

### Files involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`

### Issue

Three members were added to `IQuerySource` that are **not mentioned** in
`EXECUTION-PHASE1.md`:

- `bool CanReadInChunks => false;`
- `int? EstimatedRowCount => null;`
- `ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(...)`

These were introduced alongside `ParallelExecutionStrategy.readSourceAsync` to
support chunked parallel source reading. They go beyond the stated scope of
Task 5 ("Add IQuerySource.ExecuteAsync").

### Impact

- Default interface implementations will be invisible to older C# compilers (< C# 8) if the project ever targets them
- `ReadChunkAsync` default throws `NotSupportedException` — callers must guard with `CanReadInChunks`
- Two interface members (`CanReadInChunks`, `EstimatedRowCount`) are only consumed by `ParallelExecutionStrategy`

### Suggested fix

No immediate fix needed — this is valid code that works correctly. Consider
documenting it as a planned addition in a follow-up gap document, or updating
`EXECUTION-PHASE1.md` to reflect the actual delivered scope.

---

## Gap 6: "Concatenation" String Matching Inconsistency

### Priority

Low

### Files involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`

### Issue

`ParallelExecutionStrategy.isParallelizable` and
`executeOperationParallelSync`/Async match concatenation operations using
`StartsWith("Concatenate", StringComparison.Ordinal)`, while the plan's cost
tables and other strategies (Eager, Lazy, Streaming) use exact match
`"Concatenation"`.

The actual operation type strings used by `ConcatenationOperation` are unknown
— but the inconsistency means one strategy may match where another does not.

### Impact

- Inconsistent behavior between strategies for concatenation operations
- `ParallelExecutionStrategy` may try to parallelize operations that happen to start with "Concatenate" (false positive match)

### Suggested fix

Normalize all strategies to use the same matching strategy — either exact match
`"Concatenation"` or `StartsWith`. A constant or helper method on
`ConcatenationOperation` (e.g., `ConcatenationOperation.OperationTypeName`)
would be ideal.

---

## Future performance work, API gaps, testing gaps, and deferred design
items are tracked in `*-GAPS.md` files
