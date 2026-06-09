# Known Issues & Immediate Technical Debt

This file documents current bugs, correctness risks, and immediate technical
debt that should be addressed before relying on the affected behavior. It is
distinct from `*-GAPS.md` files that catalog forward-looking work.

---

## Gaps (reviewed against Phase 1 and Phase 2 execution plan deliverables)

> **Gap numbering note:** Gaps 1, 3, and 5 were removed as stale/resolved after
> Phase 2 and Phase 3 completion. Gap numbering is preserved to avoid confusion
> with past references. New Phase 2 gaps start at Gap 7.

---

## Gap 4: Redundant Task.Run Overhead in EagerExecutionStrategy and StreamingExecutionStrategy

### Priority

Low

### Files involved

- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

### Issue

**EagerExecutionStrategy**: `ExecuteCoreAsync` wraps each individual operation in
`Task.Run`:

```csharp
currentColumns = await Task.Run(() => operation.Execute(currentColumns), context.CancellationToken);
```

The base class `ExecutionStrategyBase.ExecuteCoreAsync` already wraps the
entire `ExecuteCore` in `Task.Run` when not overridden. Since
`IQueryOperation.Execute` is synchronous, the per-operation `Task.Run` adds
unnecessary thread-pool scheduling overhead without benefit.

**StreamingExecutionStrategy**: Both `ExecuteCoreAsync` fallback paths (lines
116 and 147) wrap the synchronous `executor.Execute(plan)` in `Task.Run`:

```csharp
return await Task.Run(() => executor.Execute(plan), context.CancellationToken);
```

Per Microsoft's [Async wrappers for sync methods](https://learn.microsoft.com/dotnet/standard/asynchronous-programming-patterns/async-wrappers-for-synchronous-methods)
guidance, wrapping synchronous CPU-bound work in `Task.Run` provides offloading
but no scalability benefit. For library code, this decision should belong to the
consumer, not the library.

### Impact

- Each operation incurs extra thread-pool scheduling overhead
- The base class default of `Task.Run(() => ExecuteCore(...))` would be sufficient
- StreamingExecutionStrategy's fallback paths consume a thread-pool thread unnecessarily

### Suggested fix

Remove `ExecuteCoreAsync` override from `EagerExecutionStrategy` and rely on
the base class default. In `StreamingExecutionStrategy`, avoid `Task.Run` in
fallback paths — call `executor.Execute(plan)` synchronously since the method
is not truly async in those branches (the non-chunked source path completes
synchronously).

---

## Gap 6: "Concatenation" String Matching Inconsistency

### Priority

Low

### Files involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`
- `src/Nivara/Operations/ConcatenationOperation.cs`

### Issue

Operation type string matching for concatenation uses different patterns across
strategies:

| Strategy | Pattern | Matches? |
|----------|---------|----------|
| `ParallelExecutionStrategy.isParallelizable` | `StartsWith("Concatenate", ...)` | **Correct** — matches "ConcatenateVertical", "ConcatenateHorizontal" |
| `ParallelExecutionStrategy.EstimateExecutionCost` | `StartsWith("Concatenate", ...)` | **Correct** |
| Eager, Lazy, Streaming | `"Concatenation"` exact match | **Never matches** — `ConcatenationOperation.OperationType` returns `"Concatenate{direction}"` |

### Phase 2 update

Phase 2 partially fixed this for `ParallelExecutionStrategy`:
- `isParallelizable` was changed from `"Concatenation"` → `StartsWith("Concatenate", ...)` (line 19)
- `EstimateExecutionCost` was similarly updated (line 672)
- These changes correctly match `"ConcatenateVertical"` / `"ConcatenateHorizontal"`

However, the other 3 strategies (Eager, Lazy, Streaming) and the cost tables
still use exact match `"Concatenation"` — they will **never** dispatch
concatenation operations.

### Phase 3 update

Phase 3 **extended** the problem to `StreamingExecutionStrategy`:
- `StreamableOperationTypes` (line 8) uses `"Concatenation"` exact match —
  never matches `"ConcatenateVertical"` or `"ConcatenateHorizontal"`
- `EstimateExecutionCost` (line 201) still uses `"Concatenation"` switch arm —
  same mismatch as Pre-Phase 2
- The `StreamableOperationTypes` field is **never referenced** anywhere (see
  also Gap 16), so this exact-match bug is currently dead code rather than
  a live logic error

### Impact

- Eager, Lazy, and Streaming strategies will never match concatenation operations
- Phase 3 added a new `StreamableOperationTypes` set with the same broken pattern
- `ParallelExecutionStrategy` uses a `StartsWith` check that is broader than exact match but correctly covers all valid operation types
- No constants or helpers encapsulate the pattern — it's repeated in string literals across files

### Suggested fix

Extract `ConcatenationOperation.OperationTypePrefix` constant or a helper
method, and normalize all 4 strategies to use it. Consider introducing
well-known constants for all operation types (see EXECUTION-PLAN Phase 5).

---

## Phase 2 Gaps (reviewed against `EXECUTION-PHASE2.md`)

---
## Gap 10: Unused ParallelExecutionHelper Methods

### Priority

Low

### Files involved

- `src/Nivara/Execution/ParallelExecutionHelper.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`

### Issue

Three public helper methods in `ParallelExecutionHelper` are **never called**
by the Phase 2 code:

| Method | In Phase 2 plan? | Actually used? |
|--------|-----------------|----------------|
| `ProcessInParallelAsync<T, TResult>` | Task 2 ("ProcessInParallelAsync over chunk ranges") | No — Phase 2 uses raw `Parallel.ForEach`/`Parallel.ForEachAsync` instead |
| `ProcessColumnsInParallelAsync` | Task 3 ("ProcessColumnsInParallelAsync over columns or sources") | No — Phase 2 uses raw `Parallel.ForEach` over columns |
| `ParallelAggregateAsync<T, TResult>` | Task 4 ("ParallelAggregateAsync with merge-combiner") | No — Phase 2 uses raw `ConcurrentBag` + manual merge |

The only Phase 2 code that calls `ParallelExecutionHelper` uses these methods:
- `CreateRowSubset` (Filter, GroupBy, Join chunking)
- `CreateChunkRanges` (all operations)
- `CalculateOptimalChunkSize` (all operations)
- `GetRecommendedParallelism` (all operations)
- `ShouldUseParallelProcessing` (all operations)
- `ConcatenateColumnDictionaries` (Filter combination, source chunk merge)
- `MergeGroupByDictionaries` (GroupBy)
- `MergeJoinHashMaps` (Join)
- `SliceColumn` (used by CreateRowSubset)

The three unused methods were added pre-Phase 2 and represent a different
abstraction level (generic chunk-process-combine) than what was actually
implemented (per-type specialized dispatch).

### Impact

- Dead code that must be maintained/fixed for compilation
- Misleading API surface — new readers may try to use these helpers and find
  they don't integrate with the actual dispatch
- The Phase 2 plan's "shared-kernel architecture" called for these helpers to
  be the dispatch mechanism, but they were bypassed

### Suggested fix

Either:
- Remove the three unused methods, or
- Retrofit them to be the actual dispatch mechanism (implementing the
  shared-kernel architecture from the Phase 2 plan), or
- Mark them `[Obsolete]` with a message directing to the direct `Parallel.ForEach` pattern

---

## Gap 13: Custom Chunk Extension Conflicts with BCL Enumerable.Chunk

### Priority

Low

### Files involved

- `src/Nivara/Execution/ParallelExecutionHelper.cs`

### Issue

The `ParallelExtensions` class (line 338) defines a custom `Chunk<T>` extension
method on `IEnumerable<T>`:

```csharp
public static IEnumerable<IEnumerable<T>> Chunk<T>(this IEnumerable<T> source, int chunkSize)
```

The project targets `net10.0` (line 4 of `Nivara.csproj`), which includes the
BCL's `System.Linq.Enumerable.Chunk<TSource>(IEnumerable<TSource>, int)` —
available since .NET 6. The custom `Chunk` has a different return type
(`IEnumerable<IEnumerable<T>>` vs BCL's `IEnumerable<TSource[]>`), which means
it won't silently shadow the BCL version in all scenarios, but:

- If `using System.Linq;` is present in the file (it's implicitly available
  via `<ImplicitUsings>enable</ImplicitUsings>`), the compiler may see two
  candidate extension methods
- The custom version returns `IEnumerable<IEnumerable<T>>` (lazy iterator of
  `List<T>` per chunk) while BCL returns `IEnumerable<TSource[]>` (arrays).
  The BCL version is more efficient (avoiding `List<T>` overhead) and is
  community-standard
- Two callers (`ProcessInParallelAsync` line 35, `ParallelAggregateAsync`
  line 167) use `source.Chunk(chunkSize)` — these would use the custom version
  when in scope

### Impact

- Code reviewers unfamiliar with the project may wonder why a custom `Chunk`
  exists when BCL provides one
- Custom version allocates `List<T>` per chunk (more GC pressure) vs BCL's
  `TSource[]` arrays
- Potential ambiguity if the file is ever moved/refactored and namespace
  resolution changes

### Suggested fix

Replace the custom `Chunk` extension method with the BCL's built-in
`System.Linq.Enumerable.Chunk`. Update callers to expect `TSource[]` instead
of `IEnumerable<T>`:
```csharp
// Was: chunks of IEnumerable<T> (List<T> internally)
// Now: chunks of TSource[] (BCL arrays)
var chunks = source.Chunk(chunkSize);  // BCL built-in
```

---

## Phase 3 Gaps (reviewed against `EXECUTION-PHASE3.md`)

---

## Gap 16: Unused StreamableOperationTypes Field

### Priority

Low

### Files involved

- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

### Issue

`StreamingExecutionStrategy` declares (line 8):

```csharp
static readonly HashSet<string> StreamableOperationTypes = new() { "Filter", "Select", "Concatenation" };
```

This field is **never referenced** anywhere in the codebase. The per-chunk
operation execution (`executeOperationsOnData`, line 29) applies **all**
operations from `plan.Operations` without filtering. The streaming suitability
check (`isSuitableForStreaming`) only checks for non-streamable operations in
`NonStreamableOperations` — it never consults `StreamableOperationTypes`.

Additionally, the `"Concatenation"` entry is incorrect (never matches
`"ConcatenateVertical"` / `"ConcatenateHorizontal"`) — see Gap 6.

### Impact

- Dead code that misleads readers into thinking per-chunk operation filtering
  is implemented
- The field suggests Task 4 (per-chunk operation execution per individual
  streamable ops) was planned but not wired in
- Bakes in the same `"Concatenation"` string-matching bug as Gap 6

### Suggested fix

Either:
- Remove `StreamableOperationTypes` entirely if per-chunk filtering is not
  needed (the plan-level `isSuitableForStreaming` guard already ensures only
  streamable operations reach the chunk loop), or
- Wire it into `executeOperationsOnData` to filter operations per-chunk,
  fixing the `"Concatenation"` → `"Concatenate"` prefix match when doing so.

---

## Gap 17: async Keyword Without Await in StubChunkedQuerySource.ReadChunkAsync

### Priority

Low

### Files involved

- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs`

### Issue

`StubChunkedQuerySource.ReadChunkAsync` (line 127) is declared `async` but
contains no `await` expression:

```csharp
public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
    int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
{
    cancellationToken.ThrowIfCancellationRequested();
    // ... all synchronous operations ...
    return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) };
}
```

The C# compiler emits warning CS1998 for this pattern. Since the method
completes synchronously every time, the `async` keyword adds unnecessary
overhead (state machine allocation).

### Impact

- Compiler warning CS1998
- Unnecessary `AsyncTaskMethodBuilder` state machine allocation on every call
- Masks the sync-over-async bug in Gap 15 — because this test helper never
  truly yields, `ExecuteCore`'s `.GetAwaiter().GetResult()` never deadlocks in
  tests, giving false confidence

### Suggested fix

Remove the `async` keyword and return `ValueTask` directly:

```csharp
public ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
    int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
{
    cancellationToken.ThrowIfCancellationRequested();
    // ... synchronous work ...
    return new ValueTask<IReadOnlyDictionary<string, IColumn>>(
        new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(data) });
}
```

Consider adding a version with `await Task.Delay(1)` or `await Task.Yield()`
for truly testing async code paths (reveals the Gap 15 deadlock in tests).

---

## Phase 4 Gaps (reviewed against `EXECUTION-PHASE4.md`)

---

## Gap 19: Public Setter on ExecutionDiagnostics.EndTime

### Priority

Low

### Files involved

- `src/Nivara/Diagnostics/ExecutionDiagnostics.cs`

### Issue

The `EndTime` property has a public setter:

```csharp
// Current implementation (line 119):
public DateTime? EndTime { get; set; }
```

The `EXECUTION-PHASE4.md` suggested implementation specified `private set`:
```csharp
// EXECUTION-PHASE4.md suggested:
public DateTime? EndTime { get; private set; }
```

Before Phase 4, the property already had `private set`. Phase 4 changed it to public. Only `EndExecution()` (an `internal` method) should set this value. The public setter allows any caller to mutate `EndTime`, which could produce incorrect diagnostic reports. There is no guard or validation on the setter.

### Impact

- Consumers can inadvertently overwrite `EndTime`, making `GenerateReport()` or `GetSummary()` return misleading results
- The public setter makes the API's contract unclear — other immutable-like properties (`StartTime`, `TotalExecutionTime`) are readonly or computed
- Minor API-surface regression from the pre-Phase 4 design

### Suggested fix

Change the setter back to `private set`:

```csharp
public DateTime? EndTime { get; private set; }
```

This restores the pre-Phase 4 encapsulation. `EndExecution()` is `internal` and already sets the property — no callers need the public setter.

---

## Future performance work, API gaps, testing gaps, and deferred design
items are tracked in `*-GAPS.md` files
