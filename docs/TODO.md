# Phase 4 — Async-First Streaming

## Problem

Async seams exist on `IQuerySource` (`ReadChunkAsync`, `ToAsyncEnumerable` at `IQueryInterfaces.cs:34-57`) but the pipeline is a **synchronous chunk puller with async wrappers**:

| # | Gap | Current reality | Impact |
|---|---|---|---|
| 1 | **No public async entry points** | `QueryFrame.Collect()` (QueryFrame.cs:396), `NivaraQuery<T>.Collect()` (NivaraQuery.cs:210) are sync-only. No `CollectAsync`/`ToListAsync`/`AsStream`. | Users block. No cancellation mid-stream. |
| 2 | **No channel-based buffering** | `StreamingExecutionStrategy.executeCoreInternalAsync` (StreamingExecutionStrategy.cs:107) accumulates ALL chunks into `List<NivaraFrame>`. No `Channel<T>`, no backpressure. | All intermediate results pile up in memory. No consumer-driven flow control. |
| 3 | **Operators are strictly synchronous** | `IQueryOperation.Execute` (IQueryInterfaces.cs:14) is sync-only. `executeOperationsOnData` (StreamingExecutionStrategy.cs:28) is a fully sync loop. | Operators cannot participate in async push pipeline. `await foreach` blocks synchronously on operator execution. |
| 4 | **No production IO source supports chunked streaming** | `CsvLazySource` (CsvDataSource.cs:133), `JsonLazySource` (JsonDataSource.cs:74) — none override `CanReadInChunks` (defaults false). | Every real-world plan hits the `!CanReadInChunks` branch → full sync materialization. Seam is dead code. |
| 5 | **StreamingBufferManager unused by streaming strategy** | `StreamingBufferManager` (Extensions/IO/StreamingBufferManager.cs:12) provides `RentBuffer`/`ReturnBuffer`, but `StreamingExecutionStrategy` only calls `calculateChunkSize` from `context.MemoryBudget`. | Memory budget advisory only; no hard enforcement on intermediate results. |
| 6 | **Cancellation checked only at iteration boundaries** | `ThrowIfCancellationRequested()` fires between chunk boundaries only (StreamingExecutionStrategy.cs:134). | No early stream termination mid-chunk; no clean `OperationCanceledException` propagation from within operator processing. |
| 7 | **No `IAsyncDisposable`** | `IQuerySource : IDisposable` (IQueryInterfaces.cs:17). `QueryFrame : IDisposable` (QueryFrame.cs:13). No `IAsyncDisposable`. | IO resources (file handles, CsvReader state) cannot be properly cleaned up across `await` boundaries. |

## Proposed Changes

### Step 1: Async operator interface
- Add `ExecuteAsync` default to `IQueryOperation` in `IQueryInterfaces.cs` (falls back to sync `Execute`)
- Override `ExecuteAsync` on streamable ops: `FilterOperation`, `SelectOperation`, `SliceOperation`, `SelectRowsOperation` (wrap existing sync logic in `ValueTask`)
- Non-streamable ops (Sort, GroupBy, Join, Distinct, Rolling, Cumulative, Shift, Rank) keep the default sync fallback

**Affected files:** `src/Nivara/Query/IQueryInterfaces.cs`, `src/Nivara/Operations/{Filter,Select,Slice,SelectRows}Operation.cs`

**Blast radius:** All `IQueryOperation` implementers (~12 ops). Default interface method means zero call-site changes. Overrides are thin wrappers — no behavior change.

### Step 2: Channel-based streaming strategy
- Rewrite `StreamingExecutionStrategy.executeCoreInternalAsync` to use `Channel<NivaraFrame>` with bounded capacity
- Producer: `await foreach` over `Source.ToAsyncEnumerable(chunkSize, ct)` → run streamable ops via `ExecuteAsync` → write to channel
- Non-streamable boundary: flush all pending channel chunks → `ConcatenateVertical` → run op synchronously → reset channel for remaining streamable ops
- Consumer: `Channel.Reader.ReadAllAsync(ct)` → accumulate → `ConcatenateVertical` at end
- Channel capacity: `clamp(memoryBudget / (estimatedBytesPerRow * chunkSize), 2, 16)`
- Use `StreamingBufferManager` for intermediate buffer tracking
- Ensure source + intermediate chunk frames are `Dispose()`d on cancellation/exception/normal completion

**Affected files:** `src/Nivara/Execution/StreamingExecutionStrategy.cs`. Uses existing `ConcatenateVertical` (NivaraFrameExtensions.cs:817), `executeOperationsOnData` helper (repurposed as `executeOperationsOnDataAsync`).

**Blast radius:** StreamingExecutionStrategy only. Sync path (`ExecuteCore`) untouched. Existing tests verify sync behavior.

### Step 3: Public async entry points
- `QueryFrame.CollectAsync(CancellationToken ct = default)` — delegates to `ExecutionEngine.ExecuteAsync`
- Make sync `QueryFrame.Collect()` a thin wrapper: `CollectAsync(default).GetAwaiter().GetResult()!`
- `NivaraFrameExtensions.CollectAsync(this NivaraFrame, CancellationToken ct = default)` — no-op for materialized frames (parallel to existing sync `Collect`)
- `NivaraQuery<T>.CollectAsync(CancellationToken ct = default)` — delegates to `frame.CollectAsync()`
- Make sync `NivaraQuery<T>.Collect()` a thin wrapper
- `NivaraQuery<T>.ToListAsync(CancellationToken ct = default)` — `CollectAsync()` → materialize rows (parallel to `ToObjects`)
- `QueryFrame.AsStream(int chunkSize, CancellationToken ct = default)` — `IAsyncEnumerable<NivaraFrame>` yielding processed chunks lazily

**Affected files:** `src/Nivara/Query/QueryFrame.cs`, `src/Nivara/Linq/NivaraQuery.cs`, `src/Nivara/NivaraFrameExtensions.cs`

**Blast radius:** Public API surface. Sync methods retain exact behavior (thin wrapper over async).

### Step 4: Chunk-capable IO sources
- **CsvLazySource** (`src/Nivara.Extensions/IO/CsvDataSource.cs:133`): override `CanReadInChunks => true`, add `EstimatedRowCount` (line-count heuristic), implement `ReadChunk`/`ReadChunkAsync` with persistent `CsvReader`/`StreamReader` state
- **JsonLazySource** (`src/Nivara/IO/JsonDataSource.cs:74`): same pattern with `Utf8JsonReader` streaming via `System.Text.Json`
- **Parquet**: add a `ParquetLazySource : IQuerySource` that streams row groups (row group boundaries = natural chunk boundaries)

**Affected files:** `src/Nivara/IO/JsonDataSource.cs`, `src/Nivara.Extensions/IO/CsvDataSource.cs`, `src/Nivara.Extensions/IO/ParquetReader.cs`

**Blast radius:** IO sources. Existing synchronous `Execute()` paths preserved — chunked methods are additive.

### Step 5: `IAsyncDisposable` and resource cleanup
- `QueryFrame` implements `IAsyncDisposable` alongside existing `IDisposable`
- `StreamingExecutionStrategy` ensures source + intermediate chunk frames are `Dispose()`d on cancellation/exception/normal completion
- All `await foreach` loops use `await using` where applicable
- Channel writer `Complete()`d on normal exit, `TryComplete(exception)` on error
- **Strategy-level casting**: `if (source is IAsyncDisposable asyncDisposable) await asyncDisposable.DisposeAsync()` — avoids churn across all `IQuerySource` implementers

**Affected files:** `src/Nivara/Query/QueryFrame.cs`, `src/Nivara/Execution/StreamingExecutionStrategy.cs`, IO sources

**Blast radius:** Resource disposal. IO sources that don't implement `IAsyncDisposable` still dispose via sync path (no behavior change for existing code).

### Step 6: `AsStream` public enumerable
- `QueryFrame.AsStream(int chunkSize, CancellationToken ct = default)` returns `IAsyncEnumerable<NivaraFrame>` — lazily yields *processed* chunks (source chunks → streamable ops applied), allowing consumers to process chunks as they arrive before concatenation

**Affected files:** `src/Nivara/Query/QueryFrame.cs`

**Blast radius:** New method only — no existing API changes.

## Verification Steps

1. Build: `dotnet build Nivara.slnx`
2. Ask user before running `dotnet test`
3. Property tests: `CollectAsync()` results match `Collect()` results (null-mask preservation via `AssertFramesEqualWithMasks`)
4. Cancellation: clean `OperationCanceledException` with no resource leaks
5. Memory: channel bounded capacity enforced; `StreamingBufferManager.IsMemoryBudgetExceeded` never true
6. All 1948 existing tests pass unchanged

## Planned Commit List

1. ✅ `feat: add ExecuteAsync to IQueryOperation with default fallback to sync Execute` (done — commit 64092b1)
2. ✅ folded into commit 1 — ExecuteAsync on streamable operations (Filter, Select, Slice, SelectRows)
3. ✅ `refactor: channel-based streaming pipeline with backpressure and flush-concatenate-resume` (done — commit 20aae2e)
4. ✅ `feat: add CollectAsync, ToListAsync, AsStream public async entry points` (done — commit b43f9fe; AsStream is on QueryFrame, collect via NivaraFrameExtensions)
5. ✅ `feat: make CsvLazySource and JsonLazySource chunk-capable with CanReadInChunks` (done — commit f6ee6ca)
6. ✅ `feat: add ParquetLazySource with row-group chunking` (done — commit 4ca14bf; bundled Steps 6+7 since IAsyncDisposable changes were already in working tree)
7. ✅ bundled into commit 4ca14bf — `feat: add IAsyncDisposable to QueryFrame and resource cleanup in streaming strategy`
8. ✅ `test: add async streaming tests covering parity, cancellation, memory budget, and IO chunking` (done — commit 7a3b2f1)

## GitHub issues log

- (none yet — will add discovered concerns as issues during execution)
