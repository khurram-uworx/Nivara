# Phase 4 — Async-First Streaming

**Status:** planning · **Scope:** async streaming pipeline (`src/Nivara`, `src/Nivara.Extensions/IO`) · **Dependencies:** Phases 1–3 (delivered)

---

## Current State (pre-Phase 4)

Async seams exist on `IQuerySource` (`ReadChunkAsync`, `ToAsyncEnumerable` at
`src/Nivara/Query/IQueryInterfaces.cs:34-57`) but the pipeline is a **synchronous
chunk puller with async wrappers**:

| # | Gap | Current reality | Impact |
|---|---|---|---|
| 1 | **No public async entry points** | `QueryFrame.Collect()` (QueryFrame.cs:396), `NivaraQuery<T>.Collect()` (NivaraQuery.cs:210), `NivaraFrameExtensions.Collect()` (NivaraFrameExtensions.cs:1168) are all sync. No `CollectAsync`/`ToListAsync`/`AsStream` anywhere. | Users with `IAsyncEnumerable`-consuming pipelines block. No way to cancel a long-running query mid-stream. |
| 2 | **No channel-based buffering** | `StreamingExecutionStrategy.executeCoreInternalAsync` (StreamingExecutionStrategy.cs:107) accumulates ALL chunks into `List<NivaraFrame>` then calls `ConcatenateVertical` at the end. No `Channel<T>`, no backpressure. | All intermediate chunk results pile up in memory. No consumer-driven flow control. |
| 3 | **Operators are strictly synchronous** | `IQueryOperation.Execute` (IQueryInterfaces.cs:14) returns `IReadOnlyDictionary<string, IColumn>` synchronously. `executeOperationsOnData` (StreamingExecutionStrategy.cs:28) is a fully sync loop. | Operators cannot participate in an async push pipeline. Even the `await foreach` over source chunks blocks synchronously on operator execution. |
| 4 | **No production IO source supports chunked streaming** | `CsvLazySource` (CsvDataSource.cs:133), `JsonLazySource` (JsonDataSource.cs:87) — none override `CanReadInChunks` (defaults false at IQueryInterfaces.cs:26). | Every real-world plan hits the `!CanReadInChunks` branch (StreamingExecutionStrategy.cs:116) which calls `executor.Execute(plan)` — full sync materialization. The `ToAsyncEnumerable`/`ReadChunkAsync` seam is dead code in production. |
| 5 | **`StreamingBufferManager` unused by streaming strategy** | `StreamingBufferManager` (StreamingBufferManager.cs:12) provides `RentBuffer`/`ReturnBuffer` with memory-budget tracking, but `StreamingExecutionStrategy` only calls `calculateChunkSize` from `context.MemoryBudget`. | Memory budget is advisory only; no hard enforcement on intermediate results. |
| 6 | **Cancellation checked only at iteration boundaries** | `ThrowIfCancellationRequested()` fires between chunk boundaries only (StreamingExecutionStrategy.cs:134). | No early stream termination mid-chunk; no clean `OperationCanceledException` propagation from within operator processing. |
| 7 | **No `IAsyncDisposable`** | `IQuerySource : IDisposable` (IQueryInterfaces.cs:17). No `IAsyncDisposable` anywhere. | IO resources (file handles, CsvReader state) cannot be properly cleaned up across `await` boundaries. |

### Infrastructure already in place (leverage, don't rebuild)

- `FusedExpressionEvaluator.EvaluateChunked` (FusedExpressionEvaluator.cs:102) — bit-identical chunked results from fused kernel IR (#167 contract)
- `TensorPrimitivesKernel.TryEvaluateChunked` (TensorPrimitivesKernel.cs:71) — SIMD-span chunked backend
- `ParallelExecutionStrategy.readSourceAsync` (ParallelExecutionStrategy.cs:519) — reference pattern for chunked source reading with `ReadChunkAsync`
- `StreamingBufferManager` (StreamingBufferManager.cs:12) — byte-buffer memory budget enforcement
- `NivaraExecutionContext` (NivaraExecutionContext.cs:43,48) — `MemoryBudget` and `CancellationToken` already exist
- `StreamingExecutionStrategy.NonStreamableOperations` (StreamingExecutionStrategy.cs:9) — already classifies which ops break streaming

---

## Design

### Project Boundary Split

| Responsibility | Project | Rationale |
|---|---|---|
| Async execution infrastructure (`IQuerySource`, `IQueryOperation`, `StreamingExecutionStrategy`, `ExecutionEngine`, `QueryFrame`, `NivaraQuery<T>`) | `src/Nivara` (core) | Core types that don't depend on any IO library |
| JSON IO source chunking (`JsonLazySource`) | `src/Nivara` (core) | Already lives in `src/Nivara/IO/JsonDataSource.cs` |
| CSV IO source chunking (`CsvLazySource`) | `src/Nivara.Extensions/IO/CsvDataSource.cs` | CsvHelper (33.1.0) is dependency-only Extensions concern |
| Parquet IO source chunking | `src/Nivara.Extensions/IO/ParquetReader.cs` | Parquet.Net (6.0.3) is dependency-only Extensions concern |
| Channel pipeline, async entry points, async operators | `src/Nivara` (core) | Independent of any IO library |

### Async Operator Interface

Extend `IQueryOperation` with a default `ExecuteAsync` that falls back to sync `Execute`:

```csharp
// src/Nivara/Query/IQueryInterfaces.cs
internal interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);

    // NEW: default falls back to sync Execute — zero call-site changes for non-overridden ops
    ValueTask<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(
        IReadOnlyDictionary<string, IColumn> input,
        CancellationToken ct = default)
        => new(Execute(input));
}
```

**Streamable ops** (`FilterOperation`, `SelectOperation`, `SliceOperation`,
`SelectRowsOperation`) override `ExecuteAsync` with their span-based kernels wrapped
in `ValueTask` for zero-overhead fast path. **Non-streamable ops** (Sort, GroupBy,
Join, Distinct, Rolling, Cumulative, Shift, Rank) keep the default sync fallback —
they force a materialization boundary.

### Channel-Based Streaming Pipeline

Replace the `List<NivaraFrame>` accumulation in `StreamingExecutionStrategy.ExecuteCoreAsync`
with a bounded `Channel<ProcessedChunk>`:

```
Source → [Producer] → chunk → [Channel (bounded)] → [Consumer] → ConcatenateVertical → NivaraFrame
              ↓                    ↑
        streamable ops via   capacity = clamp(memoryBudget /
        ExecuteAsync          estimatedBytesPerRow / chunkSize, 2, 16)
```

**Producer**: `await foreach` over `Source.ToAsyncEnumerable(chunkSize, ct)` → for each chunk,
run streamable ops via `ExecuteAsync` → write to channel.

**Non-streamable op boundary**: flush all pending chunks from channel → concatenate via
`ConcatenateVertical` → run non-streamable op synchronously → reset channel for remaining
streamable ops (re-enter the channel pipeline).

**Consumer**: `Channel.Reader.ReadAllAsync(ct)` → accumulate → `ConcatenateVertical` at end.

### Streamable vs. Non-Streamable Boundary

The existing `NonStreamableOperations` set (StreamingExecutionStrategy.cs:9) already defines
the boundary:

```
Sort, SortByExpression, GroupBy, Join, Distinct, Rolling, Cumulative, Shift, Rank
```

**Resolved: flush-concatenate-resume** — when a non-streamable op follows streamable ops,
flush all pending channel chunks, concatenate, run the op, then reset the channel for
remaining streamable ops. This preserves memory efficiency for partial-stream scenarios
(e.g., `Where` filter streams through, `Sort` triggers flush, `Select` resumes streaming).

### Public API

| Method | Type | Signature |
|---|---|---|
| `CollectAsync` | `QueryFrame` | `public Task<NivaraFrame> CollectAsync(CancellationToken ct = default)` |
| `CollectAsync` | `NivaraFrameExtensions` | `public static Task<NivaraFrame> CollectAsync(this NivaraFrame frame, CancellationToken ct = default)` |
| `ToListAsync` | `NivaraQuery<T>` | `public Task<List<T>> ToListAsync(CancellationToken ct = default)` |
| `AsStream` | `QueryFrame` | `public IAsyncEnumerable<NivaraFrame> AsStream(int chunkSize, CancellationToken ct = default)` |

Sync `Collect()` becomes a thin blocking wrapper: `CollectAsync(default).GetAwaiter().GetResult()!`

### Chunk-Capable IO Sources

**CSV** (`CsvLazySource`, src/Nivara.Extensions/IO/CsvDataSource.cs:133):
Override `CanReadInChunks => true`, `EstimatedRowCount` via line-count heuristic or
`CsvOptions.RowCountHint`, `ReadChunk`/`ReadChunkAsync` with persistent `CsvReader` +
`StreamReader` state, `ExecuteAsync` using `CsvReader.ReadAsync()`.

**JSON** (`JsonLazySource`, src/Nivara/IO/JsonDataSource.cs:74):
Same pattern with `Utf8JsonReader` streaming via `System.Text.Json`.

**Parquet** (`ParquetReader`, src/Nivara.Extensions/IO/):
Row group boundaries are natural chunk boundaries.

### Resource Disposal

- `QueryFrame` implements `IAsyncDisposable` alongside existing `Dispose()`
- `StreamingExecutionStrategy` ensures source + intermediate chunk frames are `Dispose()`d on cancellation, exception, or normal completion
- All `await foreach` loops use `await using` where applicable
- Channel writer is `Complete()`d on normal exit and `TryComplete(exception)` on error
- **Resolved: strategy-level casting** — `if (source is IAsyncDisposable asyncDisposable) await asyncDisposable.DisposeAsync()` alongside existing `source.Dispose()`. Avoids churn across all `IQuerySource` implementers and test stubs.

---

## Implementation Order

1. Async operator interface — add `ExecuteAsync` default to `IQueryOperation`, override on `FilterOperation`, `SelectOperation`, `SliceOperation`, `SelectRowsOperation`
2. Channel-based streaming strategy — rewrite `StreamingExecutionStrategy.ExecuteCoreAsync` to use `Channel<>` instead of `List<NivaraFrame>`; implement producer/consumer with bounded capacity; implement non-streamable boundary flush
3. Public async entry points — `CollectAsync` on `QueryFrame`, `NivaraFrameExtensions`, `NivaraQuery<T>`; `ToListAsync` on `NivaraQuery<T>`; make sync `Collect()` a thin wrapper
4. Chunk-capable IO sources — `CsvLazySource`, `JsonLazySource`, Parquet lazy source
5. `IAsyncDisposable` and resource cleanup — `QueryFrame`, `StreamingExecutionStrategy`, channel pipeline
6. `AsStream` public enumerable — lazy chunk yielding

---

## Acceptance Criteria

1. `CollectAsync()` results match `Collect()` results (property tests over chunk sizes, null-mask preservation via `AssertFramesEqualWithMasks`)
2. Cancellation triggers clean `OperationCanceledException` with no resource leaks; IO source readers released
3. Memory stays within configured budget; channel bounded capacity enforced; `StreamingBufferManager.IsMemoryBudgetExceeded` never true
4. Real IO sources (CSV, Parquet, JSON) with `CanReadInChunks = true` produce chunked async streams; results equal full-execution results
5. All existing tests pass unchanged (1948 baseline)
6. Sync `Collect()` behavior unchanged

---

## Resolved Decisions

1. **Non-streamable op boundary** — flush-concatenate-resume. Preserves memory efficiency for partial-stream scenarios (e.g., filter before sort on 10GB CSV).
2. **`IAsyncDisposable` on `IQuerySource`** — cast at strategy level (`is IAsyncDisposable`), not added to interface. Avoids churn across implementers.
3. **`ToListAsync` return type** — `Task<List<T>>` to match sync `ToList()` exactly.
4. **Channel capacity** — `memoryBudget / (estimatedBytesPerRow * chunkSize)` clamped to [2, 16].
5. **Post-Step-3 backlog** — `ToObjectsAsync` (async row materialization) not in initial scope.
