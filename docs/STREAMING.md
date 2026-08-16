# Streaming (AsStream) — Behavior Reference

Public entry points for chunked, lazy processing of query frames:

| API | Location |
|---|---|
| `QueryFrame.AsStream(int chunkSize = 10000, CancellationToken ct = default)` | `src/Nivara/Query/QueryFrame.cs` |
| `NivaraQuery<T>.AsStream(...)` — passthrough | `src/Nivara/Linq/NivaraQuery.cs` |
| `NivaraFrame.AsQueryFrame()` / `NivaraQuery<T>.AsQueryFrame()` | `src/Nivara/NivaraFrame.cs` / `src/Nivara/Linq/NivaraQuery.cs` |
| `Csv.ScanAsQueryFrame(string, CsvOptions?)` | `src/Nivara.Extensions/IO/CsvExtensions.cs` |
| `Json.ScanAsQueryFrame(string, JsonOptions?)` | `src/Nivara/IO/JsonExtensions.cs` |
| `Parquet.ScanAsQueryFrame(string, ParquetReadOptions?)` | `src/Nivara.Extensions/IO/NivaraParquetReader.cs` |

## When to use streaming

Prefer `AsStream` over `CollectAsync` when:

- the result is large and you want constant-memory processing (one chunk at a time), and
- the query is **fully streamable** over a **chunk-capable source** (CSV, JSON, Parquet).

When the query is not fully streamable, `AsStream` still works but degrades to a single
full-result frame — see the boundary contract below.

## The streaming contract

`AsStream` yields **one `NivaraFrame` per source chunk** only when **both** hold:

1. **Fully streamable plan** — the plan contains only streamable operations
   (`Filter`, `Select`, `Slice`, `SelectRows`) and **no window expressions**.
2. **Chunk-capable source** — the source reports `CanReadInChunks == true`
   (CSV, JSON, Parquet lazy sources do; in-memory frames and other in-memory sources do not).

Otherwise `AsStream` yields a **single frame produced from the full source** whose rows are
identical to `CollectAsync()`. The one-frame-per-chunk contract does **not** hold.

### Non-streamable boundary operations

A plan containing any of the following falls into the single-frame fallback:

```
Sort, SortByExpression, GroupBy, Join, Distinct, Rolling, Cumulative, Shift, Rank
```

Window expressions (e.g. `ColumnExpressions.RowNumber`, `.Rank`, `.DenseRank`,
`.PercentRank`) are also non-streamable and trigger the same fallback.

### Note: Collect vs AsStream

The `Collect`/`CollectAsync` path uses **segmented flush-concatenate-resume**: leading
streamable operations still run per chunk, the boundary operation runs once over the
concatenated result, and trailing streamable operations resume. `AsStream` is
**all-or-nothing**: the plan must be entirely streamable to chunk; otherwise it falls back
to a single merged frame. This asymmetry is deliberate — chunked `AsStream` output must
be independently processable, so any boundary operation (which needs the whole dataset)
defeats chunking.

## chunkSize semantics

- **Default:** 10,000 rows (`QueryFrame.AsStream` / `NivaraQuery<T>.AsStream`).
- **Row-oriented sources (CSV, JSON):** the target rows per chunk; honored by the reader.
- **Columnar sources (Parquet):** advisory — chunks align to native row-group boundaries.
- **Explicit always wins:** a caller-supplied `chunkSize` (via `AsStream` or
  `NivaraExecutionContext.ChunkSize`) overrides the budget-derived default below.

## Memory budget → chunk size

When no explicit chunk size is set anywhere, the streaming strategy derives one from the
memory budget (`StreamingExecutionStrategy.calculateChunkSize`):

```
chunkSize = clamp((memoryBudget / 10) / 100 bytes-per-row, 1_000, 100_000)
```

- `memoryBudget / 10` — only 10% of the budget is reserved for one in-flight chunk.
- `100 bytes-per-row` — fixed estimate for a typical columnar row.
- Clamped to [1000, 100,000] rows so chunking stays meaningful at either extreme
  (default budget 256 MB → `(25,600,000) / 100` → 256,000 → clamped to 100,000).

### Channel capacity (async pipeline)

The bounded producer/consumer channel between source and consumer is sized from the same
budget (`StreamingExecutionStrategy.CalculateChannelCapacity`):

```
capacity = clamp(memoryBudget / (chunkSize * 100 bytes-per-row), 2, 16)
```

This bounds how many chunk frames are in flight, keeping peak memory inside the budget.

### AC3 resolution (memory budget enforcement)

The bounded channel *is* the memory-budget enforcement in the query pipeline: at most
`capacity` row-chunk frames are accepted before the producer blocks on `WriteAsync`, so
peak in-flight memory stays inside the configured budget. This is verified by
`StreamingBackpressureTests` (formula bounds + an in-flight probe that asserts a fast
producer never exceeds capacity against a slow consumer).

`StreamingBufferManager.IsMemoryBudgetExceeded` (Nivara.Extensions) is an IO-layer-only
helper for chunk-buffered readers (CSV/Parquet). It is **intentionally not** wired into
`StreamingExecutionStrategy` — row-chunk frames plus a bounded channel replace byte-level
budgets in the core query pipeline.

## Example

```csharp
await using var query = Csv.ScanAsQueryFrame("telemetry.csv")
    .Filter("status", "OK")
    .Select("timestamp", "value");

await foreach (var chunk in query.AsStream(chunkSize: 50_000, ct))
{
    try
    {
        var sum = chunk.GetColumn<double>("value").Sum();
        Report(sum, chunk.RowCount);
    }
    finally
    {
        chunk.Dispose();
    }
}

// Non-streamable fallback: Sort needs the whole dataset → single frame.
// The single frame is consumer-owned too — dispose it when done.
await foreach (var frame in Csv.ScanAsQueryFrame("telemetry.csv")
                 .Sort("timestamp")
                 .AsStream())
{
    // frame holds ALL rows — same as CollectAsync()
    frame.Dispose();
}
```

## Resource management

- `QueryFrame` implements `IDisposable` and `IAsyncDisposable`; wrap scan/query chains in
  `await using` where possible.
- **The consumer owns each yielded chunk frame.** `AsStream` yields raw `NivaraFrame`s to the
  caller; the pipeline never disposes them (disposing the enumerator only disposes the
  enumerator). Dispose each chunk when you are done with it — wrap the loop body in
  `try/finally chunk.Dispose()` as shown above. This also applies to the single-frame
  fallback.
- Cancellation (via `ct`) propagates into the source reader and the producer loop; the
  channel is completed on normal exit and faulted on error.
