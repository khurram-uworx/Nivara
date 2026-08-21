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

`AsStream` yields **one `NivaraFrame` per source chunk** when **both** hold:

1. **Fully streamable plan** — the plan contains only streamable operations
   (`Filter`, `Select`, `Slice`, `SelectRows`) and no window expressions.
2. **Chunk-capable source** — the source reports `CanReadInChunks == true`
   (CSV, JSON, Parquet lazy sources do; in-memory frames and other in-memory sources do not).

Unpartitioned rolling, cumulative, and lag window boundaries keep this per-chunk
contract too (tier 1 below). Other plans degrade as described there: partitioned
windows yield a single drained frame; remaining boundaries yield a single frame whose
rows are identical to `CollectAsync()`.

### Boundary operations: tiered streaming

When a plan hits a boundary operation, streaming degrades in tiers rather than falling
back wholesale:

1. **Per-chunk windows** — unpartitioned rolling aggregates, cumulative aggregates, lag
   (`Shift` with non-negative periods), and lookahead windows (`Lead`, negative-period
   `Shift`) stream per chunk. Cross-chunk state is bounded: each round re-runs the
   boundary over one contiguous run of the last `max(rolling lookback, lag periods) +
   max(lead periods)` input rows plus the fresh chunk, and emits only the rows whose
   window contexts are fully satisfied by data seen so far (delayed emission for the
   lookahead kinds). The final held-back rows are flushed at drain with the operation's
   natural end-of-data semantics (nulls or fill values). Applies to window expressions
   inside `SelectOperation` as well as standalone `RollingOperation`,
   `CumulativeOperation`, and `ShiftOperation` boundaries.
2. **Partitioned windows pipelined at drain** — standalone window operations with a
   `WindowSpec` buffer their rows into per-partition lists while chunks flow, compute
   each partition once when the source drains (stable per-partition ordering by the
   spec's order keys), and restore original row order. Results arrive as a single
   frame, because partition results are only final once the source has drained.
3. **Full materialization** — everything else executes once over the concatenated data,
   exactly as before: `Sort`, `SortByExpression`, `GroupBy`, `Join`, `Distinct`,
   rank-family window expressions (`RowNumber`, `.Rank`, `.DenseRank`, `.PercentRank`),
   and broadcast aggregates (`Quantile`, `Median`).

Tiers 2 and 3 yield one frame from `AsStream`; tier 1 preserves a one-frame-per-chunk
cadence, except that plans containing lookahead windows run one chunk behind (each
yielded frame is only final once the next chunk has been read) and add one final flush
frame carrying exactly the held-back tail rows.

### Materialization diagnostics

Tier-3 boundaries are observable on `NivaraExecutionContext.ExecutionDiagnostics`:
`StreamMaterializationCount` counts how many boundary operations materialized, and
`RowsMaterializedAtBoundaries` counts the rows fed through them. Fully streamed plans
report zero materializations.

### Note: Collect vs AsStream

The `Collect`/`CollectAsync` path uses **segmented flush-concatenate-resume**: leading
streamable operations still run per chunk, tier-1 window boundaries run per chunk,
partitioned windows flush at drain, and remaining boundary operations run once over the
concatenated result before trailing streamable operations resume. `AsStream` follows
the same tiers: tier-1 plans chunk, others fall back to fewer/one merged frames. This
asymmetry is deliberate — chunked `AsStream` output must be independently processable,
so any boundary needing the whole dataset defeats chunking.

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

## Streamix bridge (`NivaraFlux`)

The `Nivara.Streamix` namespace (in `Nivara.Extensions`) bridges `QueryFrame` async streaming
to [Streamix](https://www.nuget.org/packages/Streamix)'s `IFlux<T>` ecosystem. Once wrapped in
a `Flux`, Nivara chunks gain Streamix operators: retries, backpressure, structured concurrency,
time-based windowing, observability, and ASP.NET Core streaming.

**Composition rule:** Streamix orchestrates item flow; Nivara computes inside each chunk.

### Bridge API

| Method | Signature | Description |
|--------|-----------|-------------|
| `QueryFrame.ToFlux(...)` | `→ IFlux<NivaraFrame>` | Wraps `AsStream` in `Flux.From`; optional `PipeThroughChannel` for backpressure |
| `NivaraFrame.ToFlux(...)` | `→ IFlux<NivaraFrame>` | One-shot single-chunk stream (useful for mixing with live data) |
| `QueryFrame.ToFluxRows(...)` | `→ IFlux<NivaraRow>` | Row-level bridge for live/event-oriented sources |
| `NivaraFrame.ToFluxRows(...)` | `→ IFlux<NivaraRow>` | Row-level bridge from an in-memory frame |
| `QueryFrame.ToFluxWithTimestamp(...)` | `→ IFlux<Timestamped<NivaraRow>>` | Event-time bridge for `WindowByTime` operators (lambda or column-name overload) |
| `NivaraFrame.ToFluxWithTimestamp(...)` | `→ IFlux<Timestamped<NivaraRow>>` | Event-time bridge from an in-memory frame (lambda or column-name overload) |
| `IFlux<NivaraFrame>.ToNivaraFrameAsync(...)` | `→ Task<NivaraFrame>` | Reverse terminal (frame-level) via `ConcatenateVertical` |
| `IFlux<NivaraRow>.ToNivaraFrameAsync(...)` | `→ Task<NivaraFrame>` | Reverse terminal (row-level) via schema inference |
| `IFlux<Timestamped<NivaraRow>>.ToNivaraFrameAsync(...)` | `→ Task<NivaraFrame>` | Reverse terminal for timestamped streams (collects window items into a frame) |
| `IFlux<NivaraRow>.BufferByCount(...)` | `→ IFlux<IList<NivaraRow>>` | Batch rows into fixed-size lists |
| `IFlux<NivaraRow>.BufferFrames(...)` | `→ IFlux<NivaraFrame>` | Batch rows into `NivaraFrame` instances |

All methods are in `src/Nivara.Extensions/Streamix/NivaraFlux.cs`.

### When to use the bridge vs raw `AsStream`

Use `AsStream` (raw `IAsyncEnumerable`) when:
- you control the consumption loop (`await foreach`) and don't need Streamix operators
- you want minimal dependencies (no Streamix reference)

Use `ToFlux` (Streamix bridge) when you need:
- **Backpressure** — `PipeThroughChannel(capacity, mode)` with `Wait`, `DropNewest`, `DropOldest`, `LatestOnly`, or `Fail`
- **Retries** — `.Retry(3, (attempt, ex) => delay)` for flaky sources
- **Structured concurrency** — `Flux.ScopedAsync` with fail-fast supervision
- **Time-based windowing** — `.WindowByTime(duration, slide, outOfOrderness)` for event-time processing
- **Hot-stream fan-out** — `.Publish()` / `.Replay()` / `.RefCount()` for multi-consumer scenarios
- **Observability** — `.Checkpoint()`, `.Named()`, `.Trace()`, `.Log()`
- **ASP.NET Core SSE** — `Streamix.AspNetCore` for streaming results to browsers

### Examples

**Streaming chunks with retries:**

```csharp
await Csv.ScanAsQueryFrame("telemetry.parquet")
    .Filter(ColumnExpressions.Col("cpu") > 80)
    .ToFlux(chunkSize: 50_000)
    .Named("cpu-spike-scan")
    .Retry(3, (attempt, ex) => TimeSpan.FromMilliseconds(100 * attempt))
    .Checkpoint("chunk")
    .ForEachAsync(chunk => sink.WriteAsync(chunk));
```

**Event-time windowing with `ToFluxWithTimestamp`:**

```csharp
// String-based overload (column must be DateTimeOffset)
await Csv.ScanAsQueryFrame("telemetry.csv")
    .ToFluxWithTimestamp("observed_at", chunkSize: 1000)
    .WindowByTime(TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(1))
    .FlatMap(async window =>
    {
        var frame = await window.ToNivaraFrameAsync();
        using var result = frame.AsQueryFrame()
            .RollingMean("cpu", "cpu_avg", windowSize: 10, minPeriods: 1)
            .Collect();
        var lastAvg = result.GetColumn<double>("cpu_avg").Last();
        return (Frame: result, LastRollingAvg: lastAvg);
    })
    .Where(result => result.LastRollingAvg > 90)
    .ForEachAsync(result => pager.Ping(result.Frame));
```

**Mini-batch framing for online learning:**

```csharp
var model = new Linear<float>(featureCount, 1);
var optimizer = new Adam<float>((float)1e-3);
optimizer.AddParameterGroup(model.GetParameters().Values);
var lossFn = new MSELoss<float>(Reduction.Mean);

await query
    .ToFluxRows(chunkSize: 10_000)
    .BufferFrames(batchSize: 128)
    .Map(batch =>
    {
        var features = batch.GetColumn<float>("feature").ToArray();
        var targets = batch.GetColumn<float>("target").ToArray();

        using (GradientUtils.Grad())
        {
            var input = ReverseGradTensor<float>.FromMatrix(features, features.Length, 1);
            var pred = model.Forward(input);
            var target = ReverseGradTensor<float>.FromMatrix(targets, targets.Length, 1);
            var loss = lossFn.Forward(pred, target);
            loss.Backward();
            optimizer.Step();
            optimizer.ZeroGrad();
            return loss[0];
        }
    })
    .ForEachAsync(lossValue => Console.WriteLine($"Loss: {lossValue:F4}"));
```

**Row-level reverse terminal (collect a row stream back to a frame):**

```csharp
var fluxRows = Csv.ScanAsQueryFrame("events.csv")
    .ToFluxRows(chunkSize: 5000);

using var result = await fluxRows.ToNivaraFrameAsync();
// result is the full NivaraFrame, same as CollectAsync()
```

**Reverse terminal (collect a frame stream back to a frame):**

```csharp
var flux = queryFrame.ToFlux(chunkSize: 10_000);
using var result = await flux.ToNivaraFrameAsync();
// result is the concatenated NivaraFrame, same as CollectAsync()
```

### Backpressure modes

| Mode | Behavior |
|------|----------|
| `Wait` (default) | Producer blocks when channel is full |
| `DropNewest` | Newest item is discarded when channel is full |
| `DropOldest` | Oldest buffered item is discarded when channel is full |
| `LatestOnly` | Channel keeps only the most recent item |
| `Fail` | Throws `BackpressureException` when channel is full |

### Hot-stream fan-out with `Publish` / `Replay`

Streamix's `Publish()` and `Replay()` let a single Nivara query fan out to multiple
consumers without re-executing the source:

```csharp
var shared = Csv.ScanAsQueryFrame("metrics.parquet")
    .Filter(ColumnExpressions.Col("status") == "active")
    .ToFlux(chunkSize: 50_000)
    .Publish();

shared.Subscribe(chunk => dashboard.Update(chunk));   // consumer 1
shared.Subscribe(chunk => archival.Write(chunk));      // consumer 2
shared.Connect();                                      // start the shared subscription
```

`Replay(bufferSize)` additionally replays the last N items to late subscribers:

```csharp
var replayed = query.ToFlux(chunkSize: 10_000).Replay(bufferSize: 3);
replayed.Subscribe(chunk => liveUI.Push(chunk));  // gets last 3 immediately
replayed.Connect();
```

### ASP.NET Core SSE streaming

`Streamix.AspNetCore` provides `ToSseAsync` (extension on `IFlux<T>`) and
`FluxResult<T>` (`IActionResult`) for streaming Nivara query results as Server-Sent Events.
Since `ToFlux` returns `IFlux<T>`, it plugs in directly.

Requires the `Streamix.AspNetCore` NuGet package in your web project.

**Controller pattern (`FluxResult<T>`):**

```csharp
using Nivara.Streamix;
using Streamix.AspNetCore;

[ApiController]
[Route("api/[controller]")]
public class TelemetryController : ControllerBase
{
    [HttpGet("stream")]
    public IActionResult StreamTelemetry()
    {
        var flux = Csv.ScanAsQueryFrame("telemetry.csv")
            .Filter(ColumnExpressions.Col("host") == "prod-01")
            .ToFlux(chunkSize: 1000);
        return new FluxResult<NivaraFrame>(flux);
    }
}
```

**Minimal API pattern (`ToSseAsync`):**

```csharp
using Nivara.Streamix;
using Streamix.AspNetCore;

app.MapGet("/api/telemetry/stream", async (HttpResponse response) =>
{
    await Csv.ScanAsQueryFrame("telemetry.csv")
        .Filter(ColumnExpressions.Col("host") == "prod-01")
        .ToFlux(chunkSize: 1000)
        .ToSseAsync(response);
});
```

> **Note:** Phase 3b of the Incident Lab (`samples/NivaraIncident/PHASE3B.md`) will use
> `Streamix.AspNetCore` for the SSE replay endpoint streaming live chunk results to the browser.

### Pipeline observability

Streamix's diagnostic operators compose directly with Nivara `IFlux<T>` streams.
Use them for visibility into chunk flow, latency, and pipeline health:

```csharp
await Csv.ScanAsQueryFrame("telemetry.parquet")
    .ToFlux(chunkSize: 50_000)
    .Named("telemetry-pipeline")       // appears in logs and diagnostics
    .Checkpoint("after-scan")          // logs item count + elapsed time
    .Trace("chunk-flow")               // logs OnNext/OnError/OnComplete lifecycle
    .Log()                             // logs each item's summary
    .Filter(chunk => chunk.RowCount > 0)
    .ForEachAsync(chunk => sink.WriteAsync(chunk));
```

`Checkpoint` and `Trace` use `Microsoft.Extensions.Logging.ILogger` when available,
falling back to `Console.WriteLine`. All operators are zero-cost when not subscribed
(no allocations until the pipeline is consumed).

### Known limitations

- **In-memory frames yield a single chunk.** `MemoryQuerySource.CanReadInChunks` is `false`,
  so `ToFlux()` on an in-memory frame produces a one-item stream. Use CSV/Parquet sources
  for actual multi-chunk streaming.
