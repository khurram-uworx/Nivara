# Streaming (AsStream) — Behavior Reference

Public entry points for chunked, lazy processing of query frames:

| API | Location |
|---|---|
| `QueryFrame.AsStream(int chunkSize = 10000, CancellationToken ct = default)` | `src/Nivara/Query/QueryFrame.cs` |
| `NivaraQuery<T>.AsStream(...)` — passthrough | `src/Nivara/Linq/NivaraQuery.cs` |
| `NivaraFrame.AsQueryFrame()` / `NivaraQuery<T>.AsQueryFrame()` | `src/Nivara/NivaraFrame.cs` / `src/Nivara/Linq/NivaraQuery.cs` |
| `Csv.ScanAsQueryFrame(string, CsvOptions?)` | `src/Nivara.Extensions/IO/CsvExtensions.cs` |
| `Json.ScanAsQueryFrame(string, JsonOptions?)` | `src/Nivara/IO/JsonExtensions.cs` |
| `NivaraParquetReader.ScanAsQueryFrame(string, ParquetReadOptions?)` | `src/Nivara.Extensions/IO/NivaraParquetReader.cs` |

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
- **`AsStream` always supplies one,** so its 10,000-row default shadows the budget
  derivation entirely: `resolveChunkSize` is `ChunkSize ?? <derived>`, and that parameter
  default is never null. The derivation therefore only runs when `ChunkSize` is left null,
  which on the public API means `ExecutionEngine.Execute(plan, context)` with
  `Strategy = Streaming` — see §"AC3 resolution".

## Memory budget → chunk size

When no explicit chunk size is set anywhere, the streaming strategy derives one from the
memory budget (`StreamingExecutionStrategy.CalculateChunkSize`):

```
chunkSize = clamp((memoryBudget / 10) / 100 bytes-per-row, 1_000, 100_000)
```

- `memoryBudget / 10` — only 10% of the budget is reserved for one in-flight chunk.
- `100 bytes-per-row` — fixed estimate for a typical columnar row.
- Integer division truncates at each step, and the clamp is applied to the `long` result
  *before* it is narrowed to `int`. Narrowing first is unchecked, so clamping afterwards
  would wrap above ~2.1 TB and collapse the largest budgets to the 1,000-row floor.
- Only budgets between roughly 1 MB and 100 MB land strictly inside the clamp. Worked
  values, in bytes so the arithmetic can be checked by hand:

  | `MemoryBudget` | `/ 10` | `/ 100` | `chunkSize` |
  |---|---|---|---|
  | 67,108,864 — inside the clamp | 6,710,886 | 67,108 | **67,108** |
  | 1,073,741,824 — the 1 GB default | 107,374,182 | 1,073,741 | 100,000 (clamped) |

The default `MemoryBudget` is **1 GB** (`1024 * 1024 * 1024`, assigned in
`NivaraExecutionContext.cs:17`). That is above the unclamped range, so the default always
yields the 100,000-row ceiling. It is the *core query pipeline's* budget and is unrelated
to the 256 MB default owned by the IO-layer `StreamingBufferManager` — see
§"AC3 resolution" below.

### Channel capacity (async pipeline)

The bounded producer/consumer channel between source and consumer is sized from the same
budget (`StreamingExecutionStrategy.CalculateChannelCapacity`):

```
capacity = clamp(memoryBudget / (chunkSize * 100 bytes-per-row), 2, 16)
```

This bounds how many chunk frames are in flight, keeping peak memory inside the budget.

At the 1 GB default with `AsStream`'s 10,000-row chunks this resolves to
`clamp(1,073,741,824 / 1,000,000, 2, 16)` = **16**. Because that is the ceiling, capacity
does not respond to the budget at all until it falls below 16,000,000 bytes (~15.3 MiB),
and it reaches the floor of 2 at 2,000,000 bytes. `StreamingBudgetTracker` warns at twice the
budget, i.e. **2 GB** at the default.

Both of those sit at their clamp, which is worth stating plainly: on the `AsStream` path
the budget's only observable effects are these two ceilings, so a flat result is not
evidence that the budget was ignored — nor evidence that it was honoured at the value you
expected.

### AC3 resolution (memory budget enforcement)

The bounded channel *is* the memory-budget enforcement in the query pipeline: at most
`capacity` row-chunk frames are accepted before the producer blocks on `WriteAsync`, so
peak in-flight memory stays inside the configured budget. This is verified by
`StreamingBackpressureTests` (formula bounds + an in-flight probe that asserts a fast
producer never exceeds capacity against a slow consumer).

`StreamingBufferManager.IsMemoryBudgetExceeded` (Nivara.Extensions) is an IO-layer-only
helper for chunk-buffered readers (CSV/Parquet). It is **intentionally not** wired into
`StreamingExecutionStrategy` — row-chunk frames plus a bounded channel replace byte-level
budgets in the core query pipeline. Its 256 MB default therefore describes the IO layer
only; the core pipeline's default is the 1 GB in `NivaraExecutionContext.cs:17`.

One caveat on the derivation in §"Memory budget → chunk size": its input is not reachable
from the public API today. `QueryFrame.AsStream` builds its own context, takes no context
argument, and always sets `ChunkSize`; `StreamingExecutionStrategy` is internal; and the one
public route to a custom context, `ExecutionEngine.Execute(plan, context)`, materializes one
whole frame. So a caller cannot set `MemoryBudget` for a chunked stream at all — the 1 GB
default is the only budget in play there (tracked as #514). Until that lands, `AsStream`'s
10,000-row chunk size plus the two ceilings above are the whole of the reachable contract.
See `docs/ACCELERATION.md` §"Configuration" for the same analysis.

## Example

```csharp
await using var query = Csv.ScanAsQueryFrame("telemetry.csv")
    .Filter(ColumnExpressions.Col("status") == "OK")
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
await using var sorted = Csv.ScanAsQueryFrame("telemetry.csv")
    .Sort("timestamp");

await foreach (var frame in sorted.AsStream())
{
    // frame holds ALL rows — same as CollectAsync()
    frame.Dispose();
}
```

## Resource management

- `QueryFrame` implements `IDisposable` and `IAsyncDisposable`; wrap scan/query chains in
  `await using` where possible.
- **Dispose the scan you created, not just its result.** A lazy scan holds the file open.
  `using var expected = Csv.ScanAsQueryFrame(p).Collect();` binds the `using` to the returned
  `NivaraFrame` and discards the `QueryFrame`, which is the frame that actually owns the
  source. Bind the scan to a named local and dispose that:
  `await using var scan = Csv.ScanAsQueryFrame(p); var result = scan.Collect();`
- **One source, one release.** A frame derived from another shares the same source, so
  disposing any frame in a chain releases it for the whole chain and the survivors then throw
  `ObjectDisposedException`. This is the same contract `FileStream` has, and it is deliberate —
  reference counting is not possible because derived frames are temporaries inside expressions
  that are never disposed. So you do not have to dispose each intermediate: a `using` bound to
  the terminal of a chain is sufficient, and an abandoned intermediate is not itself a leak.
  Dispose the scan you created. See [LINQ.md](LINQ.md#resource-management) for the typed-query
  form of the same contract.
- **A full read masks a missing `Dispose`.** CSV and JSON close their chunk reader on their own
  once a read reaches EOF, so reading to completion releases the handle even if the scan was
  never disposed. Partial reads (a filter, a `break`, cancellation) are where an undisposed scan
  actually leaks, and Parquet holds its reader until disposal in every case. Do not use
  "the test passes" as evidence that a scan is disposed.
- **The consumer owns each yielded chunk frame.** `AsStream` yields raw `NivaraFrame`s to the
  caller; the pipeline never disposes them (disposing the enumerator only disposes the
  enumerator). Dispose each chunk when you are done with it — wrap the loop body in
  `try/finally chunk.Dispose()` as shown above. This also applies to the single-frame
  fallback.
- **Deferred operators need the frame to outlive the setup call.** `ToFlux`, `Publish` and
  `FluxResult` all defer the read until the consumer subscribes or the response body is
  enumerated. A `using` in the method that builds the pipeline would dispose the frame first,
  and by the "one source, one release" rule above every later read would throw
  `ObjectDisposedException`. Tie the frame to whatever actually owns the subscription — for a
  response, `HttpContext.Response.RegisterForDispose(frame)`; see the `Publish` and
  `FluxResult` examples below.
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
await using var cpuSpikes = NivaraParquetReader.ScanAsQueryFrame("telemetry.parquet")
    .Filter(ColumnExpressions.Col("cpu") > 80);

await cpuSpikes.ToFlux(chunkSize: 50_000)
    .Named("cpu-spike-scan")
    .Retry(3, (attempt, ex) => TimeSpan.FromMilliseconds(100 * attempt))
    .Checkpoint("chunk")
    .ForEachAsync(chunk => sink.WriteAsync(chunk));
```

**Event-time windowing with `ToFluxWithTimestamp`:**

```csharp
// String-based overload (column must be DateTimeOffset)
await using var telemetry = Csv.ScanAsQueryFrame("telemetry.csv");

await telemetry.ToFluxWithTimestamp("observed_at", chunkSize: 1000)
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
await using var events = Csv.ScanAsQueryFrame("events.csv");
var fluxRows = events.ToFluxRows(chunkSize: 5000);

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
// Publish defers every read until Connect(), and the shared subscription outlives
// this setup - so the frame cannot be disposed here. It is owned by whoever tears
// the subscription down.
var metrics = NivaraParquetReader.ScanAsQueryFrame("metrics.parquet")
    .Filter(ColumnExpressions.Col("status") == "active");

var shared = metrics.ToFlux(chunkSize: 50_000).Publish();

// Both consumers read the one shared upstream enumeration.
var dashTask = shared.ForEachAsync(chunk => dashboard.Update(chunk));    // consumer 1
var archiveTask = shared.ForEachAsync(chunk => archival.Write(chunk));   // consumer 2

// Connect() returns the handle that owns the shared subscription.
var connection = shared.Connect();

// On shutdown:
connection.Dispose();
await Task.WhenAll(dashTask, archiveTask);
metrics.Dispose();
```

`Replay(bufferSize)` additionally replays the last N items to late subscribers:

```csharp
var replayed = query.ToFlux(chunkSize: 10_000).Replay(bufferSize: 3);

// gets the last 3 items immediately, then live
var uiTask = replayed.ForEachAsync(chunk => liveUI.Push(chunk));

var replayConnection = replayed.Connect();
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
        var telemetry = Csv.ScanAsQueryFrame("telemetry.csv")
            .Filter(ColumnExpressions.Col("host") == "prod-01");

        // ToFlux defers every read until the response body is enumerated, so the frame must
        // outlive this method. Tie it to the request, not to a using — a using here would
        // dispose the source before FluxResult ever enumerates it.
        HttpContext.Response.RegisterForDispose(telemetry);

        return new FluxResult<NivaraFrame>(telemetry.ToFlux(chunkSize: 1000));
    }
}
```

**Minimal API pattern (`ToSseAsync`):**

```csharp
using Nivara.Streamix;
using Streamix.AspNetCore;

app.MapGet("/api/telemetry/stream", async (HttpResponse response) =>
{
    // ToSseAsync is awaited to completion, so the frame can be scoped to this handler.
    await using var telemetry = Csv.ScanAsQueryFrame("telemetry.csv")
        .Filter(ColumnExpressions.Col("host") == "prod-01");

    await telemetry.ToFlux(chunkSize: 1000)
        .ToSseAsync(response);
});
```

> **Note:** Phase 3b of the Incident Lab (`samples/NivaraIncident/PHASE3B.md`) will use
> `Streamix.AspNetCore` for the SSE replay endpoint streaming live chunk results to the browser.

### Pipeline observability

Streamix's diagnostic operators compose directly with Nivara `IFlux<T>` streams.
Use them for visibility into chunk flow, latency, and pipeline health:

```csharp
await using var telemetry = NivaraParquetReader.ScanAsQueryFrame("telemetry.parquet");

await telemetry.ToFlux(chunkSize: 50_000)
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
