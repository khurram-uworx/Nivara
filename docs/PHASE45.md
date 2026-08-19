# Phase 4.5 — Streamix Bridge

**Status:** implemented · **Scope:** Nivara↔Streamix async streaming bridge (`src/Nivara.Extensions`) · **Dependencies:** Phase 4 (async streaming foundation) · **Related:** [Issue #171](https://github.com/khurram-uworx/Nivara/issues/171)

---

## Rationale

Phase 4 establishes the async streaming foundation (Channels, `CollectAsync`, chunked IO sources). Phase 4.5 layers the **Streamix** reactive-stream bridge on top, turning Nivara's chunk-based `IAsyncEnumerable<NivaraFrame>` into a `Flux<NivaraFrame>` that gains retries, backpressure, structured concurrency, event-time tooling, and ASP.NET Core streaming — all from a sister project by the same author, already published on NuGet as [`Streamix`](https://www.nuget.org/packages/Streamix).

**Why Nivara.Extensions, not core `Nivara`:** AGENTS.md rule — third-party dependencies stay in Extensions. `Nivara.Extensions` already has `InternalsVisibleTo` from core (Nivara.csproj:22), so it can access internal types (`QueryFrame`, `QueryPlan`, `ExecutionEngine`, `NivaraExecutionContext`). This follows the same pattern as `Streamix.Extensions` being separate from `Streamix` core, and mirrors `EfFlux` in Streamix.Extensions.

**What Streamix provides that Phase 4 does not:**
- `Flux<T>` with explicit backpressure (`ChannelBackpressureMode`: Wait, DropNewest, DropOldest, LatestOnly, Fail)
- `Retry` / `Retry` with backoff
- `ScopedAsync` — structured concurrency with fail-fast supervision
- `Buffer` / `BufferByTime` — time-based grouping
- `Checkpoint`, `Named`, `Trace`, `Log` — observability
- ASP.NET Core streaming (`Streamix.AspNetCore`)
- `Publish` / `Replay` / `RefCount` — hot-stream fan-out

**What Streamix does NOT replace:** Nivara's fused kernel IR, typed expression engine, null-mask model, `ConcatenateVertical`, `NivaraFrame` schema, `FusedExpressionEvaluator.EvaluateChunked`. Streamix orchestrates; Nivara computes.

**Composition rule (from #171):** Streamix groups/transports data up to a chunk; Nivara runs columnar analytics inside each chunk.

---

## Current State (pre-Phase 4.5)

After Phase 4 lands:
- `QueryFrame.CollectAsync(ct)` exists — returns `Task<NivaraFrame>`
- `QueryFrame.AsStream(chunkSize, ct)` exists — returns `IAsyncEnumerable<NivaraFrame>` (one frame per chunk)
- IO sources implement `CanReadInChunks = true` and `ReadChunkAsync`
- `StreamingBufferManager` enforces memory budgets at IO layer

What's missing:
- No `Flux<NivaraFrame>` surface — users can't apply Streamix operators (Retry, ScopedAsync, Buffer, etc.) to Nivara queries
- No row-level bridge (`IFlux<NivaraRow>`) for live/event-oriented sources
- No reverse terminal (`IFlux<NivaraFrame>` → `NivaraFrame` via `ConcatenateVertical`)

---

## Design

### Package dependency

Add to `src/Nivara.Extensions/Nivara.Extensions.csproj`:

```xml
<ItemGroup>
  <PackageReference Include="Streamix" Version="1.2.2" />
</ItemGroup>
```

The `Streamix` NuGet package (v1.2.2) provides:
- `Flux.From(IAsyncEnumerable<T>)` — universal adapter from `IAsyncEnumerable` to `IFlux<T>`
- `ChannelBackpressureMode` enum (Wait, DropNewest, DropOldest, LatestOnly, Fail)
- `Flux.ScopedAsync(Func<IStreamScope, Task>, CancellationToken)` — structured concurrency
- `IFlux<T>` : `IAsyncEnumerable<T>` — all Streamix operators as extension methods

### Bridge surface (new file: `src/Nivara.Extensions/Streaming/StreamixBridge.cs`)

Mirrors the `EfFlux` pattern (Streamix.Extensions/EfFlux.cs:10-133): factory methods that wrap Nivara's async chunk enumeration in `Flux.From(...)`.

#### 1. `QueryFrame.ToFlux` — chunk-level bridge

```csharp
// Returns one NivaraFrame per chunk, streamed through Flux<T>
public static IFlux<NivaraFrame> ToFlux(
    this QueryFrame queryFrame,
    int chunkSize = 65536,
    ChannelBackpressureMode backpressureMode = ChannelBackpressureMode.Wait,
    int channelCapacity = 2,
    string? name = null)
```

Implementation:
- Cast `backpressureMode` to `int` and use `PipeThroughChannel` for the bounded channel boundary, OR
- Build directly: `Flux.From(queryFrame.AsStream(chunkSize, ct), name)` — the simplest path since `AsStream` returns `IAsyncEnumerable<NivaraFrame>` and `Flux.From(IAsyncEnumerable<T>, string)` is the universal adapter
- The `channelCapacity` / `backpressureMode` parameters map to `Flux.PipeThroughChannel(capacity, mode)` for explicit backpressure control

**Decision:** `Flux.From(AsStream(...))` as the base path; expose `PipeThroughChannel` via an optional `boundedCapacity` parameter. Streamix's `IFlux<T>` is already `IAsyncEnumerable<T>`, so the bridge is one line at the heart.

#### 2. `NivaraFrame.ToFlux` — one-shot frame bridge

```csharp
// Treats the entire frame as a single chunk
public static IFlux<NivaraFrame> ToFlux(this NivaraFrame frame, string? name = null)
    => Flux.From(AsyncEnumerable.Just(frame), name);
```

Useful for mixing live data with a static frame (e.g., join against reference data).

#### 3. Row-level bridge: `ToFluxRows` — `IFlux<NivaraRow>`

```csharp
// Emits one NivaraRow per row across all chunks
public static IFlux<NivaraRow> ToFluxRows(
    this QueryFrame queryFrame,
    int chunkSize = 65536,
    CancellationToken ct = default)
```

Implementation pattern (from issue #171's `NivaraRow` reference):
- `await foreach (var frame in queryFrame.AsStream(chunkSize, ct))`
- `for (int i = 0; i < frame.RowCount; i++) yield return new NivaraRow(frame.Columns, frame.Schema.Map, i)`
- Wrap the resulting `IAsyncEnumerable<NivaraRow>` in `Flux.From(...)`

This enables live/event-oriented sources that feed `BufferByTime`, SSE/WebSocket, etc., while Nivara handles the columnar kernels inside each buffer.

**Note:** `NivaraRow` constructor is `internal` — accessible from `Nivara.Extensions` via `InternalsVisibleTo`.

#### 4. Reverse terminal: `CollectAsync(IFlux<NivaraFrame>)`

```csharp
// Collect a Flux<NivaraFrame> back into a single NivaraFrame
// (aggregates via ConcatenateVertical — the exact same path as StreamingExecutionStrategy)
public static async Task<NivaraFrame> ToNivaraFrameAsync(
    this IFlux<NivaraFrame> stream,
    CancellationToken ct = default)
```

Implementation:
- `await stream.ToListAsync(ct)` → `List<NivaraFrame>`
- `NivaraFrameExtensions.ConcatenateVertical(list)` (NivaraFrameExtensions.cs:817)

This is the "Nivara runs analytics inside each chunk" end of the wire.

### What does NOT change

- `QueryFrame.Collect()` / `QueryFrame.CollectAsync()` — remain as-is from Phase 4
- `StreamingExecutionStrategy` — remains as-is from Phase 4
- `Nivara` core — zero Streamix references; the bridge is entirely in `Nivara.Extensions`

---

## Usage Examples (from issue #171)

### Lazy querying + streaming chunks

```csharp
// Phase 4: chunk-based async streaming (foundation)
var stream = Csv.ScanAsQueryFrame("telemetry.parquet")
    .Filter(ColumnExpressions.Col("cpu") > 80)
    .Select("host", "ts", "cpu")
    .AsStream(chunkSize: 50_000);

// Phase 4.5: wrap in Flux<T> for Streamix operators
await Csv.ScanAsQueryFrame("telemetry.parquet")
    .Filter(ColumnExpressions.Col("cpu") > 80)
    .Select("host", "ts", "cpu")
    .ToFlux(chunkSize: 50_000)
    .Named("cpu-spike-scan")
    .Retry(3, (attempt, ex) => TimeSpan.FromMilliseconds(100 * attempt))
    .Checkpoint("chunk")
    .ForEachAsync(chunk => sink.WriteAsync(chunk));
```

### Live telemetry → event-time windows → columnar analytics

```csharp
// Streamix does event-time windowing; Nivara runs analytics inside each window
await liveMetricsStream
    .MapWithTimestamp(s => s.ObservedAt)
    .WindowByTime(TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(1), TimeSpan.FromSeconds(30))
    .Map(window => window.ToNivaraFrame())  // IFlux<NivaraRow> → NivaraFrame
    .Map(frame => frame.RollingMean("cpu", "cpu_avg", windowSize: 10))  // Nivara kernel
    .Filter(f => f["cpu_avg"].Max() > 90)
    .ForEachAsync(anomaly => pager.Ping(anomaly));
```

### Online learning: chunk frames → AutoDiff → checkpoints

```csharp
// Each Flux<NivaraFrame> chunk feeds the AutoDiff training loop (AGENTS.md product direction)
var model = new Linear<float>(features: 8, output: 1);
var optimizer = new Adam<float>(lr: 1e-3f);
optimizer.AddParameterGroup(model.GetParameters().Values);

await Flux.ScopedAsync(async scope =>
{
    await liveFeatures
        .BufferByTime(TimeSpan.FromSeconds(10), maxCount: 4096)
        .Map(async batch =>
        {
            var frame = batch.ToNivaraFrame();
            var dataset = new TensorDataset<float>(frame, featureColumns, "target");
            foreach (var batchIdx in SliceShuffled(dataset.Count, batchSize: 128))
            {
                using (GradientUtils.Grad())  // inference-default: graph only inside Grad()
                {
                    var pred = model.Forward(b.Features);
                    var loss = new MSELoss<float>().Forward(pred, b.Labels, Reduction.Mean);
                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();
                }
            }
            ModelSerializer.Save(model, checkpointPath);
            return modelState;
        }, maxConcurrency: 1)
        .DoOnError(ex => alerts.ReportDrift(ex))
        .ForEachAsync(s => monitor.Report(s));
}, cancellationToken);
```

---

## Implementation Order

1. **Add Streamix NuGet reference** to `Nivara.Extensions.csproj`
2. **`QueryFrame.ToFlux`** — wrap `AsStream` in `Flux.From`; expose backpressure options
3. **`NivaraFrame.ToFlux`** — one-shot frame → `IFlux<NivaraFrame>`
4. **`QueryFrame.ToFluxRows`** — row-level bridge (`IFlux<NivaraRow>`)
5. **`IFlux<NivaraFrame>.ToNivaraFrameAsync`** — reverse terminal via `ConcatenateVertical`
6. **Tests** — Flux/stream equivalence, backpressure, error propagation

---

## Acceptance Criteria

1. `QueryFrame.ToFlux(chunkSize)` results match `Collect()` results (property tests over chunk sizes)
2. `IFlux<NivaraFrame>.ToNivaraFrameAsync()` equals `CollectAsync()` results
3. Backpressure modes enforced — `LatestOnly` drops, `Fail` throws `BackpressureException`
4. Cancellation flows through Flux → Nivara (clean `OperationCanceledException`)
5. Existing tests green; no new dependencies in `Nivara` core

---

## Dependencies

- **Phase 4** (async streaming foundation) — prerequisite
- **GitHub Issue #171** — findings + design source

## Key Files

- `src/Nivara/Query/IQueryInterfaces.cs` — `IQuerySource` async seam
- `src/Nivara/Query/QueryFrame.cs:396` — `Collect()` / Phase 4 `CollectAsync()` / `AsStream()`
- `src/Nivara/NivaraRow.cs:14` — allocation-free row struct (internal constructor)
- `src/Nivara/NivaraFrameExtensions.cs:817` — `ConcatenateVertical`
- `src/Nivara.Extensions/Nivara.Extensions.csproj` — add `Streamix` package reference
- `src/Nivara/Nivara.csproj:22` — `InternalsVisibleTo Nivara.Extensions` (enables bridge access to internals)
- `src/Streamix/src/Streamix/Flux.cs:236` — `Flux.From(IAsyncEnumerable<T>)`
- `src/Streamix/src/Streamix/Extensions/FluxExtensions.cs:1702` — `PipeThroughChannel`
- `src/Streamix/src/Streamix/Extensions/TerminalExtensions.cs:71` — `ToListAsync`
- `src/Streamix/src/Streamix.Extensions/EfFlux.cs` — template pattern for the bridge
- `src/Streamix/src/Streamix/ChannelExecution.cs:10` — `ChannelBackpressureMode`
