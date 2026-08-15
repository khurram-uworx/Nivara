# Nivara Incident Lab

> **Production incident replay and investigation, powered entirely by .NET and Nivara.**

Nivara Incident Lab is a reference application designed to showcase Nivara as a native-.NET columnar analytics engine while simultaneously serving as a design and capability test for the library.

The sample models a realistic production environment with telemetry from services, requests, deployments, errors, latency, infrastructure signals, and dependency relationships. Engineers can load a historical incident, replay it as if it were happening live, and investigate what happened using the same Nivara analytical pipeline.

The goal is not to build another dashboard demo.

The goal is to answer:

> **Could a serious .NET engineer build a production-grade analytical application on top of Nivara?**

---

## Why Incident Lab?

Nivara's vision is:

> **Feels like LINQ. Executes like a columnar engine. Born in .NET, not ported.**

The current roadmap already provides a strong foundation:

- typed, fused expression execution
- vectorized numeric kernels
- computed sort keys
- null-aware execution
- rolling and cumulative windows
- shift/lead
- rank/dense-rank/percent-rank
- chunk-capable execution
- execution diagnostics

The roadmap's remaining work includes Roslyn source generators for typed accessors and specialized query builders.

Incident Lab intentionally exercises these capabilities in one coherent, real-world workload.

Instead of creating isolated samples for `Where`, `Select`, `OrderBy`, windows, or ranking, the application makes these features cooperate.

---

# What are we building?

A self-contained production telemetry investigation system.

Given a telemetry snapshot such as:

- HTTP/request telemetry
- service metadata
- traces
- logs
- latency measurements
- status codes
- deployment events
- regions
- dependency relationships

the application reconstructs the incident timeline and helps answer questions such as:

- Which service degraded first?
- When did the incident begin?
- Which deployment correlates with the degradation?
- Which endpoints contributed most to failures?
- Which services experienced the largest increase in error rate?
- How did latency evolve over time?
- Did retries propagate the incident?
- What was the blast radius?
- Which region was affected?
- Which dependencies were involved?

The application should work with a deterministic, generated dataset so every engineer can reproduce the same scenarios.

---

# The experience

The primary workflow is:

```text
             Historical telemetry
                     │
                     ▼
             ┌───────────────┐
             │     Nivara    │
             │  columnar     │
             │  execution    │
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Timeline   Services   Dependencies
          │          │          │
          └──────────┼──────────┘
                     ▼
             Incident analysis
```

An engineer should be able to run:

```bash
docker run -p 8080:8080 \
    -v ./data:/data \
    nivara/incident-lab
```

and open the application locally.

There should also be a CLI experience for engineers who prefer a terminal.

---

# The killer demo: Incident Replay

The application should not be limited to static analysis.

A historical incident can be replayed as though telemetry were arriving in real time.

```text
14:00 ───── 14:05 ───── 14:10 ───── 14:15 ───── 14:20 ───── 14:25
                                           ▲
                                        deploy
```

As the replay advances:

```text
14:17  deployment orders-api v4.21
14:19  latency begins increasing
14:21  payment errors spike
14:23  retry volume increases
14:25  circuit breakers open
```

The dashboard evolves with the data.

This is deliberately important to the architecture.

Historical Parquet analysis and live/replay analysis should ultimately converge on the same Nivara analytical pipeline.

```text
Parquet ───────────────┐
                       │
OpenTelemetry stream ──┼──► Nivara ──► Analysis
                       │
Synthetic replay ──────┘
```

If these paths require radically different APIs, that is valuable feedback about Nivara's design.

If they naturally converge, that validates the architecture.

---

# Example production environment

The deterministic sample environment can model services such as:

```text
gateway
catalog
inventory
orders
checkout
payments
notifications
identity
```

The generated telemetry should be large enough to make the columnar engine meaningful.

For example:

```text
10M+ requests
50+ endpoints
10 regions
100+ service instances
millions of logs
thousands of traces
dozens of deployments
multiple reproducible incidents
```

The exact scale is intentionally configurable.

The important property is that the workload is large enough to exercise:

- columnar execution
- fused expressions
- sorting
- windows
- partitioning
- ranking
- chunked execution
- memory behavior
- streaming
- cancellation
- diagnostics

### Async-native IAsyncEnumerable pipeline (Phase 4 ✓)

- `CollectAsync()` / `ToListAsync()` async entry points on `NivaraQuery<T>` and `QueryFrame`
- `IAsyncEnumerable<Chunk>` streaming via `AsStream()` on `QueryFrame`
- Bounded `Channel<T>` with consumer-driven backpressure in `StreamingExecutionStrategy`
- Chunk-capable IO sources: `CsvLazySource`, `JsonLazySource`, `ParquetLazySource`
- `IAsyncDisposable` on `QueryFrame` for async resource cleanup

---

# Reproducible incident scenarios

The sample should ship with deterministic incidents rather than relying on random failures.

## Incident A — Database degradation

```text
PostgreSQL latency
       │
       ▼
orders latency
       │
       ▼
checkout latency
       │
       ▼
payment timeout
       │
       ▼
retry storm
```

Questions:

- Which service degraded first?
- How long before downstream services were affected?
- How did the retry storm amplify the incident?

---

## Incident B — Bad deployment

```text
orders-api v4.21 deployed
          │
          ▼
exception rate increases
          │
          ▼
orders failures increase
          │
          ▼
checkout failures increase
```

Questions:

- Did the deployment precede the degradation?
- Which endpoints were affected?
- Which error types increased?
- What was the time between deployment and customer impact?

---

## Incident C — Traffic spike

```text
traffic ×8
   │
   ▼
queue depth ↑
   │
   ▼
latency ↑
   │
   ▼
timeouts ↑
```

Questions:

- Which services saturated first?
- Which latency windows crossed thresholds?
- Which services recovered first?

---

## Incident D — Regional failure

```text
us-east       healthy
eu-west       healthy
ap-south      degraded
```

This scenario is particularly useful for testing partitioning, grouping and ranking by region.

---

# Nivara capabilities exercised

The sample should deliberately use Nivara for the analytical work rather than hiding the library behind conventional LINQ-to-objects code.

## Typed expressions

For example:

```csharp
var suspiciousRequests =
    requests
        .Where(r =>
            r["Timestamp"] >= incidentStart &&
            r["Timestamp"] <= incidentEnd)
        .Select(r => new
        {
            r["Service"],
            r["Endpoint"],
            r["DurationMs"],
            r["StatusCode"],

            Slow = r["DurationMs"] > 1000,
            Error = r["StatusCode"] >= 500
        });
```

This should exercise typed expression evaluation, null handling and fused execution.

---

## Computed ordering

Example analysis:

```text
Find the slowest requests, but calculate the score first.
```

The resulting query should use Nivara's fused expression path rather than materializing unnecessary intermediate columns.

---

## Rolling analysis

A service health view might calculate:

```text
Service
14:00    0.1%
14:01    0.1%
14:02    0.2%
...
14:18    0.4%
14:19    1.7%
14:20    5.2%
14:21   14.8%   <-- anomaly
```

This exercises rolling windows and explicit null semantics.

---

## Ranking

Example:

```text
Service             Error Δ       Rank
----------------------------------------
payments-api        +418%           1
orders-api          +172%           2
checkout-api         +91%           3
inventory-api        +31%           4
catalog-api           +4%           5
```

Another analysis could rank endpoints by their contribution to total incident errors.

This exercises partitioning, ordering and rank-family semantics.

---

# Query plan visibility

The application should expose Nivara's execution plan and diagnostics.

The UI should have a **Query Plan** or **Execution** view.

Example:

```text
Logical plan
──────────────────────────

Filter
  ↓
Project
  ↓
Window
  ↓
Filter
  ↓
Sort
  ↓
Rank
```

Physical execution:

```text
FilterKernel<T>
        ↓
FusedKernel<T>
        ↓
RollingKernel<T>
        ↓
RankKernel<T>
        ↓
MultiColumnComparer<T>
```

And diagnostics:

```text
Rows read:                  12,481,992
Rows returned:                  31,842
Kernels executed:                    5
Materialized columns:                7
Intermediate allocations:            0
Peak memory:                     412 MB
Elapsed:                         1.83 s
```

The exact numbers are illustrative.

The important point is that the sample makes Nivara's execution model visible rather than treating it as a black box.

---

# Architecture

Keep the application deliberately .NET-native.

```text
                    ┌───────────────────────┐
                    │     ASP.NET Core      │
                    │       Web UI          │
                    └───────────┬───────────┘
                                │
                         Incident API
                                │
                    ┌───────────▼───────────┐
                    │   Incident Engine     │
                    │                       │
                    │       Nivara          │
                    └───────────┬───────────┘
                                │
             ┌──────────────────┼─────────────────┐
             │                  │                 │
       Trace reader        Metric reader      Log reader
             │                  │                 │
             └──────────────────┼─────────────────┘
                                │
                        Parquet / Arrow
```

Suggested project structure:

```text
samples/
└── IncidentLab/
    ├── IncidentLab.App/
    ├── IncidentLab.Core/
    ├── IncidentLab.Analysis/
    ├── IncidentLab.Ingestion/
    ├── IncidentLab.Web/
    ├── IncidentLab.Cli/
    └── IncidentLab.Tests/
```

The UI is a client.

The analytical results should come from Nivara.

A core principle:

> **Every important number shown in the UI should be calculated by Nivara.**

---

# Standalone first, cloud-ready second

The first version should run with:

```bash
dotnet run
```

and:

```bash
docker run ...
```

No database.

No Kafka.

No Redis.

No Kubernetes.

No cloud account.

No mandatory external service.

One container and one dataset should be enough to demonstrate the complete experience.

Later, an optional cloud-oriented deployment can demonstrate how the same analytical engine could consume telemetry from real infrastructure.

---

# The design-test principle

Incident Lab is not only a showcase.

It is a forcing function for Nivara's API and architecture.

The rule should be:

> **When the sample requires an awkward workaround, first ask whether Nivara itself has a missing capability or an incorrect abstraction.**

Do not immediately hide the problem inside the sample.

Examples of useful discoveries:

### Window composition

If this is difficult:

```csharp
GroupBy(x => x.Service)
    .Window(...)
    .Select(...)
```

that may indicate a missing or awkward composability primitive.

### Chunk boundaries

If a rolling calculation becomes difficult across chunks:

```text
chunk 1 ──────────┐
                  ├── rolling calculation
chunk 2 ──────────┘
```

that is exactly the kind of problem the streaming architecture should solve.

### Temporal APIs

If common incident queries require overly verbose date/time expressions, that is API feedback.

### Group → aggregate → rank → filter

If the plan layer cannot represent this cleanly, that is a useful architectural gap to resolve.

### Percentiles and distributions

Incident analysis may reveal analytical operations not currently represented by the window/aggregation model.

These discoveries are successes, not failures.

---

# Phase 4 forcing function

Phase 4 is complete. Incident Lab should now use these capabilities and expose any remaining gaps.

The desired architecture is:

```text
IAsyncEnumerable<Chunk>
          │
          ▼
      Nivara plan
          │
          ▼
    fused operators
          │
          ▼
     bounded buffers
          │
          ▼
      CollectAsync()
```

Requirements met:
- ✅ cancellation flows end-to-end (ThrowIfCancellationRequested at chunk boundaries and in executeOperationsOnDataAsync)
- ✅ memory remains bounded (bounded Channel, capacity = clamp(memoryBudget / (100 * chunkSize), 2, 16))
- ✅ chunks do not require unnecessary copies (zero-copy column extraction in ParquetLazySource)
- ✅ results agree with eager execution (CollectAsync produces identical results to Collect)
- ⚠️ window semantics across chunk boundaries — non-streamable operations (window expressions, Sort, GroupBy, Join) currently cause StreamChunksAsync to fall back to full materialization rather than computing windows across chunk boundaries
- ✅ resources disposed across async boundaries (QueryFrame : IAsyncDisposable; streams via `await using`)

Gaps for Incident Lab to expose:
1. Rolling windows on chunked Parquet data fall back to single-chunk materialization (non-streamable boundary). The sample should measure whether this prevents true streaming for window-heavy queries.
2. `ParquetLazySource.Execute()` uses sync-over-async (`.GetAwaiter().GetResult()`) for the `Execute()` interface method — safe in server/CLI contexts but not in UI SynchronizationContext.
3. `Channel<T>` provides backpressure but memory budget is advisory, not a hard limit.

---

# Phase 5 forcing function

The telemetry schema is also an ideal consumer for Roslyn source generation.

For example:

```csharp
public sealed record RequestTelemetry(
    DateTimeOffset Timestamp,
    string Service,
    string Endpoint,
    double DurationMs,
    int StatusCode,
    string Region,
    string TraceId);
```

The generator can eventually produce typed schema accessors and specialized query builders.

The sample then demonstrates the .NET-specific pipeline:

```text
C# telemetry schema
        │
        ▼
Roslyn source generator
        │
        ▼
typed Nivara accessors
        │
        ▼
typed expressions
        │
        ▼
kernel IR
        │
        ▼
fused / SIMD execution
```

This is one of the strongest differentiators in the roadmap and should be demonstrated through a real application rather than a generator-only sample.

---

# Three ways to use Incident Lab

## 1. CLI

For engineers and benchmarks:

```bash
dotnet run -- incident analyze ./data/incident-421
```

Example:

```text
NIVARA INCIDENT LAB

Dataset       4.2 GB
Rows          48,921,332
Duration      2.14 s
Streamed      48,921,332 (3 chunks)
Backpressure  4 chunks in flight (bounded channel)

TOP IMPACTED SERVICES

payments      +418% errors
orders        +172%
checkout       +91%

TOP CORRELATED EVENT

14:17:32 deployment orders-api/4.21

EXECUTION

5 operators
3 fused kernels
0 per-row boxing
412 MB peak memory
```

---

## 2. Web UI

For visual exploration:

```text
Incident
   │
   ├── Timeline
   ├── Services
   ├── Endpoints
   ├── Regions
   ├── Dependencies
   ├── Errors
   ├── Deployments
   └── Query Plan
```

The UI should make the evolution of an incident easy to understand.

---

## 3. Library API

The underlying analysis should remain visible as normal .NET/Nivara code.

This is critical for the project.

The sample must teach engineers how to use Nivara, not merely demonstrate a finished application.

---

# What makes this different?

The objective is not to claim that no other system can perform incident analytics.

The differentiating goal is the combination:

> **A production-grade, LINQ-native, columnar analytical engine running entirely in the .NET ecosystem, where the same typed query model can analyze historical Parquet data and replay/live telemetry while exposing its fused execution plan.**

The intended architecture is:

```text
                  C# / LINQ
                      │
                      ▼
          Typed expression
                      │
                      ▼
              Async execution plan
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
     SIMD kernel            Fused kernel
           │                     │
           └──────────┬──────────┘
                      ▼
                Columnar memory
                      │
                      ▼
           Async/chunked data
                      │
                      ▼
          Incident analysis
```

That is the story the sample should make tangible.

---

# What we should NOT build

Avoid turning Incident Lab into a generic analytics dashboard.

Do not optimize for:

```text
Revenue
Orders
Customers
Products
```

Those are useful for a basic DataFrame sample but do not sufficiently pressure-test Nivara's current direction.

Likewise, avoid building a fake "enterprise" architecture consisting of many unnecessary services.

A modular monolith is preferable.

The complexity should be in the **data and analytical workload**, not in infrastructure ceremony.

---

# Development milestones

## Milestone 1 — Offline Incident

Target: Phase 1–3 capabilities.

```text
Parquet/CSV
    ↓
Nivara
    ↓
Incident analysis
    ↓
CLI
```

Deliver:

- deterministic dataset generator
- incident scenarios
- ingestion
- core analytical queries
- CLI output
- Nivara diagnostics

---

## Milestone 2 — Investigation UI

```text
ASP.NET Core
    ↓
Incident API
    ↓
Nivara
    ↓
Web dashboard
```

Deliver:

- timeline
- service health
- endpoint analysis
- regional analysis
- dependency view
- query-plan view

---

## Milestone 3 — Replay

```text
Historical telemetry
        ↓
IAsyncEnumerable<Chunk>
        ↓
Nivara plan
        ↓
Live dashboard
```

This becomes the primary forcing function for Phase 4.

With Phase 4 complete, `ParquetLazySource` provides row-group chunk boundaries for
historical Parquet replay, `CsvLazySource` and `JsonLazySource` provide row-based
chunking, and `AsStream()` yields processed chunks through a bounded channel with
backpressure. Cancellation flows end-to-end.

---

## Milestone 4 — Generated schema

```text
Telemetry records
        ↓
Roslyn generator
        ↓
typed Nivara frame
        ↓
specialized query
```

This becomes the primary forcing function for Phase 5.

---

## Milestone 5 — Deployment

Provide:

```bash
dotnet run
```

and:

```bash
docker run -p 8080:8080 \
    -v ./data:/data \
    nivara/incident-lab
```

The sample should be reproducible on a developer laptop without external infrastructure.

---

# Success criteria

Incident Lab is successful when a .NET engineer can:

1. Clone the repository.
2. Run the sample with a single command.
3. Generate or load a realistic incident dataset.
4. Open the dashboard.
5. Replay an incident using `IAsyncEnumerable<Chunk>` streaming.
6. Understand how the incident evolved.
7. Inspect the Nivara query/physical execution plan.
8. See meaningful execution diagnostics.
9. Read the underlying Nivara queries.
10. Run the same analysis from the CLI.
11. Run the complete application in Docker.
12. Understand why Nivara is different from LINQ-to-Objects.

More importantly, the Nivara team should be able to use the sample to discover missing capabilities.

---

# Engineering rule

The most important rule for this project:

> **The sample is allowed to break Nivara's assumptions.**

When Incident Lab exposes a limitation:

```text
Sample requirement
       │
       ▼
Is this a sample problem?
       │
       ├── Yes → fix sample
       │
       └── No
             │
             ▼
        Is Nivara's API/
        execution model missing something?
             │
             └── Yes → improve Nivara
```

Every discovered gap should be recorded as an engineering issue or design decision.

The goal is not merely to get the sample working.

The goal is to make the library better because we tried to build the sample.

---

# Final vision

The ideal end state is a demo where an engineer can say:

> "I have a few gigabytes of production telemetry, I can run this locally in a container, replay an incident, investigate it using normal C# and LINQ-style expressions, inspect the actual columnar execution plan, and the same analytical model can eventually consume an async stream."

If we can make that experience real, Incident Lab stops being a sample.

It becomes the **reference application for what Nivara is capable of.**
