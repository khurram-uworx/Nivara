# Phase 3b — Web UI: The Incident Lab (Milestone 2)

**Status:** DEFERRED — follow-up after Phase 4 (performance assessment)
**Scope:** `samples/NivaraIncident.Web/` (ASP.NET Core minimal API + static client)
**Depends on:** Phase 3a complete (CLI works, all 4 analyses validated), Phase 4 complete (benchmarks done)
**Parent plan:** `samples/Incident-PLAN.md`
**Related:** `docs/PHASE3A.md`, `docs/PHASE4.md`, `samples/NivaraIncident/IDEA.md` (product spec)

---

## What Phase 3b adds

Phase 3a delivers the CLI with all analytical logic in `samples/Nivara.Samples/Incident/`.
Phase 4 uses the CLI to produce benchmarks and core gap evidence. Phase 3b then adds a web
surface that reuses the same analytical code. No duplication.

Execution sequence: **3a → 4 → 3b**.

| | Phase 3a (CLI) | Phase 3b (Web) |
|---|---|---|
| Entry point | `NivaraIncident.Cli/Program.cs` | `NivaraIncident.Web/Program.cs` |
| Analytical code | `Nivara.Samples/Incident/Analysis.cs` | Same (reused via project reference) |
| Output | Console text | Browser UI |
| Streaming | `--stream` flag, stdout | SSE / `IAsyncEnumerable<Chunk>` to browser |
| Execution diagnostics | CLI summary block | Query Plan view |
| New dependencies | None (refs Nivara.Samples) | `Microsoft.AspNetCore.App` (framework ref) |

---

## Architecture

```text
Browser
  │
  │  HTTP requests
  ▼
ASP.NET Core minimal API (NivaraIncident.Web)
  │
  ├── Static files (HTML/CSS/JS — inline or wwwroot)
  ├── Incident API endpoints (JSON)
  ├── SSE endpoint (live replay stream)
  │
  ▼
Nivara.Samples/Incident/Analysis.cs  (reused from 3a)
Nivara.Samples/Incident/Ingestion.cs (reused from 3a)
  │
  ▼
Nivara engine (QueryFrame, execution, diagnostics)
```

**Key principle from the IDEA:** *Every important number shown in the UI should be
calculated by Nivara.* No pre-computed summaries, no LINQ-to-objects shortcuts.

---

## Prerequisites (from 3a)

| What | Why |
|------|-----|
| `Nivara.Samples/Incident/Analysis.cs` | All 4 analyses (A–D) + top-k + computed ordering |
| `Nivara.Samples/Incident/Ingestion.cs` | `LoadParquet`, `LoadCsv`, `StreamChunks` |
| `Nivara.Samples/Incident/DatasetGenerator.cs` | Generate datasets for demo |
| `Nivara.Samples/Incident/Schema.cs` | Record types for JSON serialization |
| `QueryFrame.LastExecutionDiagnostics` | Query Plan + diagnostics view |
| `QueryFrame.ExplainPlan()` / `GetDiagnosticInfo(mode)` | Logical plan rendering |
| `QueryFrame.AsStream(chunkSize, ct)` | SSE streaming replay |

---

## Sub-steps

### 3b.1 — Project scaffolding

**Files:**
- `samples/NivaraIncident.Web/NivaraIncident.Web.csproj` (new)
- `samples/NivaraIncident.Web/Program.cs` (new, skeleton)
- `Nivara.slnx` (add project)

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\Nivara.Samples\Nivara.Samples.csproj" />
  </ItemGroup>
</Project>
```

Uses the `Microsoft.NET.Sdk.Web` SDK — no extra NuGet packages needed (ASP.NET Core is
the framework).

**Validation:** `dotnet run --project samples/NivaraIncident.Web` starts and responds on
`localhost:5000`.

---

### 3b.2 — API endpoints

**File:** `samples/NivaraIncident.Web/Program.cs` (or split into endpoint files if >300 lines)

Minimal API endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/datasets` | List available generated datasets |
| `GET` | `/api/dataset/{id}/summary` | Dataset metadata (row count, file size, scenarios) |
| `GET` | `/api/dataset/{id}/analyze?scenario=X` | Run analysis for scenario, return JSON results |
| `GET` | `/api/dataset/{id}/diagnostics?scenario=X` | Execution diagnostics (rows, kernels, memory, elapsed) |
| `GET` | `/api/dataset/{id}/plan?scenario=X` | ExplainPlan / GetDiagnosticInfo output |
| `GET` | `/api/dataset/{id}/replay?scenario=X&chunkSize=N` | SSE stream: `text/event-stream` of chunk results |

Each endpoint calls into `Nivara.Samples/Incident/` code. No analytical logic lives in
the web project — it's a thin adapter.

**SSE replay endpoint:**
```csharp
app.MapGet("/api/dataset/{id}/replay", async (string id, string scenario, int chunkSize) =>
{
    // Returns text/event-stream
    // Each event: JSON-serialized chunk result + diagnostics snapshot
    // Uses IAsyncEnumerable from Ingestion.StreamChunks
});
```

The SSE format sends one event per chunk with:
- Chunk index and row count
- Per-chunk analysis snapshot (rolling error rates, current top-k, etc.)
- Cumulative diagnostics (total rows processed, elapsed time, peak memory)

**Validation:**
- `curl /api/dataset/{id}/analyze?scenario=A` returns valid JSON with analysis results.
- `curl /api/dataset/{id}/diagnostics?scenario=A` returns diagnostics.
- SSE endpoint streams events without buffering the entire dataset.

---

### 3b.3 — Static client (HTML/CSS/JS)

**Files:**
- `samples/NivaraIncident.Web/wwwroot/index.html` (new)
- `samples/NivaraIncident.Web/wwwroot/app.js` (new)
- `samples/NivaraIncident.Web/wwwroot/style.css` (new)

No frontend framework. Plain HTML + vanilla JS + CSS. The IDEA says the UI should be
deliberately simple — the point is the Nivara analytical pipeline, not frontend polish.

**Layout** (from the IDEA §"Web UI"):

```
┌─────────────────────────────────────────────────┐
│  Nivara Incident Lab                            │
├──────────┬──────────────────────────────────────┤
│ Sidebar  │  Main content area                   │
│          │                                      │
│ Timeline │  [depends on selected view]          │
│ Services │                                      │
│ Endpoints│                                      │
│ Regions  │                                      │
│ Deps     │                                      │
│ Errors   │                                      │
│ Deploys  │                                      │
│ Plan     │                                      │
└──────────┴──────────────────────────────────────┘
```

**Views:**

| View | Data source | Visualization |
|------|------------|---------------|
| **Timeline** | Analysis A (degradation ordering) | Horizontal timeline with events plotted |
| **Services** | Analysis B (deployment correlation) | Table: service, error Δ, rank, correlated deploy |
| **Endpoints** | Analysis B breakdown | Table: endpoint, error count, status code distribution |
| **Regions** | Analysis D (regional partitioning) | Table or grouped cards per region |
| **Dependencies** | Service dependency graph | Simple adjacency list or minimal SVG graph |
| **Errors** | Analysis A+C combined | Error type distribution, retry amplification |
| **Deployments** | Analysis B events | Deployment timeline with impact markers |
| **Query Plan** | `ExplainPlan()` + diagnostics | Logical plan → physical kernels → diagnostics panel |

**Replay mode:** when the user selects a dataset + scenario and clicks "Replay", the client
opens an `EventSource` to the SSE endpoint and updates the UI incrementally as chunks arrive.
The timeline advances, service health tables update, and the diagnostics panel shows live
memory/chunk counters.

**Validation:**
- All 8 views render without errors.
- Replay mode shows incremental updates as SSE events arrive.
- Query Plan view shows the logical plan, physical kernel chain, and execution diagnostics.

---

### 3b.4 — Replay streaming (SSE to browser)

**File:** `samples/NivaraIncident.Web/wwwroot/app.js` (EventSource handling)

The SSE replay is the "killer demo" from the IDEA:

```text
Historical telemetry
        │
        ▼
IAsyncEnumerable<Chunk>  (Ingestion.StreamChunks)
        │
        ▼
ASP.NET Core SSE endpoint
        │
        ▼
Browser EventSource
        │
        ▼
Live dashboard update
```

The client-side JS:
1. Opens `EventSource` to `/api/dataset/{id}/replay?scenario=X&chunkSize=N`.
2. On each `message` event, parses the chunk result JSON.
3. Updates the timeline, service health, diagnostics panels incrementally.
4. Shows chunk progress (chunk N of M, rows processed, elapsed).
5. Handles `error` event for disconnection / cancellation.

**Backpressure:** the bounded `Channel<T>` in `QueryFrame.AsStream` handles server-side
backpressure. The SSE protocol doesn't have explicit backpressure, but the browser processes
events as fast as they arrive. For very fast chunk reads, consider a server-side delay
(50–100ms between events) to make the replay visually parseable.

**Validation:**
- Replay starts, streams chunks, and completes without buffering the entire dataset.
- Browser shows incremental updates (not a single dump at the end).
- Cancel button stops the SSE stream (server-side cancellation via `CancellationToken`).
- Peak memory stays bounded (same as CLI streaming path).

---

### 3b.5 — Execution diagnostics + query plan view

**File:** `samples/NivaraIncident.Web/wwwroot/app.js` (Plan view rendering)

The Query Plan view renders three sections:

**1. Logical plan** — from `QueryFrame.ExplainPlan()`:
```text
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

**2. Physical execution** — kernel chain:
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

**3. Diagnostics** — from `QueryFrame.GetExecutionDiagnostics()`:
```text
Rows read:            12,481,992
Rows returned:           31,842
Kernels executed:            5
Materialized columns:        7
Peak memory:             412 MB
Elapsed:                 1.83 s
```

Render as a styled panel (monospace font, no external dependencies). The diagnostics
update in real-time during replay.

**Validation:**
- Logical plan renders correctly for each analysis.
- Diagnostics match CLI output for the same dataset/scenario.
- During replay, diagnostics update incrementally.

---

### 3b.6 — Tests

**File:** `tests/Nivara.Tests/Incident/WebApiTests.cs` (new)

NUnit tests for the API endpoints (spin up a `WebApplicationFactory` in-process):

- `GET /api/datasets` returns 200 with dataset list.
- `GET /api/dataset/{id}/analyze?scenario=A` returns 200 with valid JSON.
- `GET /api/dataset/{id}/diagnostics?scenario=A` returns diagnostics.
- `GET /api/dataset/{id}/plan?scenario=A` returns plan output.
- SSE endpoint streams events (read first few events, verify format).
- Error cases: missing dataset → 404, invalid scenario → 400.

**Validation:** all web API tests pass.

---

### 3b.7 — Wire, build, end-to-end validation

- Add `NivaraIncident.Web.csproj` to `Nivara.slnx`.
- `dotnet build Nivara.slnx` clean.
- Run web server, open browser, verify all 8 views.
- Run replay end-to-end: generate → open replay → watch timeline advance.
- Verify Query Plan view shows correct plan/diagnostics for all 4 scenarios.
- Compare web diagnostics with CLI diagnostics (must match).
- Update `samples/NivaraIncident/README.md` with Web UI section.

---

## Definition of done (Phase 3b)

- [ ] `dotnet build Nivara.slnx` clean.
- [ ] Web server starts and serves the client.
- [ ] All 8 views render correctly for all 4 scenarios.
- [ ] Replay mode streams chunks via SSE with incremental UI updates.
- [ ] Query Plan view shows logical plan + physical kernels + diagnostics.
- [ ] Diagnostics match CLI output (convergence).
- [ ] All web API tests pass; no regressions in existing suite.
- [ ] `samples/NivaraIncident/README.md` updated with Web UI usage.

---

## Execution notes

- Same repo conventions as 3a: ask before `dotnet test`, build after each sub-step,
  separate commits per sub-step.
- No frontend build toolchain — vanilla HTML/JS/CSS only.
- The web project references `Nivara.Samples` (which has all the analytical code).
  The web project itself should contain zero analytical logic.
- If ASP.NET Core introduces gaps (e.g. SSE streaming quirks, `IAsyncEnumerable`
  serialization), record them as core issues per the escalation rule.
