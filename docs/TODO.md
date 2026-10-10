# Plan — #514: reachable streaming memory budget on `AsStream`

**Branch:** `khurram/514`
**Issue:** [#514 — No public knob for a streaming memory budget](https://github.com/khurram-uworx/Nivara/issues/514)

## Problem

`QueryFrame.AsStream` cannot be handed a memory budget: the streaming strategy is
internal and the public overloads build their own `NivaraExecutionContext`, so
`NivaraExecutionContext.MemoryBudget` stays at the 1 GB default and the
budget→chunk-size derivation in `StreamingExecutionStrategy.CalculateChunkSize`
(and the channel capacity in `CalculateChannelCapacity`) is unreachable from the
chunked stream path.

### Current state on `main` (important)

The capability was **already partially added** by commit `64022a646` ("cleanup",
2026-10-04), one day after the issue was filed, and never wired into tests, docs,
an ADR, or the changelog. `QueryFrame.cs` today has:

- `AsStream(NivaraExecutionContext? context = null, int? chunkSize = null, CancellationToken ct = default)` (:460)
- `AsStream(int chunkSize, CancellationToken ct = default)` (:484)
- `AsStream(int chunkSize, long memoryBudget, CancellationToken ct = default)` (:494)
- `NivaraExecutionContext.WithChunkSize(int)` (`NivaraExecutionContext.cs:119`)

Two defects were introduced with it:

1. **Silent default change.** `QueryFrame.AsStream()` (no args) now passes a null
   context and a null chunk size, so `resolveChunkSize` derives `CalculateChunkSize(1 GB)`
   = **100,000** rows, not the documented **10,000** (`STREAMING.md:87`,
   `QueryFrame.cs:448`). `NivaraQuery<T>.AsStream` still defaults to 10,000, so the
   two public entry points disagree.
2. **`AsStream(int, long memoryBudget)` cannot do what its name implies.** It forces
   `ChunkSize`, so the budget never reaches `CalculateChunkSize`; and
   `StreamChunksAsync` (the method `AsStream` calls, `StreamingExecutionStrategy.cs:501`)
   never builds a bounded channel — it consumes `plan.Source.ToAsyncEnumerable(chunkSize, ct)`
   directly. So on the `AsStream` path the budget reaches only `StreamingBudgetTracker`
   (advisory warning). `CalculateChannelCapacity` is reached only by the single-frame
   `ExecuteCoreAsync` path (`:356`). The `docs/STREAMING.md` §"AC3 resolution" claim that
   the bounded channel enforces the budget on the streaming path is inaccurate.

## Decisions (confirmed by the human)

1. **Direction:** option 1 — a context-taking `AsStream` overload; keep
   `StreamingExecutionStrategy` internal. Record as ADR-006.
2. **Default:** preserve **10,000** rows for the bare `AsStream()` call (backwards
   compatible); the budget derivation runs only when a context with `ChunkSize` left
   null is passed.
3. **Remove** the redundant `AsStream(int chunkSize, long memoryBudget)` overload.

## Proposed changes

### 1. `src/Nivara/Query/QueryFrame.cs` — reshape the surface

Replace the three current overloads with:

```csharp
// Legacy shape, unchanged: bare call stays 10,000 rows.
public IAsyncEnumerable<NivaraFrame> AsStream(int chunkSize = 10000, CancellationToken ct = default)
    => asStreamCore(null, chunkSize, ct);

// Derivation-capable: context.ChunkSize == null -> derive from context.MemoryBudget.
public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, CancellationToken ct = default)
    => asStreamCore(context, null, ct);

// Explicit override; Progress / ExecutionDiagnostics / CT ride on the context too.
public IAsyncEnumerable<NivaraFrame> AsStream(NivaraExecutionContext context, int chunkSize, CancellationToken ct = default)
    => asStreamCore(context, chunkSize, ct);
```

Private core (`asStreamCore`):

- `ObjectDisposedException.ThrowIf(handle.Released, this);`
- Build the plan + engine as today.
- **Clone** the caller's context (or start a fresh streaming context) before mutating
  `Strategy` / `ChunkSize` / `CancellationToken` / `ExecutionDiagnostics`, so a
  caller-owned context is not mutated/raced across concurrent streams.
- `streamingContext.ExecutionDiagnostics ??= new ExecutionDiagnostics();`
- Set `ChunkSize` only when an explicit `chunkSize` was supplied.
- Resolve `StreamChunksAsync` and wrap in `captureDiagnosticsOnComplete`.

Delete the two removed overloads. Overload resolution stays unambiguous:
0 args → legacy; `(ctx)` → derivation; `(ctx, n)` → explicit.

Also correct the XML doc at `QueryFrame.cs:444-453`: bare call = 10,000; derivation
runs only when a context is supplied with `ChunkSize` left null.

### 2. `tests/Nivara.Tests/Query/AsStreamBudgetTests.cs` — prove the budget arrives

- **`AsStream_ContextWithMemoryBudget_DerivesChunkSizeFromBudget`** — chunked source;
  `new NivaraExecutionContext(ExecutionStrategy.Streaming) { MemoryBudget = 2_000_000 }`
  with `ChunkSize` null; every emitted chunk except the last is
  `StreamingExecutionStrategy.CalculateChunkSize(2_000_000)` (= 2,000) rows.
- **`AsStream_BareCall_DefaultsToTenThousand`** — regression guard for the restored default.
- **`AsStream_Context_DoesNotMutateCallerContext`** — caller's context unchanged after
  full enumeration.
- No wall-clock/allocation assertions → no `[Category("Performance")]` needed.

### 3. `docs/adr/006-streaming-memory-budget-exposure.md` — record the choice

Status/Date/Context/Decision/Consequences: option 1; why the strategy stays internal;
10,000 default preserved with derivation opt-in via context; the budget bounds chunk
size on the stream path (no channel there); channel capacity remains a materializing-path
knob.

### 4. Docs + changelog

- `docs/STREAMING.md`: signature table (`:7`); §chunkSize semantics (`:87-96`); replace
  the "not reachable … tracked as #514" caveat (`:162-168`) with the working API;
  correct §"AC3 resolution" to say the bounded channel belongs to the materializing
  path and `AsStream` backpressure is pull-based.
- `docs/ACCELERATION.md`: revise the status banner/config (`:173-184`, `:208-226`,
  `:243-265`) and the table rows (`:463`, `:516`) to point at the context overload;
  leave the `BudgetEnforcement`/spill design (#325) marked unimplemented.
- `CHANGELOG.md`: the wrap-fix entry's "(#514) not settable" (`:126`) is now stale;
  add an entry for the new public overload.
- `AGENTS.md`: update the "Streaming chunk size" note to name the context overload and
  the reachable derivation.
- Gated `csharp` blocks must satisfy ADR-005 (each fence compiles alone).

### 5. Remove `docs/TODO.md` after both G2 reviews clear.

## Blast radius

- **Public API:** `Nivara.Query.QueryFrame.AsStream` — replaces one overload
  (`NivaraExecutionContext?`, `int?`) and removes `AsStream(int, long)`; adds
  `AsStream(NivaraExecutionContext, …)` and `AsStream(NivaraExecutionContext, int, …)`.
  The legacy `AsStream(int chunkSize = 10000, CancellationToken)` is unchanged.
- **All existing callers bind to the legacy `int` overload** and are unaffected:
  `NivaraQuery<T>.AsStream`/`ToObjectsAsync`, `NivaraFlux.ToFlux`/`EnumerateRows`,
  `samples/.../Ingestion.cs`, and every `AsStream(chunkSize: …)` test.
- **No callers of the removed overloads** exist outside their own definitions.
- **Tests potentially touching the surface:** `AsyncStreamingTests`,
  `QueryFrameDiagnosticsTests`, `ParquetStreamingTests`, `ScanAsQueryFrameHandleTests`,
  `NivaraQueryToObjectsAsyncTests`, `DocSnippetCompilerTests` (gated `AsStream()` block).
- **Strategy internals unchanged** (`resolveChunkSize`, `CalculateChunkSize`,
  `CalculateChannelCapacity`, `CreateBoundChannel`) — no behavior change on the
  materializing path.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` after each code change.
2. `dotnet test -c Release --filter "Category!=Performance"` (human pre-authorized this
   filter) once the test file lands; expect the new tests green and the existing
   streaming suite unchanged.
3. Confirm the gated STREAMING.md snippet still compiles (`DocSnippetCompilerTests`).

## Planned commits

1. `docs: plan 514 streaming memory budget exposure in TODO.md`
2. `fix(query): expose streaming memory budget through AsStream context overloads`
3. `test(query): assert AsStream budget reaches the chunk-size derivation`
4. `docs: record ADR-006 streaming memory-budget exposure`
5. `docs: document the reachable AsStream memory budget`
6. `docs: remove TODO.md — plan executed` (only after G2)

## GitHub issues log

- [ ] #556 — `AsStream`/`StreamChunksAsync` has no bounded channel, so `MemoryBudget`
      cannot cap in-flight rows on the chunked path (created while working on #514;
      this is the `CalculateChannelCapacity` half of #514's acceptance criterion).
- [x] #514 AC half 1 (budget → `CalculateChunkSize`) — landed in `khurram/514`; the
      §"AC3 resolution" phrase about the channel is corrected to say it belongs to the
      materializing path. No separate issue needed — folded into #556.

## Probe / harness lifecycle

Not applicable — no performance measurement or kernel probe is needed for this change.
