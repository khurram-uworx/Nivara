# Rationed Streaming — Memory-Budget Enforcement

Phase 1 of [issue #325](https://github.com/khurram-uworx/Nivara/issues/325).

## Problem

The streaming execution strategy tracks accumulated chunk memory via
`StreamingBudgetTracker` but never enforces the limit. Source readers produce chunks
regardless of downstream capacity, and boundary operations (Sort, GroupBy, Join)
accumulate all chunks before processing. Peak memory is unbounded relative to the
configured budget.

## Goal

> The reader cannot allocate another chunk unless the memory budget has capacity for it.

Backpressure at the source — not reactive spill after the fact.

---

## Two modes

The budget behavior is **configurable** and defaults to the current advisory mode for
backward compatibility.

| Mode | Behavior | Use when |
|------|----------|----------|
| **Advisory** (default) | Budget is tracked; `StreamingBudgetTracker` emits a `PerformanceWarning` when accumulated memory exceeds the threshold. No reads are blocked. | Existing code, diagnostic profiling, scenarios where over-budget is acceptable. |
| **Enforced** | Budget is a hard cap on in-flight data. Source reads are gated — the producer cannot advance until the consumer has released enough budget. | Memory-constrained environments, containerized deployments, large-file pipelines where OOM is a real risk. |

### Configuration

Mode is set via `NivaraExecutionContext` or at strategy construction time.

```csharp
// Advisory (default — same as today)
var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
{
    MemoryBudget = 256 * 1024 * 1024,   // 256 MB
    BudgetEnforcement = BudgetEnforcement.Advisory   // default, can be omitted
};

// Enforced — source reads are gated by the budget
var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
{
    MemoryBudget = 256 * 1024 * 1024,
    BudgetEnforcement = BudgetEnforcement.Enforced
};
```

```csharp
// Fluent API on QueryFrame
await foreach (var chunk in Csv.ScanAsQueryFrame("data.csv")
    .Filter("status", "OK")
    .AsStream(enforced: true))   // shorthand: sets BudgetEnforcement.Enforced
{
    // chunk memory is guaranteed to stay within the budget
    Process(chunk);
}
```

When `BudgetEnforcement` is `Advisory`, the pipeline behaves identically to today:
`StreamingBudgetTracker` records and warns, but reads are never blocked.

When `BudgetEnforcement` is `Enforced`, the `MemoryBudget` primitive gates every
chunk allocation at the source boundary.

---

## Architecture

```
                  MemoryBudget (enforced mode)
                       │
                       ▼
┌────────┐      ┌─────────────┐      ┌──────────┐
│ Source │─────►│ Chunk Buffer│─────►│ Consumer │
│ Reader │      │  (Channel)  │      │          │
└────────┘      └─────────────┘      └──────────┘
     ▲                 │
     │                 │
     └─── pause ───────┘
        (budget full)
```

### Invariant

**In enforced mode**, at any point in time:

```
Σ (estimated bytes of all in-flight chunks) ≤ MemoryBudget
```

The source cannot produce a new chunk until the consumer has released enough budget.
No chunk is ever created "then checked" — the check happens **before** the read.

### Why bytes, not item count

A bounded `Channel<T>` with item-count capacity is insufficient:

```
Channel capacity = 10

Chunk 1 = 2 MB
Chunk 2 = 3 GB    ← exceeds budget, but channel accepted it
Chunk 3 = 1 MB
```

Chunk sizes vary dramatically (different row widths, column types, compression).
The budget must be **byte-aware** — it tracks estimated memory, not item count.

---

## MemoryBudget primitive

A byte-level async resource governor. Not a tracker — a **control point**.

```csharp
public sealed class MemoryBudget : IDisposable
{
    public MemoryBudget(long capacityBytes);

    /// <summary>
    /// Waits until the budget has room for <paramref name="requestedBytes"/>,
    /// then reserves that amount. Blocks (async) when the budget is exhausted.
    /// </summary>
    public ValueTask AcquireAsync(long requestedBytes, CancellationToken ct = default);

    /// <summary>
    /// Returns previously acquired bytes to the budget, unblocking any
    /// waiting acquirer.
    /// </summary>
    public void Release(long releasedBytes);

    /// <summary>
    /// Current estimated in-flight bytes (for diagnostics).
    /// </summary>
    public long CurrentUsage { get; }

    /// <summary>
    /// Maximum capacity.
    /// </summary>
    public long Capacity { get; }

    public void Dispose();
}
```

Internally backed by `SemaphoreSlim` for async wait, with `Interlocked` for
thread-safe usage tracking. Zero-allocation on the happy path (no `Task` allocation
when the budget has room).

### Acquire/Release contract

The producer and consumer follow a strict acquire/release protocol:

```csharp
// Producer side (inside StreamingExecutionStrategy or source adapter)
await memoryBudget.AcquireAsync(chunk.EstimatedMemoryBytes, cancellationToken);
try
{
    await channel.Writer.WriteAsync(chunk, cancellationToken);
}
catch
{
    memoryBudget.Release(chunk.EstimatedMemoryBytes);
    throw;
}

// Consumer side
var chunk = await channel.Reader.ReadAsync(cancellationToken);
try
{
    await ProcessAsync(chunk);
}
finally
{
    memoryBudget.Release(chunk.EstimatedMemoryBytes);
}
```

The producer **cannot advance past budget capacity**. The consumer **must release**
when done. Ownership is unambiguous.

### Chunk memory estimation

Chunk memory is estimated via `NivaraFrame.estimateFrameMemoryUsage()`, the same
method already used by `StreamingBudgetTracker`. This returns the estimated byte
count of the frame's column data (arrays, null masks, string references).

---

## Integration with StreamingExecutionStrategy

### Enforced mode: executeCoreInternalAsync

```
1. Create MemoryBudget(context.MemoryBudget)
2. Create bounded Channel<NivaraFrame>
3. Producer Task.Run:
   for each chunk from source.ToAsyncEnumerable():
       estimate = chunk.EstimatedMemoryBytes
       budget.AcquireAsync(estimate)          // ← blocks here if budget full
       channel.Writer.WriteAsync(chunk)
4. Consumer (main thread):
   foreach chunk from channel.Reader.ReadAllAsync():
       budget.Release(previousChunk.MemorySize)  // release previous
       chunkFrames.Add(chunk)
5. After channel drains: budget is fully released
```

The key difference from today: step 3 **blocks the producer** when the consumer
hasn't kept up. Today, the producer reads unconditionally.

### Enforced mode: StreamChunksAsync (boundary operators)

For plans with Sort/GroupBy/Join, chunks must be accumulated before the boundary
operator runs. In enforced mode:

```
1. Create MemoryBudget(context.MemoryBudget)
2. For each chunk from source:
       budget.AcquireAsync(chunkMemory)
       chunkFrames.Add(chunk)
3. ConcatenateVertical(chunkFrames)          // boundary operator needs all data
4. Dispose individual chunks
5. budget.Release(all)                       // free budget after concat
6. Run boundary operator on concatenated result
```

This is inherently **not** constant-memory — the boundary operator requires the
full dataset. But the budget still provides value: it **pauses the source** during
step 2, so peak memory is bounded to `MemoryBudget + concatenation overhead`
rather than being completely uncontrolled.

The guarantee is:

> Peak memory ≤ MemoryBudget × (1 + accounting tolerance)

where accounting tolerance covers GC overhead, temporary buffers, and the
concatenation itself (documented, typically ≤ 1.5×).

### Enforced mode: StreamChunksAsync (pure streamable)

For fully streamable plans (no boundary ops), the `IAsyncEnumerable` pull semantics
already provide natural backpressure. Enforced mode adds the `MemoryBudget` gate
as an additional safeguard:

```
foreach chunk from source.ToAsyncEnumerable():
    budget.AcquireAsync(chunkMemory)
    process streamable ops
    yield return chunkFrame
    // consumer disposes → implicit release via ownership transfer
```

The yield/await enumeration means the consumer controls the pace. The budget
is a secondary guard against unexpected chunk sizes.

### Advisory mode (current behavior, unchanged)

```
1. Create StreamingBudgetTracker(context.MemoryBudget)
2. Producer reads chunks unconditionally
3. Consumer records each chunk in the tracker
4. After all chunks: tracker.RecordWarningIfExceeded(diag)
```

No `MemoryBudget` primitive is created. No blocking. Identical to today.

---

## What stays the same

| Component | Unchanged? | Notes |
|-----------|-----------|-------|
| `StreamingBudgetTracker` | ✓ | Remains the diagnostic tracker. Emits `PerformanceWarning` in both modes. |
| Source readers (`CsvLazySource`, `ParquetLazySource`, `JsonLazySource`) | ✓ | They read `chunkSize` rows as requested. Budget gating happens at the *caller*, not inside the reader. |
| `NivaraExecutionContext.MemoryBudget` | ✓ | Still a `long` in bytes. Default 1 GB. |
| `StreamingExecutionStrategy.CalculateChannelCapacity` | ✓ | Channel capacity formula is unchanged. |
| `QueryFrame.AsStream()` | ✓ | Public API unchanged. `enforced: true` is an optional parameter. |
| Streamix bridge (`NivaraFlux`) | ✓ | Uses `AsStream` under the hood. Enforced mode flows through automatically. |

---

## Migration path

1. **Default is Advisory** — existing code is unaffected. No behavior change unless
   `BudgetEnforcement.Enforced` is explicitly set.
2. **Opt-in enforcement** — users who want hard guarantees set the flag on their
   execution context.
3. **Future: Enforced may become the default** — once the enforcement is battle-tested,
   the default can flip to `Enforced` with a deprecation period for `Advisory`.

---

## What Phase 1 does NOT cover

| Concern | Phase | Why deferred |
|---------|-------|-------------|
| Spill-to-disk for boundary operators | Phase 2 | Only makes sense after sources obey backpressure. Operators that require materialization (Sort, Join) need their own external-memory algorithms — that's a separate architectural layer. |
| External sort / hash join | Phase 2 | Requires operator-level spill abstraction. |
| Hard process-level memory limits | Future | .NET GC, runtime allocations, native decoders all contribute memory outside Nivara's control. Hard process limits need OS/container enforcement. |
| Budget-adaptive chunk sizing | Future | Today chunk size is derived once at strategy start. Could adapt at runtime based on observed chunk sizes. |

---

## Acceptance criteria

- [ ] `MemoryBudget` class with `AcquireAsync`/`Release`/`CurrentUsage`/`Capacity`
- [ ] `BudgetEnforcement` enum (`Advisory`, `Enforced`) on `NivaraExecutionContext`
- [ ] `StreamingExecutionStrategy` uses `MemoryBudget` when `Enforced`, `StreamingBudgetTracker` when `Advisory`
- [ ] Producer blocks when budget is exhausted (enforced mode)
- [ ] Consumer releases budget after processing each chunk
- [ ] `QueryFrame.AsStream(enforced: true)` flows through to strategy
- [ ] Existing `StreamingBackpressureTests` pass unchanged (advisory mode)
- [ ] New tests: `MemoryBudget` unit tests (acquire/release/blocking)
- [ ] New tests: enforced-mode integration tests proving budget is respected
- [ ] Benchmark: `bench-stream` shows reduced peak memory in enforced mode for boundary-op plans

---

## Related

- **Issue:** [#325 — Spill-to-disk for streaming execution memory budget enforcement](https://github.com/khurram-uworx/Nivara/issues/325)
- **Existing streaming docs:** [`docs/STREAMING.md`](STREAMING.md)
- **Budget tracker:** `src/Nivara/Execution/StreamingBudgetTracker.cs`
- **Strategy:** `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- **Context:** `src/Nivara/Execution/NivaraExecutionContext.cs`
- **Backpressure tests:** `tests/Nivara.Tests/Execution/StreamingBackpressureTests.cs`
