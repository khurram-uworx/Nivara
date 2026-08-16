# Plan — #266 async-native CollectAsync + #267 streaming chunkSize honored

Branch: `khurram/issues` (base `main`).

## Problem

### Issue #266 — Async: CollectAsync/ToListAsync wrap the sync executor (not async-native)

- `LazyExecutionStrategy` never overrides `ExecuteCoreAsync`, so it inherits
  `ExecutionStrategyBase.ExecuteCoreAsync` = `Task.Run(() => ExecuteCore(...))`
  (`src/Nivara/Execution/ExecutionStrategyBase.cs:44-45`). The whole sync
  `QueryExecutor.Execute` pipeline runs on one thread-pool thread.
- `IQuerySource.ExecuteAsync` default = `Task.Run(() => Execute())`
  (`src/Nivara/Query/IQueryInterfaces.cs:28-29`) — even in-memory sources hop a
  thread.
- Only `ParquetLazySource` overrides `ExecuteAsync` genuinely (Parquet.Net async).
- `CsvLazySource` (`src/Nivara.Extensions/IO/CsvDataSource.cs`) has no
  `ExecuteAsync` override, and `ReadChunkAsync` calls sync `csv.Read()`.
- `JsonLazySource` (`src/Nivara/IO/JsonDataSource.cs`) has no `ExecuteAsync`
  override; `ReadChunkAsync` wraps sync `ReadChunk`. `JsonRecordStreamReader`
  (`src/Nivara/IO/JsonStreamReader.cs`) is fully sync (sync `FileStream`, sync
  `Refill`/`ReadRange`).

### Issue #267 — Streaming: memory budget is advisory; StreamingBufferManager unused; AsStream chunkSize divided by 10

- `QueryFrame.AsStream(chunkSize)` encodes the requested size via
  `MemoryBudget = chunkSize * 100` (`src/Nivara/Query/QueryFrame.cs:440`).
- `StreamingExecutionStrategy.calculateChunkSize` =
  `clamp((budget/10)/100, 1000, 100000)` (`StreamingExecutionStrategy.cs:21-27`)
  → round-trips to `chunkSize/10` (min-clamped at 1000).
- `StreamingBufferManager` (`src/Nivara.Extensions/IO/StreamingBufferManager.cs`)
  is an internal byte-buffer manager used only by Arrow/Parquet streaming IO.
  The query streaming strategy lives in dependency-free core; wiring it in is
  architecturally impossible (assembly boundary + byte-buffer vs row-chunk
  currency) → take the "document" option for that half.

## Proposed changes

### Part 1 — #266 genuinely async `CollectAsync`/`ToListAsync`

1. **`LazyExecutionStrategy.ExecuteCoreAsync` override** — real async path
   mirroring `QueryExecutor.Execute` semantics:
   validate plan, `await plan.Source.ExecuteAsync(ct)`, then per-op
   `await op.ExecuteAsync(columns, ct)`; keep diagnostics/progress/exception
   wrapping. Sync `ExecuteCore` untouched.
2. **`IQuerySource.ExecuteAsync` default** — replace `Task.Run(() => Execute(), ct)`
   with a sync-complete wrapper preserving faulted-task semantics
   (`Task.FromResult` / `Task.FromException`). Removes the unconditional thread
   hop for in-memory/non-overriding sources.
3. **`CsvLazySource` genuine async** — add `ExecuteAsync` (async
   `ReadAllChunksAsync` via `CsvReader.ReadAsync()`); fix `ReadChunkAsync` to use
   `await csv.ReadAsync()` instead of sync `csv.Read()`.
4. **JSON genuine async** — `JsonRecordStreamReader` gains async refill
   (`RefillAsync` via `FileStream.ReadAsync`, `useAsync: true`), `ReadRangeAsync`,
   and `LocateRangeAsync` (awaits only at buffer-refill boundaries). `JsonLazySource`
   gains genuine `ExecuteAsync` + genuinely-async `ReadChunkAsync`.
5. **Tests** — prove the async seam is used (source that throws on `Execute()`
   but succeeds via `ExecuteAsync()`); JSON/CSV `CollectAsync` parity; cancellation;
   rename stale `ExecuteAsync_WrapsSyncOnBackgroundThread`.

### Part 2 — #267 `AsStream` honors `chunkSize`; budget/`StreamingBufferManager` documented

1. **`NivaraExecutionContext.ChunkSize` (`int?`)** — new seam; update `Clone()`
   and `ToString()`.
2. **`QueryFrame.AsStream`** — set `context.ChunkSize = chunkSize` instead of
   `MemoryBudget = chunkSize * 100`.
3. **`StreamingExecutionStrategy`** — lines 102/170/301 use
   `context.ChunkSize ?? calculateChunkSize(context.MemoryBudget)`; `ValidatePlan`
   rejects `ChunkSize <= 0` when set. `calculateChunkSize` stays as the
   budget-derived default for the strategy-only path.
4. **Docs** — `AsStream` chunkSize is a row target honored by CSV/JSON (Parquet
   is row-group granular → advisory); `MemoryBudget`/`calculateChunkSize` derive
   chunk size only when no explicit size is set. Note `StreamingBufferManager` is
   the IO-layer byte-buffer budget manager for Arrow/Parquet streaming, not the
   query streaming strategy.
5. **Tests** — `AsStream_HonorsRequestedChunkSize` (CSV 10k rows, chunkSize 2000
   → 5 chunks), `AsStream_SmallChunkSize_NotClamped` (chunkSize 5 → ≤5-row
   chunks), `StreamingExecutionStrategy_ExplicitChunkSize_TakesPrecedenceOverBudget`.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` on `tests/Nivara.Tests` (ask human before running)
- `CHANGELOG.md` [Unreleased] entries for both fixes

## Blast radius

- **Core execution**: `ExecutionStrategyBase`, `LazyExecutionStrategy`,
  `StreamingExecutionStrategy`, `ExecutionEngine` — all strategies share the base;
  CollectAsync routes through Lazy; Parallel/Streaming async paths unaffected but
  exercise `IQuerySource.ExecuteAsync` (Parallel calls it for non-chunked sources).
- **Source seam**: `IQuerySource.ExecuteAsync` default change affects any source
  that doesn't override (in-memory `MemoryQuerySource`, test stubs). Faulted-task
  semantics preserved so `Assert.ThrowsAsync`-style callers are safe.
- **IO sources**: `CsvLazySource` (Extensions), `JsonLazySource`/`JsonRecordStreamReader`
  (core). Downstream: `ScanFrame`/`ScanJson`, Streaming/Parallel strategies, AsyncStreamingTests,
  JsonStreamingTests, IO tests.
- **Context**: `NivaraExecutionContext` used by all strategies and tests; adding
  `ChunkSize` is additive (default null → unchanged behavior).
- **AsStream**: internal on `QueryFrame`; consumers are tests only today
  (Streamix bridge is planning-only).

## Planned commit list

1. `docs: plan #266 async-native + #267 streaming chunkSize in TODO.md`
2. `refactor: LazyExecutionStrategy async-native ExecuteCoreAsync (issue #266)`
3. `refactor: drop Task.Run from default IQuerySource.ExecuteAsync (issue #266)`
4. `refactor: genuine async reads for CsvLazySource (issue #266)`
5. `refactor: genuine async reads for JsonLazySource/JsonRecordStreamReader (issue #266)`
6. `test: async seam, parity, cancellation for async-native CollectAsync (issue #266)`
7. `feat: NivaraExecutionContext.ChunkSize seam; AsStream honors chunkSize (issue #267)`
8. `feat: streaming strategy prefers explicit ChunkSize; document budget/StreamingBufferManager (issue #267)`
9. `test: AsStream chunkSize honoring + precedence (issue #267)`
10. `docs: changelog for #266 and #267`

## GitHub issues log

- [ ] #266 — Async: CollectAsync/ToListAsync wrap the sync executor (not async-native) — being implemented on this branch.
- [ ] #267 — Streaming: memory budget advisory; StreamingBufferManager unused; AsStream chunkSize divided by 10 — being implemented on this branch.

Reminder: as each task executes, if you find deferred work or a concern outside the
current plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`)
and record its number here — don't rely on memory or wait until the end of the plan.
