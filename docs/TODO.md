# TODO — Phase 4 review fixes (PR #263, branch `khurram/phase4`)

## Problem statement

PR #263 (Phase 4 — Async-First Streaming) is mergeable but NOT green and has one
confirmed performance regression. This plan fixes the blockers and hardens the
new machinery with targeted tests. Deferred items are tracked as GitHub issues
(see the log at the bottom) — they are intentionally out of scope here.

### Blockers

1. **CI red** — `build-and-test` (ubuntu) run 31910648511 fails 3 tests:
   - `ParquetLazySource_ScanQuery_PersonTypedRows` — REAL BUG: lazy Parquet
     schema reports string columns as `ReadOnlyMemory<char>` (`DataField.ClrType`)
     while data columns are `NivaraColumn<string>`; `ScanQuery<T>` /
     `TypedLinqMetadata.ValidateRowType` throws `SchemaValidationException`.
     Root cause: `ParquetLazySource.BuildSchema` falls back to `field.ClrType`
     for string fields; the eager reader normalizes via
     `TypeMapper.IsStringType` → `CreateStringColumn` → `NivaraColumn<string>`.
   - `CollectAsync_Cancellation_ThrowsOperationCanceledException` (test file
     line 71) — TEST BUG: uses sync `Assert.Throws<OperationCanceledException>`
     on the returned `Task`; the exception is captured in the task, so
     `Assert.Throws` returns null. Fix: `Assert.ThrowsAsync` in an async test.
   - `ParquetLazySource_ExecuteAsync_WithFilterAndSelect` — TEST BUG: expects
     `1000` rows; the filter `Index > 1000` over `0,2,4,…,4998` (2500 rows)
     yields rows with values 1002..4998 → 1999 rows. Fix: expect `1999`.

2. **CSV O(N²) chunk re-scan regression** — `ReadChunk`/`ReadChunkAsync` reopen
   the file and re-skip `chunkIndex * chunkSize` rows on every chunk, and
   `Execute()` routes through `ReadAllChunks()` → total re-reads ≈ N²/(2·chunkSize)
   rows (≈50× at 1M rows, ≈500× at 10M). `docs/PHASE4.md` required "persistent
   `CsvReader` + `StreamReader` state"; the implementation never delivered it.

### Non-blocker cleanup (in scope)

3. `StreamChunksAsync` (StreamingExecutionStrategy.cs) has dead `segmentsIdx`
   (never incremented) and a misleading doc comment ("Non-streamable boundary
   operations are applied after all chunks have been consumed" — they are not;
   a plan with non-streamable ops takes the single-yield fallback because
   `isSuitableForStreaming` is false). Since only streamable ops can reach the
   loop, apply `plan.Operations` per chunk and fix the comment.

### Test coverage gap (in scope)

4. The new core machinery — bounded channel + producer/consumer pipeline,
   `PartitionAtNonStreamableOps` flush-concatenate-resume, chunked CSV reads —
   has no direct tests. Add targeted parity tests (see below).

## Fixes

### 1. ParquetLazySource type fidelity (`src/Nivara.Extensions/IO/ParquetDataSource.cs`)

- `BuildSchema`: when `ResolveMetadataClrType` returns null and
  `TypeMapper.IsStringType(field.ClrType)` is true, report `typeof(string)`.
  This mirrors the eager reader's `CreateNivaraColumnFromParquetData`
  (NivaraParquetReader.cs:365) string normalization.
- Thread `reader.CustomMetadata` from `ReadRowGroupAsync` into
  `ReadRowGroupColumnsAsync` → `CreateNivaraColumnFromParquetData`, matching the
  eager `ConvertParquetToNivaraFrame` path (currently `null` is passed, so
  extended-domain CLR types like `Half`/`nint` are widened in lazy data while the
  schema claims the metadata type — same root cause as the string bug).

### 2. CSV persistent reader (`src/Nivara.Extensions/IO/CsvDataSource.cs`)

Replace per-chunk file reopen in `CsvLazySource` with persistent state:

- Fields: `StreamReader? chunkStreamReader`, `CsvReader? chunkCsvReader`,
  `int rowsConsumed`.
- `EnsureChunkPosition(chunkIndex, chunkSize, useAsync)`: if no reader exists or
  `targetRow < rowsConsumed` (backward access / re-read), reopen the file, read
  the header if configured, reset `rowsConsumed = 0`; then skip forward
  (`while (rowsConsumed < targetRow) if (!csv.Read()) return false;`). Sequential
  reads continue without re-opening → single pass.
- `ReadChunk`/`ReadChunkAsync`: call `EnsureChunkPosition`, read up to
  `chunkSize` records into the existing reader, incrementing `rowsConsumed`;
  keep cancellation checks and error wrapping as-is.
- `Dispose()`: also dispose the persistent reader pair.
- `Execute()` → `ReadAllChunks()` now runs single-pass (was the N² source).

### 3. Test fixes (`tests/Nivara.Tests/Query/AsyncStreamingTests.cs`)

- Line 71: make test `async Task` and use
  `await Assert.ThrowsAsync<OperationCanceledException>(async () => await queryFrame.CollectAsync(ct.Token))`.
- `ParquetLazySource_ExecuteAsync_WithFilterAndSelect`: expected `1000` → `1999`.

### 4. New tests (same file)

- `StreamingStrategy_ChannelPipeline_ParityWithLazy` — CSV source + Filter/Select,
  run `ExecutionEngine.ExecuteAsync` with `NivaraExecutionContext(ExecutionStrategy.Streaming)`
  vs `Lazy`; assert identical rows/values. Exercises the bounded-channel producer/
  consumer pipeline (executeCoreInternalAsync chunk-capable branch).
- `StreamingStrategy_BoundaryOperation_FlushesAndResumes` — Filter (streamable)
  then Sort (non-streamable boundary) over a chunk-capable CSV source, parity with
  Lazy. Exercises `PartitionAtNonStreamableOps` + flush-concatenate-resume.
- `CsvLazySource_ReadChunk_ReconstructsFullData` — read all chunks sequentially
  (incl. an out-of-order/backward re-read of chunk 0) and assert the concatenation
  equals `Execute()`; guards the chunk-boundary correctness the persistent reader
  preserves.

## Verification

- `dotnet build Nivara.slnx` (human-confirmed before running).
- `dotnet test` on the targeted fixtures (human-confirmed before running), in
  particular the 3 CI-failing tests, plus the 3 new tests.
- Confirm the full suite is green locally before removing this file.

## Planned commits

1. `docs: plan phase4 review fixes in TODO.md`
2. `fix: normalize ParquetLazySource schema string columns and restore extended CLR types`
3. `fix: make CsvLazySource chunk reads single-pass with persistent reader state`
4. `test: fix async cancellation and parquet filter-count test expectations`
5. `refactor: remove dead segments index from StreamChunksAsync and fix doc`
6. `test: cover streaming channel pipeline, boundary flush-resume, and CSV chunk parity`

## GitHub issues log

| Issue | Title |
|-------|-------|
| #264 | Streaming: public AsStream API and explicit non-streamable boundary behavior |
| #265 | Streaming: JsonLazySource chunking is cosmetic (whole-file load) |
| #266 | Async: CollectAsync/ToListAsync wrap the sync executor (not async-native) |
| #267 | Streaming: memory budget is advisory; StreamingBufferManager unused; AsStream chunkSize divided by 10 |
| #268 | QueryFrame: DisposeAsync disposes source, sync Dispose does not (inconsistent) |
| #269 | Streaming: sync ExecuteCore vs async ExecuteCoreAsync diverge on non-streamable plans |

## Deferred (out of scope, tracked as GitHub issues)

Deferred items are logged above in the GitHub issues log (#264–#269): public
`AsStream` + non-streamable boundary behavior, cosmetic JSON chunking, non-native
async entry points, advisory memory budget / unused `StreamingBufferManager`,
`QueryFrame` dispose inconsistency, and the sync/async streaming-strategy
divergence.
