# TODO — JSON true streaming (issue #265, branch `khurram/265`)

## Problem statement

`JsonLazySource` (src/Nivara/IO/JsonDataSource.cs) reports `CanReadInChunks = true`,
but chunking is cosmetic: `ReadChunk`/`ReadChunkAsync`/`ReadAllChunks` slice a cached
`JsonElement[]` produced by `File.ReadAllText` + `JsonSerializer.Deserialize<JsonElement[]>`
(the `lazyRecords` `Lazy`, JsonDataSource.cs:79,148). Schema inference
(JsonDataSource.cs:426) and the `JsonEagerSource` ctor (JsonDataSource.cs:693) also load
the whole file into memory. This defeats the memory purpose of streaming for large JSON
files. Issue #265 asks for true streaming reads with `Utf8JsonReader` / `JsonDocument`
parse-on-demand per chunk.

## Design

Mirror the CSV precedent (`CsvLazySource.EnsureChunkPosition`, src/Nivara.Extensions/IO/CsvDataSource.cs:295-535)
with a persistent reader whose state lives in `JsonLazySource`, plus a low-level
streaming tokenizer modeled on the MS Learn `Utf8JsonReader` partial-read pattern
(https://learn.microsoft.com/dotnet/standard/serialization/system-text-json/use-utf8jsonreader#read-from-a-stream-using-utf8jsonreader).

Key constraints discovered during planning:

- `Utf8JsonReader` is a `ref struct` — it cannot be a field. The growable-buffer/refill
  logic (carry `BytesConsumed` leftovers, reconstruct with `reader.CurrentState`) must
  live inside monolithic loops.
- `JsonDocument.ParseValue(ref reader)` needs a full record in the buffer, so each chunk
  read is **two-phase**: (1) token-walk to find the `[start,end)` byte range of the
  chunk's records, (2) seek + read that exact byte range, wrap in `[...]`, parse with
  `JsonDocument.Parse`, enumerate elements → columns. Memory stays bounded to one chunk.
- Absolute file offsets are tracked with a `baseOffset` (file offset of `buffer[0]`)
  accumulated by `BytesConsumed` on each refill; record start = `baseOffset +
  TokenStartIndex`, record end = `baseOffset + BytesConsumed` after `reader.Skip()`.
- Record boundary detection via `reader.CurrentDepth` (array mode: value tokens at
  `CurrentDepth == 1`; `IsArray=false`: the single depth-0 value). **Verify semantics
  empirically; fall back to a manual depth counter if `CurrentDepth` differs.**
- UTF-8 BOM: seek past the 3 BOM bytes on open.
- Map `JsonOptions.SerializerOptions` → `JsonReaderOptions`/`JsonDocumentOptions`
  (`ReadCommentHandling`, `AllowTrailingCommas`).
- `ReadChunk` is random-access and consumed concurrently by
  `ParallelExecutionStrategy.readSourceAsync`; a `chunkLock` serializes chunk reads so
  parallel reads stay correct (strict improvement over CSV's unguarded state).

## Blast radius

- `src/Nivara/IO/JsonDataSource.cs` — `JsonLazySource` (remove `lazyRecords`/`LoadRecords`,
  add persistent reader state + locked chunk core + streaming schema inference) and
  `JsonEagerSource` ctor validation. Public surface unchanged (`Json.ReadFrame`,
  `Json.ScanFrame`, `Json.ScanQuery<T>`, `JsonOptions`).
- `src/Nivara/IO/JsonStreamReader.cs` — NEW internal sealed helper (tokenizer/refill +
  record byte-range walker). Consumed only by `JsonLazySource`.
- Tests: `tests/Nivara.Tests/IO/LazyDataSourceTests.cs` (JSON region + eager),
  `tests/Nivara.Tests/IO/IOExceptionTests.cs`, `tests/Nivara.Tests/Execution/ExecutionIntegrationTests.cs`
  (`JsonSource_*`), `tests/Nivara.Tests/Query/NivaraQueryFeatureTests.cs` (JSON query
  tests) — all exercise the lazy/streaming paths and must stay green.
- No other production code depends on `JsonLazySource` internals (internal type).
- Core `Nivara.csproj` needs no new package: `System.Text.Json` is part of the net10.0
  BCL, and `ArrayPool<byte>.Shared` is already used elsewhere in core (AutoDiff optimizers).

## Changes

### 1. New file `src/Nivara/IO/JsonStreamReader.cs`

Internal sealed `JsonRecordStreamReader : IDisposable`:

- `FileStream` + growable rented `byte[]` (start 64 KB via `ArrayPool<byte>.Shared`,
  double on no-progress; cap growth to avoid OOM). Refill pattern per MS Learn:
  copy `BytesConsumed` leftovers to front, read more, reconstruct with
  `reader.CurrentState`; track absolute `baseOffset`.
- `OpenAt(long offset)` / BOM stripping on open.
- `FindRange(int startRecord, int count, bool isArray)` → `(long startByte, long endByte,
  int rows, bool eof)`: token-walks from the current stream position, skipping to
  `startRecord`, then locates the byte range spanning `count` records using
  `reader.Skip()` at top-level value starts.
- `Dispose()`: close stream + return rented buffer.

### 2. Rework `JsonLazySource` (JsonDataSource.cs)

- Remove `lazyRecords` + `LoadRecords` and every `File.ReadAllText`.
- Add state: `JsonRecordStreamReader? chunkReader`, `long nextRecordOffset`,
  `int recordsConsumed`, `bool eofReached`, `readonly object chunkLock`.
- `EnsureChunkPosition(chunkIndex, chunkSize)`: sequential reads continue from current
  position; backward/random access reopens the file and tokenizes-skip forward (CSV
  semantics). Returns false when EOF reached past the requested start.
- `ReadChunk`/`ReadChunkAsync` delegate to one locked core: position → find range →
  seek + read bytes into rented buffer → wrap `[ ... ]` → `JsonDocument.Parse` →
  build columns via existing `ConvertJsonValue` + `ColumnFactory.Create` → dispose →
  update `nextRecordOffset`/`recordsConsumed`/`eofReached`. Keep the existing
  `DataSourceException` wrapping ("JSON parsing error in file ...").
- `InferSchema`: read only the first `SchemaInferenceRecords` records through the same
  range reader (no whole-file load). Preserve existing `DataSourceException` messages
  (empty file / "No records found" / "Record N is not an object" / JSON parse errors).
- `Dispose`: close persistent reader (release file handle at EOF) + return buffers.
- Keep `CanReadInChunks = true`, heuristic `EstimatedRowCount`, `IsArray=false`
  single-record semantics.

### 3. `JsonEagerSource` ctor

Replace whole-file validation (`File.ReadAllText` + `JsonSerializer.Deserialize`)
with the streaming checks, preserving error messages ("JSON file is empty", "No records
found in JSON file for schema inference").

### 4. Tests — new `tests/Nivara.Tests/IO/JsonStreamingTests.cs`

- `JsonLazySource_ReadChunk_ReconstructsFullData` — mirror CSV chunk test
  (`CsvLazySource_ReadChunk_ReconstructsFullData` in AsyncStreamingTests.cs): chunks of
  a 2500-record file equal `Execute()`, incl. a backward re-read of chunk 0 after EOF.
- `JsonLazySource_ReadChunkAsync_Chunks` — async parity + out-of-range → empty chunk.
- `JsonLazySource_SchemaInference_ReadsSampleOnly` — malformed record beyond
  `SchemaInferenceRecords`: `Schema` succeeds, `Execute` throws (proves no whole-file
  read during inference).
- `JsonLazySource_FileHandleReleased_AtEndOfFile` — after streaming to EOF the file can
  be deleted (persistent handle closed), mirroring the CSV fix commit `ce4edb4`.
- `JsonSource_StreamingStrategy_ParityWithLazy` + `JsonSource_ParallelStrategy_ParityWithLazy`
  — Filter/Select over a JSON file via `Streaming`/`Parallel` vs `Lazy` strategies.
- `JsonLazySource_IsArrayFalse_SingleRecordChunking` — `IsArray=false` single object.

### 5. CHANGELOG

Add an entry under the current unreleased section referencing #265.

## Verification

- `dotnet build Nivara.slnx` (human-confirmed before running).
- `dotnet test` on targeted fixtures (human-confirmed): new `JsonStreamingTests` plus the
  existing JSON-touching suites (`LazyDataSourceTests`, `ExecutionIntegrationTests`,
  `NivaraQueryFeatureTests`, `AsyncStreamingTests`).
- Confirm the full suite is green locally before removing this file.

## Planned commits

1. `docs: plan JSON true streaming (issue #265) in TODO.md`
2. `feat: add JsonRecordStreamReader for streaming JSON array tokenization`
3. `refactor: make JsonLazySource chunk reads truly streaming with persistent reader`
4. `refactor: drop whole-file load from JsonEagerSource validation`
5. `test: cover JSON chunk parity, sample-bounded schema inference, and strategy parity`
6. `docs: changelog for JSON true streaming (issue #265)`
7. `docs: remove TODO.md -- plan executed`

## GitHub issues log

- [ ] #265 — Streaming: JsonLazySource chunking is cosmetic (whole-file load) — being
      implemented by this plan.
