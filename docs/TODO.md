# Plan: Streamix Bridge Integration Tests with Real Chunkable Sources (Issue #314)

## Problem

The existing `StreamixBridgeTests` only use in-memory `NivaraFrame` objects. Since
`MemoryQuerySource.CanReadInChunks` is `false`, `AsStream()` always yields the entire
frame as a single chunk. This means:

- The multi-chunk streaming path through `Flux<T>` is never exercised
- Backpressure/cancellation tests use synthetic `Flux.Range` / custom async enumerables, not the actual bridge
- The end-to-end flow (file -> QueryFrame -> AsStream -> Flux -> operators -> collect) is untested

## Proposed changes

**New file:** `tests/Nivara.Tests/Streamix/StreamixBridgeIntegrationTests.cs`

No production code changes needed — this is purely test additions.

### Helper methods

- `CreateTempDir()` — GUID-named temp directory under `%TEMP%/NivaraStreamixIntegrationTests/`
- `CreateCsvFile(dir, rowCount)` — CSV with columns `Id` (int), `Name` (string), `Value` (double)
- `CreateParquetFile(dir, rowCount, rowGroupSize)` — Parquet with same schema via `NivaraParquetWriter.WriteParquet`

### Test methods

1. **`ToFluxCsvSource_ProducesMultipleChunks`** — 50-row CSV, chunkSize=10, verify N chunks + total rows
2. **`ToFluxParquetSource_ProducesRowGroupAlignedChunks`** — 2500-row Parquet (rowGroup=1000), verify 3 chunks
3. **`ToFluxCsvSource_WithFilter_MatchesCollect`** — 30-row CSV, Filter(Id>10), chunkSize=5
4. **`BackpressureWaitMode_RealCsvSource_CompletesEndToEnd`** — 25-row CSV, channel backpressure with real source
5. **`Cancellation_PropagatesThroughRealCsvSource`** — 100-row CSV, cancel after 5 rows
6. **`ToFluxRowsCsvSource_YieldsAllRows`** — 20-row CSV, chunkSize=5, verify individual rows
7. **`ToFluxCsvSource_ToNivaraFrameAsync_RoundTrips`** — 15-row CSV, round-trip through bridge

### Blast radius

- Only touches `tests/Nivara.Tests/Streamix/` (new file)
- Tests exercise: `NivaraFlux`, `QueryFrame.AsStream`, `StreamingExecutionStrategy`, `CsvLazySource`, `ParquetLazySource`, `NivaraParquetWriter`
- No changes to any production code

## Verification

- `dotnet build Nivara.slnx` — confirm no build errors
- `dotnet test --filter "StreamixBridgeIntegrationTests"` — run only the new tests
- Ask human before running tests

## Commit plan

1. `docs: plan issue 314 in TODO.md`
2. `test: add Streamix bridge integration tests with real CSV/Parquet sources (closes #314)`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none discovered yet)
