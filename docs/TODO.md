# Plan: #1077 — chunk-count growth policy on concat

## Problem

`ColumnFilterHelper.ConcatenateColumns` preserves chunk boundaries when any input is chunked, and
`ChunkedColumn<T>.Create` appends every input chunk. Chunk count is additive across concat with
no coalescing. Streaming-window accumulation (last-run input is chunked) adds one chunk per window,
degrading `FindChunk` (O(log n) index) into a large fan-out and growing the prefix-offset array
unbounded even though the data is contiguous.

Arrow import/export is intentionally exempt: preserving per-column chunk counts is the point of
`PreserveChunkStructure`.

## Decision

Coalesce on concat: when concatenating chunked inputs, merge *adjacent* chunks whose combined length
is at or below the `AsStream` 10,000-row default. Arrow import/export keeps using
`FromChunkStorages` unchanged, so its chunk counts are untouched. Recorded as ADR-010.

## Proposed changes

### code
- `src/Nivara/Helpers/ColumnFilterHelper.cs` — in `concatenateColumnsTyped`, the `anyChunked` branch
  no longer calls `ChunkedColumn<T>.Create(storages)` directly. Instead it flattens each input into
  `ColumnStorage<T>` chunks, greedily packs adjacent chunks while `running + next <= 10_000`,
  allocates one merged `ColumnStorage<T>` per pack (values + OR-ed null mask), and collapses via
  `ChunkedColumn<T>.FromChunkStorages`. Add `internal const int ConcatCoalesceMaxChunkRows = 10_000`.
- `ChunkedColumn<T>.Create` and `FromChunkStorages` usage for Arrow remain untouched.

### docs
- `docs/adr/010-chunk-count-growth-policy-on-concat.md` (new) — decision and exemption rationale.
- `AGENTS.md` — ADR-010 one-line entry.
- `docs/STREAMING.md` + `GETTING-STARTED.md` — chunk-count growth policy subsection.

### tests
- Update `ChunkedConcatTests` assertions where chunks fall under the threshold.
- Add `ChunkedConcatGrowthTests` (N sequential concats bounded chunk count + correct values).

## Blast radius

- Primary: `concatenateColumnsTyped` (all `ConcatenateColumns` callers: window processor, parallel
  exec helper/strategy, ConcatenationOperation). Values are unchanged; only storage chunk topology
  changes. Existing callers do not depend on exact chunk counts except tests.
- Arrow boundary (`ArrowInterop.ConvertArrowColumnToNivaraColumnTyped`, `FromChunkStorages`) untouched; doc-only.
- `ChunkedColumnTests.Create_FlattensChunkedAndFlatInputs_PreservingBoundaries` untouched (still raw Create).

## GitHub issues log

- #1077 — this issue.
