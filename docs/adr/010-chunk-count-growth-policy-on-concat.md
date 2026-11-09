# ADR-010: Chunk-count growth policy on concat

**Status:** Accepted
**Date:** 2026-11-09
**Issue:** #1077

## Context

`ColumnFilterHelper.ConcatenateColumns` preserves chunk boundaries whenever an input is chunked,
and `ChunkedColumn<T>.Create` appends every input chunk. Chunk count is therefore additive across
concat with no coalescing. Streaming-window accumulation concatenates a chunked accumulator with a
slice per window, so after N windows the accumulator carries ~N chunks: `FindChunk` degrades from
O(log n) into a wide fan-out and the prefix-offset array grows unbounded even though the data is
contiguous.

## Decision

**Coalesce on concat.** In `concatenateColumnsTyped`, the chunked branch flattens each input into
`ColumnStorage<T>` chunks and greedily packs adjacent chunks whose combined length is at or below
the `AsStream` default of 10,000 rows, allocating one merged `ColumnStorage<T>` per pack (values plus
an OR-ed null mask). The policy is encoded as
`ColumnFilterHelper.ConcatCoalesceMaxChunkRows = 10_000`.

**Arrow import/export is exempt.** That path builds storage with `ChunkedColumn<T>.FromChunkStorages`
directly, which preserves chunk boundaries and never runs the concat coalescing policy; preserving
per-column chunk counts is the point of `ArrowConversionOptions.PreserveChunkStructure`. `Create`
remains a pure flatten for the same reason, so non-concat callers that need structure are unaffected.

## Consequences

- Repeated concat of small chunks collapses toward one chunk per ~10,000 rows, bounding indexing
  and the prefix-offset array.
- Chunks that are already large (source/Arrow chunks) are never split or merged, so zero-copy chunk
  boundaries survive concat.
- Storage topology of concat results changes (chunk count and, occasionally, a single coalesced
  storage); values and null masks are unchanged.
- `ChunkedColumnTests.Create_FlattensChunkedAndFlatInputs_PreservingBoundaries` still uses `Create`,
  so raw flatten behavior remains covered.
