# Plan: Window Overlap Buffer for Per-Chunk Streaming (Issue #322)

## Problem

After removing the LazyExecutionStrategy fallback for window expressions (PR #324),
window operations (Rolling, Cumulative, Shift) are handled as segment boundaries in
`StreamingExecutionStrategy`. The streaming strategy accumulates all chunks, concatenates
them, then runs the window op on the full result. This is correct but defeats per-chunk
streaming — the full column must be materialized before any windowed result is produced.

For time-series workloads (like NivaraIncident analyses), data is often pre-sorted at the
source. With an overlap buffer, Rolling/Shift/Cumulative could process chunks
incrementally instead of materializing the full column at the boundary.

## Approach: Prepend-and-Trim Overlap

For each chunk, prepend `overlapSize` trailing rows from the previous chunk's processed
data to ALL columns in the input dict, run the boundary op on the extended data, then trim
the first `overlapSize` rows from every result column.

**Why this works without modifying any window kernels:**
- Rolling prefix sums/monotonic deques naturally extend over prepended data
- Shift lookback correctly references prepended rows
- PartitionedWindowEngine's group-by/sort handles prepended rows correctly
- The boundary op (SelectOperation with WindowExpression) is a black box — it just sees
  a slightly longer column

**Overlap sizes by operation type:**
| Operation | Overlap Size | Rationale |
|-----------|-------------|-----------|
| Rolling (sum/mean/max/min), windowSize=W | W - 1 | Need W rows for full window at chunk start |
| Shift (lag, periods=P) | P | Need P lookback rows |
| Shift (lead, periods<0) | 0 | Lookahead can't use trailing overlap |
| Cumulative (sum/max/min/product/count) | 1 | Need running accumulator state |
| Rank/RowNumber/DenseRank | 0 | Non-streamable, full materialization |
| Quantile/Median | 0 | Non-streamable, full materialization |

**Cumulative special case:** The simple "prepend raw row" approach needs adjustment for
cumulative ops — we carry the accumulated STATE (not raw data) as the overlap value.
The overlap row's value for the cumulative source column is set to the previous chunk's
final cumulative result, so the cumulative kernel builds on it correctly.

## Files to Create

### 1. `src/Nivara/Execution/WindowOverlapBuffer.cs`

Non-generic class managing overlap state across chunks:

```
WindowOverlapBuffer(int overlapSize)
  - tailColumns: Dictionary<string, IColumn> (last N rows per column)
  - HasData: bool
  - OverlapSize: int

  UpdateFromChunk(dict) — extract tail rows from all columns via IColumn.Slice
  PrependToChunk(dict) — concatenate overlap + chunk via ColumnFilterHelper
  TrimFirstN(dict, n) — slice first N rows off all columns via IColumn.Slice

  DetermineOverlapSize(IQueryOperation?) — inspect boundary op for window expressions,
    return max overlap across all WindowExpression nodes
  GetOverlapForWindowExpression(WindowExpression) — per-expression overlap calculation
```

The boundary op is typically a `SelectOperation` containing `WindowExpression` nodes.
`DetermineOverlapSize` walks the SelectOperation's Columns, finds WindowExpressions via
expression tree inspection, and computes the max overlap needed.

For Cumulative ops, the overlap value is a synthetic row where the cumulative source
column carries the accumulated state and other columns carry nulls (or the last value).

### 2. `src/Nivara/Execution/StreamingExecutionStrategy.cs` (modify)

Changes to `ExecuteCore`, `executeCoreInternalAsync`, and `StreamChunksAsync`:

**Before chunk loop:**
- Find the first boundary op that supports overlap (overlapSize > 0)
- If found: create `WindowOverlapBuffer`
- Track which segment index holds the overlapable boundary

**In chunk loop:**
```
for each chunk from source:
    processedData = applyLeadingStreamableOps(chunk)

    if overlapBuffer active:
        if overlapBuffer.HasData:
            extendedData = overlapBuffer.PrependToChunk(processedData)
            windowResult = overlapBoundaryOp.Execute(extendedData)
            trimmed = TrimFirstN(windowResult, overlapSize)
            chunkFrame = NivaraFrame.Create(trimmed)
        else:
            // First chunk: run boundary directly (no overlap yet)
            windowResult = overlapBoundaryOp.Execute(processedData)
            chunkFrame = NivaraFrame.Create(windowResult)

        overlapBuffer.UpdateFromChunk(processedData)  // raw data, not result
        chunkFrames.Add(chunkFrame)
    else:
        // Existing behavior: accumulate for later concatenation
        chunkFrame = NivaraFrame.Create(processedData)
        chunkFrames.Add(chunkFrame)
```

**After chunk loop:**
- If overlap was used: chunkFrames already contain window results
  - Skip the overlapable boundary in the post-loop segment iteration
  - Still run any remaining non-window boundaries (Sort, Join, Rank) and trailing ops
- If no overlap: existing behavior (concatenate + run all boundaries)

**For `StreamChunksAsync` specifically:**
- With overlap, each yielded frame includes the window result — true per-chunk streaming!
- The "no boundary" fast path (lines 372-392) should also check for overlapable windows

## Blast Radius

- **Modified:** `StreamingExecutionStrategy.cs` — all three execution paths
  (ExecuteCore, executeCoreInternalAsync, StreamChunksAsync)
- **New:** `WindowOverlapBuffer.cs` — self-contained, no external callers yet
- **Tests:** `StreamingExecutionStrategyTests.cs` — new overlap-specific tests
- **Unchanged:** Window kernel code (`WindowFunctions.cs`), `WindowOperationBase`,
  `PartitionedWindowEngine`, `FusedExpressionEvaluator` — zero modifications
- **Risk:** Medium — changes the chunk processing loop which is shared by all streaming
  execution paths. Must preserve existing behavior when no overlap is applicable.

## Verification

1. New tests verify per-chunk streaming results match lazy execution for:
   - RollingSum, RollingMean, RollingMax, RollingMin
   - Shift (lag)
   - CumulativeSum
   - Partitioned Rolling (RollingMax with PartitionBy)
2. Existing streaming tests continue to pass (no behavioral regression)
3. Window larger than chunk size edge case
4. Rank/Quantile still falls back to full materialization

## Planned Commits

1. `docs: plan window overlap buffer in TODO.md`
2. `feat: add WindowOverlapBuffer for per-chunk window streaming`
3. `feat: integrate overlap buffer into StreamingExecutionStrategy`
4. `test: add streaming window overlap tests`
5. `docs: remove TODO.md — plan executed`

## GitHub Issues Log

- [ ] #322 — Window overlap buffer for per-chunk streaming (original issue)
