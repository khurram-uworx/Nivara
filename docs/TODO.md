# Plan: Streaming window functions without full materialization

## Problem

`AsStream` streams leading operations (Filter, Select) per-chunk, then materializes the
full concatenated frame at window boundaries (Rolling, Rank, Cumulative, Shift) before
resuming. The window operation runs on the full column for correctness. Queries with Sort
before windows also materialize at the Sort boundary
(`samples/NivaraIncident/README.md` Limitations; measured in issue #307).

Current mechanics (`src/Nivara/Execution/StreamingExecutionStrategy.cs`):

- `PartitionAtNonStreamableOps` splits the plan into `(StreamableOps, BoundaryOp?)`
  segments at non-streamable ops and window expressions.
- First-segment boundary ops that are `SelectOperation`s containing window expressions
  stream per-chunk via `WindowOverlapBuffer` (prepend last-N rows, execute, trim first N).
- Everything else materializes: all chunks are collected, concatenated vertically, the
  boundary op runs over the full frame, trailing streamable ops resume.

### Latent bug found while planning (fix in Phase 2)

`WindowOverlapBuffer.getOverlapForWindowExpression` returns overlap `1` for cumulative
kinds. Prepending only the previous chunk's last row and computing cumulative over
`[lastRow] + chunk` yields `lastRow + prefixSums(chunk)` after trimming — **not** the true
global cumulative (which needs the running total of *all* prior rows). Correct only by
accident at chunk size 1. No test covers cumulative-vs-lazy across multiple chunks
(existing overlap tests pin `RollingSum` only). Rolling/shift are correct because their
lookback is bounded by `WindowSize`/`Periods`.

## How Polars does it (reference)

From pola-rs/polars#20947 (new streaming engine tracking), #27303, #25058, #25013:

1. **Lowering, not special-casing** — `over()` decomposes into group-by + join, both of
   which have native streaming nodes (with out-of-core hash spill).
2. **Sorted-run exploitation** — sorted group-by + merge-sorted nodes avoid hash tables;
   `LazyFrame.rolling` has a native streaming node.
3. **Bounded intermediates** — queries run larger-than-RAM as long as the *intermediate*
   (one accumulator per group) fits; sort/group-by/join spill to disk.
4. **Per-node transparent fallback** — ops without streaming impls fall back to the
   in-memory engine for just that subplan, visible in `explain(streaming=True)`.
5. Even Polars has not made `rank`/`arg_sort`/general rolling streamable — whole-partition
   buffering is the floor for rank.

Nivara analogs: Phases 1–3 below (fallback transparency → true incremental state →
per-partition pipelining). Out-of-core spill and sink terminals are deferred.

## Blast radius

| Area | Files | Change |
|---|---|---|
| Streaming strategy | `src/Nivara/Execution/StreamingExecutionStrategy.cs` | Phases 1–3: diagnostics hooks, carry-state processor, per-partition pipelining |
| Overlap buffer | `src/Nivara/Execution/WindowOverlapBuffer.cs` | Phase 2: stop returning overlap 1 for cumulative; extend detection to standalone window ops |
| New | `src/Nivara/Execution/StreamingWindowProcessor.cs` (name TBD) | Phase 2: cumulative carry-state; Phase 3: per-partition pipelined boundary execution |
| Diagnostics | `src/Nivara/Diagnostics/ExecutionDiagnostics.cs` | Phase 1: additive counters (`StreamMaterializationCount`, `RowsMaterializedAtBoundaries`) |
| Tests | `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`, new files | regression + equivalence tests per phase |
| Docs | `docs/STREAMING.md`, `samples/NivaraIncident/README.md` | contract + Limitations update |

Downstream callers: `ExecutionEngine` (strategy routing), `QueryFrame.AsStream` /
`StreamChunksAsync` consumers, `NivaraIncident` bench-stream. No public API removals;
diagnostics additions are additive. Existing tests (~60 in
`StreamingExecutionStrategyTests`) are the consistency guardrail.

---

## Phase 1 — Materialization diagnostics (transparency)

Polars-style "this node fell back" visibility. No behavior change.

1. Add to `ExecutionDiagnostics`: `int StreamMaterializationCount` and
   `long RowsMaterializedAtBoundaries` (additive, default 0).
2. In `StreamingExecutionStrategy`, wherever a boundary op executes over concatenated
   data (sync `ExecuteCore` segment loop, async `executeCoreInternalAsync`,
   `StreamChunksAsync` legacy + trailing-boundary paths):
   - increment the counters,
   - record a `PerformanceWarning` (Info severity):
     `"Streaming materialized {n} rows at boundary '{op}'"`,
   - emit `ExecutionProgress("Materializing boundary '<op>' over n rows", ...)`.
3. Overlap-streamed boundaries do NOT count as materializations.

Tests:
- Filter→Sort over chunked source: diagnostics report exactly 1 materialization with
  rows == filtered row count (sync + async + StreamChunksAsync).
- Fully streamable plan: 0 materializations.
- First-boundary window select (overlap path): 0 materializations.

Commit: `feat(execution): report window/sort boundary materializations in streaming diagnostics`

## Phase 2 — Correct cumulative streaming + standalone window ops

2a. **Fix the cumulative overlap bug with true carry-state.**
    New internal `StreamingWindowProcessor`:
    - Cumulative kinds maintain cross-chunk state: `hasValue` + running
      sum/product/min/max/count. Per chunk: run the existing cumulative kernel on the
      raw chunk column, then combine elementwise with carried state
      (`out = cumChunk + carried` for sum, `*` for product, `min`/`max` with carried
      extreme, `+` for count); null masks preserved (null positions stay null; carried
      state updates only from non-null values; empty-so-far state = identity until
      first non-null).
    - `WindowOverlapBuffer` stops claiming overlap for cumulative kinds (returns 0);
      rolling/shift keep the overlap path.
    - A boundary `SelectOperation` whose window columns mix rolling/shift and cumulative
      is handled by splitting into an overlap-executed select and a carry-executed
      select, merging result columns per chunk. Pure-cumulative selects skip the
      overlap buffer entirely.
2b. **Standalone window operations stream too.**
    `QueryFrame.RollingMean(...)` etc. add standalone `RollingOperation` /
    `CumulativeOperation` / `ShiftOperation` instances — `DetermineOverlapSize` currently
    inspects only `SelectOperation`, so these always materialize. Extend detection:
    unpartitioned (`Spec` null/empty) rolling → `WindowSize - 1`, shift(lag) → `Periods`,
    lead → 0, cumulative → carry-state path, rank → 0 (materialize).
    Partitioned (`Spec` non-empty) standalone ops wait for Phase 3.

Tests:
- Regression: CumulativeSum/Max/Min/Product/Count window selects match lazy across
  multiple chunks at several chunk sizes (would fail before the fix).
- Standalone `RollingOperation`/`CumulativeOperation`/`ShiftOperation` boundary plans
  yield >1 frames and match lazy results.
- Null-mask propagation across chunk boundaries (nulls in first/middle/last chunks).

Commits:
- `fix(execution): stream cumulative windows with true cross-chunk carry state`
- `feat(execution): stream standalone rolling/cumulative/shift boundary operations`

## Phase 3 — Per-partition pipelined windows

Target: boundary ops that are window operations **with partition keys** — standalone
`RollingOperation`/`CumulativeOperation`/`ShiftOperation`/`RankOperation` with non-empty
`Spec` (or `PartitionBy`). These materialize today even though they are partition-local.

Design (Polars' group-by lowering analog):
1. Stream chunks through leading streamable ops as today.
2. Instead of collecting all chunks: for each chunk, hash-group row indices by the
   partition key columns (reuse `GroupByOperation.CreateGroupsInternal` per chunk) and
   append each group's rows (bulk `Slice` copies) to per-key column builders
   (`Dictionary<key, List<IColumn>>` slices per column).
3. After the source drains: for each partition (one at a time), build the partition
   sub-frame, run the boundary op on it, append result columns to output builders, then
   release the partition's input buffers. Memory → O(largest partition) instead of
   O(dataset).
4. Concatenate per-partition results vertically; resume trailing streamable ops as today.
5. Unpartitioned windows degenerate to one giant partition → fall back to current
   materialization (correct; Phase 2 already covers unpartitioned rolling/shift/
   cumulative). Record a Phase 1 materialization diagnostic in that case.

Correctness argument: window ops with a `Spec` are partition-local and length-preserving
(`PartitionedWindowEngine` scatters back to original row order), so executing the op per
partition slice and concatenating vertically equals full-frame execution.

Tests:
- Partitioned Rolling/Cumulative/Shift/Rank plans over multi-chunk sources match eager
  full-frame results (multiple partition cardinalities, partitions spanning chunk
  boundaries, null partition keys).
- High-cardinality partitions (many small partitions) yield per-partition processing
  without materializing the full frame (assert via diagnostics counter staying 0 where
  applicable / frame-yield behavior).

Commit: `feat(execution): per-partition pipelined streaming for partitioned window operations`

## Phase 4 — Docs

1. `docs/STREAMING.md`: update the boundary contract — what streams per-chunk now
   (unpartitioned rolling/shift/cumulative incl. standalone ops; partitioned windows via
   per-partition pipelining), what still materializes (Sort/GroupBy/Join/Distinct
   boundaries, unpartitioned rank/broadcast aggregates), and the new diagnostics fields.
2. `samples/NivaraIncident/README.md` **Limitations section**: rewrite the
   "Window operations materialize at boundary" bullet to reflect the new behavior
   (partitioned windows and unpartitioned rolling/cumulative/shift stream; Sort-boundary
   materialization remains; diagnostics now expose it). Cross-link issue #307 resolution.

Commit: `docs: update streaming contract and incident sample limitations for window streaming`

## Verification

- `dotnet build Nivara.slnx` after each phase (ask before `dotnet test`).
- Full test suite before final review (with human confirmation).
- Equivalence property: every streaming path must produce values + null masks identical
  to `LazyExecutionStrategy` for the same plan.

## Planned commit list

1. `docs: plan streaming window functions in TODO.md`
2. Phase 1 diagnostics commit
3. Phase 2a cumulative carry-state fix commit
4. Phase 2b standalone window ops commit
5. Phase 3 per-partition pipelining commit
6. Phase 4 docs commit
7. `git rm docs/TODO.md` → `docs: remove TODO.md — plan executed`

## GitHub issues log

As each task executes, if deferred work or an out-of-plan concern is found, create an
issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it here —
do not rely on memory; compaction can lose it.

- [ ] (none yet)
