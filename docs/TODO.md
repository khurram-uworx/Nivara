# Plan: #358 — StreamingWindowProcessor per-run re-materialization can overflow int-family CumulativeProduct

**Decision (confirmed with user):** Direction 1 — preserve per-chunk streaming for cumulative
product by NOT re-materializing the carried cumulative windows over the mid-run boundary op.

## Problem

`StreamingWindowProcessor.ProcessChunk` (line 240) re-executes the whole boundary op over
`run` = carried context + fresh chunk. For a cumulative **product** window over int-family data,
that re-execution computes the product starting from the **run's first value** (never the
dataset's row-0 seed). From chunk 2 onward the run starts mid-column, so the per-run product of
large values trips the *checked* `long` accumulator in `WindowFunctions.cumulativeScan`
(`WindowFunctions.cs` lines ~402 / ~452), the guard added for #248. The exception is thrown
*before* the correct carry-computed column replaces the (discarded) run value at
`StreamingWindowProcessor.cs:254`.

Key facts that make the fix safe:

- **Carry-slot cumulative outputs are never used.** In `ProcessChunk`, `emitted[slot.OutputName]`
  is always overwritten by `carryColumnForEmission` (line 254). In `Flush`, carry outputs come from
  `buildDeferredColumn(slot)` (line 285), never from `lastRunResult`. So the boundary op's value for
  a carry window column is pure wasted computation — and the source of the overflow — for *every*
  cumulative kind (Sum/Max/Min/Product/Count). Sum over int-family uses the same checked `long`
  path and can overflow in principle too.
- **The carry path itself is correct.** `ComputeCarryColumn` (lines 508-513) seeds a `[state]`
  column and computes the cumulative over the *fresh chunk only*, then carries `state` forward via
  `UpdateCarryState`. For well-defined data (leading zero, or bounded carried product) it matches
  the lazy engine exactly and cannot overflow beyond the true product.
- The first chunk is unaffected because `contextLength == 0`, so `run` starts at dataset row 0. The
  bug only manifests from chunk 2 onward.

## Files

- `src/Nivara/Execution/StreamingWindowProcessor.cs` — carries state, boundary re-materialization.
- `src/Nivara/Operations/SelectOperation.cs` — boundary op for select-based windows.
- `src/Nivara/Operations/WindowOperations.cs` — `CumulativeOperation` (standalone boundary op).
- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs` — regression tests.

## Core change — `StreamingWindowProcessor.cs`

Introduce a **reduced boundary op** that evaluates only the boundary columns NOT owned by a
carry slot, so the per-run materialization no longer computes any carried cumulative window.
The resultant emitted dictionary still gets every output: non-carry columns come from the reduced
boundary op, carry columns come from `carryColumnForEmission` (overwriting the slot's entry, as
today).

### New/updated fields

```csharp
// Per-run op evaluating only the non-carry boundary columns. Null when every boundary
// column is a carry slot (no boundary re-computation needed at all, e.g. a standalone
// CumulativeOperation, or a SelectOperation whose columns are all cumulative windows).
readonly IQueryOperation? reRunBoundaryOp;

// Output names owned by carry slots; carry columns overwrite these in the emitted dict.
readonly HashSet<string> carryOutputNames = new(StringComparer.OrdinalIgnoreCase);
```

`carryOutputNames` is populated in `CollectCarrySlots` (each `Add` also registers
`slot.OutputName`).

### `TryCreate` — build the reduced boundary op

- **Select path** (`boundaryOp is SelectOperation select`): build a pruned `SelectOperation`
  whose columns replace every top-level carry-slot cumulative window expression
  (Sum/Max/Min/Product/Count) with its `Source` projection, carrying explicit `outputNames`
  equal to the original effective output names. Prune exactly the windows that
  `CollectCarrySlots` collects (top-level `WindowExpression` of a cumulative kind), so the slot
  set and the pruned set always agree. If the pruned select would have zero columns (a select of
  only cumulative windows), set `reRunBoundaryOp = null`.
- **Standalone cumulative path** (`boundaryOp is CumulativeOperation`): its only output is a carry
  slot, so `reRunBoundaryOp = null` (the boundary re-run is fully redundant).
- **Rolling / Shift paths** are unaffected (no carry slots; re-run context supplies correct
  lookback/lookahead history, no cumulative re-materialization).

### `ProcessChunk` — skip boundary re-run when there are no real columns

Replace the unconditional `result = boundaryOp.Execute(run)` with:

```csharp
var runLength = contextLength + chunkLength;
var runStart = totalRowsSeen - contextLength;

IReadOnlyDictionary<string, IColumn> result = reRunBoundaryOp is not null
    ? reRunBoundaryOp.Execute(run)
    : new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
// (reRunBoundaryOp being null means every boundary column is owned by a carry slot.)
```

Build `emitted` only from non-carry keys of `result`:

```csharp
var emitted = new Dictionary<string, IColumn>(result.Count + carrySlots.Count, StringComparer.OrdinalIgnoreCase);
foreach (var kvp in result)
    if (!carryOutputNames.Contains(kvp.Key))
        emitted[kvp.Key] = sliceRange(kvp.Value, from, emitCount);

foreach (var slot in carrySlots)
    emitted[slot.OutputName] = carryColumnForEmission(slot, processedChunk, emitCount);
```

Set `lastRunResult = result` as today. **`Flush` is unchanged**: when `reRunBoundaryOp == null`
every boundary column is a carry slot, which implies `leadDistance == 0`, so `Flush` early-returns
at `emittedCount >= totalRowsSeen` and never reaches `getRowLength(lastRunResult)`. In mixed cases
the pruned `result` is non-empty (carry slots excluded from emission but present in the dict) and
`Flush` slices non-carry keys from it exactly as before.

## Correctness notes / edge cases

- **First chunk:** `contextLength == 0`, `reRunBoundaryOp` (pruned) runs over `processedChunk`.
  Same output as before for non-carry columns; carry columns unaffected.
- **Null propagation:** carry columns come from `ComputeCarryColumn` (mask-preserving), unchanged.
  Pruned select columns are non-carry to begin with.
- **`lastRunInput` (context for the next chunk)** is `takeLastRows(run, contextSize)` from `run`
  (the full concatenated dict), independent of `result` — unaffected by the reduction.
- **All-carry + lead:** impossible today (a select whose columns are all cumulative windows has
  `leadDistance == 0`, and a standalone cumulative has `leadDistance == 0`), so `Flush` never needs
  non-carry rows from an empty `lastRunResult` in that case.
- **No regression to #356/#357:** the delayed-emission `PendingColumnBuffer` staging and the
  `AddConstant` path are untouched; this change only stops re-computing carry windows, replacing
  the redundant value with the (overwritten) source projection in the pruned select.

## Test additions — `StreamingExecutionStrategyTests.cs`

Add streaming-vs-lazy equivalence tests, mirroring
`Property_StreamingVsLazy_AllCumulativeKinds_OnDoubleSource_MatchesLazy` but on int-family data
whose full-column product is well-defined. The canonical repro is `data[i] = i` (0..N → product 0
from row 0), chunked so `ChunksRead.Count > 1` (already-used helpers: `ExecutionTestHelpers
.CreateLargeChunkedSource` / a small int `IQuerySource`, `ChunkSize`, `AssertFramesEqualWithMasks`).

1. `Property_StreamingVsLazy_CumulativeProduct_IntSource_LeadingZero_MatchesLazy`
   — `SelectOperation` with `Col<int>("A")`, `CumulativeProduct(Col("A"))`, `Lead(Col("A"), 2)`
   (mirrors the issue repro), `data[i] = i` over ~6000 rows, chunk size ~333; assert
   streaming matches lazy and chunked.
2. `Property_StreamingVsLazy_CumulativeProduct_IntSource_BoundedValues_MatchesLazy`
   — int source whose values stay small (e.g. alternating `2/3` like the double test) so the true
   product stays bounded; assert no `OverflowException` and equivalence for Sum and Product kinds.
3. `Property_StreamingVsLazy_StandaloneCumulativeProduct_IntSource_LeadingZero_MatchesLazy`
   — standalone `CumulativeOperation` with `Kind = Product` over an int source with a leading zero;
   assert chunked streaming and lazy equivalence (guards the `reRunBoundaryOp = null` path).

All use `AssertFramesEqualWithMasks` and dispose lazy results, per file convention. (Per AGENTS.md,
ask before running `dotnet test`.)

## Verification

1. `dotnet build Nivara.slnx` — 0 warnings / 0 errors.
2. Run `StreamingExecutionStrategyTests` (existing suite + 3 new) — all pass, including the
   existing `Property_StreamingVsLazy_AllCumulativeKinds_OnDoubleSource_MatchesLazy` (proves no
   regression to the double/streaming-equivalence behavior).
3. Confirm the repro from the issue no longer throws for the leading-zero case and that
   `ChunksRead.Count > 1` (streaming is preserved — Direction 1).

## Known edge (out of scope, document not change)

`CollectCarrySlots` only captures **top-level** cumulative windows. A cumulative product nested
*inside* a larger expression (e.g. `CumulativeProduct(Col("A")) + Col("B")`) is currently treated
as streamable by `isStreamableNode` but is **not** a carry slot, so its per-run re-materialization
still overflows. Recommend a follow-up: exclude non-slot nested cumulative windows from streaming
(`isStreamableNode` returning `false` for a cumulative product that is not a top-level carry slot),
or extend carry-slot collection to nested carry windows. Not part of #358's repro / acceptance
criteria (which use a top-level `CumulativeProduct`).

## Open API / compatibility

No public API change. `StreamingWindowProcessor`, `SelectOperation`, `CumulativeOperation` are all
internal; `StreamingWindowProcessor` is consumed only by `StreamingExecutionStrategy`.

## Planned commits

1. `docs: plan #358 streaming cumulative product overflow in TODO.md`
2. `fix: stop re-materializing carried cumulative windows in StreamingWindowProcessor`
3. `test: pin streaming cumulative product equivalence vs lazy for #358`
4. `docs: remove TODO.md — #358 plan executed`

## Blast radius

- `StreamingWindowProcessor.cs` — internal, consumed only by `StreamingExecutionStrategy`.
  Changes the per-run boundary execution (skips redundant carry-window computation) and how
  `Flush` computes run length; no public output shape or API change.
- `StreamingExecutionStrategy.cs` — indirect consumer via `StreamingWindowProcessor`; already
  exercised by the full `StreamingExecutionStrategyTests` suite (94 existing + new).
- The double-source cumulative equivalence test
  (`Property_StreamingVsLazy_AllCumulativeKinds_OnDoubleSource_MatchesLazy`) guards against
  regressing all cumulative kinds; existing 94 streaming tests guard rolling/shift/lead/lazy-parity.

## GitHub issues log

- [ ] #358 — Streaming cumulative product can overflow per-run (this plan).
- [x] #360 — nested (non-slot) cumulative product windows still per-run re-materialize and can overflow (created while working on #358; out of scope, direction decided later).
