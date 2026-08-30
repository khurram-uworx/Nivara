# TODO — #356: StreamingWindowProcessor delayed-emission path still boxes pending cumulative values

## Problem

`src/Nivara/Execution/StreamingWindowProcessor.cs` boxes every cumulative-window element when a
lookahead (Lead / negative Shift) window delays emission:

- `carryColumnForEmission` enqueues `corrected.GetValue(i)` (boxed `object?`) into
  `CarrySlot.PendingValues` (`Queue<object?>`), one box per element per chunk.
- `buildDeferredPrefix` / `buildDeferredColumn` dequeue into `object?[]` and materialize through
  `ColumnFactory.Create` (boxed `MakeGenericMethod` invocation + per-element cast).

This is exactly the per-element boxing #343 removed from `AddConstant` (PR #357); it remains only on
the delayed-emission staging path (issue #356).

## Proposed changes

### 1. `src/Nivara/Execution/StreamingWindowProcessor.cs`

Replace the boxed `Queue<object?> PendingValues` carried per slot with a lazily-created, typed
pending buffer dispatched on the corrected column's element type (mirroring the #343
`AddConstant` pattern):

- `CarrySlot.PendingValues` (`Queue<object?>`, eagerly created) → `PendingBuffer`
  (`PendingColumnBuffer?`, created lazily on first carry when `leadDistance > 0`). Keep
  `ElementType` (used for the defensive empty-column fallback).
- New nested types after `CarrySlot`:
  - `abstract class PendingColumnBuffer` — `int Count`, `void Enqueue(IColumn corrected)`,
    `IColumn Dequeue(int count)`.
  - `sealed class TypedPendingColumnBuffer<T> where T : struct` — `Queue<T?>` staging; enqueues via
    the typed `IColumn<T>` indexer (`column.IsNull(i) ? null : column[i]`, no boxing); dequeues into
    an owned `T[]` + `bool[]` null mask and materializes with
    `NivaraColumn<T>.CreateFromOwnedArrays` / `CreateFromOwnedArray`.
  - `sealed class BoxedPendingBuffer` — the current boxed behavior (`Queue<object?>` +
    `ColumnFactory.Create(elementType, ...)`) for reference / non-`NivaraColumn<T>` columns.
- New `static PendingColumnBuffer createPendingBuffer(IColumn corrected)` — 18-arm type switch over
  the numeric `NivaraColumn<T>` domain (int, long, float, double, decimal, byte, sbyte, short,
  ushort, uint, ulong, char, nint, nuint, Int128, UInt128, Half, BFloat16), `_ => BoxedPendingBuffer`
  fallback.
- `carryColumnForEmission`: after the `leadDistance == 0` early return, set
  `slot.ElementType ??= corrected.ElementType`, create the buffer once
  (`slot.PendingBuffer ??= createPendingBuffer(corrected)`), `Enqueue(corrected)`, return
  `buildDeferredPrefix(slot, emitCount)`.
- `buildDeferredPrefix(slot, count)` → `slot.PendingBuffer!.Dequeue(count)`.
- `buildDeferredColumn(slot)` → `slot.PendingBuffer` non-null: `buffer.Dequeue(buffer.Count)`;
  defensive `ColumnFactory.Create(slot.ElementType ?? typeof(long), Array.Empty<object?>())`
  otherwise (mirrors current behavior).

**Deliberate deviation from the issue's literal `T[]` suggestion:** the corrected cumulative
columns carry null positions (null source cells "stay null", `WindowFunctions.cs`), and the current
queue preserves them via null entries → `ColumnFactory.Create`. A plain `Queue<T>` + `T[]` +
`CreateFromOwnedArray` would silently drop those null cells and break the existing
`Property_StreamingVsLazy_LeadWithCumulativeCount_NullableSource_MatchesWithMasks` test. Staging as
`Queue<T?>` + `CreateFromOwnedArrays(data, nullMask)`/`CreateFromOwnedArray(data)` removes all
boxing while preserving null semantics (same shape as `addConstant<T>` in this file).

### 2. `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

Add `using Nivara;` and four streaming-vs-lazy equivalence tests in the "Delayed-emission
streaming" section (after the existing cumulative+lead tests, ~line 1897):

1. `Property_StreamingVsLazy_LeadWithCumulativeMaxMinProduct_MatchesLazy` — int source, max/min/
   product + `Lead(A,2)`, looped over `chunkEquivalenceBudgets`.
2. `Property_StreamingVsLazy_LeadWithCumulativeSum_NullableSource_MatchesWithMasks` — nullable int
   source (`B`), sum + `Lead(B,2)`, `ChunkSize = 512` (pins null preservation through the buffer).
3. `Property_StreamingVsLazy_NegativeShiftWithCumulativeSumAndMax_MatchesLazy` — int source, sum +
   max + `Shift(A,-2)`, `ChunkSize = 271`.
4. `Property_StreamingVsLazy_LeadWithCumulativeSum_DecimalSource_MatchesLazy` — `decimal` source
   (non-vectorizable numeric type) via a small nested `DecimalChunkedSource : IQuerySource`
   (schema `("A", typeof(decimal))`, `CanReadInChunks = true`, chunked reads mirroring
   `ExecutionTestHelpers.StubChunkedQuerySource`).

All assert `ExecutionTestHelpers.AssertFramesEqualWithMasks(lazyResult, result)`.

## Verification

- `dotnet build Nivara.slnx` — 0 warnings, 0 errors.
- `StreamingExecutionStrategyTests` — 94 existing + 4 new = 98 passing.
- Code-inspection check: no `object?[]` allocation or `GetValue(i)` boxing remains on the
  delayed-emission path for numeric/vectorizable cumulative columns (boxing is structural).

## Planned commits

1. `docs: plan #356 delayed-emission de-boxing in TODO.md`
2. `perf: remove per-element boxing on StreamingWindowProcessor delayed-emission path`
   (`src/Nivara/Execution/StreamingWindowProcessor.cs`)
3. `test: cover cumulative delayed-emission equivalence for max/min/product, nulls, negative shift, decimal`
   (`tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`)
4. `docs: remove TODO.md — #356 plan executed`

## Blast radius

- **File changed:** `src/Nivara/Execution/StreamingWindowProcessor.cs` (internal class). Its only
  external surface used by `StreamingExecutionStrategy` is `TryCreate` / `ProcessChunk` / `Flush`
  / `OverlapSize` — none change.
- **Downstream:** `StreamingExecutionStrategy` sync `Execute`, `StreamChunksAsync`, and the
  segmented `QueryFrame.AsStream` path consume the processor; behavior must be bit-identical.
- **Tests covering it:** `StreamingExecutionStrategyTests` (94 tests, incl. delayed-emission
  cumulative+lead/count equivalence across chunk sizes and budgets, nullable sources, async flush).
- **Risk:** low. The change is structurally similar to #343's already-merged `AddConstant` fix.
  Null semantics are preserved via `Queue<T?>` (see deliberate deviation), guarded by existing and
  new nullable-source tests.

## GitHub issues log

- (empty — no deferred work expected so far; create and record issues here at discovery time)