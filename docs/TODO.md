# Plan — Issue #360: Nested cumulative windows overflow per-run in streaming

Branch: `khurram/360` (off `main`).
Issue: https://github.com/khurram-uworx/Nivara/issues/360

## Problem

`StreamingWindowProcessor` streams a first-boundary `SelectOperation` per chunk by
re-executing a *reduced* select over each carried run. Carry-slot collection
(`CollectCarrySlots(SelectOperation)`, `src/Nivara/Execution/StreamingWindowProcessor.cs`
L531) and the reduced select builder (`BuildReducedSelect`, L218) only recognize
**top-level** `WindowExpression` cumulative nodes.

`isStreamableNode` (L384) however recurses through `Scalar/Binary/Comparison/Not/
Conditional` nodes and treats every cumulative window as streamable. So a select like
`CumulativeProduct(Col("A")).Add(Col("B"))` passes `hasOnlyStreamableWindows` but has **no**
carry slot and its nested cumulative node is **not** neutralized by `BuildReducedSelect`
(it only substitutes for top-level window nodes). `ProcessChunk` therefore re-materializes
`cumulativeScan` (checked `long` accumulator, `src/Nivara/Tensors/WindowFunctions.cs`
L386–419) over each mid-run starting at the run's first value rather than dataset row 0,
reproducing the #358 overflow for nested products. For nested cumulative **sums/max/min**
the per-run re-scan omits the dataset row-0 seed and produces **wrong values** (not just
overflow).

`WindowExpressionInspector.HasWindowExpression` recurses the tree (via
`FusedExpressionEvaluator.ContainsWindowExpression`), so the boundary op is still detected —
the gap is purely the streamable-vs-carried decision in `StreamingWindowProcessor`.

## Decision (per issue direction 2)

Exclude **non-slot nested cumulative windows** from the streamable set. Simplest fix,
matches how rank-family/broadcast windows are already excluded. `TryCreate` then returns
`null` for such a boundary, and `StreamingExecutionStrategy` (`ExecuteCore`, L222–236)
materializes the boundary op over the full concatenated frame — exact, non-chunked, no
carry machinery, no overflow.

Tradeoff (accepted): a `SelectOperation` containing a **nested** cumulative window no longer
per-chunk-streams; it materializes at the boundary. Top-level cumulative windows are
unaffected and keep carry-slot streaming. Mixed selects (rolling/shift/lead nested beside a
top-level cumulative) still stream. Nested **rolling / shift / lead** windows remain
streamable — they are pure per-position functions that read history from the carried
context (rolling/lag) or are covered by the existing recursive `determineLeadDistance`
delayed emission (lead). Only cumulative kinds accumulate from dataset row 0, so only they
are excluded when nested.

Why not Direction 1: nested cumulative windows have no own output name to overwrite, so a
carry slot cannot simply replace the emitted column. Full direction 1 would require
recomputing the entire outer expression over the run with the window node *bound* to
carry-seeded columns (extended to run length), i.e. retained per-row carry tails or a bound
hydration path in `FusedExpressionEvaluator` — substantially more machinery for a niche case
the issue flags as "more complex carry collection".

## Proposed changes

### 1. `src/Nivara/Execution/StreamingWindowProcessor.cs`

- `hasOnlyStreamableWindows` (L373): pass `isRoot: true` into `isStreamableNode`.
- `isStreamableNode` (L384): add an `isRoot` parameter. Cumulative kinds return `isRoot`
  only:
  ```csharp
  static bool isStreamableNode(ColumnExpression node, bool isRoot = false)
      => node switch
      {
          WindowExpression window => window.Kind switch
          {
              WindowFunctionKind.RollingSum or WindowFunctionKind.RollingMean
                  or WindowFunctionKind.RollingMin or WindowFunctionKind.RollingMax
                  or WindowFunctionKind.CumulativeSum or WindowFunctionKind.CumulativeMax
                  or WindowFunctionKind.CumulativeMin or WindowFunctionKind.CumulativeProduct
                  or WindowFunctionKind.CumulativeCount
                  => isRoot,          // cumulative windows stream only as top-level carry slots (#360)
              WindowFunctionKind.Shift => true,
              WindowFunctionKind.Lead => true,
              _ => false              // rank family + broadcast aggregates
          },
          ScalarExpression scalar => isStreamableNode(scalar.Column),
          BinaryExpression binary => isStreamableNode(binary.Left) && isStreamableNode(binary.Right),
          ComparisonExpression comparison => isStreamableNode(comparison.Left) && isStreamableNode(comparison.Right),
          NotExpression not => isStreamableNode(not.Operand),
          ConditionalExpression conditional => isStreamableNode(conditional.Test)
              && isStreamableNode(conditional.TrueValue)
              && isStreamableNode(conditional.FalseValue),
          _ => true
      };
  ```
  (recursion passes default `isRoot: false`; no other call sites change.)
- `BuildReducedSelect` / `CollectCarrySlots` / `CollectCarrySlots(CumulativeOperation)`:
  unchanged — direction 2 requires no carry changes.
- Update the `TryCreate` doc comment (L249–253) to note nested cumulative windows also
  disqualify a boundary from streaming (one sentence).

### 2. `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

Mirror the #358 property-test pattern (L1511+):

- `Property_StreamingVsLazy_NestedCumulativeProduct_IntSource_LeadingZero_MatchesLazy`
  — source `data[i] = i` (6000 rows, chunked 333); select
  `Col<int>("A"), CumulativeProduct(Col("A")).Add(Col<int>("B"))`; assert no throw, result
  equals lazy (`AssertFramesEqualWithMasks`), `StreamMaterializationCount > 0`.
- `Property_StreamingVsLazy_NestedCumulativeSum_IntSource_MatchesLazy` — same shape with
  `CumulativeSum(Col("A")).Add(Col<int>("B"))`; pins the wrong-values (missing row-0 seed)
  aspect shared by all nested cumulative kinds.
- `Property_StreamingVsLazy_NestedCumulativeProduct_NestedTwice_IntSource_MatchesLazy`
  — `CumulativeProduct(Col("A")).Add(Col<int>("B")).Multiply(CumulativeProduct(Col("C")))`:
  two nested cumulative nodes under one column, both must take the fallback.
- Each test asserts `source.ChunksRead.Count > 1` (source still read in chunks; the
  *boundary* materializes).

### 3. `CHANGELOG.md`

Unreleased **Changed** entry mirroring the #358 entry style.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests --filter "FullyQualifiedName~StreamingExecutionStrategyTests"` (requires human confirmation before running)
- Review `git diff` per commit; sanity-check no regression in existing #358 tests.

## Planned commits

1. `docs: plan #360 nested-cumulative window streaming in TODO.md`
2. `test: pin nested-cumulative window streaming equivalence for #360` (tests first, expect red)
3. `fix: exclude non-slot nested cumulative windows from streaming boundary (#360)`
4. `docs: changelog #360 nested-cumulative window boundary fallback`
5. G2 review, then `git rm docs/TODO.md` → `docs: remove TODO.md — #360 plan executed`
6. Offer push + PR (human-confirmed only).

## Blast radius

- `StreamingWindowProcessor.isStreamableNode` is internal, static, called only from
  `hasOnlyStreamableWindows` (same class). Changing the signature is fully internal.
- `TryCreate` returns `null` for selects with a nested cumulative window → the first
  boundary op materializes over the full frame in all four streaming paths
  (`ExecuteCore` L222–236, and the async path at L326–330, parallel-streaming paths
  L459/L596/L728/L812 via the same mechanism). Result is exact and matches lazy — this is
  the pre-existing behavior for rank/broadcast windows.
- Downstream consumers: `StreamingExecutionStrategy` (only caller of `TryCreate`);
  tests in `StreamingExecutionStrategyTests.cs`. No public API changes.
- Existing #358 tests (top-level cumulative product) must keep streaming with
  `StreamMaterializationCount == 0`; they guard against regressing the carry-slot path.

## GitHub issues log

- [ ] (none yet — record any deferred work discovered during execution here)