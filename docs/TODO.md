# Plan: issues #277 (expression quantile/median) and #281 (streaming AC3 backpressure)

Working branch: `khurram/issues`

## Problem

- **#277** — `NivaraSeries<T>.Quantile/Median`, the `QuantileAggregation`/`MedianAggregation`
  classes, and the typed LINQ group-by path are implemented, but the dynamic expression DSL
  (`ColumnExpressions`) has no way to reference quantile/median as a broadcast aggregate in a
  `QueryFrame` column expression. Sum/Mean/Min/Max are not exposed as broadcast window
  aggregations either; this plan adds the broadcast-aggregate node for quantile/median only
  (scope matches the issue's acceptance criteria).
- **#281** — Phase 4 AC3 ("Memory stays within configured budget; channel bounded capacity
  enforced; `StreamingBufferManager.IsMemoryBudgetExceeded` never true") was never verifiably
  closed. `StreamingExecutionStrategy.ExecuteCoreAsync` creates a bounded channel with capacity
  `CalculateChannelCapacity(memoryBudget, chunkSize)` clamped to `[2, 16]`, but no test proves the
  channel is actually bounded under load or that `MemoryBudget`/`ChunkSize` influence the bound.
  `StreamingBufferManager` (Extensions) is intentionally not wired into the core streaming
  strategy; AC3 as literally written is untestable in the current design.

## Proposed changes

### Issue #277 — `ColumnExpressions.Quantile(source, q)` / `.Median(source)`

Extend the existing `WindowExpression` node (the whole-column-materialization node) with two new
`WindowFunctionKind` members. Broadcast aggregates are window-over-frame by construction; window
hydration rewrites window nodes into synthetic `ColumnReference`s *before* kernel planning, so
`ExpressionTypeInferer`, `KernelLowerer`, and the compiled/span/tensorprim kernel backends need no
changes.

1. `src/Nivara/Expressions/WindowExpression.cs`
   - Add `Quantile` and `Median` to `WindowFunctionKind`.
   - Add `WindowFunctionHelpers.IsBroadcastAggregate(kind)`.
   - `GetResultType`: return `typeof(double)` for `Quantile`/`Median`.
   - New ctor `WindowExpression(kind, ColumnExpression source, double q)`:
     validates `IsBroadcastAggregate(kind)` and (for `Quantile`) `q in [0,1]`;
     sets `Quantile` property. New `double? Quantile` property (null for non-quantile kinds).
   - `Name`: `Quantile(A, 0.25)` / `Median(A)`.
2. `src/Nivara/Expressions/ColumnExpression.cs` — `ColumnExpressions` factories:
   - `public static ColumnExpression Quantile(ColumnExpression source, double q)`
   - `public static ColumnExpression Median(ColumnExpression source)`
3. `src/Nivara/Expressions/FusedExpressionEvaluator.cs` — `MaterializeWindow`: new cases for
   `Quantile`/`Median`:
   - evaluate `window.Source` to a column;
   - compute the scalar via `QuantileAggregation.Apply(source, allIndices)` /
     `MedianAggregation.Apply(source, allIndices)` (routes through the aggregation classes per
     the issue; both return `double?`, null when the column is empty/all-null);
   - broadcast to a full-length `NivaraColumn<double>` (all-null mask when the value is null).
   `CollectColumnReferences`/`ContainsWindowExpression` are already generic over `WindowExpression`
   — no change.

Blast radius (#277): `WindowExpression` ctor family and `WindowFunctionKind` switch sites
(`GetResultType`, `ToRollingKind`, `ToCumulativeKind`, `ToRankKind` — new kinds fall through the
defaults/throws safely); `ColumnExpressions` API surface (additive); `FusedExpressionEvaluator`
window hydration (additive). Downstream callers of `WindowExpression`/`ColumnExpressions` are the
expression DSL (`QueryFrame.Select`/`Filter`, typed LINQ) and `WindowExpressionInspector`, which
already falls back for any window expression — a Quantile/Median select is automatically
non-streamable. Existing tests that enumerate `WindowFunctionKind` switches remain unaffected
(no existing switch matches the new members).

### Issue #281 — streaming AC3 verification

Keep `StreamingBufferManager` unwired (its class remarks + `STREAMING.md` already argue row-chunk
units + bounded channel replace byte-level budgets). Prove the bounded channel directly and amend
the docs.

1. `src/Nivara/Execution/StreamingExecutionStrategy.cs`
   - Make `CalculateChannelCapacity` `internal` (was `private static`).
   - Extract `internal static Channel<NivaraFrame> CreateBoundChannel(long memoryBudget, int
     chunkSize)` returning `Channel.CreateBounded<NivaraFrame>(capacity)`; use it at the
     `ExecuteCoreAsync` channel creation site (currently inline at line 225).
2. New `tests/Nivara.Tests/Execution/StreamingBackpressureTests.cs`
   - Formula tests: capacity clamps to `[2, 16]`; capacity is non-increasing when the budget
     shrinks or the chunk size grows (both knobs influence the bound).
   - Backpressure probe (deterministic): build the channel via `CreateBoundChannel`, run a fast
     producer writing N frames against a deliberately slow consumer (`Task.Delay` per read),
     track an `Interlocked` in-flight counter (producer increments before `WriteAsync`, consumer
     decrements after `ReadAsync`); assert peak in-flight equals capacity and never exceeds it.
     The producer blocks on `WriteAsync` at capacity+1, so this is deterministic.
   - Parity: `ExecuteCoreAsync` over `StubChunkedQuerySource` with a small budget still returns
     the full result (bounded channel + concat correctness).
3. `docs/STREAMING.md` — add an AC3-resolution note: in the query pipeline the bounded channel
   (row-chunk units) enforces the memory budget; `StreamingBufferManager.IsMemoryBudgetExceeded`
   remains IO-layer-only and is intentionally not used by `StreamingExecutionStrategy`.

Blast radius (#281): `StreamingExecutionStrategy` internals only (`ExecuteCoreAsync` call site,
one line); the extracted factory is additive. Existing streaming tests (`StreamingExecutionStrategyTests`,
`AsyncStreamingTests`, cancellation PerfScenario) exercise the same channel via `ExecuteCoreAsync`
and are the regression guardrail.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests --filter "FullyQualifiedName~BroadcastAggregateExpressionTests|FullyQualifiedName~StreamingBackpressureTests|FullyQualifiedName~ColumnExpressionTests"` (ask before running)
- Full `dotnet test tests/Nivara.Tests` (ask before running)

## Planned commit list

1. `docs: plan issues #277 and #281 in TODO.md`
2. `feat(expressions): WindowExpression broadcast-aggregate node (Quantile/Median) + ColumnExpressions factories` (#277, src part 1)
3. `feat(expressions): wire Quantile/Median through FusedExpressionEvaluator.MaterializeWindow` (#277, src part 2)
4. `test(expressions): BroadcastAggregateExpressionTests for quantile/median DSL` (#277)
5. `feat(execution): expose streaming bounded-channel factory + internal capacity calc` (#281, src)
6. `test(execution): StreamingBackpressureTests (formula bounds + in-flight probe)` (#281)
7. `docs: resolve streaming AC3 in STREAMING.md (bounded channel replaces byte budgets)` (#281)
8. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet — create via `gh issue create --repo khurram-uworx/Nivara` at discovery time and record here)
