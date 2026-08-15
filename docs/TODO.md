# Plan: POLARS-ROADMAP Phase 1-3 high-priority bugfixes (#245-#248)

Branch: `khurram/issues` (created off `main` @ bac3c7c).
Single PR closing #245, #246, #247, #248. One commit per issue, then docs.

## Problem

Four independent high-priority bugs surfaced in the Phase 1-3 review of the
POLARS-ROADMAP (fused expression engine, window functions, execution strategies):

- **#246** — Compiled-delegate cache keyed by `plan.Signature`; `FormatValue`
  drops the literal runtime type, so `0.1f` vs `0.1`, `1.1m` vs `1.1`,
  `nint`/`int` literals collide and reuse a wrong delegate.
- **#247** — Compiled path evaluates every position, then ORs leaf masks in a
  separate `ComputeMask` pass. Masked `decimalCol / intCol` (null in int)
  throws `DivideByZeroException`; span kernel short-circuits to `default(T)`.
  Masked-position backing values diverge.
- **#245** — `Select` ops whose expression embeds a `WindowExpression` are
  treated as streamable per-chunk; `Filter` with a window runs per-slice under
  the parallel strategy. `ContainsWindowExpression` is private, so strategies
  cannot see op expressions.
- **#248** — `buildPrefix` / `cumulativeScan` accumulate in element type `T`
  with no `checked`; `RollingSum`/`RollingMean`/`CumulativeSum`/`CumulativeProduct`
  silently wrap on prefix overflow. Int property tests use small values and
  can't detect it.

## Proposed changes

### Commit 1 — #246 literal-type-aware signatures

`src/Nivara/Expressions/ExpressionTypeInferer.cs` (`BuildSignature` :165-177,
`FormatValue` :181-190):

- Append the literal runtime type to the signature fragment so two literals
  that stringify identically but differ in type never share a cache key:
  `return $"{text}:{value.GetType().FullName}";` in `FormatValue` (covers both
  `LiteralExpression` :170 and `ScalarExpression` :172).

Tests in `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`:
- `Col + 0.1f` vs `Col + 0.1` and `Col + 1.1m` vs `Col + 1.1` on matching
  input → distinct `plan.Signature`s, correct per-type results through the
  shared static `compiledKernelCache`.

### Commit 2 — #247 mask-before-value compiled evaluation

`src/Nivara/Expressions/FusedExpressionEvaluator.cs`:

- Compute the OR'd leaf mask (`ComputeMask` :595-607) **before** the value pass
  in `EvaluateCompiled` (:271-315) instead of after.
- Extend `CompiledFusedInvoke` (:56) and `BuildCompiledDelegate` (:684) with a
  `bool[]? mask` parameter. The delegate loop writes `default(T)` at masked
  positions and skips evaluation there, matching the span kernel's
  short-circuit semantics.
- Drop the post-pass bool zeroing (:302-308) — the delegate now writes
  `default(false)` at masked positions.

Tests in `FusedExpressionEvaluatorTests`:
- `decimalCol / intCol` with nulls in `intCol`: no exception, null at masked
  positions, correct values elsewhere, bit-identical across compiled / span /
  TensorPrimitives backends (raw-indexer-backed masked values pinned to
  `default(T)`).

### Commit 3 — #245 window expressions under streaming/parallel

- Make `FusedExpressionEvaluator.ContainsWindowExpression` (:331) `internal static`.
- Add op-level inspector `HasWindowExpression(IQueryOperation)` handling
  `SelectOperation.Columns` (:49) and `FilterOperation.Condition` (:26).
- `src/Nivara/Execution/StreamingExecutionStrategy.cs` — `isSuitableForStreaming`
  (:10) returns false when any operation carries a window expression → falls
  back to `LazyExecutionStrategy`.
- `src/Nivara/Execution/ParallelExecutionStrategy.cs` — add op-aware
  `isParallelizable(IQueryOperation)` used at the sync (:64) and async (:330)
  dispatch gates and the final Filter routes (:87, :353) so window-bearing
  operations execute whole-column.

Tests in `StreamingExecutionStrategyTests` / `ParallelExecutionStrategyTests`:
- `Select(over(...))` and `Filter(over(...))` produce eager-equivalent results
  (no per-chunk/per-slice window evaluation); streaming falls back to lazy.

### Commit 4 — #248 widen int-family window accumulators

`src/Nivara/Tensors/WindowFunctions.cs`:

- `buildPrefix` (:349-371): for int-family `T`
  (`sbyte/byte/short/ushort/int/uint/char`) accumulate in `long`; keep `long`/
  float/double/etc. unchanged (matches `NivaraSeries` promotion rules).
- `RollingSum` (:114-130) / `RollingMean` (:155-167): read the widened prefix,
  convert per-window sums via `T.CreateChecked` / `double.CreateChecked`
  (defined `OverflowException` on genuine overflow instead of silent wrap).
- `cumulativeScan` (:309-347): accumulate `isSum`/`isProduct` in `long` for
  int-family `T`; record via `T.CreateChecked`.

Tests in `tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`:
- `RollingSum([int.MaxValue, 0, int.MaxValue, 0], window 2)` → correct (prefix
  wraps in `int`, per-window sums fit in `long`).
- `CumulativeSum([int.MaxValue, 1])` → `OverflowException`.
- `CumulativeProduct` int-family widening with defined-throw case.

### Commit 5 — docs/CHANGELOG

- `CHANGELOG.md` `[Unreleased]` entries for the four fixes.

## Blast radius

- `FusedExpressionEvaluator` / `ExpressionTypeInferer` / `FusedKernel` /
  `KernelLowerer`: touched by #246, #247, #245 (inspector). Internal only;
  callers are the query pipeline and `NivaraColumn`/`NivaraFrame` expression
  entry points. Guardrails: `FusedExpressionEvaluatorTests` (606 lines),
  `WindowExpressionEvaluationTests`, `WindowExpressionOperationTests`.
- `StreamingExecutionStrategy` / `ParallelExecutionStrategy`: gate decisions
  only; no data-path change. Guardrails: `StreamingExecutionStrategyTests`,
  `ParallelExecutionStrategyTests`, `ExecutionTestHelpers`.
- `WindowFunctions`: kernel math for int-family types only; float/double/long
  unchanged. Guardrails: `WindowFunctionsTests`, `WindowOperationTests`,
  `WindowExpressionOperationTests` (window kind parity), eager/lazy parity.
- Public API surface: **none** — all changes are internal to `src/Nivara`.

## Verification

- `dotnet build Nivara.slnx` — 0 warnings, 0 errors (before each commit).
- `dotnet test` — run only on explicit human confirmation (repo workflow).
  Targeted filters first: `FusedExpressionEvaluatorTests|WindowFunctionsTests|StreamingExecutionStrategyTests|ParallelExecutionStrategyTests|WindowExpressionEvaluationTests|WindowOperationTests`.

## Commits

1. `docs: plan bugfix batch #245-#248 in TODO.md`
2. `fix: include literal runtime type in fused plan signatures (#246)` ✓ `a3cba89`
3. `fix: mask before value pass in compiled fused kernel (#247)` ✓ `d1ff242`
4. `fix: run window-bearing operations whole-column in streaming/parallel (#245)` ✓ `8367bce`
5. `fix: widen int-family window accumulators to avoid silent wrap (#248)`
6. `docs: changelog for #245-#248`

## GitHub issues log

- [x] #246 — no new issue; fixed FormatValue literal-type collision.
- [x] #247 — no new issue; fixed compiled-path masked evaluation.
- [x] #245 — no new issue; op-level window inspector added (WindowExpressionInspector).
- [ ] #248 — in progress.
