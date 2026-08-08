# Plan: Remove legacy `ExpressionEvaluator`, fix fused-path promotion, add Modulo (#152)

Branch: `khurram/152` (off `main`). Commits are local only; the human reviews, pushes, and opens the PR.

## Problem

GitHub issue #152: the legacy `ExpressionEvaluator` (`src/Nivara/Helpers/ExpressionEvaluator.cs`) is
dead weight — every production query op (`FilterOperation`, `SelectOperation`, `SortByExpressionOperation`)
and `ParallelExecutionStrategy` routes through the fused evaluator (`FusedExpressionEvaluator` +
`FusedKernel` + `ExpressionTypeInferer`). The legacy evaluator is referenced only by its own test file,
the perf benchmark's multi-pass baseline, and docs. Meanwhile, three real defects live on the **production
fused path**:

1. **Same-type small-integral promotion is wrong.** `NumericPromoter.GetPromotedType` returns `left` for
   equal operand types (`src/Nivara/Helpers/NumericPromoter.cs:26`), so `byte + byte` → `byte` instead of
   the C# §12.4.7.3 result `int`. The legacy evaluator hid this with an `IsSmallIntegralType` fast switch;
   the fused evaluator inherits the bug (e.g. `double` was seen in old repros because unrelated columns
   widened to the frame's first column type).
2. **Schema vs evaluated-column type divergence.** `BinaryExpression.DetermineResultType`
   (`src/Nivara/Expressions/ColumnExpression.cs:467`) and `ScalarExpression`'s inline promotion use a
   stale 6-type table (`double,float,long,int,short,byte`) that diverges from `NumericPromoter`
   (which already handles `uint`, `decimal`, `Half`, `nint`, `Int128`…). Evaluated columns are correct
   (the frame is built from actual columns in `QueryExecutor.cs:82`), but `QueryPlan.ResultSchema` /
   validation sees the wrong type.
3. **No `%` (Modulo) anywhere.** `BinaryOperator` enum, `ColumnExpression` `%` operators, operator
   symbols, `FusedExpressionEvaluator.ApplyArithmetic`, `FusedKernel.ApplyArithmetic`, and the typed
   LINQ translator (`TypedExpressionTranslator.cs:108` throws) all lack modulo.

## Decisions (human-confirmed)

- Delete the legacy `ExpressionEvaluator` entirely (file + tests + perf baseline).
- Perf benchmark: **drop** the fused-vs-multi-pass comparison scenario (`CreateMultiPassChainScenario`);
  keep `CreateFusedChainScenario` standalone.
- Modulo is added to **both** the `ColumnExpression` DSL and the typed LINQ translator; update
  `docs/LINQ.md` (remove the "% fails fast" claim).

## Changes

1. **Remove legacy evaluator** — delete `src/Nivara/Helpers/ExpressionEvaluator.cs` and
   `tests/Nivara.Tests/Query/ExpressionEvaluatorTests.cs`. Update `docs/LINQ.md:105,147,596` and
   `docs/plan/POLARS-ROADMAP.md:24,43,46,50,67,78` to describe the fused evaluator as the sole engine.
2. **Same-type promotion fix** — in `NumericPromoter.GetPromotedType`, replace the `left == right`
   early-return with: `sbyte`/`byte`/`short`/`ushort`/`char` same-type pairs → `int`; all other same-type
   pairs keep the type (`decimal/decimal`→decimal, `uint/uint`→uint, `float/float`→float, etc.). This
   flows automatically into `ExpressionTypeInferer.InferBinary`/`InferScalar`
   (`ExpressionTypeInferer.cs:132,141`) and the compiled kernel target
   (`FusedExpressionEvaluator.cs:495-496`).
3. **Unify result-type logic** — `BinaryExpression.DetermineResultType` and `ScalarExpression` result
   type delegate to `NumericPromoter.GetPromotedType(left, right) ?? typeof(object)` so the plan schema
   matches evaluated columns.
4. **Add Modulo** — `BinaryOperator.Modulo` enum member, `%` operators on `ColumnExpression` (binary +
   scalar), `GetOperatorSymbol` → `"%"`, `FusedExpressionEvaluator.ApplyArithmetic` → `Expression.Modulo`,
   `FusedKernel.ApplyArithmetic<T>` → generic `left % right`, and
   `TypedExpressionTranslator.cs:108` `ExpressionType.Modulo` → `NivaraBinaryExpression(BinaryOperator.Modulo, …)`.
5. **Rework `FusedExpressionEvaluatorTests`** — remove the ~7 legacy `new ExpressionEvaluator()` oracle
   comparisons; assert hand-computed values. Add: `byte+byte`→`NivaraColumn<int>`; `decimal/decimal`→decimal;
   `uint/uint`→uint; `%` binary + scalar (values, null-OR mask propagation, diagnostics/EvaluationMode
   guardrails); schema `ResultType` consistency.
6. **Perf benchmark** — drop `CreateMultiPassChainScenario` + its registration
   (`tests/Nivara.PerformanceTests/Program.cs`).
7. **CHANGELOG** — `[Unreleased]` Breaking (legacy `ExpressionEvaluator` removed), Fixed (same-type
   small-integral promotion), Added (`%` in DSL + LINQ). Update the stale line-15 claim to point at the
   fused evaluator.

## Planned commits

1. `docs: plan removal of legacy ExpressionEvaluator + #152 fixes in TODO.md`
2. `Remove legacy ExpressionEvaluator and its tests`
3. `Fix same-type small-integral promotion in NumericPromoter (byte+byte -> int)`
4. `Unify expression result-type inference with NumericPromoter`
5. `Add Modulo support to the expression DSL and typed LINQ surface`
6. `Rework fused evaluator tests without the legacy oracle`
7. `Drop the fused-vs-multi-pass benchmark comparison`
8. `docs: update CHANGELOG for ExpressionEvaluator removal, promotion fix, and modulo`
9. `docs: remove TODO.md — plan executed`

## Blast radius

- **Deletions:** `src/Nivara/Helpers/ExpressionEvaluator.cs`, `tests/Nivara.Tests/Query/ExpressionEvaluatorTests.cs`.
  Only consumers were tests + perf + docs (verified by grep; production ops already use the fused path).
- **`NumericPromoter.GetPromotedType`** is used by: `ExpressionTypeInferer` (fused plans),
  `FusedExpressionEvaluator.BuildNode` (kernel conversions), `BuildComparison`
  (`FusedExpressionEvaluator.cs:543`), and `NivaraColumn`/`NivaraSeries` arithmetic dispatch. The
  byte/byte→int change only affects same-type small integrals; no current test asserts byte/byte→byte
  (`NumericPromoterTests` covers only mixed pairs + non-numeric).
- **`BinaryExpression.DetermineResultType` / `ScalarExpression`** affect `TransformSchema` result types
  (plan validation/diagnostics), not evaluated columns. Tests: query pipeline tests + schema tests.
- **`BinaryOperator` enum + symbols + evaluator switches** — the node-tree `FusedKernel` and compiled
  `FusedExpressionEvaluator` switches both gain a Modulo arm; the `_ => throw` guards remain the safety net.
- **Tests:** `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs` (12 tests), `ExpressionEvaluatorTests.cs`
  (15 tests, deleted), `NumericPromoterTests.cs`, `tests/Nivara.PerformanceTests/Program.cs`.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests` (human-confirmed before running)

## GitHub issues log

- [ ] #152 — Remove legacy ExpressionEvaluator + fix byte+byte promotion + add modulo (this plan).
- [ ] #157 — NivaraColumn arithmetic kernels not yet collapsed onto generic math (pre-existing, referenced in POLARS-ROADMAP).
- (Create any newly discovered issues here as work proceeds.)
