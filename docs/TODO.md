# TODO — Issue #159: Window-Function Expressions in the Expression DSL

## Goal

Make window functions first-class, composable `ColumnExpression`s so they can be
embedded in `Select` / `Filter` / `SortBy` and in window ops over *computed* sources:

- `Select(RollingSum(Col("Salary"), 2) * 2)` — window result fused with elementwise math.
- `RollingSum(Col("A") * 2, "r", 2)` — window over a computed source in the pipeline.
- `Rank(..., orderBy: [SortExpressionKey(Col("B") * 1)], partitionBy: [Col("Dept")])` — rank over expression keys.
- Plan layer: visitors and diagnostics recognize window ops (currently they fall to "unknown").

## Design (single window kernel per kind, hydration into the fused evaluator)

1. New `WindowFunctionKind` enum + `WindowExpression : ColumnExpression` AST
   (`src/Nivara/Expressions/WindowExpression.cs`), plus `ColumnExpressions` factories.
   - Non-rank kinds carry a `Source` sub-expression (+ rolling `WindowSize`/`MinPeriods`,
     `NullHandler`; shift/lead `Periods`/`FillValue`).
   - Rank kinds carry `PartitionBy` (`IReadOnlyList<ColumnExpression>`) and
     `OrderBy` (`IReadOnlyList<SortExpressionKey>`).
   - `ResultType`: rolling mean → `double`, cumulative count → `long`, rank → `long`
     (`PercentRank` → `double`), everything else = source result type.
   - `Validate(schema)`: validates sub-expressions and recomputes `ResultType`.
   - Shared `WindowFunctionHelpers` (internal) maps kind → `RollingKind`/`CumulativeKind`/
     `RankKind` and `GetResultType` — used by both the AST and the ops so schema and
     execution agree.
2. **Fused evaluator hydration** (`src/Nivara/Expressions/FusedExpressionEvaluator.cs`).
   - Before the passthrough/inference in `EvaluateCore`, if the tree contains any
     `WindowExpression`, rewrite it bottom-up: each window node is evaluated via the
     existing kernels (`CalculateRolling`/`CalculateCumulative`/`CalculateCumulativeCount`/
     `CalculateShift`/`RankKernel.Compute` over materialized expression keys) and replaced
     by a `ColumnReference` to a synthetic column injected into the input dictionary.
   - Nested windows compose because window sub-evaluations recurse through `Evaluate`.
   - A bare window expression short-circuits to the direct-column passthrough (no outer
     kernel), so standalone `Select(RollingSum(...))` stays a single materialization.
   - `CollectColumnReferences` (vacuous-empty check) gains a `WindowExpression` case.
3. **Window ops over expression sources** (`src/Nivara/Operations/WindowOperations.cs`,
   `RankOperations.cs`).
   - `WindowOperationBase` gains optional `SourceExpression`; `Source` becomes `string?`.
     `TransformSchema`/`Execute` prefer the expression when present (validate + evaluate it,
     then run the same `Compute`/kernel). `AddWindowOperation` validation in
     `src/Nivara/Query/QueryFrame.cs` relaxed accordingly.
   - `RankOperation` gains an expression ctor (`IReadOnlyList<SortExpressionKey>` orderBy,
     `IReadOnlyList<ColumnExpression>?` partitionBy) that delegates to the fused evaluator
     via a constructed `WindowExpression`.
4. **Plan layer** (`src/Nivara/Query/QueryPlanVisitor.cs`, `QueryPlan.cs`).
   - Add `OperationType.Rolling`/`.Cumulative`/`.Shift`/`.Rank` dispatch → new
     `protected virtual VisitWindow(IQueryOperation)` in both the visitor and the
     transformer (default no-op / return-unchanged), so window ops are no longer "unknown".
   - `QueryPlan.GetOperationDetails` gains window/rank cases (kind, window size, periods,
     order/partition) for `Describe()`.
   - `WindowNode` already models name-based windows; no structural change (plan nodes are
     constructed by consumers, not from ops). Schema propagation is covered by the ops'
     `TransformSchema` + `WindowExpression.ResultType` work above.
5. Tests + docs (see change units).

## Change units (each ends with a green build)

- [ ] **U1 — WindowExpression AST + factories.** New `WindowFunctionKind`,
  `WindowExpression`, `WindowFunctionHelpers`, `ColumnExpressions` window factories
  (`RollingSum`/`RollingMean`/`RollingMin`/`RollingMax`, `CumulativeSum`/`Max`/`Min`/
  `Product`/`Count`, `Shift`/`Lead`, `RowNumber`/`Rank`/`DenseRank`/`PercentRank`).
  Schema/result-type unit tests. No behavior change to existing paths.
- [ ] **U2 — Fused evaluator hydration.** Window-in-`Select` composition; nested windows;
  standalone window short-circuit; fused-path guardrail test; vacuous-empty reference
  collection covers windows.
- [ ] **U3 — Window ops over computed sources.** `SourceExpression` on Rolling/Cumulative/
  Shift + expression ctor for `RankOperation`; `QueryFrame` expression overloads
  (`RollingSum(ColumnExpression, ...)`, `CumulativeSum(expr, ...)`, `Shift(expr, ...)`,
  `Lead(expr, ...)`, `Rank(resultColumn, orderBy: SortExpressionKey[], partitionBy:
  ColumnExpression[])`). Pipeline tests comparing expression vs. name-based results.
- [ ] **U4 — Plan layer.** Visitor/transformer window dispatch + `GetOperationDetails`.
  Tests exercising `QueryPlanVisitor`/`Describe()` with window ops.
- [ ] **U5 — Docs.** `CHANGELOG.md` Unreleased entry; `docs/LINQ.md` window-expression
  section.
- [ ] Full `dotnet build Nivara.slnx` + targeted test run; confirm all existing tests green.

## Verification

- `dotnet build Nivara.slnx` after each unit.
- Ask before running `dotnet test` (long-running); targeted test files per unit are OK after
  explicit confirmation.
- Guardrail: assert `FusedExpressionEvaluator.FusedPathEvaluationCount > 0` for
  window-composed elementwise expressions.
