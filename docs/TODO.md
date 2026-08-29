# Plan: BFloat16 in the column / query-expression path (Phase 2 residual)

**Branch:** `khurram/bfloat16` (off `main`)
**Detailed plan:** `docs/plan/POLARS-PHASE2-BFLOAT16.md`
**Closes:** Phase 2 acceptance criterion *"Half/BFloat16 columns execute through the generic path"*

> As each task executes, if deferred work or a concern is found that is outside this plan,
> create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record
> its number in the GitHub issues log below. Do not rely on memory — compaction can lose items.

---

## Problem

`Half` flows through the entire columnar layer; `BFloat16` does not. Issue #137 admitted
`BFloat16` to the **AutoDiff** domain only. The column/query layer still excludes it:
`NumericKernelDispatcher.arithmeticDomain` omits `BFloat16`, and `ExpressionTypeInferer`
intentionally excludes it from the fused evaluator.

## Blocker / Step 0 — verify BFloat16 + `TensorPrimitives` on the active SDK

This is the **gate**. `NumericTensorKernels<T>` calls `TensorPrimitives.Add/Subtract/
Multiply/Divide/Sum/Min/Max` directly. Adding `BFloat16` to `arithmeticDomain` makes the
dispatcher instantiate `NumericTensorKernels<BFloat16>` → `TensorPrimitives.Add(...)`, which
**only compiles if the BCL ships those overloads**.

**Step 0 probe:** confirm `typeof(System.Numerics.BFloat16)` implements
`IBinaryFloatingPointIeee754<BFloat16>` **and** that `TensorPrimitives` exposes `BFloat16`
overloads for the arithmetic surface used by `NumericTensorKernels`.

Branch on result:
- **Branch A (net11, full overloads):** add `BFloat16` to the domain; SIMD path works.
- **Branch B (only `Dot`, or none):** `NumericTensorKernels<BFloat16>` cannot use
  `TensorPrimitives` for arithmetic → add a `BFloat16`-specific scalar `INumber<T>` fallback
  kernel set (operator loops, like existing `SubtractFrom`/`DivideBy`). Correct, no SIMD.

Per the build state, Branch B is expected, but Step 0 must confirm before any code change.

## Changes (per detailed plan doc)

1. **Step 0** — probe + record Branch (A/B).
2. **Numeric dispatch core** — `Helpers/NumericKernelDispatcher.cs` add `typeof(BFloat16)` to
   `arithmeticDomain`; `Helpers/NumericTensorKernels.cs` (Branch B) scalar fallback set.
3. **Promotion & compatibility** — `Helpers/NumericPromoter.cs`, `Helpers/TypeCompatibilityValidator.cs`.
4. **Column factory & series** — `Helpers/ColumnFactory.cs`, `NivaraSeries.cs`.
5. **Window functions** — `WindowFrameExtensions.cs` (rolling/cumulative/count/shift arms).
6. **Sort & comparison** — `Operations/SortOperation.cs`, `Operations/SingleColumnComparers.cs`.
7. **Aggregation & quantile** — `Operations/AggregationFunction.cs` (Sum/Mean → `double`),
   `Helpers/QuantileKernel.cs`.
8. **Fused evaluator** — `Expressions/ExpressionTypeInferer.cs` comment,
   `Expressions/FusedKernel.cs` `CoerceLiteral`, `Expressions/FusedExpressionEvaluator.cs` doc.
9. **Tests** — extend existing suites with `BFloat16` cases (arithmetic, window, fused eval,
   aggregation, quantile); assert bit-equivalence + null-mask preservation.
10. **Roadmap update** — mark Phase 2 BFloat16 bullet ✅ Delivered.

## Blast radius

- **Entry point:** `NivaraColumn<T>`/`NivaraSeries<T>` arithmetic & comparison →
  `NumericKernelDispatcher` → `NumericTensorKernels<T>`. Adding `BFloat16` to the domain
  changes dispatch for that type only; no impact on existing types.
- **Downstream:** window functions, sort/comparers, aggregation, quantile, fused expression
  evaluator all gain a new supported type; existing `Half` paths are the template and are
  unchanged.
- **Tests covering changes:** `tests/Nivara.Tests/Helpers/NumericTensorKernelsTests.cs`,
  `tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`, `tests/Nivara.Tests/Query/*`,
  `tests/Nivara.Tests/Expressions/FusedExpressionEvaluatorTests.cs`,
  `tests/Nivara.Tests/Operations/AggregationTests.cs`, `QuantileKernelTests.cs`.
- **Risk:** a wrong Branch choice breaks the build (`NumericTensorKernels<BFloat16>` won't
  compile without BCL overloads). Step 0 eliminates this.

## Verification

- `dotnet build Nivara.slnx` green (per Branch).
- `dotnet test` green, including new `BFloat16` cases (ask before running).
- Roadmap Phase 2 bullet updated to ✅ Delivered.

## Planned commit list (one logical change per commit)

1. `docs: plan BFloat16 column wiring in TODO.md`
2. `chore: probe TensorPrimitives BFloat16 coverage (record Branch)` — doc/research commit
3. `feat: add BFloat16 to numeric dispatch domain (+ scalar kernel fallback if Branch B)`
4. `feat: BFloat16 numeric promotion & compatibility`
5. `feat: BFloat16 column factory & NivaraSeries conversion`
6. `feat: BFloat16 window functions (rolling/cumulative/shift)`
7. `feat: BFloat16 sort & comparers`
8. `feat: BFloat16 aggregation & quantile promotion`
9. `feat: BFloat16 fused expression evaluator wiring`
10. `test: add BFloat16 coverage to existing suites`
11. `docs: mark Phase 2 BFloat16 delivered in roadmap`

## GitHub issues log

- [ ] #NNN — one-line description (created while working on <task>)
