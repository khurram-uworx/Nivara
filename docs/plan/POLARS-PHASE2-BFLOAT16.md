# Phase 2 — BFloat16 in the column / query-expression path

**Parent:** `docs/plan/POLARS-ROADMAP.md` (Phase 2 "Scope (remaining)")
**Status:** plan · **Not yet executed**
**Closes:** the residual Phase 2 acceptance criterion *"Half/BFloat16 columns execute through the generic path"*

---

## 0. Context

`Half` already flows through the entire columnar layer (`NumericKernelDispatcher`,
`NumericTensorKernels`, `WindowFrameExtensions`, sort/comparers, aggregation, quantile,
`NivaraSeries`, fused evaluator). `BFloat16` does **not** — issue #137 admitted it to the
**AutoDiff** domain only (`TypeValidator` + optimizer/kernels on net11). The column/query
layer still excludes it:

- `src/Nivara/Helpers/NumericKernelDispatcher.cs:42-48` — `arithmeticDomain` and
  `comparisonDomain` omit `typeof(BFloat16)`, so `NivaraColumn<T>`/`NivaraSeries<T>`
  arithmetic/comparison throw `NotSupportedException` for it.
- `src/Nivara/Expressions/ExpressionTypeInferer.cs:68-70` — comment states BFloat16 is
  "intentionally excluded from the fused expression engine".
- Every type-switch site that has a `NivaraColumn<Half>` arm lacks a `NivaraColumn<BFloat16>` arm.

## 1. Blocker / precondition (gate before any code)

> Per the current build state, the active SDK does **not** yet expose usable `BFloat16`
> support. This plan is gated on the net11 BCL landing `System.Numerics.BFloat16` with
> `IBinaryFloatingPointIeee754<BFloat16>` (AutoDiff #137 already relies on exactly this).

The **single hard gate** is `TensorPrimitives` BFloat16 coverage. `NumericTensorKernels<T>`
(`src/Nivara/Helpers/NumericTensorKernels.cs:21-135`) calls
`TensorPrimitives.Add/Subtract/Multiply/Divide/Sum/Min/Max` **directly**. Adding
`BFloat16` to `arithmeticDomain` makes the dispatcher instantiate
`createArithmetic<BFloat16>` → `NumericTensorKernels<BFloat16>` → `TensorPrimitives.Add(...)`.
That **only compiles** if the BCL ships those overloads for `BFloat16`.

**Step 0 — verify on the active SDK (do this first):**
1. Confirm `typeof(System.Numerics.BFloat16).IsAssignableTo(typeof(IFloatingPointIeee754<BFloat16>))`.
2. Probe `TensorPrimitives` for `BFloat16` overloads: `Add`, `Subtract`, `Multiply`,
   `Divide`, `Sum`, `Min`, `Max` (the set used by `NumericTensorKernels`). Note that AutoDiff
   already uses `TensorPrimitives.Dot` for `BFloat16` (CHANGELOG #137), so `Dot` is known-good;
   the arithmetic surface is what must be confirmed.

**Branch on the result:**

- **Branch A — full BFloat16 `TensorPrimitives` overloads present (net11):** add `BFloat16`
  to the domains and let the generic kernel compile. SIMD path works out of the box.
- **Branch B — only `Dot` (or nothing) present:** `NumericTensorKernels<BFloat16>` cannot use
  `TensorPrimitives` for arithmetic. Provide a `BFloat16`-specific scalar-fallback kernel set
  (operator/`INumber<T>` loops, exactly like the existing `SubtractFrom`/`DivideBy` scalar
  loops in `NumericTensorKernels.cs:38-63`) and route `BFloat16` to it instead of the generic
  `TensorPrimitives` methods. No SIMD, but correct and within the generic-math collapse design.

Do **not** proceed past Step 0 until Branch A or B is confirmed; the rest of the plan is
identical except for the `NumericTensorKernels` change.

## 2. Changes (independent of branch unless noted)

All sites below mirror the existing `Half` handling.

### 2.1 Numeric dispatch core
- `Helpers/NumericKernelDispatcher.cs:42-48` — add `typeof(BFloat16)` to `arithmeticDomain`.
  Leave `comparisonDomain` as-is unless BFloat16 comparisons are required (see §2.5); window/
  sort go through typed comparers, not this domain.
- `Helpers/NumericTensorKernels.cs` — **Branch A:** nothing (generic `TensorPrimitives` works).
  **Branch B:** add `BFloat16`-typed scalar overloads (or a constrained secondary generic) for
  `Add/Subtract/Multiply/Divide/Sum/Min/Max` using `INumber<T>` loops.

### 2.2 Numeric promotion & compatibility
- `Helpers/NumericPromoter.cs:34` — extend the `Half` promotion rule to `BFloat16`
  (BFloat16 implicitly converts to float/double; no implicit conversion to/from integer types,
  mirrors `Half`).
- `Helpers/TypeCompatibilityValidator.cs:236` — add `typeof(BFloat16)` to the arithmetic
  compatibility list alongside `Half`.

### 2.3 Column factory & series
- `Helpers/ColumnFactory.cs` — add `BFloat16` to the extended CLR domain the factory handles.
- `NivaraSeries.cs:924` — add a `BFloat16 v => (double)(BFloat16)(object)value!` arm to the
  numeric-conversion switch (mirror `Half` → double).

### 2.4 Window functions
- `WindowFrameExtensions.cs:299,322,345,370` — add `NivaraColumn<BFloat16>` arms to the
  rolling / cumulative / cumulative-count / shift dispatch (mirror `Half`).

### 2.5 Sort & comparison
- `Operations/SortOperation.cs:307` — add `NivaraColumn<BFloat16> c => compareTyped(c, ...)`.
- `Operations/SingleColumnComparers.cs:56` — add `NivaraColumn<BFloat16> c => Create(c, sortKey)`.

### 2.6 Aggregation & quantile
- `Operations/AggregationFunction.cs:194,236,271` — add `BFloat16` to the Sum/Mean
  result-type promotion (BFloat16 Sum → `double`, Mean → `double`, mirroring `Half`).
- `Helpers/QuantileKernel.cs:58,140` — add `BFloat16` arm routing through `(double)v`
  (mirror `Half`).

### 2.7 Fused expression engine
- `Expressions/ExpressionTypeInferer.cs:68-70` — update the comment: BFloat16 is no longer
  excluded. `IsGenericMathType(typeof(BFloat16))` already returns `true` via the interface
  check (BFloat16 implements `INumber<BFloat16>`), so no code change is needed there once the
  kernel path below supports it.
- `Expressions/FusedKernel.cs:310` — add `BFloat16 v => T.CreateChecked(v)` to
  `CoerceLiteral<T>` (mirror `Half v => T.CreateChecked(v)`).
- `Expressions/FusedExpressionEvaluator.cs:930` — extend the extended-domain comment to list
  `BFloat16` alongside `Half` (doc only; the `CreateChecked` fallback already covers it).

## 3. Tests (mirror the `Half` coverage)

Add `BFloat16` cases to the existing suites rather than new files:
- `tests/Nivara.Tests/Helpers/NumericTensorKernelsTests.cs` — arithmetic/comparison for
  `BFloat16` columns (null-mask propagation preserved).
- `tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs` — rolling/cumulative/shift for
  `NivaraColumn<BFloat16>`.
- `tests/Nivara.Tests/Query/` window + rank suites — one `BFloat16` expression case.
- `tests/Nivara.Tests/Expressions/FusedExpressionEvaluatorTests.cs` — a fused expression over
  a `BFloat16` column (e.g. `r => r["x"] * 1.1f + 1000`) with null-mask assertions.
- `tests/Nivara.Tests/Operations/AggregationTests.cs` + `QuantileKernelTests.cs` — Sum/Mean →
  `double`, Quantile `BFloat16` arm.

All new tests must assert **bit-equivalence to the eager per-operator result** and **null-mask
preservation**, per the Phase 2 acceptance criteria.

## 4. Acceptance criteria (from POLARS-ROADMAP Phase 2)

- [ ] `BFloat16` columns execute through the generic/column kernel path (no
      `NotSupportedException`), matching `Half`'s behavior.
- [ ] Fused expression output over `BFloat16` columns is bit-equivalent to the per-operator
      result, null-mask preserving.
- [ ] `OrderBy(r => r["BFloat16Col"] * 1.1f)` works and routes through the fused evaluator.
- [ ] Existing `QueryOptimizationPropertyTests` / `QueryExecutionPropertyTests` stay green;
      `Half` results identical to today.

## 5. Definition of done

- `dotnet build Nivara.slnx` green on the active SDK (Branch A or B resolved per Step 0).
- `dotnet test` green, including the new `BFloat16` cases above.
- `POLARS-ROADMAP.md` Phase 2 "Scope (remaining)" BFloat16 bullet marked ✅ Delivered, linking
  this plan.
- No new `object?`-boxing expression paths introduced.

## 6. Risk register

| Risk | Mitigation |
| --- | --- |
| `TensorPrimitives` lacks BFloat16 arithmetic overloads (Branch B) | scalar `INumber<T>` fallback kernel set; correct, no SIMD |
| BFloat16 precision (lower than float) surprises in aggregation promotion | Sum/Mean → `double`, mirroring `Half` |
| Build break when dispatcher instantiates `NumericTensorKernels<BFloat16>` | Step 0 gate; never add to `arithmeticDomain` before Branch resolved |
| Fused evaluator `Expression.Compile` target over `T[]` (not spans) for BFloat16 | acceptable per #155; IR/chunked span path already covers streaming |
