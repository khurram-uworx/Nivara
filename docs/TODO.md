# Plan: Add ConditionalExpression (ternary `?:`) support to the LINQ expression engine

## Problem

`TypedExpressionTranslator` explicitly rejects `ConditionalExpression` (C# ternary `?:`) at
`TypedExpressionTranslator.cs:51`, causing `UnsupportedQueryExpressionException` for patterns like:

```csharp
g.Sum(r => r.StatusCode >= 500 ? 1 : 0)
g.Average(r => r.StatusCode >= 500 ? 1.0 : 0.0)
```

This breaks `Analysis.AnalyzeRegionalPartitioning` and `Analysis.AnalyzeGroupedAggregation`,
affecting 8+ tests across scenarios A–D.

## Root cause

Line 51 of `TypedExpressionTranslator.cs`:
```csharp
ConditionalExpression _ => throw Unsupported(expression, "ternary (?:) expressions are not supported"),
```

The C# compiler compiles `?:` into `System.Linq.Expressions.ConditionalExpression` (printed as
`IIF(...)`), but the Nivara expression model has no `ConditionalExpression` type to represent it.

## Blast radius

**Modified files** (7 source + 1 test):
- `src/Nivara/Expressions/ColumnExpression.cs` — new type, new factory method
- `src/Nivara/Expressions/KernelIR.cs` — new `KernelOp.Conditional`, routing guard
- `src/Nivara/Expressions/KernelLowerer.cs` — new lowering case
- `src/Nivara/Linq/TypedExpressionTranslator.cs` — translate instead of throw
- `src/Nivara/Expressions/ExpressionTypeInferer.cs` — type inference + signature
- `src/Nivara/Expressions/FusedExpressionEvaluator.cs` — compiled path, window hydration, column refs
- `tests/Nivara.Tests/Query/TypedLinqTests.cs` — new tests

**NOT modified**: `src/Nivara/Expressions/FusedKernel.cs` (span kernel) — conditional plans are
excluded from `IsUniformNumeric` routing, so they never reach the span kernel.

**Downstream callers**: `Analysis.cs` (samples) — unchanged, already uses the correct pattern.

## Implementation steps

### Step 1: Add `ConditionalExpression` to the expression model (`ColumnExpression.cs`)
- New `ConditionalExpression(ColumnExpression test, ColumnExpression trueValue, ColumnExpression falseValue)`
  - Properties: `Test`, `TrueValue`, `FalseValue`
  - `ResultType` = promoted type of true/false (via `NumericPromoter`); null → `typeof(object)`
  - `Validate()`: validate all three sub-expressions; verify test resolves to bool; verify true/false are type-compatible
  - `Name` → `"(test ? trueValue : falseValue)"`
- Add `ColumnExpressions.Conditional(test, trueValue, falseValue)` factory

### Step 2: Add `KernelOp.Conditional` to kernel IR (`KernelIR.cs`)
- Add `Conditional` to `KernelOp` enum
- Update `KernelPlan.IsUniformNumeric`: exclude plans containing `Conditional` (the span kernel's
  `EvalNodes<T>` stack is `T[]` and can't handle the bool-producing test sub-expression)
- `MaxStackDepth` already computed as `nodes.Count(... Column/Literal)` — conditional has no extra
  Column/Literal children beyond its three sub-expression nodes, so no change needed

### Step 3: Add lowering in `KernelLowerer.cs`
- In `LowerNode` switch, add `ConditionalExpression` case:
  - Lower test, trueValue, falseValue children
  - Emit `KernelOp.Conditional` node with ComputeType = promoted type of true/false

### Step 4: Translate in `TypedExpressionTranslator.cs`
- Change `ConditionalExpression` case from throwing to:
  ```csharp
  ConditionalExpression cond => new ConditionalExpression(
      Translate(cond.Test),
      Translate(cond.IfTrue),
      Translate(cond.IfFalse)),
  ```

### Step 5: Add type inference (`ExpressionTypeInferer.cs`)
- In `InferNode` switch, add `ConditionalExpression` case:
  - Infer true type, false type
  - Return `NumericPromoter.GetPromotedType(trueType, falseType)` or null if incompatible
- In `BuildSignature`, add `ConditionalExpression` case

### Step 6: Add compiled evaluation support (`FusedExpressionEvaluator.cs`)
- `CollectColumnReferences` — add `ConditionalExpression` case, recurse into all three children
- `ContainsWindowExpression` — add `ConditionalExpression` case, recurse into all three children
- `HydrateWindows` — add `ConditionalExpression` case, recurse into all three children
- `BuildCompiledNode` — add `KernelOp.Conditional` case:
  - Build test (bool), true (converted to ComputeType), false (converted to ComputeType)
  - Emit `Expression.Condition(test, true, false)`
- `BuildSignature` — add `ConditionalExpression` case

### Step 7: Add tests (`TypedLinqTests.cs`)
- `GroupBy_TernaryInSum_ComputesCorrectly`
- `GroupBy_TernaryInAverage_ComputesCorrectly`
- `Select_Ternary_ProducesCorrectValues`

### Step 8: Build + verify
- `dotnet build Nivara.slnx` — verify no compilation errors
- `dotnet test` — verify ScenarioD_RegionalPartitioning_ApSouth1Present and all related tests pass

## Deferred work / concerns

(None identified yet)

## GitHub issues log

- (none yet)
