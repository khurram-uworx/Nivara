# Plan: Issue #1157 — Typed null comparison (`r.Name == null`)

## Problem

The typed LINQ path cannot express a null test. `Where(r => r.Name == null)` builds a
`ComparisonExpression` against an `object`-typed null literal, which fails schema validation with
"Left operand type: String, Right operand type: Object", throwing at `Collect()`.

Root cause:
- `TypedExpressionTranslator.TranslateBinary` maps `Equal`/`NotEqual` straight to `ComparisonExpression`.
- `LiteralExpression` types a null literal as `object`.
- `ComparisonExpression.Validate` only coerces non-null literals.

## Proposed changes

1. Add `IsNullExpression` / `IsNotNullExpression` to `ColumnExpression.cs` (internal sealed, `ResultType == bool`),
   plus `ColumnExpressions.IsNull` / `IsNotNull` factories.
2. Rewrite null comparisons in `TypedExpressionTranslator.TranslateBinary`; reject non-nullable value-type
   properties with a clear `UnsupportedQueryExpressionException`.
3. Evaluate via hydration in `FusedExpressionEvaluator`: `ContainsNullCheckExpression` + `HydrateNullChecks`
   produce a null-free synthetic `bool` column from the operand's leaf null masks.
4. Update `CollectColumnReferences` for the new nodes.
5. Document the form in `docs/LINQ.md`.

## Blast radius

- `src/Nivara/Expressions/ColumnExpression.cs` — new classes (additive).
- `src/Nivara/Linq/TypedExpressionTranslator.cs` — null-comparison branch.
- `src/Nivara/Expressions/FusedExpressionEvaluator.cs` — hydration pass + collect.
- `docs/LINQ.md` — documentation.

Downstream: `FilterOperation`, streaming/parallel strategies route through `FusedExpressionEvaluator`,
so they inherit the fix. No kernel-IR changes.

## GitHub issues log

(none created yet)

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted: `ColumnExpressionTests`, `LinqQueryTests`
- Docs namespace if `docs/LINQ.md` edited.