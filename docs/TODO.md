# TODO — Typed LINQ object model `frame.Query<T>()` (issue #130)

## Problem

IDEA.md §6.1/§6.2 describes a typed object LINQ model that is not implemented. The current LINQ
surface is expression-based (`QueryFrame` + `RowExpressionBuilder`/`ColumnExpression`). Issue #130
is decided as **implemented**: add a typed `frame.Query<T>()` object model as an ergonomic layer on
top of the existing expression engine.

```csharp
var query =
    frame.Query<Person>()
         .Where(p => p.Age > 30)
         .GroupBy(p => p.City)
         .Select(g => new
         {
             g.Key,
             AvgSalary = g.Average(p => p.Salary)
         });
```

LINQ rules (§6.2): allowed = property access, arithmetic, comparisons, boolean logic, aggregates;
rejected = loops, method calls, closures, side effects; unsupported expressions fail fast with
clear diagnostics.

## Design decisions (confirmed)

1. Results materialize both ways: `Collect()`/`ToList()` → `NivaraFrame`; `ToObjects()`/`ToRows()`
   → `IReadOnlyList<TResult>` via a compiled, cached per-type row factory.
2. GroupBy scope: `GroupBy(...).Select(g => new { g.Key, ...aggregates })` + bare `GroupBy().Collect()`
   (distinct keys). Any other op after GroupBy fails fast.
3. Everything is additive; no public API breaks.

## Proposed changes

### 1. `NivaraQuery<T>` — typed wrapper
`src/Nivara/Linq/TypedQuery.cs` (+ `TypedLinqExtensions.cs`)
- Immutable wrapper around a `QueryFrame`; new `QueryFrame` per op.
- State: wrapped frame, `T` property→column map, optional GroupBy metadata.
- Entry: `NivaraFrame.Query<T>()` — eager validation: T non-primitive class with ≥1 public
  instance property; every property maps (case-insensitive) to a column with exact or
  nullable-underlying type, else `SchemaValidationException` (fail fast, no data access).

### 2. Expression translator + validator
`src/Nivara/Linq/TypedExpressionTranslator.cs`
- Converts `Expression<Func<...>>` → `ColumnExpression`; throws
  `UnsupportedQueryExpressionException` at build time.
- Allowed: parameter, direct member access → `ColumnReference`; constant literal (closure-root
  constant rejected); `+ - * /` → Binary/Scalar; `&& ||` → Binary(And/Or); comparisons;
  unary `!` → new `NotExpression`; benign Convert unwrapped; `MemberInit`/anonymous `New` in
  Select (member name = result column name).
- Rejected (with clear message): method calls, captured variables/closures, nested property
  access, array/index access, `InvocationExpression`, ternary, `%`, string `+`, nested lambdas.

### 3. GroupBy + aggregation
`src/Nivara/Operations/GroupByOperation.cs`, `src/Nivara/Operations/AggregationFunction.cs`
- Add `GroupedAggregation(string ResultColumnName, ColumnExpression Source, AggregationFunction Function)`.
- New ctor `GroupByOperation(ColumnExpression[] keys, string[] keyOutputNames, IReadOnlyList<GroupedAggregation> aggs)`;
  existing single-arg ctor unchanged.
- `TransformSchema`: keys + `Function.GetResultType(sourceType)` per agg.
- `Execute`: `CreateGroupsInternal` → `ExtractDistinctKeyValues` per key + `ApplyToGroups` per agg.
- Add `RowCountAggregation` (for `g.Count()` → `(long)groupIndices.Count`).
- Parallel strategy (`ParallelExecutionStrategy.cs`): route aggregate GroupBy through serial
  `op.Execute(input)` (fallback, like streaming computed sort keys).
- Typed flow: `GroupBy` records metadata (no op appended); `Select` on a grouped query appends a
  single `GroupByOperation` with key output names + aggregations; bare `Collect()` appends key-only
  GroupBy.

### 4. `NotExpression`
`src/Nivara/Expressions/ColumnExpression.cs` + `src/Nivara/Helpers/ExpressionEvaluator.cs`
- New `NotExpression(ColumnExpression operand)`, bool result, `Name = "!(...)"`, evaluator branch
  (`result = !value`, null mask = operand mask).

### 5. Marker grouping type
`src/Nivara/Linq/Grouping.cs` — `Grouping<TKey,T>` with `Key` + `Average/Sum/Min/Max/Count`
signatures for C# type inference (never invoked).

### 6. Infrastructure
- `src/Nivara/Query/QueryFrame.cs`: internal `WithOperation(IQueryOperation)` (expose source/ops
  internally) for the typed layer.
- `src/Nivara/Exceptions/QueryEngineExceptions.cs`: `UnsupportedQueryExpressionException : Exception`.
- `docs/LINQ.md`: "Typed Object LINQ" section.

## Tests (NUnit, `tests/Nivara.Tests/`)
- `TypedLinqQueryTests.cs`: issue end-to-end example; Where arithmetic/comparison/`&&`/`||`/`!`/
  string equality; Select projections (computed + anonymous); OrderBy/ThenBy; GroupBy aggregates
  (Average/Sum/Count/Min/Max, `Count()` row count); bare GroupBy distinct keys; `ToObjects()`
  round-trip; nullable columns; frame/objects agreement.
- `TypedLinqValidationTests.cs`: fail-fast diagnostics (missing/mismatched column, method calls,
  closures, nested access, ternary, nested lambda); message key phrases.
- Extend `ExpressionEvaluatorTests` (NotExpression) and `GroupByOperationTests` (aggregations +
  key output names); parallel-strategy serial fallback test.

## Verification
- `dotnet build Nivara.slnx` after each logical change.
- `dotnet test` only after explicit human confirmation (repo convention).

## Planned commits (logical units)
1. ✓ `docs: plan typed LINQ object model (issue #130) in TODO.md` — `f68519f`
2. ✓ Add `UnsupportedQueryExpressionException` + `NotExpression` + evaluator branch — `31134a0`
3. ✓ Extend `GroupByOperation` with aggregations + key output names; add `RowCountAggregation` — `369a660`
4. ✓ `QueryFrame.WithOperation` internal helper — `7c442a6`
5. ✓ Typed LINQ layer (`Grouping`, `TypedExpressionTranslator`, `NivaraQuery<T>`, `Query<T>()`, row factory) — done; 2298 tests pass
6. Document typed API in `docs/LINQ.md` — pending
7. ✓ Tests — `tests/Nivara.Tests/Query/TypedLinqTests.cs` (28 tests: entry validation, Where, unsupported expressions, Select, OrderBy/ThenBy, Skip/Take, GroupBy aggregates, distinct keys, nullable columns, Schema/ExplainPlan)
8. `docs: remove TODO.md — plan executed`
9. Offer push + PR (human-confirmed)

## Notes from implementation
- `NivaraQuery<T>` and `NivaraGroupedQuery<TKey,T>` live in `src/Nivara/Linq/NivaraQuery.cs`; grouping marker in `Grouping.cs`; translator in `TypedExpressionTranslator.cs`; metadata/validation in `TypedLinqMetadata.cs`; projection building in `TypedProjectionBuilder.cs`; compiled row factory in `TypedRowFactory.cs`; `Query<T>()` entry in `TypedLinqExtensions.cs`.
- Grouped query holds the base frame without a GroupBy op; `Select` appends `GroupByOperation(keys, keyNames, aggregations)` + `SelectOperation`, bare `Collect` appends a key-only `GroupByOperation`.
- Aggregate result column name = projection member name; `g.Key` → `ColumnReference(keyColumnName)`; aggregate pass-through uses `ColumnReference(memberName)`.
- `Grouping<TKey,T>` deliberately omits `Count(Func<T,bool>)` so count-with-predicate is a compile error; Sum overloads limited to types `SumAggregation.Apply` supports.
- `p.City == null` comparisons fail at Collect (literal coercion only supports `lit.Value != null`) — follow-up for null-check semantics.

## Follow-ups (out of scope)
- Ops after GroupBy other than aggregate-`Select`/`Collect` (Where/OrderBy on groups).
- Nested property access, ternary, string `+`, `%`, `First/Last`, typed joins.
- Parallel aggregation merging for GroupBy-with-aggregates (serial fallback for now).
