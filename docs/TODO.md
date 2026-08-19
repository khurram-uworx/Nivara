# TODO — Fix #305: GroupBy(keys, aggregations) silently drops aggregations

## Problem

`NivaraFrameExtensions.GroupBy(string[] keyColumns, Dictionary<string, AggregationFunction> aggregations)` at `src/Nivara/NivaraFrameExtensions.cs:1123` validates the `aggregations` parameter but never passes it to `GroupByOperation`. The method creates a `GroupByOperation` using only key columns (line 1148), so callers silently get only distinct keys with no aggregated values.

Root cause: line 1148 calls `new GroupByOperation(columnExpressions)` — the simple constructor — instead of the full constructor that accepts aggregations.

## Proposed Changes

### Step 1: Wire aggregations through

In `src/Nivara/NivaraFrameExtensions.cs`, replace lines 1147-1154:

```csharp
// Current (broken):
var groupByOperation = new GroupByOperation(columnExpressions);
var resultColumns = groupByOperation.Execute(columns);
// Apply aggregations to the grouped data
// Note: This is a simplified implementation. In a full implementation,
// we would need to modify GroupByOperation to support aggregations directly
// For now, we just return the grouped keys
```

With:

```csharp
// Convert AggregationFunction dictionary to GroupedAggregation list
var groupedAggregations = aggregations
    .Select(kvp => new GroupedAggregation(
        ResultColumnName: kvp.Key,
        Source: ColumnExpressions.Col(kvp.Key),
        Function: kvp.Value))
    .ToList();

// Create and execute group by operation with aggregations
var groupByOperation = new GroupByOperation(columnExpressions, null, groupedAggregations);
var resultColumns = groupByOperation.Execute(columns);
```

**Key types:**
- `GroupedAggregation` is `internal sealed record` in `src/Nivara/Operations/GroupByOperation.cs:199` — same assembly, accessible
- `ColumnExpressions.Col(name)` creates a `ColumnReference` — same pattern as LINQ path (`src/Nivara/Linq/NivaraQuery.cs:461`)
- `GroupByOperation(columns, keyOutputNames, aggregations)` constructor already validates and executes aggregations

### Step 2: Add tests

Add tests to `tests/Nivara.Tests/Operations/GroupByOperationTests.cs` or a new test class:

1. `GroupBy_WithAggregations_IncludesAggregatedColumns` — Sum on a numeric column, verify result has key + aggregated columns
2. `GroupBy_WithMultipleAggregations_ComputesAll` — Multiple aggregations (Sum + Mean + Max) on same/different columns
3. `GroupBy_WithAggregations_ResultValuesCorrect` — Verify actual computed values match expected

## Blast Radius

- **Changed file:** `src/Nivara/NivaraFrameExtensions.cs` — one public method body changed
- **No callers exist** — grep confirmed no code calls this overload today
- **Downstream:** `GroupByOperation.Execute` already handles aggregations correctly (line 369-386) — no changes needed there
- **Tests:** existing `GroupByOperationTests.cs` tests the operation directly — new tests will cover the extension method path

## Verification

1. `dotnet build Nivara.slnx` — confirm no compilation errors
2. `dotnet test --filter "GroupBy"` — run GroupBy-related tests
3. `dotnet test` — full suite (ask human before running)

## Planned Commits

1. `fix: wire aggregations through GroupBy extension method` — the core fix
2. `test: add tests for GroupBy with aggregations` — coverage for the fix

## GitHub Issues Log

- [ ] #305 — GroupBy(keys, aggregations) silently drops aggregations (this issue)
