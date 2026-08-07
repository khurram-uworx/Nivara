# Nivara LINQ Query Engine

Nivara provides a deferred, plan-based LINQ-like query engine over typed columnar DataFrames. Queries are built immutably via `QueryFrame` and materialized lazily on demand.

---

## Architecture

```
User API              QueryFrame / Linq Extensions (Where, Select, OrderBy, GroupBy)
Plan Layer            QueryPlan (IQuerySource + ordered IQueryOperation[])
Execution Layer       QueryExecutor (internal) / IExecutionStrategy (Eager, Lazy, Streaming, Parallel)
```

**Key design principles:**
- Queries are **deferred** — nothing executes until `.Collect()` is called
- Each operation is an immutable node in a pipeline; no mutation
- Schema is validated eagerly when the `QueryPlan` is constructed (before data flows)
- Two API surfaces: expression-based (`ColumnExpression` operator overloads) and lambda-based (`RowExpressionBuilder`)

---

## Entry Points

### From a NivaraFrame

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 70000, 50000, 90000 }))
);

var query = frame.AsQueryFrame();  // QueryFrame — deferred, nothing executed yet
```

### From file sources

```csharp
var query = Json.ScanJsonAsQueryFrame("data.json");
```

### From custom IQuerySource

```csharp
public interface IQuerySource : IDisposable
{
    Schema Schema { get; }
    bool IsLazy { get; }
    IReadOnlyDictionary<string, IColumn> Execute();

    // Chunked and async members (used by streaming/parallel strategies)
    Task<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(CancellationToken cancellationToken = default);
    bool CanReadInChunks => false;
    int? EstimatedRowCount => null;
    IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize);
    ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(int chunkIndex, int chunkSize, CancellationToken cancellationToken = default);
    IAsyncEnumerable<IReadOnlyDictionary<string, IColumn>> ToAsyncEnumerable(int chunkSize, CancellationToken cancellationToken = default);
}
```

The `Execute()` and `Schema` members are required for basic operation. The chunked and async members enable streaming and parallel execution strategies.

---

## Core Types

| Type | Role |
|------|------|
| `QueryFrame` | Deferred query builder; stores `IQuerySource` + `List<IQueryOperation>` |
| `QueryPlan` | Validated plan with `Source`, `Operations` list, and computed `ResultSchema` |
| `IQueryOperation` | Single pipeline step — `TransformSchema()` + `Execute()` |
| `IQuerySource` | Data source — materializes columns on demand |
| `QueryExecutor` | Executes a `QueryPlan` by piping columns through operations |
| `ColumnExpression` | Composable expression tree (reference, literal, binary, comparison, scalar) |
| `RowExpressionBuilder` | Lambda-entry point for column access — `row["ColumnName"]` |

---

## Expression System

`ColumnExpression` is an abstract base with concrete variants:

| Expression | Purpose |
|------------|---------|
| `ColumnReference` | References a column by name |
| `LiteralExpression` | A constant value |
| `BinaryExpression` | Arithmetic/logical operation between two expressions |
| `ComparisonExpression` | Comparison (`>`, `<`, `==`, etc.) returning bool |
| `ScalarExpression` | Column-scalar arithmetic |

`ColumnExpressions` static factory:
```csharp
ColumnExpressions.Col("Name")       // column reference
ColumnExpressions.Col<int>("Id")    // typed column reference
ColumnExpressions.Lit(42)           // literal
```

C# operator overloads enable natural syntax:
```csharp
ColumnExpressions.Col("Price") > 50.0                  // ComparisonExpression
ColumnExpressions.Col("Qty") * ColumnExpressions.Col("Price")  // BinaryExpression
ColumnExpressions.Col("Score") + 10                     // ScalarExpression
```

The `ExpressionEvaluator` (in `Helpers/ExpressionEvaluator.cs`) walks the tree at execution time and produces result `IColumn` instances from input column dictionaries.

---

## Operations Reference

| Operation | `OperationType` | Class | Effect |
|-----------|-----------------|-------|--------|
| **Filter** | `"Filter"` | `FilterOperation` | Keeps rows where a boolean expression is true |
| **Select** | `"Select"` | `SelectOperation` | Projects a subset of columns/expressions |
| **Sort** | `"Sort"` | `SortOperation` | Reorders rows by one or more sort keys |
| **GroupBy** | `"GroupBy"` | `GroupByOperation` | Groups rows by key columns, returns distinct keys |
| **Join** | `"Join"` | `JoinOperation` | Hash-based join between two frames (Inner/Left/Right/FullOuter) |
| **Projection** | `"Projection"` | `ProjectionOperation` | Renames and/or selects columns |
| **Slice** | `"Slice"` | `SliceOperation` | Row-range slicing by index |
| **Distinct** | `"Distinct"` | `DistinctOperation` | Removes duplicate rows (all columns or specified subset) |
| **SelectRows** | `"SelectRows"` | `SelectRowsOperation` | Selects specific rows by positional index |
| **Concatenate** | `"ConcatenateVertical"` / `"ConcatenateHorizontal"` | `ConcatenationOperation` | Vertical (row append) or horizontal (column append) concatenation |
| **Rolling** | `"Rolling"` | `RollingOperation` | Appends a rolling sum/mean/min/max column over a fixed trailing window |
| **Cumulative** | `"Cumulative"` | `CumulativeOperation` | Appends a cumulative sum/max/min/product/count column |
| **Shift** | `"Shift"` | `ShiftOperation` | Appends a lag (`Shift`) or lead (`Lead`) column |

Each operation implements `IQueryOperation`:
```csharp
public interface IQueryOperation
{
    string OperationType { get; }
    Schema TransformSchema(Schema inputSchema);
    IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input);
}
```

### Filter

Builds a boolean mask by evaluating the condition, then applies it to all columns.

```csharp
public QueryFrame Filter(ColumnExpression condition)
```

Execution:
1. `ExpressionEvaluator.EvaluateBoolean(condition, input)` → `NivaraColumn<bool>`
2. Iterates rows, collecting indices where mask is `true`
3. Creates new columns from those indices (type-dispatched per element type)

### Select

Evaluates each `ColumnExpression` against the input and returns only those columns.

```csharp
public QueryFrame Select(params ColumnExpression[] columns)
public QueryFrame Select(params string[] columnNames)
```

### Sort

Creates an index array and sorts it using `MultiColumnComparer`, then reorders all columns.

```csharp
public QueryFrame Sort(string columnName, SortDirection direction, NullOrdering nullOrdering, bool stable)
public QueryFrame Sort(IEnumerable<SortKey> sortKeys, bool stable)
```

- Supports stable vs unstable sorting
- `NullOrdering.NullsFirst` / `NullOrdering.NullsLast`
- Multi-key sorting via `SortKey[]`

### GroupBy

Hash-based grouping using `GroupKey` (composite key with precomputed hash). Returns distinct key values.

```csharp
public QueryFrame GroupBy(params string[] columnNames)
public QueryFrame GroupBy(params ColumnExpression[] columns)
```

### Join

Hash-map join between two frame snapshots. Builds a right-side hash table, probes with left rows.

**Supported join types:**
- `JoinType.Inner`
- `JoinType.Left`
- `JoinType.Right`
- `JoinType.FullOuter`

**Column disambiguation strategies:**
- `ColumnDisambiguationStrategy.Prefix` → `left_Name`, `right_Name`
- `ColumnDisambiguationStrategy.Suffix` → `Name_left`, `Name_right`
- `ColumnDisambiguationStrategy.Error` → throw on conflict

Join key columns are coalesced in outer joins (left value wins, falls back to right).

### Distinct

Removes duplicate rows. Without arguments, deduplicates on all columns. With column names, deduplicates on the specified subset.

```csharp
public QueryFrame Distinct()
public QueryFrame Distinct(params string[] columnNames)
```

### SelectRows

Selects specific rows by positional index.

```csharp
public QueryFrame SelectRows(params int[] indices)
```

### Window functions

Whole-column window operations append a computed result column while preserving all input columns. Rolling/cumulative/shift operations are **non-parallelizable** and **non-streamable** (they require the full column).

```csharp
public QueryFrame RollingSum(source, resultColumn, windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
public QueryFrame RollingMean(source, resultColumn, windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
public QueryFrame RollingMin(source, resultColumn, windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
public QueryFrame RollingMax(source, resultColumn, windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
public QueryFrame CumulativeSum(source, resultColumn, Func<object?>? nullHandler = null)
public QueryFrame CumulativeMax / CumulativeMin / CumulativeProduct(source, resultColumn, Func<object?>? nullHandler = null)
public QueryFrame CumulativeCount(source, resultColumn)          // long result, type-agnostic
public QueryFrame Shift(source, resultColumn, periods, object? fillValue = null)
public QueryFrame Lead(source, resultColumn, periods, object? fillValue = null)
```

Rolling output is null until the window holds at least `minPeriods` valid values (default: the full window). Cumulative ops skip nulls (value carries forward); null positions stay null. `Shift`/`Lead` move in nulls (or `fillValue`) at the boundaries. When `nullHandler` is set, nulls are replaced by the handler output and every position satisfies the window. `RollingMean` returns a `double` column.

```csharp
var result = frame.AsQueryFrame()
    .RollingSum("Price", "priceSum3", 3)
    .CumulativeCount("Price", "tickCount")
    .Shift("Price", "prevPrice", 1)
    .Collect();
```

The same signatures are available directly on `NivaraFrame` (see `src/Nivara/Tensors/WindowFunctions.cs` and `src/Nivara/WindowFrameExtensions.cs`).

### Skip / Take / Slice

Row-range operations for pagination and windowing.

```csharp
public QueryFrame Skip(int count)
public QueryFrame Take(int count)
public QueryFrame Slice(int skip, int take)
```

`Skip` skips the first N rows, `Take` takes the first N rows, `Slice` combines both into a single operation.

---

## API Surfaces

### 1. Expression-based API (on QueryFrame)

Every `Filter`/`Select`/`Sort`/`GroupBy`/`Distinct`/`Skip`/`Take`/`Slice`/`SelectRows` call returns a **new** `QueryFrame` with the operation appended.

```csharp
// Basic filter + select
var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Salary") > 50000)
    .Select("Name", "Salary")
    .Collect();

// Row-range operations
var page = frame.AsQueryFrame().Skip(10).Take(5).Collect();

// Distinct on specific columns
var unique = frame.AsQueryFrame().Distinct("Department").Collect();

// Select specific rows by index
var sampled = frame.AsQueryFrame().SelectRows(0, 2, 4).Collect();
```

### 2. Lambda-based LINQ API (extension methods)

```csharp
using Nivara.Linq;

var result = frame.AsQueryFrame()
    .Where(row => row["Salary"] > 50000)
    .Select(row => row["Name"], row => row["Salary"])
    .OrderBy(row => row["Name"])
    .ToList();  // alias for Collect()
```

The `RowExpressionBuilder` singleton's indexer `this[string]` returns `ColumnExpressions.Col(name)`, so the lambda composes into the same `ColumnExpression` tree.

Available LINQ extensions (`NivaraLinqExtensions` in `src/Nivara/Linq/NivaraLinqExtensions.cs`):
| Method | Maps to | Notes |
|--------|---------|-------|
| `Where(predicate)` | `Filter(expression)` | Accepts `Func<RowExpressionBuilder, ColumnExpression>` |
| `Select(selectors...)` | `Select(expressions)` | Accepts `params Func<RowExpressionBuilder, ColumnExpression>[]` |
| `Select(columnNames...)` | `Select(expressions)` | String overload |
| `OrderBy(keySelector)` | `Sort(columnName, Ascending)` | Supports descending via `OrderByDescending` |
| `OrderByDescending(keySelector)` | `Sort(columnName, Descending)` | |
| `ToList()` | `Collect()` | Materializes to `NivaraFrame` |
| `ToNivaraFrame()` | `Collect()` | Alias for `ToList` |

### 3. Frame-level operations (extension methods on NivaraFrame)

Join operations are extension methods in `NivaraFrameExtensions`, not through QueryFrame:

```csharp
var joined = left.InnerJoin(right, "KeyColumn");                    // same column name
var leftJoin = left.LeftJoin(right, "OrderId", "Id");              // different column names
var rightJoin = left.RightJoin(right, "Id");
var fullOuter = left.FullOuterJoin(right, "Id");
```

All join methods accept additional optional parameters: `ColumnDisambiguationStrategy`, `leftPrefix`, and `rightPrefix`.

### 4. Typed Object LINQ API (`frame.Query<T>()`)

`NivaraFrame.Query<T>()` (requires `T : class, new()`; `using Nivara.Linq;`) returns an immutable
`NivaraQuery<T>`. Every property of `T` maps to a column with the same name (case-insensitive) and an
exact or nullable-underlying type, validated eagerly at `Query<T>()` (no data access) with
`SchemaValidationException` on mismatch.

```csharp
public sealed class Person
{
    public string Name { get; set; }
    public string City { get; set; }
    public int Age { get; set; }
    public double Salary { get; set; }
}

var result = frame.Query<Person>()
    .Where(p => p.Age > 30 && p.City == "NYC")
    .OrderByDescending(p => p.Salary)
    .Select(p => new { p.Name, SalaryRaised = p.Salary * 1.1 })
    .ToObjects();   // IReadOnlyList<anonymous>
```

Supported operators: property access, constant literals, `+ - * /` arithmetic, comparisons,
`&&`/`||`/`!` boolean logic. Method calls, captured variables/closures, nested property access,
array/index access, ternary, `%`, and string `+` fail fast at build time with
`UnsupportedQueryExpressionException`.

Materialization:
- `Collect()` / `ToList()` → `NivaraFrame` (pipeline runs lazily like the expression API).
- `ToObjects()` / `ToRows()` → `IReadOnlyList<TResult>` via a compiled, cached per-type row factory.
- `Schema` / `ExplainPlan()` behave like the expression API.
- `AsQueryFrame()` exposes the underlying expression pipeline for further composition.

Grouping:
- `GroupBy(key)` returns `NivaraGroupedQuery<TKey,T>`; bare `Collect()` yields the distinct keys.
- Aggregate `Select` reads `g.Key` plus `g.Average/Sum/Count/Min/Max(p => ...)` (the `Grouping<TKey,T>`
  marker exposes these for C# type inference; they are never invoked at runtime).
- Any operation other than an aggregate `Select`/`Collect` after `GroupBy` fails fast.

```csharp
var stats = frame.Query<Person>()
    .Where(p => p.Age > 18)
    .GroupBy(p => p.City)
    .Select(g => new { g.Key, AvgSalary = g.Average(p => p.Salary), People = g.Count() })
    .ToObjects();
```

Row factories require a public parameterless constructor (anonymous types via `Select` work
automatically); a null cell materialized into a non-nullable value-type property throws at
materialization time.

---

## Execution Pipeline

### QueryPlan construction

When `.Collect()` is called, a `QueryPlan` is created:

1. `Source` + `Operations` are captured
2. `ResultSchema` is computed by piping `Source.Schema` through each operation's `TransformSchema()`
3. Schema validation errors are thrown eagerly (before data access)

### QueryExecutor.Execute()

`QueryExecutor` is an `internal sealed class` invoked by `QueryFrame.Collect()`:

```csharp
internal NivaraFrame Execute(QueryPlan plan)
```

1. Validates the plan schema
2. Calls `plan.Source.Execute()` to get initial `IReadOnlyDictionary<string, IColumn>`
3. Iterates `plan.Operations`, piping columns through each `operation.Execute(input)`
4. Wraps final columns into a `NivaraFrame`

### Lazy validation

Operations validate schemas in `TransformSchema()` and defer row-level checks to `Execute()`, providing fast-failure for schema mismatches while allowing lazy sources to avoid materializing data during validation.

### Execution Strategies

The `ExecutionEngine` supports four strategies (not used by default `Collect()`):

| Strategy | Class | Behavior |
|----------|-------|----------|
| **Eager** | `EagerExecutionStrategy` | Immediate execution (same as default) |
| **Lazy** | `LazyExecutionStrategy` | Defers plan execution with optimization |
| **Streaming** | `StreamingExecutionStrategy` | Chunk-based processing with memory budget |
| **Parallel** | `ParallelExecutionStrategy` | Multi-threaded chunk dispatch |

---

## Diagnostics and Optimization

### ExplainPlan

```csharp
Console.WriteLine(query.ExplainPlan());
```

Outputs a tree view of the query plan:
```
Query Execution Plan:
├─ Source: MemorySource
│  └─ Schema: Name: String, Salary: Double
├─ Operations:
│  ├─ 1. Filter
│  │  └─ Condition: (Salary > 50000)
│  └─ 2. Select
│     └─ Schema: Name: String, Salary: Double
└─ Result Schema: Name: String, Salary: Double
```

### AnalyzeOptimizations

```csharp
var suggestions = query.AnalyzeOptimizations();
// Returns: ["Multiple filter operations detected...",
//           "Filter operations on lazy source..."]
```

Checks for:
- Multiple filter operations (suggests combining)
- Multiple select operations (suggests combining)
- Predicate pushdown opportunities on lazy sources
- Unused columns (suggests explicit selection)

### QueryPlan serialization

```csharp
var json = query.ToQueryPlan().Serialize();
```

Produces a JSON representation with source info, operation details, and schemas.

---

## Complete Examples

### Basic filter + select

```csharp
var frame = NivaraFrame.Create(
    ("Product", NivaraColumn<string>.Create(new[] { "A", "B", "C", "D" })),
    ("Price", NivaraColumn<double>.Create(new[] { 10.0, 25.0, 5.0, 30.0 })),
    ("InStock", NivaraColumn<bool>.Create(new[] { true, false, true, true }))
);

var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Price") > 10.0)
    .Filter(ColumnExpressions.Col("InStock") == true)
    .Select("Product", "Price")
    .Collect();
```

### LINQ-lambda syntax with computed columns

```csharp
var result = frame.AsQueryFrame()
    .Where(row => row["Price"] > 10.0)
    .Select(
        row => row["Product"],
        row => row["Price"] * 1.1   // computed scalar expression
    )
    .OrderByDescending(row => row["Price"])
    .ToList();
```

### GroupBy distinct values

```csharp
var items = NivaraFrame.Create(
    ("Category", NivaraColumn<string>.Create(new[] { "A", "B", "A", "C", "B" })),
    ("Value", NivaraColumn<int>.Create(new[] { 10, 20, 30, 40, 50 }))
);

var groups = items.AsQueryFrame()
    .GroupBy("Category")
    .Collect();
// Category: ["A", "B", "C"]
```

### Typed object LINQ (issue end-to-end)

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Carol", "Dan", "Eve" })),
    ("City", NivaraColumn<string>.Create(new[] { "NYC", "LA", "NYC", "SF", "LA" })),
    ("Age", NivaraColumn<int>.Create(new[] { 34, 28, 41, 25, 33 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 90000, 65000, 120000, 55000, 78000 }))
);

var query = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .GroupBy(p => p.City)
    .Select(g => new { g.Key, AvgSalary = g.Average(p => p.Salary) });

var stats = query.Collect();             // NivaraFrame: Key | AvgSalary
var objects = query.ToObjects();         // IReadOnlyList<anonymous>: NYC -> 105000, LA -> 78000
```

### Multi-column sort

```csharp
var result = frame.AsQueryFrame()
    .Sort(
        new SortKey("Department", SortDirection.Ascending),
        new SortKey("Salary", SortDirection.Descending, NullOrdering.NullsFirst)
    )
    .Collect();
```

### Inner join two frames

```csharp
var orders = NivaraFrame.Create(
    ("OrderId", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
    ("CustomerId", NivaraColumn<int>.Create(new[] { 101, 102, 103 })),
    ("Amount", NivaraColumn<double>.Create(new[] { 50.0, 75.0, 100.0 }))
);

var customers = NivaraFrame.Create(
    ("CustomerId", NivaraColumn<int>.Create(new[] { 101, 102, 104 })),
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Dave" }))
);

var joined = orders.InnerJoin(customers, "CustomerId");
// Result columns: OrderId, CustomerId, Amount, Name
// Inner join on CustomerId=CustomerId, 2 result rows
```

### Plan inspection without execution

```csharp
var query = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Price") > 10.0)
    .Select("Product", "Price");

Console.WriteLine(query.ExplainPlan());
var diagnostics = query.GetDiagnosticInfo();
var suggestions = query.AnalyzeOptimizations();
```

---

## Implementation Map

| Component | File |
|-----------|------|
| QueryFrame | `src/Nivara/Query/QueryFrame.cs` |
| QueryPlan | `src/Nivara/Query/QueryPlan.cs` |
| QueryExecutor | `src/Nivara/Query/QueryExecutor.cs` |
| IQueryOperation / IQuerySource | `src/Nivara/Query/IQueryInterfaces.cs` |
| OperationType constants | `src/Nivara/Query/OperationType.cs` |
| FilterOperation | `src/Nivara/Operations/FilterOperation.cs` |
| SelectOperation | `src/Nivara/Operations/SelectOperation.cs` |
| SortOperation | `src/Nivara/Operations/SortOperation.cs` |
| GroupByOperation | `src/Nivara/Operations/GroupByOperation.cs` |
| JoinOperation | `src/Nivara/Operations/JoinOperation.cs` |
| ProjectionOperation | `src/Nivara/Operations/ProjectionOperation.cs` |
| SliceOperation | `src/Nivara/Operations/SliceOperation.cs` |
| DistinctOperation | `src/Nivara/Operations/DistinctOperation.cs` |
| SelectRowsOperation | `src/Nivara/Operations/SelectRowsOperation.cs` |
| ConcatenationOperation | `src/Nivara/Operations/ConcatenationOperation.cs` |
| WindowOperationBase / RollingOperation / CumulativeOperation / ShiftOperation | `src/Nivara/Operations/WindowOperations.cs` |
| Column window primitives | `src/Nivara/Tensors/WindowFunctions.cs` |
| NivaraFrame window extensions | `src/Nivara/WindowFrameExtensions.cs` |
| AggregationFunction | `src/Nivara/Operations/AggregationFunction.cs` |
| ColumnExpression | `src/Nivara/Expressions/ColumnExpression.cs` |
| ExpressionEvaluator | `src/Nivara/Helpers/ExpressionEvaluator.cs` |
| RowExpressionBuilder | `src/Nivara/Linq/RowExpressionBuilder.cs` |
| NivaraLinqExtensions | `src/Nivara/Linq/NivaraLinqExtensions.cs` |
| NivaraQuery<T> / NivaraGroupedQuery<TKey,T> | `src/Nivara/Linq/NivaraQuery.cs` |
| Grouping<TKey,T> marker | `src/Nivara/Linq/Grouping.cs` |
| TypedExpressionTranslator | `src/Nivara/Linq/TypedExpressionTranslator.cs` |
| TypedLinqMetadata / TypedProjectionBuilder | `src/Nivara/Linq/TypedLinqMetadata.cs` / `TypedProjectionBuilder.cs` |
| TypedRowFactory | `src/Nivara/Linq/TypedRowFactory.cs` |
| Query<T>() entry | `src/Nivara/Linq/TypedLinqExtensions.cs` |
| ExecutionEngine | `src/Nivara/Execution/ExecutionEngine.cs` |
| QueryOptimizer | `src/Nivara/Query/QueryOptimizer.cs` |
| QueryPlanAnalyzer | `src/Nivara/Query/QueryPlan.cs` |
| QueryDiagnostics | `src/Nivara/Query/QueryDiagnosticMode.cs` |
| NivaraFrame.AsQueryFrame | `src/Nivara/NivaraFrame.cs` |
| NivaraFrameExtensions (joins) | `src/Nivara/NivaraFrameExtensions.cs` |
