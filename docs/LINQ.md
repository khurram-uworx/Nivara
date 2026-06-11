# Nivara LINQ Query Engine

Nivara provides a deferred, plan-based LINQ-like query engine over typed columnar DataFrames. Queries are built immutably via `QueryFrame` and materialized lazily on demand.

---

## Architecture

```
User API              QueryFrame / Linq Extensions (Where, Select, OrderBy, GroupBy)
Plan Layer            QueryPlan (IQuerySource + ordered IQueryOperation[])
Execution Layer       QueryExecutor / IExecutionStrategy (Eager, Lazy, Streaming, Parallel)
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
var query = JsonExtensions.ScanJsonAsQueryFrame("data.json");
```

### From custom IQuerySource

```csharp
public interface IQuerySource : IDisposable
{
    Schema Schema { get; }
    bool IsLazy { get; }
    IReadOnlyDictionary<string, IColumn> Execute();
}
```

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
| **Concatenate*** | `"Concatenate"` | `ConcatenationOperation` | Vertical concatenation |

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

---

## API Surfaces

### 1. Expression-based API (on QueryFrame)

```csharp
var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Salary") > 50000)
    .Select("Name", "Salary")
    .Collect();
```

Every `Filter`/`Select`/`Sort`/`GroupBy` call returns a **new** `QueryFrame` with the operation appended.

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

Available LINQ extensions (`QueryFrameExtensions`):
| Method | Maps to | Notes |
|--------|---------|-------|
| `Where(predicate)` | `Filter(expression)` | Accepts `Func<RowExpressionBuilder, ColumnExpression>` |
| `Select(selectors...)` | `Select(expressions)` | Accepts `params Func<RowExpressionBuilder, ColumnExpression>[]` |
| `Select(columnNames...)` | `Select(expressions)` | String overload |
| `OrderBy(keySelector)` | `Sort(columnName, Ascending)` | Supports descending via `OrderByDescending` |
| `OrderByDescending(keySelector)` | `Sort(columnName, Descending)` | |
| `ToList()` | `Collect()` | Materializes to `NivaraFrame` |
| `ToNivaraFrame()` | `Collect()` | Alias for `ToList` |

### 3. Frame-level operations (on NivaraFrame directly)

```csharp
// Join operations — not through QueryFrame
var joined = left.InnerJoin(right, "KeyColumn");
var leftJoin = left.LeftJoin(right, new JoinKey("OrderId", "Id"));
var rightJoin = left.RightJoin(right, "Id");
var fullOuter = left.FullOuterJoin(right, "Id");
```

---

## Execution Pipeline

### QueryPlan construction

When `.Collect()` is called, a `QueryPlan` is created:

1. `Source` + `Operations` are captured
2. `ResultSchema` is computed by piping `Source.Schema` through each operation's `TransformSchema()`
3. Schema validation errors are thrown eagerly (before data access)

### QueryExecutor.Execute()

```csharp
public NivaraFrame Execute(QueryPlan plan)
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
| ConcatenationOperation | `src/Nivara/Operations/ConcatenationOperation.cs` |
| ColumnExpression | `src/Nivara/Expressions/ColumnExpression.cs` |
| ExpressionEvaluator | `src/Nivara/Helpers/ExpressionEvaluator.cs` |
| RowExpressionBuilder | `src/Nivara/Linq/RowExpressionBuilder.cs` |
| QueryFrameExtensions | `src/Nivara/Linq/QueryFrameExtensions.cs` |
| ExecutionEngine | `src/Nivara/Execution/ExecutionEngine.cs` |
| QueryOptimizer | `src/Nivara/Query/QueryOptimizer.cs` |
| QueryPlanAnalyzer | `src/Nivara/Query/QueryPlan.cs` |
| NivaraFrame.AsQueryFrame | `src/Nivara/NivaraFrame.cs` |
