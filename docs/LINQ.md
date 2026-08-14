# Nivara LINQ Query Engine

Nivara provides a deferred, plan-based LINQ query engine over typed columnar DataFrames. The
**typed object layer (`NivaraQuery<T>`)** is the single public query API: strongly typed lambdas over
a POCO are compiled to query plans and executed lazily. The lower-level `QueryFrame` /
`ColumnExpression` engine is internal plumbing and is not part of the public surface.

Queries are built immutably and materialize lazily when `Collect()`, `ToObjects()`, or `ToList()` is
called.

---

## Architecture

```
Public API            NivaraQuery<T> (Where, Select, OrderBy/ThenBy, Distinct, GroupBy, ...)
Internal Engine       QueryFrame / IQueryOperation / ColumnExpression (Query/, Expressions/, Operations/)
Plan Layer            QueryPlan (IQuerySource + ordered IQueryOperation[])
Execution Layer       QueryExecutor (internal) / IExecutionStrategy (Eager, Lazy, Streaming, Parallel)
```

**Key design principles:**
- Queries are **deferred** — nothing executes until `Collect()` / `ToObjects()` / `ToList()`.
- Each operation is an immutable node in a pipeline; no mutation.
- Schema is validated eagerly when the query is constructed (before data flows).
- The public surface is **typed only** — engine types (`QueryFrame`, `IQueryOperation`,
  `ColumnExpression`, operations) are `internal`.

---

## Entry Points

### From a NivaraFrame

```csharp
using Nivara;
using Nivara.Linq;

public sealed class Person
{
    public string Name { get; set; } = string.Empty;
    public string City { get; set; } = string.Empty;
    public int Age { get; set; }
    public double Salary { get; set; }
}

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 70000, 50000, 90000 }))
);

var query = frame.Query<Person>();   // NivaraQuery<Person> — deferred, nothing executed yet
```

`frame.Query<T>()` (requires `T : class, new()`; `using Nivara.Linq;`) returns an immutable
`NivaraQuery<T>`. Every property of `T` maps to a column with the same name (case-insensitive) and an
exact or nullable-underlying type, validated eagerly at `Query<T>()` (no data access) with
`SchemaValidationException` on mismatch.

### From file sources (lazy)

```csharp
// Nivara.IO.Json (core) and Nivara.Extensions.IO.Csv (Extensions)
var query = Json.ScanQuery<Person>("data.json");

var csvQuery = Csv.ScanQuery<Person>("data.csv");
```

These create a lazy typed query over the file — the schema is inferred from a sample on construction,
and data is read only when the query is executed. Row types must map to the inferred column types
(CSV integers infer as `int`; JSON numbers infer as `double`).

---

## Typed Query Reference

### Where

```csharp
var rows = frame.Query<Person>()
    .Where(p => p.Age > 30 && p.City == "NYC")
    .ToObjects();
```

### Select

```csharp
var projected = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Select(p => new { p.Name, Raised = p.Salary * 1.1 })
    .ToObjects();   // IReadOnlyList<anonymous>
```

### OrderBy / ThenBy — typed multi-key sort

Direction and null ordering are per-key and optional:

```csharp
var rows = frame.Query<Person>()
    .OrderBy(p => p.City)                                     // ascending, NullsLast (default)
    .ThenBy(p => p.Salary, direction: SortDirection.Descending,
            nullOrdering: NullOrdering.NullsFirst)
    .ToObjects();
```

| Method | Defaults |
|--------|----------|
| `OrderBy(key, direction = Ascending, nullOrdering = NullsLast)` | primary ascending key |
| `OrderByDescending(key, nullOrdering = NullsLast)` | primary descending key |
| `ThenBy(key, direction = Ascending, nullOrdering = NullsLast)` | secondary ascending key |
| `ThenByDescending(key, nullOrdering = NullsLast)` | secondary descending key |

`SortDirection` and `NullOrdering` come from `Nivara.Operations` (also used by the frame-level
`Rank`/`DenseRank`/`PercentRank` window API). Computed keys (e.g. `p => p.Age * 2`) are materialized
into a column at execution time and sorted on.

### Distinct / DistinctBy

```csharp
var unique = frame.Query<Person>().Distinct().ToObjects();       // dedup on all columns
var byCity = frame.Query<Person>().DistinctBy(p => p.City).ToObjects();  // dedup on a column
```

`DistinctBy` requires a direct column reference; a computed key fails fast with
`UnsupportedQueryExpressionException`.

### SelectRows

```csharp
var sampled = frame.Query<Person>().SelectRows(4, 0, 2).ToObjects();   // preserves the given order
```

### Skip / Take / Slice

```csharp
var page = frame.Query<Person>().Skip(10).Take(5).ToObjects();
var slice = frame.Query<Person>().Slice(2, 8).ToObjects();
```

### GroupBy

`GroupBy(key)` returns `NivaraGroupedQuery<TKey, T>`. A bare `Collect()` yields the distinct keys;
an aggregate `Select` reads `g.Key` plus `g.Average/Sum/Count/Min/Max(p => ...)` (the
`Grouping<TKey, T>` marker exposes these for C# type inference; they are never invoked at runtime).
Any operation other than an aggregate `Select` or `Collect` after `GroupBy` fails fast.

```csharp
var stats = frame.Query<Person>()
    .Where(p => p.Age > 18)
    .GroupBy(p => p.City)
    .Select(g => new { g.Key, AvgSalary = g.Average(p => p.Salary), People = g.Count() })
    .ToObjects();
```

### Supported expressions

Property access, constant literals, `+ - * / %` arithmetic, comparisons, and `&&`/`||`/`!` boolean
logic. Method calls, captured variables/closures, nested property access, array/index access,
ternary, and string `+` fail fast at build time with `UnsupportedQueryExpressionException`.

### Window functions in the expression DSL

Rolling / cumulative / shift / lead / rank windows are first-class `ColumnExpression`s; see the
[Window Functions](#window-functions) section for the full surface.

---

## Materialization

| Member | Returns |
|--------|---------|
| `Collect()` | `NivaraFrame` (pipeline runs lazily) |
| `ToObjects()` | `IReadOnlyList<TResult>` via a compiled, cached per-type row factory |
| `ToList()` | `List<TResult>` |
| `ToRows()` | `IReadOnlyList<TResult>` (alias for `ToObjects`) |
| `Schema` | the resulting schema |
| `ExplainPlan()` | a tree view of the query plan |
| `IsLazy` | whether the query uses a lazy data source |

Row factories require a public parameterless constructor (anonymous types via `Select` work
automatically); a null cell materialized into a non-nullable value-type property throws at
materialization time.

---

## Frame-Level Operations

Join operations are extension methods in `NivaraFrameExtensions`, not part of the query engine:

```csharp
var joined = left.InnerJoin(right, "KeyColumn");                    // same column name
var leftJoin = left.LeftJoin(right, "OrderId", "Id");              // different column names
var rightJoin = left.RightJoin(right, "Id");
var fullOuter = left.FullOuterJoin(right, "Id");
```

All join methods accept optional `ColumnDisambiguationStrategy`, `leftPrefix`, and `rightPrefix`
parameters. Join types: `Inner`, `Left`, `Right`, `FullOuter`. Disambiguation strategies: `Prefix`
(`left_Name`/`right_Name`), `Suffix` (`Name_left`/`Name_right`), `Error` (throw on conflict). Join
key columns are coalesced in outer joins (left value wins, falls back to right).

Window functions (rolling/cumulative/shift/rank) live directly on `NivaraFrame` via
`WindowFrameExtensions` and are unaffected by the query layer; see the
[Window Functions](#window-functions) section.

---

## Window Functions

Nivara exposes window functions on `NivaraFrame` (eager) and `QueryFrame` (lazy). Both layers
share the same kernels, so results agree between eager and lazy paths.

Every window function returns a new `NivaraFrame` / `QueryFrame` with the result appended as a
new column; the input column is untouched.

### The `WindowSpec` builder

`Over()` starts a `WindowSpec`, the recommended way to describe partitioning and ordering. The
builder is immutable — each call returns a new spec.

```csharp
using Nivara.Operations; // WindowSpec, SortKey, SortDirection, NullOrdering

var spec = frame.Over()
    .PartitionBy("dept")              // one or more column names
    .OrderBy("salary");               // ascending, NULLS LAST by default
```

`OrderBy` has three overloads:

```csharp
spec.OrderBy(new SortKey("salary", SortDirection.Descending));
spec.OrderBy("salary", SortDirection.Descending, NullOrdering.NullsFirst);
spec.OrderBy("salary");                       // ascending, NULLS LAST
```

- Partition keys are optional; with no partition keys the whole frame is one partition.
- Order keys are optional for rolling / cumulative / shift / lead and `RowNumber`, and required
  for `Rank` / `DenseRank` / `PercentRank`.
- A spec with no keys at all (`new WindowSpec()`) is valid and matches the un-partitioned,
  row-order behavior of the plain overloads.
- Partition keys and order keys are validated up front: missing columns and non-comparable
  order-key columns throw `ArgumentException` (eager) or a wrapped `SchemaValidationException`
  (lazy `Collect()`).

### Semantics

A window function with a spec runs in four phases:

1. **Partition** — rows are grouped by the `PartitionBy` keys (stability preserved).
2. **Sort** — each partition is ordered by the `OrderKeys` (SQL `NULLS LAST` default).
3. **Compute** — the window kernel runs over the sorted partition (per-partition reset).
4. **Scatter** — results are mapped back to original row positions.

Order-key nulls participate as ordinary rows — with `NULLS LAST` the null-key rows sort to the
end of their partition and are not dropped from the result.

### Eager `NivaraFrame` surface

All eager members are extension methods on `NivaraFrame`. Every function has both a plain
overload and a `WindowSpec` overload; the plain overloads are equivalent to passing
`new WindowSpec()`.

#### Rolling

| Member | Result type |
|--------|-------------|
| `RollingSum(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |
| `RollingMean(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | `double` |
| `RollingMin(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |
| `RollingMax(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |

The leading `windowSize - 1` rows of each partition yield null (or `minPeriods` if set).

#### Cumulative

| Member | Result type |
|--------|-------------|
| `CumulativeSum(source, result, spec?, nullHandler?)` | source type |
| `CumulativeMax(source, result, spec?, nullHandler?)` | source type |
| `CumulativeMin(source, result, spec?, nullHandler?)` | source type |
| `CumulativeProduct(source, result, spec?, nullHandler?)` | source type |
| `CumulativeCount(source, result, spec?)` | `long` |

#### Shift / Lead

`Shift(source, result, periods, spec?, fillValue?)` and `Lead(...)` move values within each
partition. Rows that would cross a partition boundary (or the frame edge) become null unless
`fillValue` is supplied.

#### Rank family

| Member | Result type |
|--------|-------------|
| `RowNumber(result, spec?)` | `long` |
| `Rank(result, spec)` | `long` |
| `DenseRank(result, spec)` | `long` |
| `PercentRank(result, spec)` | `double` |

The rank family reuses the existing `RankKernel`, so a null order-key produces a null rank
result (unchanged semantics). `RowNumber` accepts a spec without order keys (partition order
within a partition is insertion order); the other rank functions require order keys.

### Lazy `QueryFrame` surface

`QueryFrame` mirrors the eager surface: `Over()`, plus the same overloads (`RollingSum`,
`CumulativeCount`, `Shift`, `Rank`, ...). Because the pipeline is lazy, partition/order-key
validation happens at `Collect()` time (thrown as `QueryExecutionException` wrapping a
`SchemaValidationException`), or immediately when `Schema` is read.

### Window functions in the expression DSL

Rolling / cumulative / shift / lead / rank windows are first-class `ColumnExpression`s and can be
embedded in `Select` / `Filter` / `SortBy`, or composed with elementwise math:

```csharp
// Window result fused with elementwise math.
var result = frame.AsQueryFrame()
    .Select(ColumnExpressions.RollingSum(ColumnExpressions.Col("Salary"), 2) * 2)
    .Collect();

// Rank over expression partition/order keys.
var ranked = frame.AsQueryFrame()
    .Rank("rn", orderBy: new[] { new SortExpressionKey(ColumnExpressions.Col("Score") * 1) },
                partitionBy: new[] { ColumnExpressions.Col("Dept") })
    .Collect();
```

Factory surface on `ColumnExpressions`: `RollingSum` / `RollingMean` / `RollingMin` /
`RollingMax`, `CumulativeSum` / `CumulativeMax` / `CumulativeMin` / `CumulativeProduct` /
`CumulativeCount`, `Shift` / `Lead`, and the rank family `RowNumber` / `Rank` / `DenseRank` /
`PercentRank`. Nested windows compose (a window inside a window is evaluated bottom-up and
injected as a synthetic column). Result types: rolling mean → `double`, cumulative count and
rank-family → `long` (`PercentRank` → `double`), everything else matches the source column type.

Window pipeline ops also accept computed sources and keys directly:

```csharp
var r = frame.AsQueryFrame()
    .RollingSum(ColumnExpressions.Col("A") * 2, "r", 2)   // window over a computed source
    .CumulativeCount(ColumnExpressions.Col("A"), "cnt")
    .Shift(ColumnExpressions.Col("A"), "lag1", 1, fillValue: -1)
    .Lead(ColumnExpressions.Col("A"), "lead1", 1)
    .Collect();
```

The expression DSL reuses the same kernels as the eager/lazy surfaces, so results agree.

### Example

```csharp
using var frame = new NivaraFrame(("dept", deptColumn), ("t", timeColumn), ("v", valueColumn));

var spec = frame.Over().PartitionBy("dept").OrderBy("t");

var eager = frame
    .RollingSum("v", "roll2", 2, spec)
    .CumulativeSum("v", "cum", spec)
    .Shift("v", "lag1", 1, spec)
    .Rank("rnk", spec);

var lazy = frame.AsQueryFrame()
    .RollingSum("v", "roll2", 2, spec)
    .CumulativeSum("v", "cum", spec)
    .Shift("v", "lag1", 1, spec)
    .Rank("rnk", spec)
    .Collect();
```

Implementation: `src/Nivara/Operations/WindowSpec.cs` (builder),
`src/Nivara/Tensors/PartitionedWindowEngine.cs` (partition/sort/scatter kernel),
`src/Nivara/WindowFrameExtensions.cs` (eager overloads), `src/Nivara/Query/QueryFrame.cs`
(lazy overloads).

---

## Execution Pipeline

### QueryPlan construction

When the query executes, a `QueryPlan` is created:

1. `Source` + `Operations` are captured.
2. `ResultSchema` is computed by piping `Source.Schema` through each operation's `TransformSchema()`.
3. Schema validation errors are thrown eagerly (before data access).

### QueryExecutor.Execute()

`QueryExecutor` is an `internal sealed class` invoked by `QueryFrame.Collect()`:

1. Validates the plan schema.
2. Calls `plan.Source.Execute()` to get the initial column dictionary.
3. Iterates `plan.Operations`, piping columns through each `operation.Execute(input)`.
4. Wraps final columns into a `NivaraFrame`.

### Execution Strategies

The `ExecutionEngine` supports four strategies:

| Strategy | Class | Behavior |
|----------|-------|----------|
| **Eager** | `EagerExecutionStrategy` | Immediate execution |
| **Lazy** | `LazyExecutionStrategy` | Defers plan execution with optimization |
| **Streaming** | `StreamingExecutionStrategy` | Chunk-based processing with memory budget |
| **Parallel** | `ParallelExecutionStrategy` | Multi-threaded chunk dispatch |

---

## Diagnostics and Optimization

```csharp
Console.WriteLine(query.ExplainPlan());
// Query Execution Plan:
// ├─ Source: MemorySource
// │  └─ Schema: Name: String, Age: Int32
// └─ Operations:
//    └─ 1. Filter
//       └─ Condition: (Age > 30)
```

Optimization is automatic (predicate pushdown, projection pushdown, operation fusion). Query-plan
diagnostics are exposed on the public typed layer via `ExplainPlan()`; the underlying optimization
analysis and plan serialization live on the internal engine (`QueryFrame.AnalyzeOptimizations()` /
`ToQueryPlan().Serialize()`), which `Nivara.Extensions` and the test suite use directly.

---

## Complete Examples

### Filter + project + sort

```csharp
using Nivara;
using Nivara.Linq;
using Nivara.Operations;

public sealed class Product
{
    public string Name { get; set; } = string.Empty;
    public double Price { get; set; }
    public bool InStock { get; set; }
}

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "C", "D" })),
    ("Price", NivaraColumn<double>.Create(new[] { 10.0, 25.0, 5.0, 30.0 })),
    ("InStock", NivaraColumn<bool>.Create(new[] { true, false, true, true }))
);

var result = frame.Query<Product>()
    .Where(p => p.Price > 10.0 && p.InStock)
    .OrderByDescending(p => p.Price)
    .Select(p => new { p.Name, MarkedUp = p.Price * 1.1 })
    .ToObjects();
```

### Deduplicate and sample

```csharp
var uniqueByCategory = frame.Query<Product>().DistinctBy(p => p.Name).ToObjects();
var sampled = frame.Query<Product>().SelectRows(3, 0).ToObjects();
```

### Lazy CSV query

```csharp
using Nivara.IO;

var adults = Csv.ScanQuery<Person>("people.csv")
    .Where(p => p.Age > 30)
    .ToObjects();
```

### GroupBy aggregate

```csharp
var stats = frame.Query<Person>()
    .GroupBy(p => p.City)
    .Select(g => new { g.Key, AvgSalary = g.Average(p => p.Salary) })
    .ToObjects();
// NYC -> average salary, LA -> average salary, ...
```

---

## Implementation Map

| Component | File |
|-----------|------|
| `NivaraQuery<T>` / `NivaraGroupedQuery<TKey,T>` | `src/Nivara/Linq/NivaraQuery.cs` |
| `frame.Query<T>()` entry | `src/Nivara/Linq/TypedLinqExtensions.cs` |
| Internal `FromFrame<T>` factory | `src/Nivara/Linq/TypedLinqExtensions.cs` |
| `Grouping<TKey,T>` marker | `src/Nivara/Linq/Grouping.cs` |
| TypedExpressionTranslator | `src/Nivara/Linq/TypedExpressionTranslator.cs` |
| TypedLinqMetadata / TypedProjectionBuilder | `src/Nivara/Linq/TypedLinqMetadata.cs` / `TypedProjectionBuilder.cs` |
| TypedRowFactory | `src/Nivara/Linq/TypedRowFactory.cs` |
| Lazy JSON typed queries | `src/Nivara/IO/JsonExtensions.cs` |
| Lazy CSV typed queries | `src/Nivara.Extensions/IO/CsvExtensions.cs` |
| QueryFrame (internal) | `src/Nivara/Query/QueryFrame.cs` |
| QueryPlan / QueryExecutor (internal) | `src/Nivara/Query/QueryPlan.cs` / `QueryExecutor.cs` |
| IQueryOperation / IQuerySource (internal) | `src/Nivara/Query/IQueryInterfaces.cs` |
| ColumnExpression (internal) | `src/Nivara/Expressions/ColumnExpression.cs` |
| Operations (internal) | `src/Nivara/Operations/` |
| ExecutionEngine | `src/Nivara/Execution/ExecutionEngine.cs` |
| NivaraFrameExtensions (joins) | `src/Nivara/NivaraFrameExtensions.cs` |
| Window functions | `src/Nivara/WindowFrameExtensions.cs`, `src/Nivara/Tensors/WindowFunctions.cs`, `src/Nivara/Tensors/PartitionedWindowEngine.cs`, `src/Nivara/Operations/WindowSpec.cs` |
