# Nivara

A high-performance, columnar DataFrame library for .NET, focused on **type safety**, **explicit null semantics**, and **vectorized execution**.

Nivara is designed for developers who want predictable behavior, strong typing, and performance-oriented data processing without relying on dynamic or NaN-based conventions.

---

## Why Nivara

Most DataFrame-style libraries trade correctness and type safety for convenience. Nivara takes a different approach:

- **Strong typing end-to-end** — column types are explicit and enforced
- **Explicit null handling** — no NaN-based semantics or hidden behavior
- **Immutable data model** — operations return new data structures
- **Vectorized execution where it matters** — SIMD via `System.Numerics`
- **Schema-aware query planning** — errors surface early, not at runtime

If you care about correctness, debuggability, and performance in .NET data processing, Nivara is built for you.

---

## Installation

Core library:

```bash
dotnet add package Nivara
```

Optional extensions and I/O integrations (install when you need file formats, Arrow interoperability, or ML integration):

```bash
dotnet add package Nivara.Extensions
```

---

## Quick Example

```csharp
using Nivara;

var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });

var doubled = column * 2;

Console.WriteLine(doubled[0]); // 2
```

---

## Core Concepts

### Typed Columns

Nivara columns are strongly typed and immutable:

```csharp
var ages = NivaraColumn<int>.Create(new[] { 25, 30, 35 });
Console.WriteLine(ages.Length); // 3
```

### Explicit Null Semantics

Nulls are tracked explicitly using validity masks, not sentinel values:

```csharp
var data = new int?[] { 1, null, 3 };
var column = NivaraColumn<int>.CreateFromNullable(data);

Console.WriteLine(column.HasNulls);   // True
Console.WriteLine(column.NullCount);  // 1
```

Null-aware operations behave predictably:

```csharp
var filled = column.FillNull(0);  // [1, 0, 3]
var dropped = column.DropNulls(); // [1, 3]
```

---

## Vectorized Operations

For vectorizable types, Nivara automatically uses SIMD-accelerated kernels backed by `System.Numerics.Tensors`:

```csharp
var a = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
var b = a * 1.5;
var c = a + b;
```

Nivara automatically selects the optimal storage backend:
- **TensorStorage**: For vectorizable types (`int`, `float`, `double`, `bool`) using `System.Numerics.Tensors`
- **MemoryStorage**: For non-vectorizable types (`string`, `Guid`, reference types) using `Memory<T>`

Vectorization is applied opportunistically and safely, with scalar fallbacks when required.

---

## Frames and Schemas

Multiple columns can be combined into a schema-aware frame:

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob" })),
    ("Age",  NivaraColumn<int>.Create(new[] { 30, 40 }))
);

Console.WriteLine(frame.RowCount);    // 2
Console.WriteLine(frame.ColumnCount); // 2
```

Schemas are immutable and validated on every transformation.

---

## Lazy Query API

Nivara provides a fluent, LINQ-like query API with lazy execution:

```csharp
using Nivara.Expressions;

var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name")
    .Collect();
```

Queries are planned and validated before execution, allowing early detection of schema and type errors.

---

## Lazy Data Sources (CSV / JSON)

File-based data sources support lazy scanning with automatic schema inference:

```csharp
using Nivara.IO;

// Create a QueryFrame that scans the CSV lazily
var query = Csv.ScanAsQueryFrame("employees.csv")
    .Filter(ColumnExpressions.Col("Salary") > 70000)
    .Select("Name", "Salary");

var result = query.Collect();
```

File I/O is deferred until `Collect()` is called.

---

## Grouping and Aggregation

Nivara provides comprehensive grouping and aggregation operations with hash-based grouping and vectorized aggregations:

### GroupBy Operations

Group data by one or more columns with efficient hash-based grouping:

```csharp
using Nivara.Expressions;

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Alice", "Charlie" })),
    ("Department", NivaraColumn<string>.Create(new[] { "IT", "HR", "IT", "Finance" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 75000, 65000, 78000, 85000 }))
);

// Group by single column
var groupedByName = frame.AsQueryFrame()
    .GroupBy("Name")
    .Collect();

// Group by multiple columns
var groupedByNameAndDept = frame.AsQueryFrame()
    .GroupBy("Name", "Department")
    .Collect();
```

GroupBy operations handle:
- **Multiple grouping keys** with composite key generation
- **Null values** as distinct groups
- **Efficient hashing** for all data types
- **Schema preservation** with proper type inference

### Aggregation Functions

Apply aggregations to grouped or ungrouped data with automatic vectorization:

```csharp
// Built-in aggregation functions
var countAgg = AggregationFunctions.Count();
var sumAgg = AggregationFunctions.Sum();
var avgAgg = AggregationFunctions.Mean();
var minAgg = AggregationFunctions.Min();
var maxAgg = AggregationFunctions.Max();

// Apply to column data
var salaryColumn = frame.GetColumn<double>("Salary");
var allIndices = Enumerable.Range(0, salaryColumn.Length).ToList();

Console.WriteLine($"Total salary: {sumAgg.Apply(salaryColumn, allIndices)}");
Console.WriteLine($"Average salary: {avgAgg.Apply(salaryColumn, allIndices)}");
Console.WriteLine($"Employee count: {countAgg.Apply(salaryColumn, allIndices)}");
```

Aggregation features:
- **Vectorized operations** using TensorPrimitives for float/double types
- **Null-aware computation** (nulls are ignored in calculations)
- **Type-safe results** with appropriate return types (e.g., Sum returns long for integers)
- **Extensible architecture** for custom aggregation functions
- **Automatic fallbacks** to scalar operations when vectorization isn't available

### Custom Aggregation Functions

Create custom aggregations by extending the base class:

```csharp
public class MedianAggregation : AggregationFunction
{
    public override string Name => "Median";
    
    public override Type GetResultType(Type inputType) => inputType;
    
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        // Extract valid values and compute median
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null) validValues.Add(value);
        }
        
        if (validValues.Count == 0) return null;
        
        // Implement median calculation logic
        // ...
    }
}
```

---

## Row Operations

Nivara provides efficient row-level operations for filtering and slicing DataFrames while preserving schema and handling null values correctly.

### Filtering with Boolean Masks

Filter rows using boolean masks for precise control over which rows to include:

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000, 80000 }))
);

// Create a boolean mask (e.g., from a condition)
var mask = NivaraColumn<bool>.Create(new[] { true, false, true, false });

// Filter the frame using the mask
var filtered = frame.FilterByMask(mask);
// Result: Alice (25, 50000) and Charlie (35, 70000)
```

Boolean masks can be generated from any condition or expression evaluation, providing flexible filtering capabilities.

### Row Slicing Operations

Take the first n rows from a DataFrame:

```csharp
// Take first 3 rows
var firstThree = frame.Take(3);
Console.WriteLine(firstThree.RowCount); // 3
```

Skip the first n rows and return the rest:

```csharp
// Skip first 2 rows, return remaining
var remaining = frame.Skip(2);
Console.WriteLine(remaining.RowCount); // 2 (Charlie and Diana)
```

Combine Skip and Take for precise row ranges:

```csharp
// Skip first 1 row, then take next 2 rows
var middle = frame.Skip(1).Take(2);
// Result: Bob (30, 60000) and Charlie (35, 70000)
```

### Sorting Operations

Sort DataFrames by one or multiple columns with full control over sort direction and null handling:

```csharp
using Nivara.Operations;

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie", "Alice", "Bob", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 35, 25, 30, 40 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 70000, 50000, 60000, 80000 }))
);

// Single column sorting (ascending by default)
var sortedByAge = frame.AsQueryFrame()
    .Sort("Age")
    .Collect();
// Result: Alice (25), Bob (30), Charlie (35), Diana (40)

// Single column sorting with explicit direction
var sortedBySalaryDesc = frame.AsQueryFrame()
    .Sort("Salary", SortDirection.Descending)
    .Collect();
// Result: Diana (80000), Charlie (70000), Bob (60000), Alice (50000)
```

Multi-column sorting with priority order:

```csharp
// Sort by Department first, then by Salary within each department
var multiSorted = frame.AsQueryFrame()
    .Sort(new[]
    {
        new SortKey("Department", SortDirection.Ascending),
        new SortKey("Salary", SortDirection.Descending)
    })
    .Collect();
```

Null value handling in sorting:

```csharp
var frameWithNulls = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Score", NivaraColumn<int>.CreateFromNullable(new int?[] { 85, null, 92 }))
);

// Sort with nulls first
var nullsFirst = frameWithNulls.AsQueryFrame()
    .Sort("Score", SortDirection.Ascending, NullOrdering.NullsFirst)
    .Collect();
// Result: Bob (null), Alice (85), Charlie (92)

// Sort with nulls last (default)
var nullsLast = frameWithNulls.AsQueryFrame()
    .Sort("Score", SortDirection.Ascending, NullOrdering.NullsLast)
    .Collect();
// Result: Alice (85), Charlie (92), Bob (null)
```

Sorting features:
- **Multi-column sorting** with configurable priority order
- **Stable sorting** preserves relative order of equal elements
- **Null ordering control** (nulls first or nulls last)
- **Type-safe comparisons** with validation of comparable types
- **Efficient implementation** using index-based reordering
- **Schema preservation** maintains all column types and metadata

### Arbitrary Row Ranges

Use Slice for arbitrary row ranges with start index and length:

```csharp
// Get rows 1-2 (0-based indexing)
var slice = frame.Slice(1, 2);
// Result: Bob (30, 60000) and Charlie (35, 70000)
```

### Null Value Handling

All row operations properly handle null values and preserve null masks:

```csharp
var nullableData = new int?[] { 1, null, 3, null, 5 };
var column = NivaraColumn<int>.CreateFromNullable(nullableData);
var frame = NivaraFrame.Create(("Numbers", column));

// Filter preserves null semantics
var mask = NivaraColumn<bool>.Create(new[] { true, true, false, true, false });
var filtered = frame.FilterByMask(mask);
// Result: [1, null, null] with proper null tracking
```

### Edge Case Handling

Row operations handle edge cases gracefully:

- **Empty results**: Return empty DataFrames with preserved schema
- **Out-of-bounds parameters**: Clamp to valid ranges or return empty results
- **Zero-length operations**: Return appropriate empty or full results

---

## Column Transformations and Projections

Nivara provides powerful column transformation and projection capabilities that maintain type safety while enabling flexible data manipulation.

### Column Transformations

Transform individual columns with element-wise operations while preserving null semantics:

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000 }))
);

// Transform a single column
var frameWithAgeInMonths = frame.WithTransformedColumn<int, int>(
    "Age", 
    age => age * 12, 
    "AgeInMonths"
);
// Result: AgeInMonths column with values [300, 360, 420]

// Transform and replace existing column
var frameWithUpdatedSalary = frame.WithTransformedColumn<double, double>(
    "Salary",
    salary => salary * 1.1 // 10% raise
);
// Result: Salary column updated with 10% increase
```

Multi-column transformations for computed columns:

```csharp
// Create computed column from two source columns
var frameWithBonus = frame.WithComputedColumn<int, double, double>(
    "Age",
    "Salary", 
    (age, salary) => age > 30 ? salary * 0.1 : salary * 0.05,
    "Bonus"
);
// Result: Bonus column based on age and salary
```

### Column Projections and Selection

Select specific columns from a DataFrame:

```csharp
// Select specific columns by name
var nameAndAge = frame.Select("Name", "Age");
// Result: DataFrame with only Name and Age columns

// Select columns using IEnumerable
var selectedColumns = frame.Select(new[] { "Name", "Salary" });
// Result: DataFrame with Name and Salary columns
```

Column selection with renaming:

```csharp
// Select and rename columns
var renamedFrame = frame.SelectAndRename(new Dictionary<string, string?>
{
    { "Name", "EmployeeName" },
    { "Age", "YearsOld" },
    { "Salary", null } // Keep original name
});
// Result: Columns renamed to EmployeeName, YearsOld, Salary
```

### Column Renaming

Rename individual or multiple columns:

```csharp
// Rename a single column
var renamedSingle = frame.RenameColumn("Age", "YearsOld");

// Rename multiple columns
var renamedMultiple = frame.RenameColumns(new Dictionary<string, string>
{
    { "Name", "EmployeeName" },
    { "Age", "YearsOld" }
});
```

### Column Exclusion

Remove unwanted columns from a DataFrame:

```csharp
// Exclude specific columns
var withoutAge = frame.Exclude("Age");
// Result: DataFrame with Name and Salary columns only

// Exclude multiple columns
var nameOnly = frame.Exclude("Age", "Salary");
// Result: DataFrame with only Name column
```

### Null Handling in Transformations

Column transformations properly handle null values with predictable behavior:

```csharp
var nullableData = new int?[] { 1, null, 3, null, 5 };
var column = NivaraColumn<int>.CreateFromNullable(nullableData);
var frame = NivaraFrame.Create(("Numbers", column));

// Transform with null propagation
var doubled = frame.WithTransformedColumn<int, int>(
    "Numbers",
    x => x * 2,
    "Doubled"
);
// Result: Doubled column = [2, null, 6, null, 10]
// Nulls are preserved without applying the transformation function
```

Multi-column computations handle nulls by propagating them:

```csharp
var frame = NivaraFrame.Create(
    ("A", NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 })),
    ("B", NivaraColumn<int>.CreateFromNullable(new int?[] { 2, 4, null }))
);

var sum = frame.WithComputedColumn<int, int, int>(
    "A", "B",
    (a, b) => a + b,
    "Sum"
);
// Result: Sum column = [3, null, null]
// Any null input produces null output
```

### Transformation Features

Column transformation capabilities include:

- **Type-safe transformations** with compile-time type checking
- **Null propagation** - null inputs always produce null outputs
- **Exception handling** with clear error context including row indices
- **Performance optimization** for vectorizable operations
- **Flexible API** supporting both simple and complex transformation scenarios
- **Schema validation** ensuring result columns have appropriate types
- **Memory efficiency** through lazy evaluation and minimal copying
- **Schema preservation**: All operations maintain column names and types

Row operations are designed to be composable, efficient, and predictable across all data types and scenarios.

---

## Aggregate Functions

Nivara provides efficient aggregate functions with automatic vectorization:

```csharp
var data = new[] { 1, 2, 3, 4, 5 };
var series = NivaraSeries<int>.Create(data);

Console.WriteLine(series.Sum());     // 15
Console.WriteLine(series.Average()); // 3
Console.WriteLine(series.Min());     // 1
Console.WriteLine(series.Max());     // 5
```

Aggregate functions automatically use `TensorPrimitives` for vectorizable types (float, double) and handle null values gracefully:

```csharp
// Vectorized operations for float/double
var floats = NivaraSeries<float>.Create(new[] { 1.5f, 2.5f, 3.5f });
var sum = floats.Sum(); // Uses TensorPrimitives.Sum for performance

// Null-aware aggregation
var nullableData = new int?[] { 1, null, 3, null, 5 };
var column = NivaraColumn<int>.CreateFromNullable(nullableData);
var series = new NivaraSeries<int>(column);

Console.WriteLine(series.Sum()); // 9 (ignores nulls)
```

---

## Extensions & Integrations

Additional I/O adapters and integrations are provided in the separate `Nivara.Extensions` package (install when you need these features):

```bash
dotnet add package Nivara.Extensions
```

The integrations are shipped as a separate package so the core Nivara runtime remains small and focused.

### Apache Arrow Interoperability

Convert between Nivara and Apache Arrow formats for cross-language data exchange:

```csharp
using Nivara.IO;

// Convert NivaraFrame to Arrow Table
var arrowTable = frame.ToArrowTable();

// Convert Arrow Table back to NivaraFrame
var restoredFrame = arrowTable.FromArrowTable();

// Series-level conversions
var arrowArray = series.ToArrowArray();
var restoredSeries = arrowArray.FromArrowArray<int>();
```

Arrow integration supports zero-copy operations when possible and handles all supported data types including proper null semantics.

### Parquet File I/O

Read and write Parquet files with columnar compression and schema preservation:

```csharp
using Nivara.IO;

// Write to Parquet file
frame.ToParquet("data.parquet");

// Read from Parquet file
var loadedFrame = NivaraFrameExtensions.LoadParquet("data.parquet");

// Async operations
await frame.ToParquetAsync("data.parquet");
var asyncFrame = await NivaraFrameExtensions.LoadParquetAsync("data.parquet");

// Stream-based operations
using var stream = new FileStream("data.parquet", FileMode.Create);
frame.ToParquetStream(stream);
```

Parquet I/O supports:
- Configurable compression algorithms (snappy, gzip, lz4)
- Custom row group sizes for optimal performance
- Batch writing of multiple frames
- Schema validation and type mapping
- Streaming operations for large datasets

### Configuration Options

Both Arrow and Parquet operations support extensive configuration:

```csharp
// Arrow conversion options
var arrowOptions = new ArrowConversionOptions
{
    UseZeroCopy = true,
    ValidateTypes = true,
    TimeZone = TimeZoneInfo.Utc,
    StringEncoding = Encoding.UTF8
};

// Parquet write options
var parquetOptions = new ParquetWriteOptions
{
    Compression = "snappy",
    RowGroupSize = 10000,
    ValidateSchema = true
};

frame.ToArrowTable(arrowOptions);
frame.ToParquet("data.parquet", parquetOptions);
```

---

## Current Capabilities

Nivara currently supports:

- **Core Data Structures**: Typed, immutable columns and frames with automatic storage selection
- **Null Handling**: Explicit null handling with fill and drop operations, comprehensive null mask tracking
- **Performance**: Vectorized arithmetic and comparisons using `System.Numerics.Tensors`
- **Storage**: High-performance tensor-backed storage for numeric types, memory-based storage for reference types
- **Query Engine**: Schema-aware lazy query construction with diagnostics and plan inspection
- **Data Sources**: CSV and JSON lazy data sources with automatic schema inference
- **Row Operations**: Filtering with boolean masks, slicing with Take/Skip operations, and arbitrary row range selection
- **Sorting Operations**: Multi-column sorting with configurable direction, null ordering, and stable sort semantics
- **Column Transformations**: Type-safe element-wise transformations with null propagation and exception handling
- **Column Projections**: Flexible column selection, renaming, exclusion, and computed column generation
- **Join Operations**: Inner, Left, Right, and Full Outer joins with flexible key mapping, column disambiguation, and null-aware matching
- **Aggregate Functions**: Sum, Average, Min, Max with vectorized operations and null-aware computation
- **Grouping Operations**: Hash-based GroupBy with composite key support and efficient group management
- **Aggregation Framework**: Extensible aggregation system with built-in functions (Count, Sum, Min, Max, Mean) and vectorized execution
- **Parquet I/O**: Full read/write support with compression, streaming, and batch operations (via `Nivara.Extensions`)
- **Apache Arrow**: Bidirectional conversion with zero-copy optimization support (via `Nivara.Extensions`)
- **ML.NET Integration**: Tensor conversion helpers for machine learning workflows (via `Nivara.Extensions`)
- **Performance Optimization**: Buffer pooling, memory management, and async I/O operations

---

## Join Operations

Nivara provides comprehensive join operations for combining data from multiple DataFrames with full control over join types, key matching, and column name resolution.

### Basic Join Operations

Join DataFrames using common column names:

```csharp
var employees = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "David" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 }))
);

var departments = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 2, 3, 4, 5 })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "HR", "IT", "Finance", "Marketing" })),
    ("Salary", NivaraColumn<decimal>.Create(new[] { 50000m, 60000m, 70000m, 80000m }))
);

// Inner join - returns only matching rows
var innerJoin = employees.InnerJoin(departments, "Id");
// Result: 3 rows (Id: 2, 3, 4) with columns from both DataFrames

// Left join - returns all left rows with matching right rows
var leftJoin = employees.LeftJoin(departments, "Id");
// Result: 4 rows, Alice (Id=1) has null values for Department and Salary

// Right join - returns all right rows with matching left rows  
var rightJoin = employees.RightJoin(departments, "Id");
// Result: 4 rows, Marketing department (Id=5) has null values for Name and Age

// Full outer join - returns all rows from both DataFrames
var fullJoin = employees.FullOuterJoin(departments, "Id");
// Result: 5 rows, includes Alice (left only) and Marketing (right only)
```

### Join with Different Column Names

Join DataFrames when join keys have different names:

```csharp
var customers = NivaraFrame.Create(
    ("CustomerId", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
    ("CustomerName", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" }))
);

var orders = NivaraFrame.Create(
    ("OrderId", NivaraColumn<int>.Create(new[] { 101, 102, 103 })),
    ("CustId", NivaraColumn<int>.Create(new[] { 2, 3, 4 })),
    ("Amount", NivaraColumn<decimal>.Create(new[] { 100m, 200m, 300m }))
);

// Join using different column names
var customerOrders = customers.InnerJoin(orders, "CustomerId", "CustId");
// Result: Matches customers with their orders using CustomerId = CustId
```

### Column Name Conflict Resolution

Handle column name conflicts with configurable disambiguation strategies:

```csharp
var leftFrame = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 1, 2 })),
    ("Value", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B" }))
);

var rightFrame = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 1, 2 })),
    ("Value", NivaraColumn<string>.CreateForReferenceType(new[] { "X", "Y" }))
);

// Suffix disambiguation (default)
var suffixResult = leftFrame.InnerJoin(rightFrame, "Id");
// Result: Columns are Id, Value_left, Value_right

// Prefix disambiguation
var prefixResult = leftFrame.InnerJoin(rightFrame, "Id", 
    ColumnDisambiguationStrategy.Prefix, "L", "R");
// Result: Columns are Id, L_Value, R_Value

// Error on conflicts
try 
{
    var errorResult = leftFrame.InnerJoin(rightFrame, "Id", 
        ColumnDisambiguationStrategy.Error);
}
catch (SchemaValidationException ex)
{
    // Exception thrown due to column name conflict
}
```

### Null Value Handling in Joins

Join operations handle null values with SQL-like semantics:

```csharp
var leftWithNulls = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.CreateFromNullable(new int?[] { 1, 2, null, 4 })),
    ("Name", NivaraColumn<string>.CreateForReferenceType(new string?[] { "Alice", "Bob", null, "David" }!))
);

var rightWithNulls = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.CreateFromNullable(new int?[] { 2, null, 4, 5 })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new string?[] { "HR", null, "Finance", "Marketing" }!))
);

// Inner join with nulls
var result = leftWithNulls.InnerJoin(rightWithNulls, "Id");
// Result: Only rows with matching non-null IDs (Id=2 and Id=4)
// Null values never match other null values
```

### Join Key Coalescing

For outer joins, join key values are intelligently coalesced to show meaningful results:

```csharp
var left = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
    ("LeftData", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "C" }))
);

var right = NivaraFrame.Create(
    ("Id", NivaraColumn<int>.Create(new[] { 2, 3, 4 })),
    ("RightData", NivaraColumn<string>.CreateForReferenceType(new[] { "X", "Y", "Z" }))
);

var rightJoin = left.RightJoin(right, "Id");
// Result: Id column shows [2, 3, 4] (actual join key values)
// Not [2, 3, null] - the Id=4 row shows 4, not null
```

### Join Features

Join operations provide:

- **All SQL join types**: Inner, Left, Right, Full Outer joins
- **Flexible key mapping**: Join on columns with different names
- **Column disambiguation**: Configurable strategies for handling name conflicts
- **Null-aware matching**: SQL-like null semantics (nulls don't match nulls)
- **Type validation**: Ensures join key types are compatible
- **Schema preservation**: Maintains column types and metadata
- **Efficient algorithms**: Hash-based joins for optimal performance
- **Key coalescing**: Intelligent join key handling in outer joins
- **Error handling**: Clear validation messages for missing columns or type mismatches

---

## DataFrame Concatenation

Nivara provides flexible concatenation operations for combining multiple DataFrames either vertically (appending rows) or horizontally (appending columns).

### Vertical Concatenation (Row Append)

Combine DataFrames by appending rows, with configurable handling of schema mismatches:

```csharp
var frame1 = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
);

var frame2 = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 35, 28 }))
);

// Simple vertical concatenation
var combined = frame1.ConcatenateVertical(frame2);
// Result: 4 rows with Alice, Bob, Charlie, Diana

// Alternative syntax
var appended = frame1.Append(frame2);
// Same result as ConcatenateVertical
```

### Handling Schema Mismatches

Control how schema differences are handled during vertical concatenation:

```csharp
var employees = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
);

var contractors = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000.0 }))
    // Missing Age column, has Salary column instead
);

// Fill missing columns with nulls (default behavior)
var combined = employees.ConcatenateVertical(contractors, ConcatenationMismatchHandling.FillWithNulls);
// Result: 3 rows, Charlie has null Age, Alice/Bob have null Salary

// Error on schema mismatch
try 
{
    var strict = employees.ConcatenateVertical(contractors, ConcatenationMismatchHandling.Error);
}
catch (SchemaValidationException ex)
{
    // Exception thrown due to missing columns
}
```

### Horizontal Concatenation (Column Append)

Combine DataFrames by appending columns, requiring matching row counts:

```csharp
var names = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
);

var details = NivaraFrame.Create(
    ("Salary", NivaraColumn<double>.Create(new[] { 50000.0, 60000.0 })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "Engineering", "Sales" }))
);

// Horizontal concatenation
var complete = names.ConcatenateHorizontal(details);
// Result: 2 rows with Name, Age, Salary, Department columns

// Alternative syntax
var combined = names.Combine(details);
// Same result as ConcatenateHorizontal
```

### Multiple DataFrame Concatenation

Concatenate multiple DataFrames in a single operation:

```csharp
var frames = new[] { frame1, frame2, frame3 };

// Vertical concatenation of multiple frames
var allRows = NivaraFrameExtensions.ConcatenateVertical(frames);

// Horizontal concatenation of multiple frames
var allColumns = NivaraFrameExtensions.ConcatenateHorizontal(frames);
```

### Null Value Preservation

Concatenation operations properly preserve null values and null masks:

```csharp
var frameWithNulls = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new string?[] { "Alice", null }!)),
    ("Age", NivaraColumn<int>.CreateFromNullable(new int?[] { 25, null }))
);

var frameWithoutNulls = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Bob", "Charlie" })),
    ("Age", NivaraColumn<int>.Create(new[] { 30, 35 }))
);

var result = frameWithNulls.ConcatenateVertical(frameWithoutNulls);
// Result: Null values are preserved in their original positions
// Row 1: Alice (25), Row 2: null (null), Row 3: Bob (30), Row 4: Charlie (35)
```

### Error Handling

Concatenation operations provide clear error messages for common issues:

```csharp
// Row count mismatch in horizontal concatenation
var twoRows = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2 })));
var oneRow = NivaraFrame.Create(("B", NivaraColumn<int>.Create(new[] { 3 })));

try 
{
    var invalid = twoRows.ConcatenateHorizontal(oneRow);
}
catch (QueryExecutionException ex)
{
    // Clear error message about row count mismatch
}

// Column name conflicts in horizontal concatenation
var left = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice" })));
var right = NivaraFrame.Create(("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Bob" })));

try 
{
    var conflict = left.ConcatenateHorizontal(right);
}
catch (QueryExecutionException ex)
{
    // Clear error message about column name conflict
}
```

### Concatenation Features

DataFrame concatenation provides:

- **Flexible direction control**: Vertical (row append) and horizontal (column append)
- **Schema mismatch handling**: Configurable strategies for missing columns
- **Type safety**: Validates column type compatibility during concatenation
- **Null preservation**: Maintains null values and null masks correctly
- **Empty DataFrame handling**: Gracefully handles empty inputs
- **Multiple frame support**: Concatenate many DataFrames in single operations
- **Clear error messages**: Descriptive errors for invalid operations
- **Performance optimization**: Efficient column copying and memory management
- **Convenient aliases**: `Append()` and `Combine()` methods for common scenarios

---

## Roadmap (Planned)

The following features are still planned or in-progress:

- Query optimization engine
- Streaming and parallel execution strategies

---

## Stability

Nivara is under active development.

- Core concepts are stabilizing
- APIs may evolve as additional features are added
- Feedback and experimentation are encouraged

---

## Documentation

- [**ARCHITECTURE**](ARCHITECTURE.md) — design and internal architecture
- [**CONTRIBUTING**](CONTRIBUTING.md) — how to contribute to the project
- [**GUIDELINES**](GUIDELINES.md) — architectural rationale, lessons learned, and known gotchas

---

Nivara aims to bring **predictable, high-performance data processing** to the .NET ecosystem — without sacrificing correctness or clarity.

