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
- **Aggregate Functions**: Sum, Average, Min, Max with vectorized operations and null-aware computation
- **Grouping Operations**: Hash-based GroupBy with composite key support and efficient group management
- **Aggregation Framework**: Extensible aggregation system with built-in functions (Count, Sum, Min, Max, Mean) and vectorized execution
- **Parquet I/O**: Full read/write support with compression, streaming, and batch operations (via `Nivara.Extensions`)
- **Apache Arrow**: Bidirectional conversion with zero-copy optimization support (via `Nivara.Extensions`)
- **ML.NET Integration**: Tensor conversion helpers for machine learning workflows (via `Nivara.Extensions`)
- **Performance Optimization**: Buffer pooling, memory management, and async I/O operations

---

## Roadmap (Planned)

The following features are still planned or in-progress:

- Join operations (Inner, Left, Right, Full Outer)
- Sorting operations (single and multi-column)
- Column transformations and projections
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

