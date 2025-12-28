# Nivara

A high-performance columnar DataFrame library for .NET, designed for efficient data processing with vectorized operations and explicit null handling.

## Overview

Nivara provides strongly-typed, immutable columns backed by optimized storage implementations. The library automatically selects between tensor-backed storage for vectorizable types and memory-backed storage for non-vectorizable types, ensuring optimal performance while maintaining a clean, type-safe API.

## Key Features

- **Automatic Storage Selection**: Vectorizable types (int, float, double, etc.) use optimized tensor storage, while non-vectorizable types (string, Guid, etc.) use memory storage
- **Vectorized Operations**: SIMD-accelerated arithmetic operations using System.Numerics.Tensors
- **Explicit Null Handling**: Boolean tensor masks for tracking null values, avoiding NaN-based semantics
- **Type Safety**: Strongly-typed columns with compile-time type checking
- **Immutability**: All operations return new instances without modifying originals
- **Memory Efficiency**: Views and slicing operations minimize copying
- **Performance Diagnostics**: Comprehensive diagnostic APIs for performance analysis and kernel selection optimization

## Quick Start

### Installation

```bash
dotnet add package Nivara
```

### Basic Usage

```csharp
using Nivara;

// Create columns from arrays
var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
var floatColumn = NivaraColumn<float>.Create(new[] { 1.5f, 2.5f, 3.5f });
var stringColumn = NivaraColumn<string>.Create(new[] { "hello", "world", "test" });

// Create series with labels
var labeledSeries = NivaraSeries<int>.Create(
    new[] { 100, 200, 300 },
    new[] { "first", "second", "third" });

// Access elements
Console.WriteLine(intColumn[0]);         // 1
Console.WriteLine(intColumn.Length);     // 5
Console.WriteLine(labeledSeries["first"]); // 100

// Arithmetic operations (vectorizable types only)
var doubled = intColumn * 2;                    // Scalar multiplication
var sum = intColumn + intColumn;                // Element-wise addition
var result = intColumn.Multiply(3);             // Method syntax

// Working with nulls (reference types)
var stringWithNulls = NivaraColumn<string>.CreateForReferenceType(
    new[] { "hello", null, "world" });
    
Console.WriteLine(stringWithNulls.HasNulls);    // True
Console.WriteLine(stringWithNulls.IsNull(1));   // True

// Working with nullable value types
var nullableInts = new int?[] { 1, null, 3, null, 5 };
var intColumnWithNulls = NivaraColumn<int>.CreateFromNullable(nullableInts);

Console.WriteLine(intColumnWithNulls.NullCount);     // 2
var nullIndices = intColumnWithNulls.GetNullIndices(); // [1, 3]

// Null handling operations
var filled = intColumnWithNulls.FillNull(0);           // Replace nulls with 0
var forwardFilled = intColumnWithNulls.FillNullForward(); // Forward fill
var backwardFilled = intColumnWithNulls.FillNullBackward(); // Backward fill
var dropped = intColumnWithNulls.DropNulls();          // Remove null values
```

### Advanced Null Handling

Nivara provides comprehensive null handling capabilities:

```csharp
// Create from nullable arrays
var nullableData = new double?[] { 1.5, null, 3.14, null, 2.71 };
var column = NivaraColumn<double>.CreateFromNullable(nullableData);

// Null inspection
Console.WriteLine(column.HasNulls);        // True
Console.WriteLine(column.NullCount);       // 2
Console.WriteLine(column.Length);          // 5

// Get null positions
int[] nullIndices = column.GetNullIndices(); // [1, 3]

// Fill strategies
var constantFilled = column.FillNull(0.0);           // [1.5, 0.0, 3.14, 0.0, 2.71]
var forwardFilled = column.FillNullForward();        // [1.5, 1.5, 3.14, 3.14, 2.71]
var backwardFilled = column.FillNullBackward();      // [1.5, 3.14, 3.14, 2.71, 2.71]

// Remove nulls entirely
var cleaned = column.DropNulls();                    // [1.5, 3.14, 2.71] (length: 3)

// Convert to array (preserves nulls as default values)
double[] array = column.ToArray();                  // [1.5, 0.0, 3.14, 0.0, 2.71]
```

### Supported Operations

#### Arithmetic Operations
- Scalar multiplication: `column * scalar` or `column.Multiply(scalar)`
- Element-wise addition: `left + right` or `left.Add(right)`
- Proper null propagation in all operations

#### Comparison Operations
- Scalar comparisons: `column.Equals(value)`, `column.GreaterThan(value)`, `column.LessThan(value)`
- Element-wise comparisons: `column.Equals(other)`, `column.GreaterThan(other)`, `column.LessThan(other)`
- Returns boolean columns with proper null handling
- Supports all comparable types (numeric, string, DateTime, etc.)

#### Column Operations
- Indexer access: `column[index]`
- Length property: `column.Length`
- Null checking: `column.IsNull(index)`, `column.HasNulls`, `column.NullCount`
- Null inspection: `column.GetNullIndices()`
- Null handling: `column.FillNull(value)`, `column.FillNullForward()`, `column.FillNullBackward()`, `column.DropNulls()`
- Array conversion: `column.ToArray()`
- Slicing: `column.Slice(start, length)`

#### Series Operations
- Position-based access: `series[position]`
- Label-based access: `series[label]` or `series.GetByLabel(label)`
- Safe label access: `series.TryGetByLabel(label, out value)`
- Label checking: `series.ContainsLabel(label)`
- Index inspection: `series.GetLabel(position)`
- Slicing with index preservation: `series.Slice(start, length)`

## Architecture

### Storage Types

Nivara automatically selects the appropriate storage implementation:

- **TensorStorage**: For vectorizable types (int, float, double, long, short, byte, bool)
  - Uses System.Numerics.Tensors for SIMD acceleration
  - Optimized for numerical computations
  
- **MemoryStorage**: For non-vectorizable types (string, Guid, DateTime, custom objects)
  - Uses Memory<T> for efficient memory management
  - Supports null detection for reference types

### Query Engine Foundation

Nivara includes a foundational query engine infrastructure for building DataFrame-like operations:

- **Schema System**: ✅ **Complete** - Immutable schema management with column metadata, type information, and transformation methods (WithColumn, WithoutColumn, SelectColumns)
- **Column Expressions**: Composable expression system for building queries with operator overloading
- **Query Planning**: Infrastructure for building and optimizing query execution plans
- **Type-Safe Interfaces**: Both generic and non-generic column interfaces for compile-time and runtime type handling
- **Error Handling**: Comprehensive exception hierarchy with context-specific error messages

#### Schema System Features

The Schema system provides comprehensive metadata management for columnar data:

```csharp
// Create schema from column definitions
var schema = new Schema(new[]
{
    ("Name", typeof(string)),
    ("Age", typeof(int)),
    ("Salary", typeof(double))
});

// Access schema information
Console.WriteLine($"Columns: {schema.ColumnNames.Count}");
Console.WriteLine($"Has Age column: {schema.HasColumn("Age")}");
Console.WriteLine($"Age type: {schema.GetColumnType("Age")}");

// Transform schemas immutably
var withBonus = schema.WithColumn("Bonus", typeof(double));
var withoutAge = schema.WithoutColumn("Age");
var projected = schema.SelectColumns(new[] { "Name", "Salary" });

// Schema compatibility validation
var otherSchema = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });
bool compatible = schema.IsCompatibleWith(otherSchema);
bool flexibleMatch = schema.IsCompatibleWith(otherSchema, requireExactMatch: false);

// Column metadata support
var metadata = new ColumnMetadata(
    isNullable: false,
    defaultValue: 0,
    description: "Employee age in years");
var schemaWithMetadata = schema.WithColumn("Age", typeof(int), metadata);
```

The query engine foundation supports the upcoming NivaraFrame implementation, which will provide lazy query execution, data source scanning, and advanced DataFrame operations.

### Type Support

**Vectorizable Types** (support arithmetic operations):
- `int`, `uint`, `long`, `ulong`
- `float`, `double`
- `short`, `ushort`, `byte`, `sbyte`
- `bool`

**Non-Vectorizable Types** (storage only):
- `string`, `Guid`, `DateTime`
- Custom reference types and value types

### Null Handling

- **Value Types**: Cannot contain nulls (use nullable value types for optional values)
- **Reference Types**: Automatic null detection with boolean masks
- **Null Propagation**: Arithmetic operations correctly propagate nulls

## Performance

Nivara is designed for high-performance data processing:

- **SIMD Acceleration**: Vectorized operations using System.Numerics.Tensors
- **Memory Efficiency**: Views and slicing minimize allocations
- **Batch Processing**: Optimized for columnar data access patterns
- **Zero-Copy Operations**: Slicing creates views without copying data

## Examples

### Working with Numeric Data

```csharp
// Create numeric columns
var prices = NivaraColumn<double>.Create(new[] { 10.50, 25.75, 15.25, 30.00 });
var quantities = NivaraColumn<double>.Create(new[] { 2.0, 1.0, 3.0, 1.5 });

// Calculate total values using vectorized operations
var totals = prices * quantities;

// Apply discount
var discountedPrices = prices * 0.9;

// Sum columns
var combined = prices + discountedPrices;
```

### Working with Comparisons

```csharp
// Create numeric data
var scores = NivaraColumn<int>.Create(new[] { 85, 92, 78, 95, 88 });
var threshold = 90;

// Scalar comparisons
var highScores = scores.GreaterThan(threshold);     // [false, true, false, true, false]
var exactMatches = scores.Equals(92);               // [false, true, false, false, false]
var belowThreshold = scores.LessThan(threshold);    // [true, false, true, false, true]

// Element-wise comparisons
var otherScores = NivaraColumn<int>.Create(new[] { 80, 95, 75, 90, 85 });
var comparisons = scores.GreaterThan(otherScores);  // [true, false, true, true, true]

// String comparisons
var names = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie" });
var startsWithB = names.GreaterThan("B");           // Lexicographic comparison
var exactName = names.Equals("Bob");                // [false, true, false]

// Working with comparison results
for (int i = 0; i < highScores.Length; i++)
{
    if (highScores[i])
    {
        Console.WriteLine($"Score {scores[i]} is above threshold");
    }
}
```

### Working with String Data

```csharp
// Create string column with potential nulls
var names = NivaraColumn<string>.CreateForReferenceType(
    new[] { "Alice", "Bob", null, "Charlie" });

// Check for nulls
if (names.HasNulls)
{
    for (int i = 0; i < names.Length; i++)
    {
        if (names.IsNull(i))
        {
            Console.WriteLine($"Position {i} contains null");
        }
        else
        {
            Console.WriteLine($"Position {i}: {names[i]}");
        }
    }
}

// String comparisons with null handling
var searchName = "Bob";
var matches = names.Equals(searchName);  // [false, true, null, false]

// Null comparisons return null (represented as false with null mask)
Console.WriteLine(matches.IsNull(2));    // True - null comparison yields null
```

### Working with Series (Labeled Data)

```csharp
// Create series with default integer index
var prices = NivaraSeries<double>.Create(new[] { 10.50, 25.75, 15.25 });

// Access by position
Console.WriteLine(prices[0]);    // 10.50
Console.WriteLine(prices[1]);    // 25.75

// Access by default integer labels
Console.WriteLine(prices[(object)0]);  // 10.50 (explicit label access)

// Create series with custom string labels
var namedPrices = NivaraSeries<double>.Create(
    new[] { 10.50, 25.75, 15.25 },
    new[] { "apple", "banana", "cherry" });

// Access by label
Console.WriteLine(namedPrices["apple"]);   // 10.50
Console.WriteLine(namedPrices["banana"]);  // 25.75

// Check if label exists
if (namedPrices.ContainsLabel("apple"))
{
    Console.WriteLine($"Apple price: {namedPrices["apple"]}");
}

// Try to get value by label
if (namedPrices.TryGetByLabel("orange", out var orangePrice))
{
    Console.WriteLine($"Orange price: {orangePrice}");
}
else
{
    Console.WriteLine("Orange not found");
}

// Create series with mixed object labels
var mixedSeries = NivaraSeries<string>.Create(
    new[] { "value1", "value2", "value3" },
    new object[] { 1, "key", DateTime.Today });

Console.WriteLine(mixedSeries.GetByLabel(1));           // "value1"
Console.WriteLine(mixedSeries.GetByLabel("key"));       // "value2"
Console.WriteLine(mixedSeries.GetByLabel(DateTime.Today)); // "value3"

// Slice series (preserves index-value relationships)
var sliced = namedPrices.Slice(1, 2);  // Gets "banana" and "cherry"
Console.WriteLine(sliced["banana"]);   // 25.75
Console.WriteLine(sliced["cherry"]);   // 15.25
```

### Slicing and Views

```csharp
var data = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

// Create a view of elements 2-5 (zero-based indexing)
var slice = data.Slice(2, 4); // Contains [3, 4, 5, 6]

Console.WriteLine(slice.Length); // 4
Console.WriteLine(slice[0]);     // 3
```

### Performance Analysis and Diagnostics

Nivara provides comprehensive diagnostic information for performance analysis and optimization:

```csharp
// Get column diagnostic information
var column = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
var diagnostics = column.Diagnostics;

Console.WriteLine($"Storage Type: {diagnostics.StorageType}");           // Memory or Tensor
Console.WriteLine($"Is Vectorizable: {diagnostics.IsVectorizable}");     // True for numeric types
Console.WriteLine($"Element Type: {diagnostics.ElementType.Name}");      // Int32
Console.WriteLine($"Length: {diagnostics.Length}");                      // 5
Console.WriteLine($"Has Nulls: {diagnostics.HasNulls}");                 // False
Console.WriteLine($"Hardware Accelerated: {diagnostics.IsHardwareAccelerated}"); // SIMD support
Console.WriteLine($"Recommended Kernel: {diagnostics.RecommendedKernel}"); // Vectorized or Scalar
Console.WriteLine($"Estimated Memory: {diagnostics.EstimatedMemoryUsage} bytes");

// Performance characteristics
var performance = diagnostics.Performance;
Console.WriteLine($"Throughput Multiplier: {performance.ThroughputMultiplier}x");
Console.WriteLine($"Memory Efficiency: {performance.MemoryEfficiency:P1}");
Console.WriteLine($"Supports Vectorization: {performance.SupportsVectorization}");

// Track operations for performance analysis
DiagnosticsTracker.IsEnabled = true;

// Perform operations - they will be automatically tracked
var result1 = column.Add(column);
var result2 = column.Multiply(2);
var result3 = column.Equals(3);

// Get operation statistics
var operations = DiagnosticsTracker.GetRecordedOperations();
foreach (var op in operations)
{
    Console.WriteLine($"Operation: {op.OperationType}");
    Console.WriteLine($"Kernel Used: {op.KernelUsed}");
    Console.WriteLine($"Input Length: {op.InputLength}");
    Console.WriteLine($"Had Nulls: {op.HadNulls}");
    Console.WriteLine($"Selection Reason: {op.KernelSelectionReason}");
}

// Get summary statistics
var summary = DiagnosticsTracker.GetSummary();
Console.WriteLine($"Total Operations: {summary.TotalOperations}");
Console.WriteLine($"Vectorized: {summary.VectorizedOperations}");
Console.WriteLine($"Scalar: {summary.ScalarOperations}");
Console.WriteLine($"Vectorization Rate: {summary.VectorizationRate:F1}%");

// Clear tracking data
DiagnosticsTracker.ClearRecordedOperations();
DiagnosticsTracker.IsEnabled = false;
```

## Error Handling

Nivara provides clear error messages for common mistakes:

```csharp
var stringColumn = NivaraColumn<string>.Create(new[] { "1", "2", "3" });

// This will throw InvalidOperationException with clear message
try
{
    var result = stringColumn * "2";
}
catch (InvalidOperationException ex)
{
    Console.WriteLine(ex.Message);
    // "Arithmetic operations are not supported for type String. 
    //  Only numeric types support arithmetic operations."
}

// Comparison operations work on all types
var stringComparison = stringColumn.Equals("2");        // Works fine
var stringOrdering = stringColumn.GreaterThan("2");     // Works fine (lexicographic)

// But some types don't support ordering comparisons
var guidColumn = NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() });
var guidEquals = guidColumn.Equals(Guid.Empty);         // Works fine
var guidOrdering = guidColumn.GreaterThan(Guid.Empty);  // Works (Guid implements IComparable)
```

## Building from Source

### Prerequisites

- .NET 10.0 SDK or later
- System.Numerics.Tensors package

### Build Commands

```bash
# Clone the repository
git clone https://github.com/khurram-uworx/nivara.git
cd nivara

# Build the solution
dotnet build

# Run tests
dotnet test

# Create NuGet package
dotnet pack
```

### Project Structure

```
src/
├── Nivara/                 # Core library (dependency-free)
│   ├── Diagnostics/        # Performance analysis and diagnostic tools
│   ├── Exceptions/         # Custom exception hierarchy
│   ├── Expressions/        # Query expression system
│   ├── IO/                 # Built-in IO functionality (JSON, etc.)
│   ├── Memory/             # Memory-based storage implementations
│   ├── Tensors/            # Tensor-based storage implementations
│   └── [Root]              # Core interfaces and main classes
├── Nivara.Extensions/      # Extension methods and third-party integrations
│   └── IO/                 # Third-party IO functionality (CSV, Parquet, etc.)
samples/
├── Nivara.SampleApp/       # Sample applications
tests/
├── Nivara.Tests/           # Unit and property tests
│   ├── Diagnostics/        # Tests for diagnostic functionality
│   ├── Memory/             # Tests for memory storage
│   ├── Tensors/            # Tests for tensor storage
│   └── [Root]              # Tests for core functionality
```

### Extensions Package

The `Nivara.Extensions` package provides additional functionality that requires third-party dependencies:

- **CSV Support**: Reading and writing CSV files using CsvHelper
- **Parquet Support**: Integration with Parquet.Net for columnar file format
- **Arrow Support**: Apache Arrow integration for interoperability
- **ML.NET Integration**: Machine learning pipeline integration

To use CSV functionality:

```bash
dotnet add package Nivara.Extensions
```

```csharp
using Nivara.IO;

// CSV functionality (requires Nivara.Extensions package)
var csvSource = Csv.Scan("data.csv");
var csvColumns = Csv.Read("data.csv");

// JSON functionality (built into core Nivara package)
var jsonSource = Json.Scan("data.json");
var jsonColumns = Json.Read("data.json");

// With custom options
var csvOptions = new CsvOptions 
{ 
    HasHeaderRecord = true,
    Delimiter = ";",
    SchemaInferenceRows = 50
};
var csvData = Csv.Read("data.csv", csvOptions);

var jsonOptions = new JsonOptions
{
    IsArray = true,
    SchemaInferenceRecords = 100
};
var jsonData = Json.Read("data.json", jsonOptions);
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Guidelines

- Follow the patterns documented in [GUIDELINES.md](GUIDELINES.md)
- Write comprehensive tests for new features
- Ensure all tests pass before submitting PRs
- Use property-based testing for universal behaviors

## Roadmap

### Current Status
- ✅ Core column types (`NivaraColumn<T>`)
- ✅ Automatic storage selection
- ✅ Basic arithmetic operations
- ✅ Comparison operations (equals, greater than, less than)
- ✅ Null handling for reference types
- ✅ Series with indexing (`NivaraSeries<T>`)
- ✅ Performance diagnostics and kernel selection analysis
- ✅ Query engine foundation and interfaces
- ✅ Schema system with column metadata and transformation methods
- ✅ Column expression system for query building
- ✅ Comprehensive test suite with 241 passing tests

### Upcoming Features
- 🔄 NivaraFrame (DataFrame-like structure)
- 🔄 Lazy query execution with optimization
- 🔄 Data source scanning (CSV, JSON)
- 🔄 Advanced null handling
- 🔄 Performance optimizations
- 📋 I/O operations (CSV, JSON, Parquet)
- 📋 Aggregation functions
- 📋 Grouping and joining operations

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- Built on top of [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors/) for vectorized operations
- Inspired by modern columnar data processing libraries
- Designed for high-performance .NET applications

---

**Note**: Nivara is currently in active development. APIs may change between versions until v1.0 is released.