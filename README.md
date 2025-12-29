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

Nivara includes a comprehensive query engine infrastructure for building DataFrame-like operations:

- **Schema System**: ✅ **Complete** - Immutable schema management with column metadata, type information, and transformation methods (WithColumn, WithoutColumn, SelectColumns)
- **NivaraFrame**: ✅ **Complete** - Multi-column DataFrame-like structure with schema management, column validation, and immutable transformations
- **QueryFrame**: ✅ **Complete** - Lazy query construction with fluent API for building complex data processing pipelines
- **Column Expressions**: ✅ **Complete** - Composable expression system for building queries with operator overloading
- **Query Operations**: ✅ **Complete** - Filter, Select, and GroupBy operations with schema transformation and execution
- **Query Planning**: ✅ **Complete** - Infrastructure for building and optimizing query execution plans
- **DataFrame Operations Infrastructure**: ✅ **Complete** - Core infrastructure for advanced DataFrame operations including:
  - **ExecutionStrategy**: Enumeration for different execution approaches (Lazy, Eager, Streaming, Parallel)
  - **ExecutionContext**: Configuration class for execution parameters, memory budgets, parallelism, and progress reporting
  - **Generic IQueryOperation<T>**: Type-safe interface for composable query operations with transformation support
  - **DataFrameOperation**: Abstract base class for DataFrame operations with execution strategy support
  - **QueryNode Hierarchy**: Structured representation of query plans with visitor pattern support for optimization
- **Type-Safe Interfaces**: Both generic and non-generic column interfaces for compile-time and runtime type handling
- **Error Handling**: Comprehensive exception hierarchy with context-specific error messages

#### QueryFrame Usage

The QueryFrame provides a LINQ-like fluent API for building complex data processing pipelines:

```csharp
using Nivara;
using Nivara.Expressions;

// Create a frame with sample data
var employees = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie", "David", "Eve" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 28, 32 })),
    ("Department", NivaraColumn<string>.Create(new[] { "Engineering", "Sales", "Engineering", "Marketing", "Sales" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 75000, 65000, 85000, 55000, 70000 }))
);

// Build complex queries using fluent API
var result = employees.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)                    // Filter employees over 30
    .Filter(ColumnExpressions.Col("Salary") >= 65000)            // And earning at least 65k
    .Select("Name", "Department", "Salary")                       // Select specific columns
    .GroupBy("Department")                                        // Group by department (distinct values)
    .Collect();                                                   // Execute the query

// Query building is lazy - no execution until Collect()
var lazyQuery = employees.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Department") == "Engineering")
    .Select(ColumnExpressions.Col("Name"), ColumnExpressions.Col("Salary") * 1.1); // 10% raise

// Inspect the query plan before execution
Console.WriteLine(lazyQuery.ExplainPlan());
var optimizations = lazyQuery.AnalyzeOptimizations();

// Execute when ready
var engineersWithRaise = lazyQuery.Collect();

// Complex expressions with arithmetic and comparisons
var complexQuery = employees.AsQueryFrame()
    .Filter((ColumnExpressions.Col("Age") + 5) > 35)             // Complex arithmetic
    .Filter(ColumnExpressions.Col("Salary") / 1000 > 60)        // Salary in thousands
    .Select("Name", "Age", "Salary")
    .Collect();

// Method chaining supports multiple operations
var pipeline = employees.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") >= 25)
    .Filter(ColumnExpressions.Col("Age") <= 35)
    .Select("Name", "Department")
    .GroupBy("Department")
    .Collect();

Console.WriteLine($"Result has {pipeline.RowCount} rows and {pipeline.ColumnCount} columns");
```

#### Lazy Data Sources

Nivara provides comprehensive lazy data source capabilities for CSV and JSON files with automatic schema inference:

```csharp
using Nivara.IO;
using Nivara.Expressions;

// Lazy CSV scanning with schema inference
var csvQuery = CsvExtensions.ScanCsvAsQueryFrame("employees.csv");
Console.WriteLine($"CSV Schema: {string.Join(", ", csvQuery.Schema.ColumnNames)}");
// Schema inferred automatically: Name (string), Age (int), Salary (int)

// Lazy JSON scanning with schema inference  
var jsonQuery = JsonExtensions.ScanJsonAsQueryFrame("employees.json");
Console.WriteLine($"JSON Schema: {string.Join(", ", jsonQuery.Schema.ColumnNames)}");
// Schema inferred automatically: Name (string), Age (double), Salary (double)

// Build complex queries on lazy sources
var filteredEmployees = csvQuery
    .Filter(ColumnExpressions.Col("Age") > 30)                    // Only employees over 30
    .Filter(ColumnExpressions.Col("Salary") > 70000)             // Earning more than 70k
    .Select("Name", "Department", "Salary")                       // Select specific columns
    .Collect();                                                   // Execute query and materialize results

Console.WriteLine($"Found {filteredEmployees.RowCount} senior high-earners");

// Lazy evaluation - no file I/O until Collect()
var lazyPipeline = jsonQuery
    .Filter(ColumnExpressions.Col("Department") == "Engineering")
    .Select(ColumnExpressions.Col("Name"), ColumnExpressions.Col("Salary") * 1.1); // 10% raise

// File is not read until this point
var engineersWithRaise = lazyPipeline.Collect();

// Static factory classes for convenience
var csvData = Csv.ScanAsQueryFrame("data.csv");
var jsonData = Json.ScanAsQueryFrame("data.json");

// Eager reading for immediate materialization
var csvFrame = CsvExtensions.ReadCsvAsFrame("data.csv");        // Immediate file reading
var jsonFrame = JsonExtensions.ReadJsonAsFrame("data.json");    // Immediate file reading

// Compare lazy vs eager - same results, different execution timing
var lazyResult = Csv.ScanAsQueryFrame("large_file.csv")
    .Filter(ColumnExpressions.Col("Status") == "Active")
    .Collect();                                                  // File read only when needed

var eagerFrame = Csv.ReadAsFrame("large_file.csv");            // File read immediately
var eagerResult = eagerFrame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Status") == "Active")
    .Collect();

// Results are identical, but timing and memory usage differ
Console.WriteLine($"Lazy and eager results match: {lazyResult.RowCount == eagerResult.RowCount}");
```

**Schema Inference Characteristics**:
- **CSV Files**: Conservative inference (string, int types)
  - Numeric columns that parse as integers become `int` type
  - All other columns become `string` type
  - Analyzes up to 100 sample rows for type detection

- **JSON Files**: JSON-native inference (string, double types)  
  - JSON numbers become `double` type (JSON standard)
  - String values become `string` type
  - Analyzes up to 100 sample objects for type detection

**Lazy vs Eager Evaluation**:
- **Lazy Sources** (`ScanCsv`, `ScanJson`): File I/O deferred until `Collect()`
  - Minimal upfront cost (schema inference only)
  - Memory efficient for large files
  - Best for filtered queries or conditional processing

- **Eager Sources** (`ReadCsvAsFrame`, `ReadJsonAsFrame`): Immediate file I/O
  - Full file processing upfront
  - Better for small files or guaranteed data usage
  - Simpler memory model (data already materialized)

**Error Handling**:
```csharp
try
{
    var source = CsvExtensions.ScanCsv("nonexistent.csv");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.Message}");
}

try
{
    var source = JsonExtensions.ScanJson("empty.json");  // Empty JSON array
}
catch (DataSourceException ex)
{
    Console.WriteLine($"Cannot infer schema: {ex.Message}");
    // Empty JSON arrays cannot have schema inferred
}

// Null parameter validation
try
{
    var source = CsvExtensions.ScanCsv(null!);
}
catch (ArgumentNullException ex)
{
    Console.WriteLine($"Invalid parameter: {ex.Message}");
}
```

#### Query Operations

The query engine supports several core operations:

**Filter Operations**: Apply conditions to filter rows
```csharp
// Simple comparisons
.Filter(ColumnExpressions.Col("Age") > 30)
.Filter(ColumnExpressions.Col("Name") == "Alice")

// Complex expressions
.Filter((ColumnExpressions.Col("Age") * 2) > ColumnExpressions.Col("Salary") / 1000)
.Filter(ColumnExpressions.Col("Department") != "Sales")
```

**Select Operations**: Choose specific columns or expressions
```csharp
// Select by column names
.Select("Name", "Age", "Salary")

// Select with expressions
.Select(
    ColumnExpressions.Col("Name"),
    ColumnExpressions.Col("Salary") * 1.1,  // 10% salary increase
    ColumnExpressions.Col("Age") + 1        // Age next year
)
```

**GroupBy Operations**: Group by columns (returns distinct values)
```csharp
// Group by single column
.GroupBy("Department")

// Group by multiple columns
.GroupBy("Department", "Age")

// Group by expressions
.GroupBy(ColumnExpressions.Col("Department"), ColumnExpressions.Col("Age") / 10)
```

#### Query Diagnostics and Optimization

The query engine provides comprehensive diagnostic information:

```csharp
var query = employees.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 25)
    .Filter(ColumnExpressions.Col("Salary") > 60000)
    .Select("Name", "Department");

// Explain the query execution plan
string explanation = query.ExplainPlan();
Console.WriteLine(explanation);
/* Output:
Query Execution Plan:
├─ Source: MemoryQuerySource
│  └─ Schema: Name: String, Age: Int32, Department: String, Salary: Double
│  └─ Lazy: False
├─ Operations:
│  ├─ 1. Filter
│  ├─ 2. Filter
│  └─ 3. Select
│     └─ Schema: Name: String, Department: String
└─ Result Schema: Name: String, Department: String
*/

// Analyze optimization opportunities
var optimizations = query.AnalyzeOptimizations();
foreach (var suggestion in optimizations)
{
    Console.WriteLine($"Optimization: {suggestion}");
}
/* Possible output:
Optimization: Found 2 filter operations - consider combining them for better performance
*/

// Schema information is available before execution
var resultSchema = query.Schema;
Console.WriteLine($"Result will have columns: {string.Join(", ", resultSchema.ColumnNames)}");
Console.WriteLine($"Result will have {resultSchema.ColumnNames.Count} columns");
```

#### NivaraFrame Usage

```csharp
using Nivara;

// Create frame from named columns
var numbers = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 });
var names = NivaraColumn<string>.Create(new[] { "Alice", "Bob", "Charlie", "David" });
var scores = NivaraColumn<double>.Create(new[] { 95.5, 87.2, 92.1, 88.9 });

var frame = NivaraFrame.Create(
    ("ID", numbers),
    ("Name", names), 
    ("Score", scores)
);

// Frame properties
Console.WriteLine(frame.RowCount);      // 4
Console.WriteLine(frame.ColumnCount);   // 3
Console.WriteLine(string.Join(", ", frame.ColumnNames)); // ID, Name, Score

// Type-safe column access
var idColumn = frame.GetColumn<int>("ID");
var nameColumn = frame.GetColumn<string>("Name");

// Schema information
var schema = frame.Schema;
Console.WriteLine(schema.GetColumnType("Score")); // System.Double

// Immutable transformations
var frameWithAge = frame.WithColumn("Age", NivaraColumn<int>.Create(new[] { 25, 30, 28, 35 }));
var frameWithoutScore = frame.WithoutColumn("Score");
var selectedFrame = frame.SelectColumns("Name", "Score");

// Case-insensitive column access
var column = frame.GetColumn<string>("name"); // Works with different case

// Convert to QueryFrame for complex operations
var queryFrame = frame.AsQueryFrame();
var filtered = queryFrame.Filter(ColumnExpressions.Col("Score") > 90).Collect();
```
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

#### Column Expression System Features

The Column Expression system provides a composable way to build query conditions and transformations:

```csharp
using Nivara.Expressions;

// Create column references
var ageExpr = ColumnExpressions.Col("Age");
var salaryExpr = ColumnExpressions.Col<double>("Salary");
var nameExpr = ColumnExpressions.Col("Name");

// Create literal values
var threshold = ColumnExpressions.Lit(30);
var multiplier = ColumnExpressions.Lit(1.1);

// Build arithmetic expressions
var salaryIncrease = salaryExpr * multiplier;           // Scalar multiplication
var totalCompensation = salaryExpr + ageExpr;          // Column addition
var complexCalc = (ageExpr + 5) * salaryExpr;          // Complex composition

// Build comparison expressions
var seniorEmployees = ageExpr > 30;                     // Age greater than 30
var highEarners = salaryExpr >= 75000;                  // Salary threshold
var nameMatch = nameExpr == "John";                     // String equality

// Combine expressions for complex conditions
var eligibleForBonus = (ageExpr > 25) & (salaryExpr < 100000);
var promotionCandidate = (ageExpr >= 30) | (salaryExpr > 80000);

// Expression validation against schema
var schema = new Schema(new[]
{
    ("Name", typeof(string)),
    ("Age", typeof(int)),
    ("Salary", typeof(double))
});

// Validate expressions
ageExpr.Validate(schema);        // Passes - Age column exists
salaryExpr.Validate(schema);     // Passes - Salary column exists and type matches

try
{
    var invalidExpr = ColumnExpressions.Col("Department");
    invalidExpr.Validate(schema);  // Throws SchemaValidationException
}
catch (SchemaValidationException ex)
{
    Console.WriteLine(ex.Message); // "Column 'Department' not found in schema. Available columns: Name, Age, Salary"
}

// Expression composition and type inference
var expr = ageExpr + 5;          // ScalarExpression: Age + 5
var comparison = expr > 30;      // ComparisonExpression: (Age + 5) > 30
var complex = (ageExpr * 2) + (salaryExpr / 1000); // BinaryExpression with type promotion

// Expression properties
Console.WriteLine(ageExpr.Name);         // "Age"
Console.WriteLine(ageExpr.ResultType);   // System.Int32
Console.WriteLine(comparison.ToString()); // "(Age + 5) > 30"
```

The expression system supports:
- **Arithmetic Operations**: `+`, `-`, `*`, `/` for both column-to-column and column-to-scalar operations
- **Comparison Operations**: `>`, `<`, `>=`, `<=`, `==`, `!=` with automatic type promotion
- **Expression Composition**: Build complex nested expressions through operator chaining
- **Type Safety**: Compile-time and runtime type checking with clear error messages
- **Schema Validation**: Validate column references against frame schemas
- **Immutable Design**: All operations create new expression instances

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

Nivara provides comprehensive error handling with clear, context-specific error messages:

### Core Operations Error Handling

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

### I/O Operations Error Handling

The Arrow/Parquet I/O system provides detailed error context with specific exception types:

```csharp
using Nivara.IO;

try
{
    // Parameter validation with specific error messages
    frame.ToParquet(""); // ArgumentException: File path cannot be empty or whitespace
    
    // Type validation with helpful suggestions
    var unsupportedFrame = NivaraFrame.Create(
        ("ID", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() }))
    );
    unsupportedFrame.ToParquet("output.parquet");
}
catch (UnsupportedTypeException ex)
{
    Console.WriteLine($"Unsupported type: {ex.UnsupportedType.Name}");
    Console.WriteLine($"Suggested alternatives: {string.Join(", ", ex.SuggestedAlternatives)}");
    // Output: "Suggested alternatives: string, byte[]"
}
catch (ArgumentException ex)
{
    Console.WriteLine($"Invalid parameter: {ex.Message}");
}

try
{
    // File I/O errors include operation context and file paths
    var frame = await NivaraFrameExtensions.LoadParquetAsync("nonexistent.parquet");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.Message}");
}
catch (NivaraIOException ex)
{
    Console.WriteLine($"I/O Error: {ex.Message}");
    Console.WriteLine($"Operation: {ex.OperationContext}");
    Console.WriteLine($"File: {ex.FilePath}");
}

try
{
    // Data corruption errors include affected columns and row ranges
    var corruptedData = /* some corrupted data source */;
    var frame = await ParquetReader.ReadParquetAsync(corruptedData);
}
catch (DataCorruptionException ex)
{
    Console.WriteLine($"Data corruption detected: {ex.Message}");
    Console.WriteLine($"Affected columns: {string.Join(", ", ex.AffectedColumns)}");
    Console.WriteLine($"Affected rows: {ex.AffectedRowRange}");
    Console.WriteLine($"Operation: {ex.OperationContext}");
}

try
{
    // Schema validation errors provide detailed mismatch information
    var incompatibleFrames = new[] { frame1, frame2 }; // Different schemas
    await ParquetWriter.WriteParquetBatchAsync(incompatibleFrames, "batch.parquet");
}
catch (SchemaValidationException ex)
{
    Console.WriteLine($"Schema validation failed: {ex.Message}");
    Console.WriteLine($"Expected schema: {ex.ExpectedSchema}");
    Console.WriteLine($"Actual schema: {ex.ActualSchema}");
    Console.WriteLine($"Type mismatches: {string.Join(", ", ex.TypeMismatches)}");
}
```

### Exception Hierarchy

Nivara uses a comprehensive exception hierarchy for precise error handling:

- **`NivaraIOException`**: Base exception for all I/O operations with operation context and file path information
- **`UnsupportedTypeException`**: Thrown when encountering unsupported types, includes suggested alternatives
- **`SchemaValidationException`**: Thrown when schema validation fails, includes detailed mismatch information
- **`DataCorruptionException`**: Thrown when data corruption is detected, includes affected columns and row ranges

### Parameter Validation

All public methods include comprehensive parameter validation:

```csharp
// Null parameter validation
frame.ToParquet(null);              // ArgumentNullException
arrowTable.FromArrowTable(null);    // ArgumentNullException

// String parameter validation
frame.ToParquet("");                // ArgumentException: cannot be empty or whitespace
frame.ToParquet("   ");             // ArgumentException: cannot be empty or whitespace

// Stream parameter validation
var readOnlyStream = new MemoryStream(new byte[0], false);
frame.ToParquetStream(readOnlyStream); // ArgumentException: Stream must be writable

var writeOnlyStream = new MemoryStream();
writeOnlyStream.Close();
NivaraFrameExtensions.LoadParquetFromStream(writeOnlyStream); // ArgumentException: Stream must be readable
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
- ✅ Arrow/Parquet I/O: ✅ **Complete** - Comprehensive Arrow and Parquet I/O capabilities with fluent extension methods
  - CLR ↔ Arrow type conversion with support for all primitive types
  - CLR ↔ Parquet type conversion with nullable type handling
  - ✅ **Parquet Reader**: File and stream reading with async support
  - ✅ **Parquet Writer**: File and stream writing with configurable compression and batch operations
  - ✅ **Arrow Interoperability**: Bidirectional conversion between NivaraFrame/NivaraSeries and Apache Arrow formats
  - ✅ **Extension Methods**: Fluent API with both instance extension methods and static factory methods
  - ✅ **Schema Validation**: Type compatibility checking with detailed error messages
  - ✅ **Error Handling**: Comprehensive exception hierarchy with context information
  - ✅ **Advanced Configuration**: Zero-copy optimization support with fallback mechanisms
  - ✅ **Performance Optimization**: Configurable validation options for performance-critical scenarios
  - Timezone-aware DateTime conversion
  - Context-specific error messages with suggested alternativesc error messages with type suggestions
- **ML.NET Integration**: Machine learning pipeline integration

#### Arrow/Parquet Type Mapping

The type mapping system provides seamless conversion between .NET types and columnar formats:

```csharp
using Nivara.IO;

// Supported types for Arrow/Parquet conversion:
// - Primitives: bool, int, long, float, double, byte, short, uint, ulong, ushort, sbyte
// - DateTime with timezone handling
// - string with Unicode support
// - Nullable value types (int?, bool?, DateTime?, etc.)
// - Reference types (inherently nullable)

// Type mapping validation
bool isArrowSupported = TypeMapper.IsArrowSupported(typeof(int));     // true
bool isParquetSupported = TypeMapper.IsParquetSupported(typeof(Guid)); // false

// Error handling provides helpful suggestions
try 
{
    var arrowType = TypeMapper.MapClrToArrow(typeof(Guid));
}
catch (UnsupportedTypeException ex)
{
    // ex.SuggestedAlternatives contains ["string", "byte[]"]
    Console.WriteLine($"Consider using: {string.Join(", ", ex.SuggestedAlternatives)}");
}
```

#### Performance Optimizations

Nivara's Arrow/Parquet I/O includes comprehensive performance optimizations for large dataset processing:

**Async Operations with Cancellation Support**:
```csharp
// All I/O operations support cancellation tokens
var cancellationToken = new CancellationTokenSource(TimeSpan.FromMinutes(5)).Token;

try
{
    var frame = await NivaraFrameExtensions.LoadParquetAsync("large_file.parquet", cancellationToken: cancellationToken);
    await frame.ToParquetAsync("output.parquet", cancellationToken: cancellationToken);
}
catch (OperationCanceledException)
{
    Console.WriteLine("Operation was cancelled");
}
```

**Memory Optimization with Buffer Pooling**:
```csharp
// Automatic buffer pooling for large datasets (> 1024 elements)
// Reduces memory allocations and garbage collection pressure
var largeFrame = NivaraFrame.Create(
    ("Data", NivaraColumn<int>.Create(Enumerable.Range(0, 100000).ToArray()))
);

// Buffer pooling is automatically used during I/O operations
await largeFrame.ToParquetAsync("large_output.parquet");

// Get buffer pool statistics for monitoring
var (byteBuffers, intBuffers, doubleBuffers) = BufferPool.GetPoolStatistics();
Console.WriteLine($"Buffer pools: {byteBuffers} byte, {intBuffers} int, {doubleBuffers} double");
```

**Streaming with Memory Budget Management**:
```csharp
// Process large files with bounded memory usage
var memoryBudget = 256L * 1024 * 1024; // 256MB limit

foreach (var chunk in ParquetReader.ReadParquetStreaming("huge_file.parquet", memoryBudget: memoryBudget))
{
    Console.WriteLine($"Processing chunk: {chunk.RowCount} rows");
    // Memory usage stays within budget
    
    // Process chunk and write results
    var processed = ProcessChunk(chunk);
    await processed.ToParquetAsync($"output_chunk_{DateTime.Now.Ticks}.parquet");
}
```

**Performance Monitoring**:
```csharp
// Monitor I/O performance and memory usage
using var bufferManager = new StreamingBufferManager(memoryBudget: 512L * 1024 * 1024);

Console.WriteLine($"Memory budget: {bufferManager.MemoryBudget:N0} bytes");
Console.WriteLine($"Current usage: {bufferManager.CurrentMemoryUsage:N0} bytes");
Console.WriteLine($"Budget exceeded: {bufferManager.IsMemoryBudgetExceeded}");

// Automatic garbage collection when memory usage is high
bufferManager.TryCollectGarbage(); // Triggers GC at 80% of budget
```

**Optimization Features**:
- **Buffer Pooling**: Automatic reuse of byte, int, and double arrays for large datasets
- **Memory Budget Management**: Configurable memory limits with automatic garbage collection
- **Cancellation Support**: All async operations support cancellation tokens with proper cleanup
- **Streaming Operations**: Process large files with bounded memory usage
- **Performance Thresholds**: Optimizations automatically enabled for arrays > 1024 elements
- **Zero-Copy Operations**: Planned future enhancement for memory-mapped file access

#### Arrow Interoperability

Convert between NivaraFrame/NivaraSeries and Apache Arrow formats:

```csharp
using Nivara.IO;
using Apache.Arrow;

// Create sample data
var intData = new[] { 1, 2, 3, 4, 5 };
var stringData = new[] { "apple", "banana", "cherry", "date", "elderberry" };
var boolData = new[] { true, false, true, false, true };

var frame = NivaraFrame.Create(
    ("Numbers", NivaraColumn<int>.Create(intData)),
    ("Fruits", NivaraColumn<string>.Create(stringData)),
    ("Flags", NivaraColumn<bool>.Create(boolData))
);

// Convert NivaraFrame to Arrow Table using extension methods
var arrowTable = frame.ToArrowTable();
Console.WriteLine($"Arrow table: {arrowTable.RowCount} rows, {arrowTable.ColumnCount} columns");

// Convert Arrow Table back to NivaraFrame using extension methods
var restoredFrame = arrowTable.FromArrowTable();
Console.WriteLine($"Restored frame: {restoredFrame.RowCount} rows, {restoredFrame.ColumnCount} columns");

// Convert NivaraSeries to Arrow Array using extension methods
var series = NivaraSeries<double>.Create(new[] { 1.1, 2.2, 3.3, 4.4 });
var arrowArray = series.ToArrowArray();
Console.WriteLine($"Arrow array length: {arrowArray.Length}");

// Convert Arrow Array back to NivaraSeries using extension methods
var restoredSeries = arrowArray.ToNivaraSeries<double>();
Console.WriteLine($"Restored series length: {restoredSeries.Length}");

// Static methods are also available for explicit calls
var arrowTableStatic = ArrowInterop.ToArrowTable(frame);
var restoredFrameStatic = ArrowInterop.FromArrowTable(arrowTable);

// Handle nullable data
var nullableData = new int?[] { 1, null, 3, null, 5 };
var nullableColumn = NivaraColumn<int>.CreateFromNullable(nullableData);
var frameWithNulls = NivaraFrame.Create(("NullableNumbers", nullableColumn));

// Arrow conversion preserves null information
var arrowTableWithNulls = frameWithNulls.ToArrowTable();
var restoredFrameWithNulls = arrowTableWithNulls.FromArrowTable();

// Verify null preservation
var restoredColumn = restoredFrameWithNulls.GetColumn("NullableNumbers");
Console.WriteLine($"Nulls preserved: {restoredColumn.HasNulls}"); // True
Console.WriteLine($"Null count: {restoredColumn.NullCount}");     // 2

// Custom conversion options
var options = new ArrowConversionOptions
{
    TimeZone = TimeZoneInfo.Local,      // Handle DateTime timezone conversion
    UseZeroCopy = true,                 // Enable zero-copy optimizations (with fallback)
    ValidateTypes = true,               // Validate type compatibility (can be disabled for performance)
    StringEncoding = Encoding.UTF8      // String encoding for text data
};

var customArrowTable = frame.ToArrowTable(options);

// Performance optimization - disable validation for trusted data
var fastOptions = new ArrowConversionOptions
{
    ValidateTypes = false,              // Skip type validation for better performance
    UseZeroCopy = true                  // Attempt zero-copy operations
};

var fastConversion = frame.ToArrowTable(fastOptions);
```

#### Parquet File I/O

Read and write Parquet files with comprehensive schema validation and error handling:

```csharp
using Nivara.IO;

// Write Parquet file using extension methods (async)
var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d", "e" });
var frame = NivaraFrame.Create(
    ("Numbers", intColumn),
    ("Letters", stringColumn)
);

await frame.ToParquetAsync("output.parquet");
Console.WriteLine("Parquet file written successfully");

// Write Parquet file using extension methods (sync)
frame.ToParquet("output_sync.parquet");

// Write to stream using extension methods
using var memoryStream = new MemoryStream();
await frame.ToParquetStreamAsync(memoryStream);

// Load Parquet file using extension methods (async)
var readFrame = await NivaraFrameExtensions.LoadParquetAsync("output.parquet");
Console.WriteLine($"Loaded {readFrame.RowCount} rows, {readFrame.ColumnCount} columns");

// Load Parquet file using extension methods (sync)
var frameSync = NivaraFrameExtensions.LoadParquet("output.parquet");

// Load from stream using extension methods
using var fileStream = File.OpenRead("output.parquet");
var frameFromStream = await NivaraFrameExtensions.LoadParquetFromStreamAsync(fileStream);

// Static methods are also available for explicit calls
await ParquetWriter.WriteParquetAsync(frame, "output.parquet");
var readFrameStatic = await ParquetReader.ReadParquetAsync("output.parquet");

// Custom writing options
var writeOptions = new ParquetWriteOptions
{
    RowGroupSize = 5000,           // Rows per row group (default: 10000)
    Compression = "snappy",        // Compression algorithm (default: "snappy")
    ValidateSchema = true,         // Validate schema before writing (default: true)
    WriteMetadata = true           // Include metadata (default: true)
};

await frame.ToParquetAsync("custom_output.parquet", writeOptions);

// Performance optimization - disable validation for trusted data
var fastWriteOptions = new ParquetWriteOptions
{
    ValidateSchema = false,        // Skip schema validation for better performance
    Compression = "none"           // No compression for fastest writing
};

await frame.ToParquetAsync("fast_output.parquet", fastWriteOptions);

// Batch writing - write multiple frames to single file
var frame1 = NivaraFrame.Create(
    ("ID", NivaraColumn<int>.Create(new[] { 1, 2 })),
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" }))
);

var frame2 = NivaraFrame.Create(
    ("ID", NivaraColumn<int>.Create(new[] { 3, 4 })),
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie", "David" }))
);

var frames = new[] { frame1, frame2 };
await ParquetWriter.WriteParquetBatchAsync(frames, "batch_output.parquet");

// Handle nullable data
var nullableData = new int?[] { 1, null, 3, null, 5 };
var nullableColumn = NivaraColumn<int>.CreateFromNullable(nullableData);
var frameWithNulls = NivaraFrame.Create(("NullableNumbers", nullableColumn));

// Parquet writing preserves null information
await frameWithNulls.ToParquetAsync("with_nulls.parquet");

// Custom reading options
var readOptions = new ParquetReadOptions
{
    ValidateSchema = true,      // Validate schema compatibility (default: true)
    BatchSize = 1000,          // Batch size for processing (default: 1000)
    StreamRowGroups = false    // Stream row groups for large files (default: false)
};

var frameWithOptions = await NivaraFrameExtensions.LoadParquetAsync("output.parquet", readOptions);

// Performance optimization - disable validation for trusted files
var fastReadOptions = new ParquetReadOptions
{
    ValidateSchema = false,     // Skip schema validation for better performance
    BatchSize = 10000          // Larger batch size for better throughput
};

var fastFrame = await NivaraFrameExtensions.LoadParquetAsync("trusted_data.parquet", fastReadOptions);

// Streaming for large files (simplified implementation)
foreach (var chunk in ParquetReader.ReadParquetStreaming("huge_data.parquet"))
{
    Console.WriteLine($"Processing chunk with {chunk.RowCount} rows");
    // Process chunk...
}

// Error handling with detailed context
try
{
    // Schema validation during writing
    var unsupportedFrame = NivaraFrame.Create(
        ("GuidColumn", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() }))
    );
    await unsupportedFrame.ToParquetAsync("invalid.parquet");
}
catch (SchemaValidationException ex)
{
    Console.WriteLine($"Schema validation failed: {ex.Message}");
    Console.WriteLine($"Expected: {ex.ExpectedSchema}");
    Console.WriteLine($"Actual: {ex.ActualSchema}");
}
catch (DataCorruptionException ex)
{
    Console.WriteLine($"Data corruption detected: {ex.Message}");
    Console.WriteLine($"Affected columns: {string.Join(", ", ex.AffectedColumns)}");
}
catch (NivaraIOException ex)
{
    Console.WriteLine($"I/O error: {ex.Message}");
    Console.WriteLine($"File: {ex.FilePath}");
    Console.WriteLine($"Operation: {ex.OperationContext}");
}
```

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
- ✅ Column expression system with operator overloading and composition
- ✅ NivaraFrame (DataFrame-like structure) with column management and validation
- ✅ QueryFrame with fluent API for lazy query construction and execution
- ✅ Query operations (Filter, Select, GroupBy) with schema transformation
- ✅ Query planning and execution infrastructure with optimization analysis
- ✅ Query execution engine with comprehensive optimization passes (predicate pushdown, operation fusion, column elimination, operation reordering)
- ✅ DataFrame operations infrastructure with execution strategies and query node hierarchy
- ✅ Arrow/Parquet type mapping system with comprehensive CLR ↔ Arrow ↔ Parquet conversion
- ✅ Arrow interoperability with bidirectional conversion (ToArrowTable, FromArrowTable, ToArrowArray, FromArrowArray)
- ✅ Parquet I/O operations with reading and writing support (file/stream, async/sync, batch operations)
- ✅ Extension methods and fluent API for Arrow and Parquet operations with method chaining support
- ✅ Lazy data sources (CSV, JSON) with automatic schema inference and lazy evaluation
- ✅ Eager data sources (CSV, JSON) with immediate loading and data consistency validation

### Upcoming Features
- ✅ Data source scanning (CSV, JSON) with lazy evaluation - **Complete**
- ✅ Query optimization passes (predicate pushdown, operation fusion) - **Complete**
- ✅ Query execution engine with comprehensive optimization - **Complete**
- 🔄 Advanced aggregation functions (Sum, Count, Average, etc.)
- 🔄 Parquet streaming and batch operations (advanced features)
- 📋 Grouping with aggregation functions
- 📋 Joining operations between frames

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- Built on top of [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors/) for vectorized operations
- Inspired by modern columnar data processing libraries
- Designed for high-performance .NET applications

---

**Note**: Nivara is currently in active development. APIs may change between versions until v1.0 is released.
