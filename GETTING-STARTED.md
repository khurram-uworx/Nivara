# Getting Started with Nivara

This guide provides comprehensive examples and tutorials for using Nivara's DataFrame library. For a high-level overview, see [README.md](README.md). For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Working with Columns](#working-with-columns)
- [DataFrames and Schemas](#dataframes-and-schemas)
- [Query API](#query-api)
- [Data Sources](#data-sources)
- [Row Operations](#row-operations)
- [Column Operations](#column-operations)
- [Joins and Concatenation](#joins-and-concatenation)
- [Grouping and Aggregation](#grouping-and-aggregation)
- [Advanced Features](#advanced-features)
- [Extensions and I/O](#extensions-and-io)

---

## Core Concepts

### Typed Columns

Nivara columns are strongly typed and immutable:

```csharp
using Nivara;

// Create columns with explicit types
var ages = NivaraColumn<int>.Create(new[] { 25, 30, 35 });
var names = NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" });

Console.WriteLine(ages.Length); // 3
Console.WriteLine(names[0]);    // "Alice"
```

### Explicit Null Semantics

Nulls are tracked explicitly using validity masks, not sentinel values:

```csharp
// Create column with nullable data
var data = new int?[] { 1, null, 3 };
var column = NivaraColumn<int>.CreateFromNullable(data);

Console.WriteLine(column.HasNulls);   // True
Console.WriteLine(column.NullCount);  // 1
Console.WriteLine(column.IsNull(1));  // True (index 1 is null)
```

Null-aware operations behave predictably:

```csharp
var filled = column.FillNull(0);  // [1, 0, 3]
var dropped = column.DropNulls(); // [1, 3]
```

### Vectorized Operations

For vectorizable types, Nivara automatically uses SIMD-accelerated kernels:

```csharp
var a = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
var b = a * 1.5;  // Vectorized multiplication
var c = a + b;    // Vectorized addition

// Results: a=[1.0, 2.0, 3.0], b=[1.5, 3.0, 4.5], c=[2.5, 5.0, 7.5]
```

Nivara automatically selects the optimal storage backend:
- **TensorStorage**: For vectorizable types (`int`, `float`, `double`, `bool`)
- **MemoryStorage**: For non-vectorizable types (`string`, `Guid`, reference types)

---

## Working with Columns

### Creating Columns

```csharp
// From arrays
var integers = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
var doubles = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3 });

// For reference types (strings, objects)
var strings = NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "C" });

// From nullable arrays
var nullableInts = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 });
```

### Column Operations

```csharp
var numbers = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });

// Arithmetic operations
var doubled = numbers * 2;           // [2, 4, 6, 8, 10]
var incremented = numbers + 1;       // [2, 3, 4, 5, 6]

// Comparison operations
var mask = numbers > 3;              // [false, false, false, true, true]

// Aggregations
var sum = numbers.Sum();             // 15
var average = numbers.Average();     // 3.0
var min = numbers.Min();             // 1
var max = numbers.Max();             // 5
```

### Working with Null Values

```csharp
var data = new int?[] { 1, null, 3, null, 5 };
var column = NivaraColumn<int>.CreateFromNullable(data);

// Check for nulls
Console.WriteLine(column.HasNulls);     // True
Console.WriteLine(column.NullCount);    // 2
Console.WriteLine(column.ValidCount);   // 3

// Handle nulls
var filled = column.FillNull(0);        // [1, 0, 3, 0, 5]
var dropped = column.DropNulls();       // [1, 3, 5]

// Null-aware operations
var sum = column.Sum();                 // 9 (ignores nulls)
var doubled = column * 2;               // [2, null, 6, null, 10]
```

---

## DataFrames and Schemas

### Creating DataFrames

```csharp
// Create from columns
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000 }))
);

Console.WriteLine(frame.RowCount);      // 3
Console.WriteLine(frame.ColumnCount);   // 3
Console.WriteLine(frame.ColumnNames);   // ["Name", "Age", "Salary"]
```

### Accessing Data

```csharp
// Get columns
var nameColumn = frame.GetColumn<string>("Name");
var ageColumn = frame.GetColumn<int>("Age");

// Get column by index
var firstColumn = frame.GetColumn(0);

// Check if column exists
bool hasAge = frame.HasColumn("Age");

// Get schema information
var schema = frame.Schema;
Console.WriteLine(schema.GetColumnType("Age")); // System.Int32
```

### Schema Validation

Schemas are immutable and validated on every transformation:

```csharp
try
{
    // This will fail - column doesn't exist
    var invalidColumn = frame.GetColumn<string>("InvalidColumn");
}
catch (ArgumentException ex)
{
    Console.WriteLine($"Column not found: {ex.Message}");
}

try
{
    // This will fail - wrong type
    var wrongType = frame.GetColumn<string>("Age");
}
catch (InvalidCastException ex)
{
    Console.WriteLine($"Type mismatch: {ex.Message}");
}
```

---

## Query API

### Basic Queries

```csharp
using Nivara.Expressions;

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000, 80000 }))
);

// Using LINQ extensions (Recommended)
using Nivara.Linq;

// Filter rows
var adults = frame.AsQueryFrame()
    .Where(x => x["Age"] > 30)
    .ToNivaraFrame();
// Result: Charlie (35) and Diana (40)

// Select columns
var names = frame.AsQueryFrame()
    .Select(x => x["Name"])
    .ToNivaraFrame();
// Result: DataFrame with only Name column

// Chain operations
var result = frame.AsQueryFrame()
    .Where(x => x["Salary"] > 55000)
    .Select(x => x["Name"], x => x["Age"])
    .ToNivaraFrame();
// Result: Bob, Charlie, Diana with Name and Age columns

// Legacy Fluent API
var legacyResult = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Salary") > 55000)
    .Select("Name", "Age")
    .Collect();

```

### Complex Expressions

```csharp
// Multiple conditions
// Multiple conditions using LINQ
var complexFilter = frame.AsQueryFrame()
    .Where(x => x["Age"] > 25 & x["Salary"] < 75000)
    .ToNivaraFrame();

// Arithmetic in expressions
var bonusQuery = frame.AsQueryFrame()
    .Where(x => x["Salary"] * 0.1 > 6000) // 10% bonus > $6000
    .ToNivaraFrame();
```

### Lazy Evaluation

Queries are planned and validated before execution:

```csharp
// Build query (no execution yet)
var query = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name", "Salary");

// Inspect the query plan
var plan = query.GetQueryPlan();
Console.WriteLine(plan.ToString());

// Execute the query
var result = query.Collect();
```

---

## Data Sources

### CSV Data Sources

```csharp
using Nivara.IO;

// Lazy CSV scanning with schema inference
var csvQuery = Csv.ScanAsQueryFrame("employees.csv")
    .Filter(ColumnExpressions.Col("Salary") > 70000)
    .Select("Name", "Department", "Salary");

var result = csvQuery.Collect();

// Custom CSV options
var csvOptions = new CsvScanOptions
{
    HasHeader = true,
    Delimiter = ',',
    Quote = '"',
    NullValue = "NULL"
};

var customCsv = Csv.ScanAsQueryFrame("data.csv", csvOptions)
    .Collect();
```

### JSON Data Sources

```csharp
// Lazy JSON scanning
var jsonQuery = Json.ScanAsQueryFrame("data.json")
    .Filter(ColumnExpressions.Col("active") == true)
    .Select("id", "name", "email");

var jsonResult = jsonQuery.Collect();
```

### Schema Inference

Data sources automatically infer schemas:

```csharp
// Get inferred schema without loading data
var schema = Csv.InferSchema("employees.csv");
foreach (var column in schema.Columns)
{
    Console.WriteLine($"{column.Name}: {column.Type}");
}
```

---

## Row Operations

### Filtering with Boolean Masks

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 })),
    ("Active", NivaraColumn<bool>.Create(new[] { true, false, true, false }))
);

// Create boolean mask
var mask = NivaraColumn<bool>.Create(new[] { true, false, true, false });

// Filter using mask
var filtered = frame.FilterByMask(mask);
// Result: Alice and Charlie

// Filter using column values
var activeMask = frame.GetColumn<bool>("Active");
var activeUsers = frame.FilterByMask(activeMask);
// Result: Alice and Charlie
```

### Row Slicing

```csharp
// Take first n rows
var firstThree = frame.Take(3);
Console.WriteLine(firstThree.RowCount); // 3

// Skip first n rows
var remaining = frame.Skip(2);
Console.WriteLine(remaining.RowCount); // 2

// Combine Skip and Take for ranges
var middle = frame.Skip(1).Take(2);
// Result: Bob and Charlie

// Arbitrary slicing
var slice = frame.Slice(1, 2); // Start at index 1, take 2 rows
// Result: Bob and Charlie
```

### Sorting

```csharp
using Nivara.Operations;

// Single column sorting
// Single column sorting
var sortedByAge = frame.AsQueryFrame()
    .OrderBy(x => x["Age"])
    .ToNivaraFrame();
// Result: Alice (25), Bob (30), Charlie (35), Diana (40)

// Descending sort
// Descending sort
var sortedBySalaryDesc = frame.AsQueryFrame()
    .OrderByDescending(x => x["Salary"])
    .ToNivaraFrame();

// Multi-column sorting
var multiSorted = frame.AsQueryFrame()
    .Sort(new[]
    {
        new SortKey("Department", SortDirection.Ascending),
        new SortKey("Salary", SortDirection.Descending)
    })
    .Collect();
```

### Null Handling in Sorting

```csharp
var frameWithNulls = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Score", NivaraColumn<int>.CreateFromNullable(new int?[] { 85, null, 92 }))
);

// Nulls first
var nullsFirst = frameWithNulls.AsQueryFrame()
    .Sort("Score", SortDirection.Ascending, NullOrdering.NullsFirst)
    .Collect();
// Result: Bob (null), Alice (85), Charlie (92)

// Nulls last (default)
var nullsLast = frameWithNulls.AsQueryFrame()
    .Sort("Score", SortDirection.Ascending, NullOrdering.NullsLast)
    .Collect();
// Result: Alice (85), Charlie (92), Bob (null)
```

---

## Column Operations

### Column Transformations

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000 }))
);

// Transform single column (create new column)
var frameWithAgeInMonths = frame.WithTransformedColumn<int, int>(
    "Age", 
    age => age * 12, 
    "AgeInMonths"
);
// Result: New column AgeInMonths with values [300, 360, 420]

// Transform and replace existing column
var frameWithRaise = frame.WithTransformedColumn<double, double>(
    "Salary",
    salary => salary * 1.1 // 10% raise
);
// Result: Salary column updated with 10% increase

// Multi-column transformation
var frameWithBonus = frame.WithComputedColumn<int, double, double>(
    "Age",
    "Salary", 
    (age, salary) => age > 30 ? salary * 0.1 : salary * 0.05,
    "Bonus"
);
// Result: Bonus column based on age and salary
```

### Column Selection and Projection

```csharp
// Select specific columns
var nameAndAge = frame.Select("Name", "Age");
// Result: DataFrame with only Name and Age columns

// Select with array
var selectedColumns = frame.Select(new[] { "Name", "Salary" });

// Select and rename
var renamedFrame = frame.SelectAndRename(new Dictionary<string, string?>
{
    { "Name", "EmployeeName" },
    { "Age", "YearsOld" },
    { "Salary", null } // Keep original name
});
// Result: Columns renamed to EmployeeName, YearsOld, Salary
```

### Column Renaming

```csharp
// Rename single column
var renamedSingle = frame.RenameColumn("Age", "YearsOld");

// Rename multiple columns
var renamedMultiple = frame.RenameColumns(new Dictionary<string, string>
{
    { "Name", "EmployeeName" },
    { "Age", "YearsOld" }
});
```

### Column Exclusion

```csharp
// Exclude specific columns
var withoutAge = frame.Exclude("Age");
// Result: DataFrame with Name and Salary only

// Exclude multiple columns
var nameOnly = frame.Exclude("Age", "Salary");
// Result: DataFrame with only Name column
```

### Null Handling in Transformations

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
// Result: Doubled = [2, null, 6, null, 10]
// Nulls are preserved without applying the transformation

// Multi-column with null propagation
var frameWithNulls = NivaraFrame.Create(
    ("A", NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3 })),
    ("B", NivaraColumn<int>.CreateFromNullable(new int?[] { 2, 4, null }))
);

var sum = frameWithNulls.WithComputedColumn<int, int, int>(
    "A", "B",
    (a, b) => a + b,
    "Sum"
);
// Result: Sum = [3, null, null]
// Any null input produces null output
```

---

## Joins and Concatenation

### Basic Join Operations

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

// Inner join - only matching rows
var innerJoin = employees.InnerJoin(departments, "Id");
// Result: 3 rows (Id: 2, 3, 4) with all columns from both DataFrames

// Left join - all left rows, matching right rows
var leftJoin = employees.LeftJoin(departments, "Id");
// Result: 4 rows, Alice (Id=1) has null Department and Salary

// Right join - all right rows, matching left rows
var rightJoin = employees.RightJoin(departments, "Id");
// Result: 4 rows, Marketing (Id=5) has null Name and Age

// Full outer join - all rows from both DataFrames
var fullJoin = employees.FullOuterJoin(departments, "Id");
// Result: 5 rows, includes Alice (left only) and Marketing (right only)
```

### Join with Different Column Names

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
// Result: Matches customers with orders using CustomerId = CustId
```

### Column Name Conflict Resolution

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
    Console.WriteLine($"Column conflict: {ex.Message}");
}
```

### DataFrame Concatenation

#### Vertical Concatenation (Row Append)

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
```

#### Handling Schema Mismatches

```csharp
var employees = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
);

var contractors = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000.0 }))
);

// Fill missing columns with nulls (default)
var combined = employees.ConcatenateVertical(contractors, ConcatenationMismatchHandling.FillWithNulls);
// Result: Charlie has null Age, Alice/Bob have null Salary

// Error on mismatch
try 
{
    var strict = employees.ConcatenateVertical(contractors, ConcatenationMismatchHandling.Error);
}
catch (SchemaValidationException ex)
{
    Console.WriteLine($"Schema mismatch: {ex.Message}");
}
```

#### Horizontal Concatenation (Column Append)

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
```

---

## Grouping and Aggregation

### GroupBy Operations

```csharp
using Nivara.Expressions;

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Alice", "Charlie" })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "IT", "HR", "IT", "Finance" })),
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

// Access group information
foreach (var group in groupedByName.Groups)
{
    Console.WriteLine($"Group: {group.Key}, Count: {group.Indices.Count}");
}
```

### Aggregation Functions

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

// Apply to groups
var grouped = frame.AsQueryFrame().GroupBy("Department").Collect();
foreach (var group in grouped.Groups)
{
    var groupSalaries = group.Indices.Select(i => salaryColumn.GetValue(i)).Cast<double>();
    Console.WriteLine($"{group.Key}: Avg = {groupSalaries.Average():C}");
}
```

### Custom Aggregation Functions

```csharp
public class MedianAggregation : AggregationFunction
{
    public override string Name => "Median";
    
    public override Type GetResultType(Type inputType) => inputType;
    
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var validValues = new List<double>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null && value is double d) 
                validValues.Add(d);
        }
        
        if (validValues.Count == 0) return null;
        
        validValues.Sort();
        int mid = validValues.Count / 2;
        
        return validValues.Count % 2 == 0 
            ? (validValues[mid - 1] + validValues[mid]) / 2.0
            : validValues[mid];
    }
}

// Use custom aggregation
var medianAgg = new MedianAggregation();
var salaryColumn = frame.GetColumn<double>("Salary");
var allIndices = Enumerable.Range(0, salaryColumn.Length).ToList();
var median = medianAgg.Apply(salaryColumn, allIndices);
Console.WriteLine($"Median salary: {median:C}");
```

### Vectorized Aggregations

```csharp
// Vectorized operations for float/double types
var floats = NivaraSeries<float>.Create(new[] { 1.5f, 2.5f, 3.5f, 4.5f });
var sum = floats.Sum(); // Uses TensorPrimitives.Sum for performance

// Null-aware aggregation
var nullableData = new int?[] { 1, null, 3, null, 5 };
var column = NivaraColumn<int>.CreateFromNullable(nullableData);
var series = new NivaraSeries<int>(column);

Console.WriteLine(series.Sum());     // 9 (ignores nulls)
Console.WriteLine(series.Average()); // 3.0 (9/3, ignores nulls)
Console.WriteLine(series.Count());   // 5 (includes nulls)
Console.WriteLine(series.ValidCount()); // 3 (excludes nulls)
```

---

## Advanced Features

### Fluent API

```csharp
var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Alice" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 28 })),
    ("Score", NivaraColumn<double>.Create(new[] { 85.5, 92.0, 78.5, 88.0 }))
);

// Chain operations fluently
var result = frame
    .Where(row => row.GetValue<int>("Age") > 25)
    .OrderBy("Score", ascending: false)
    .GroupBy("Name")
    .Take(10);

// Mix materialized and lazy operations
var queryResult = frame
    .AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 27)
    .Sort("Score", SortDirection.Descending)
    .Collect();
```

### Query Optimization

```csharp
using Nivara.Expressions;

// Query with optimization opportunities
var query = Csv.ScanAsQueryFrame("employees.csv")
    .Select("Name", "Age", "Salary", "Department")  // Select all first
    .Filter(ColumnExpressions.Col("Age") > 25)      // Filter after selection
    .Filter(ColumnExpressions.Col("Salary") > 50000) // Multiple filters
    .Select("Name", "Salary");                       // Final projection

// The optimizer automatically:
// 1. Pushes filters closer to data source (predicate pushdown)
// 2. Combines multiple filters (operation fusion)
// 3. Eliminates unused columns early (projection pushdown)

var result = query.Collect(); // Optimizations applied during execution

// Get optimization details
var optimizer = new QueryOptimizer();
var plan = query.GetQueryPlan();
var optimizationResult = optimizer.OptimizeWithStatistics(plan);
Console.WriteLine(optimizationResult.GenerateReport());
```

### Execution Strategies

```csharp
using Nivara;
using Nivara.Execution;

var engine = new ExecutionEngine();
var plan = query.GetQueryPlan();

// Lazy execution (default) - optimal for most cases
var lazyContext = new ExecutionContext(ExecutionStrategy.Lazy);
var lazyResult = engine.Execute(plan, lazyContext);

// Parallel execution - for CPU-intensive operations
var parallelContext = ExecutionContext.WithParallelism(Environment.ProcessorCount);
var parallelResult = engine.Execute(plan, parallelContext);

// Streaming execution - for large datasets
var streamingContext = ExecutionContext.WithMemoryBudget(512 * 1024 * 1024); // 512MB
streamingContext.Strategy = ExecutionStrategy.Streaming;
var streamingResult = engine.Execute(plan, streamingContext);

// Async execution with progress reporting
var cancellationTokenSource = new CancellationTokenSource();
var progress = new Progress<ExecutionProgress>(p => 
{
    Console.WriteLine($"{p.OperationName}: {p.PercentComplete:P1}");
});

var asyncContext = new ExecutionContext
{
    Strategy = ExecutionStrategy.Parallel,
    CancellationToken = cancellationTokenSource.Token,
    Progress = progress
};

var asyncResult = await engine.ExecuteAsync(plan, asyncContext);
```

### Error Handling and Diagnostics

```csharp
// Structured exception handling
try
{
    var result = leftFrame.InnerJoin(rightFrame, "InvalidKey");
}
catch (JoinException ex)
{
    Console.WriteLine($"Join failed: {ex.Message}");
    Console.WriteLine($"Join type: {ex.AttemptedJoinType}");
    Console.WriteLine($"Left keys: {string.Join(", ", ex.LeftKeys)}");
    Console.WriteLine($"Right keys: {string.Join(", ", ex.RightKeys)}");
    Console.WriteLine(ex.GetDetailedContext());
}

// Performance diagnostics
var diagnostics = new ExecutionDiagnostics();

var result = DiagnosticHelper.ExecuteWithDiagnostics(
    diagnostics,
    "ComplexQuery",
    () => frame
        .Filter(row => row.GetValue<int>("Age") > 25)
        .Sort("Name")
        .GroupBy("Department"),
    frame.RowCount);

Console.WriteLine(diagnostics.GenerateReport());

var summary = diagnostics.GetSummary();
Console.WriteLine($"Processed {summary.TotalRowsProcessed:N0} rows in {summary.TotalExecutionTime.TotalMilliseconds:F2}ms");
Console.WriteLine($"Throughput: {summary.AverageThroughput:F0} rows/sec");
```

---

## Extensions and I/O

### Apache Arrow Interoperability

```csharp
using Nivara.IO;

// Convert NivaraFrame to Arrow Table
var arrowTable = frame.ToArrowTable();

// Convert Arrow Table back to NivaraFrame
var restoredFrame = arrowTable.FromArrowTable();

// Series-level conversions
var series = new NivaraSeries<int>(NivaraColumn<int>.Create(new[] { 1, 2, 3 }));
var arrowArray = series.ToArrowArray();
var restoredSeries = arrowArray.FromArrowArray<int>();

// Custom conversion options
var arrowOptions = new ArrowConversionOptions
{
    UseZeroCopy = true,
    ValidateTypes = true,
    TimeZone = TimeZoneInfo.Utc
};

var customArrowTable = frame.ToArrowTable(arrowOptions);
```

### Parquet File I/O

```csharp
using Nivara.IO;

// Write to Parquet file
frame.ToParquet("employees.parquet");

// Read from Parquet file
var loadedFrame = NivaraFrameExtensions.LoadParquet("employees.parquet");

// Async operations
await frame.ToParquetAsync("employees_async.parquet");
var asyncFrame = await NivaraFrameExtensions.LoadParquetAsync("employees_async.parquet");

// Stream-based operations
using var fileStream = new FileStream("employees_stream.parquet", FileMode.Create);
frame.ToParquetStream(fileStream);

// Custom Parquet options
var parquetOptions = new ParquetWriteOptions
{
    Compression = "snappy",
    RowGroupSize = 10000,
    ValidateSchema = true
};

frame.ToParquet("employees_custom.parquet", parquetOptions);

// Batch operations
var frames = new[] { frame1, frame2, frame3 };
NivaraFrameExtensions.WriteParquetBatch("batch.parquet", frames, parquetOptions);
```

### Configuration and Performance Tuning

```csharp
// Memory management
var memoryManager = new NivaraMemoryManager();
memoryManager.SetMemoryBudget(1024 * 1024 * 1024); // 1GB
memoryManager.EnableBufferPooling(true);

// Performance monitoring
var performanceMonitor = new PerformanceMonitor();
performanceMonitor.EnableDetailedMetrics(true);
performanceMonitor.SetSamplingInterval(TimeSpan.FromSeconds(1));

// Resource management
using var resourceManager = new ResourceManager();
resourceManager.SetMaxParallelism(Environment.ProcessorCount);
resourceManager.EnableAutoGarbageCollection(true);
```

---

## Best Practices

### Performance Tips

1. **Use lazy evaluation** for complex queries to enable optimization
2. **Prefer vectorizable types** (int, float, double) when possible
3. **Use appropriate execution strategies** based on data size and operations
4. **Enable query optimization** for file-based data sources
5. **Monitor memory usage** with streaming execution for large datasets

### Error Handling

1. **Validate schemas early** before expensive operations
2. **Use structured exception handling** for operation-specific errors
3. **Enable diagnostics** for performance troubleshooting
4. **Check for null values** when working with external data sources

### Code Organization

1. **Separate query construction from execution** for better testability
2. **Use fluent API** for readable data processing pipelines
3. **Create reusable aggregation functions** for domain-specific calculations
4. **Leverage type safety** to catch errors at compile time

---

This guide covers the essential patterns for working with Nivara. For more advanced scenarios and architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).