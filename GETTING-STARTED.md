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
- [Tensor Interop](#tensor-interop)
- [Automatic Differentiation](#automatic-differentiation)
- [Extensions and I/O](#extensions-and-io)
- [Real World Examples](EXAMPLES.md)

---

## Core Concepts

### Typed Columns

Nivara columns are strongly typed and immutable:

```csharp
using Nivara;

// Create columns with explicit types
NivaraColumn<int> ages = [25, 30, 35];
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

For vectorizable types, Nivara uses SIMD-accelerated kernels where semantics are simple and null handling is explicit:

```csharp
var a = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
var b = a * 1.5;  // Vectorized multiplication
var c = a + b;    // Vectorized addition

// Results: a=[1.0, 2.0, 3.0], b=[1.5, 3.0, 4.5], c=[2.5, 5.0, 7.5]
```

All columns share a single storage class (`ColumnStorage<T>`, a sole-owner `T[]` plus an optional `bool[]` null mask). Whether an operation dispatches to a SIMD kernel is decided at runtime by `KernelSelector` — vectorized when `T` is vectorizable (`int`, `float`, `double`, `bool`, etc.), hardware acceleration is available, and the column is large enough to make vectorization worthwhile. This is transparent to users.

Use `System.Numerics.Tensors` directly for tensor math such as dot products, norms, cosine similarity, and model-facing APIs. Nivara's role is to preserve typed columns, schemas, labels, and null masks at the tabular boundary.

---

## Working with Columns

### Creating Columns

```csharp
// From arrays
var integers = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
var doubles = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3 });

// From collection expressions
NivaraColumn<int> ids = [101, 102, 103];
NivaraSeries<float> scores = [0.8f, 0.5f, 0.9f];

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
using Nivara.Linq;

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Diana" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000, 80000 }))
);

// Row type whose properties map to columns (case-insensitive, validated eagerly)
public sealed class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
    public double Salary { get; set; }
}

// Filter rows with typed predicates
var adults = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Collect();
// Result: Charlie (35) and Diana (40)

// Project columns
var names = frame.Query<Person>()
    .Select(p => new { p.Name })
    .Collect();
// Result: DataFrame with only Name column

// Chain operations
var result = frame.Query<Person>()
    .Where(p => p.Salary > 55000)
    .Select(p => new { p.Name, p.Age })
    .Collect();
// Result: Bob, Charlie, Diana with Name and Age columns
```

### Complex Expressions

```csharp
// Multiple conditions
var complexFilter = frame.Query<Person>()
    .Where(p => p.Age > 25 && p.Salary < 75000)
    .Collect();

// Arithmetic in expressions
var bonusQuery = frame.Query<Person>()
    .Where(p => p.Salary * 0.1 > 6000) // 10% bonus > $6000
    .Collect();
```

### Typed Object LINQ (`frame.Query<T>()`)

The typed object model layers strongly typed lambdas over the same query engine — no string column names and no `RowExpressionBuilder`:

```csharp
// Define a row type whose properties map to columns (case-insensitive)
public sealed class Person
{
    public string Name { get; set; }
    public string Department { get; set; }
    public int Age { get; set; }
    public double Salary { get; set; }
}

var people = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Diana" })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "IT", "HR", "IT", "Finance" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 40 })),
    ("Salary", NivaraColumn<double>.Create(new[] { 50000, 60000, 70000, 80000 }))
);

// Typed predicates and projections — validated eagerly at Query<T>()
var result = people.Query<Person>()
    .Where(p => p.Age > 30 && p.Salary > 55000)
    .OrderByDescending(p => p.Salary)
    .Select(p => new { p.Name, AnnualBonus = p.Salary * 0.1 })
    .ToObjects();
// IReadOnlyList<anonymous>: Diana (8000), Charlie (7000)

// Materialize as a NivaraFrame instead (lazy, ExplainPlan available)
var resultFrame = people.Query<Person>()
    .Where(p => p.Age > 30)
    .Select(p => new { p.Name, p.Age })
    .Collect();

// GroupBy aggregates via Grouping<TKey,T>: g.Key, g.Average/Sum/Count/Min/Max
var byDept = people.Query<Person>()
    .GroupBy(p => p.Department)
    .Select(g => new { g.Key, AvgSalary = g.Average(p => p.Salary), People = g.Count() })
    .ToObjects();
```

Notes:
- `Query<T>()` requires `T : class, new()`. `Collect()`/`ToList()` return a `NivaraFrame`; `ToObjects()`/`ToRows()` return `IReadOnlyList<TResult>`.
- Supported: property access, literals, `+ - * /`, comparisons, `&&`/`||`/`!`. Method calls, captured variables/closures, nested property access, and ternary fail fast with `UnsupportedQueryExpressionException` at translation time.
- `GroupBy` accepts an aggregate `Select` or a bare `Collect` of distinct keys; any other operation after `GroupBy` fails fast.
- See the full typed-query example in [EXAMPLES.md](EXAMPLES.md#5c-typed-object-linq--framequeryt).

### Lazy Evaluation

Queries are planned and validated before execution:

```csharp
// Build query (no execution yet) — typed queries are lazy and inspectable
var query = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Select(p => new { p.Name, p.Salary });

// Inspect the query plan
Console.WriteLine(query.ExplainPlan());

// Execute the query
var result = query.Collect();
```

---

## Data Sources

### CSV Data Sources

```csharp
using Nivara.IO;

public sealed class Employee
{
    public string Name { get; set; }
    public string Department { get; set; }
    public int Salary { get; set; }
}

// Lazy CSV scanning with schema inference (CSV integers infer as int)
var csvQuery = Csv.ScanQuery<Employee>("employees.csv")
    .Where(e => e.Salary > 70000)
    .Select(e => new { e.Name, e.Department, e.Salary });

var result = csvQuery.Collect();

// Custom CSV options
var csvOptions = CsvOptions.Default.With(
    hasHeaderRecord: true,
    delimiter: ",",
    trimOptions: CsvTrimOptions.Trim);

var customCsv = Csv.ScanQuery<Employee>("data.csv", csvOptions)
    .Collect();
```

### JSON Data Sources

```csharp
using Nivara.IO;

public sealed class User
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
    public bool Active { get; set; }
}

// Lazy JSON scanning (JSON numbers infer as double)
var jsonQuery = Json.ScanQuery<User>("data.json")
    .Where(u => u.Active)
    .Select(u => new { u.Id, u.Name, u.Email });

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
using Nivara.Linq;
using Nivara.Operations;

// Single column sorting
var sortedByAge = frame.Query<Person>()
    .OrderBy(p => p.Age)
    .Collect();
// Result: Alice (25), Bob (30), Charlie (35), Diana (40)

// Descending sort
var sortedBySalaryDesc = frame.Query<Person>()
    .OrderByDescending(p => p.Salary)
    .Collect();

// Multi-column sorting with per-key direction
var multiSorted = frame.Query<Person>()
    .OrderBy(p => p.Name, direction: SortDirection.Ascending)
    .ThenBy(p => p.Salary, direction: SortDirection.Descending)
    .Collect();
```

### Null Handling in Sorting

```csharp
public sealed class Player
{
    public string Name { get; set; }
    public int? Score { get; set; }
}

var frameWithNulls = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" })),
    ("Score", NivaraColumn<int>.CreateFromNullable(new int?[] { 85, null, 92 }))
);

// Nulls first
var nullsFirst = frameWithNulls.Query<Player>()
    .OrderBy(p => p.Score, nullOrdering: NullOrdering.NullsFirst)
    .Collect();
// Result: Bob (null), Alice (85), Charlie (92)

// Nulls last (default)
var nullsLast = frameWithNulls.Query<Player>()
    .OrderBy(p => p.Score, nullOrdering: NullOrdering.NullsLast)
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
using Nivara.Linq;

public sealed class Employee
{
    public string Name { get; set; }
    public string Department { get; set; }
    public double Salary { get; set; }
}

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Alice", "Charlie" })),
    ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "IT", "HR", "IT", "Finance" })),
    ("Salary", NivaraColumn<double>.Create(new[] { 75000, 65000, 78000, 85000 }))
);

// Group by a single key column — collect the distinct keys
var groupedByName = frame.Query<Employee>()
    .GroupBy(e => e.Name)
    .Collect();

// Group by with aggregation: g.Key, g.Average/Sum/Count/Min/Max
var byDept = frame.Query<Employee>()
    .GroupBy(e => e.Department)
    .Select(g => new
    {
        g.Key,
        EmployeeCount = g.Count(),
        TotalSalary = g.Sum(e => e.Salary),
        AvgSalary = g.Average(e => e.Salary)
    })
    .ToObjects();
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
var grouped = frame.Query<Employee>()
    .GroupBy(e => e.Department)
    .Select(g => new { g.Key, AvgSalary = g.Average(e => e.Salary) })
    .ToObjects();
foreach (var group in grouped)
{
    Console.WriteLine($"{group.Key}: Avg = {group.AvgSalary:C}");
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
using Nivara.Linq;
using Nivara.Operations;

public sealed class Contestant
{
    public string Name { get; set; }
    public int Age { get; set; }
    public double Score { get; set; }
}

var frame = NivaraFrame.Create(
    ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie", "Alice" })),
    ("Age", NivaraColumn<int>.Create(new[] { 25, 30, 35, 28 })),
    ("Score", NivaraColumn<double>.Create(new[] { 85.5, 92.0, 78.5, 88.0 }))
);

// Chain operations fluently — typed predicates and projections
var result = frame.Query<Contestant>()
    .Where(c => c.Age > 25)
    .OrderByDescending(c => c.Score)
    .Select(c => new { c.Name, c.Score })
    .Collect();
```

### Query Optimization

```csharp
using Nivara.IO;

public sealed class Employee
{
    public string Name { get; set; }
    public int Age { get; set; }
    public int Salary { get; set; }
    public string Department { get; set; }
}

// Queries are optimized automatically before execution:
// 1. Filters are pushed closer to the data source (predicate pushdown)
// 2. Multiple filters are combined (operation fusion)
// 3. Unused columns are eliminated early (projection pushdown)

var query = Csv.ScanQuery<Employee>("employees.csv")
    .Where(e => e.Age > 25)
    .Where(e => e.Salary > 50000)
    .Select(e => new { e.Name, e.Salary });

var result = query.Collect(); // Optimizations applied during execution

// Inspect the optimized plan
Console.WriteLine(query.ExplainPlan());
```

### Execution

```csharp
// Execution is lazy by default. The engine optimizes the plan (predicate
// pushdown, operation fusion, projection pushdown) before any data is
// touched, and the plan can be inspected with ExplainPlan().

var query = frame.Query<Contestant>()
    .Where(c => c.Age > 25)
    .Select(c => new { c.Name, c.Score });

Console.WriteLine(query.ExplainPlan());
var result = query.Collect();
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

// Query diagnostics — inspect the optimized plan before execution
var query = frame.Query<Contestant>()
    .Where(c => c.Age > 25)
    .OrderBy(c => c.Name)
    .Select(c => new { c.Name, c.Score });

Console.WriteLine(query.ExplainPlan());
var result = query.Collect();
```

---

## Tensor Interop

Nivara provides tensor interop for moving tabular data into platform tensor APIs. It does not replace `Tensor<T>` or `TensorPrimitives` for numerical kernels.

### Null-Preserving Tensor Export

```csharp
using Nivara.Tensors;

var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });
NullableTensor<float> tensor = column.ToNullableTensor();

Console.WriteLine(tensor.Data.Lengths[0]);       // 3
Console.WriteLine(tensor.NullMask!.AsTensorSpan()[1]); // True
```

For null-free data, use `ToTensor()` when you only need tensor data:

```csharp
var values = NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f });
var tensor = values.ToTensor();
```

### 2D Tensor Import With Labels

```csharp
var matrix = Tensor.Create(
    new[] { 0.9f, 0.2f, 0.1f, 0.8f },
    new nint[] { 2, 2 });

using var frame = NivaraFrame.FromMatrix(
    matrix,
    columnNames: ["e0", "e1"],
    rowLabels: ["doc-1", "doc-2"],
    rowLabelColumnName: "DocumentId");
```

### Labeled Row Vector Ingestion

```csharp
using var embeddings = NivaraFrame.FromRows(
    [
        ("doc-1", new[] { 0.9f, 0.2f, 0.5f }),
        ("doc-2", new[] { 0.1f, 0.8f, 0.4f })
    ],
    columnNames: ["e0", "e1", "e2"],
    labelColumnName: "DocumentId");
```

After ingestion, use Nivara for filtering, joining, grouping, and schema validation. Extract spans or tensors and call `TensorPrimitives` when you need vector math.

---

## Automatic Differentiation

Nivara provides automatic differentiation (AutoDiff) on DataFrames, enabling gradient-based optimization and machine learning workflows directly on your data.

Inference is the default. A normal `model.Forward(input)` or tensor operation computes values without building a training graph. When you are writing a manual training step, wrap the forward/loss/backward/update code in `using (GradientUtils.Grad())`. Built-in training loops enter that scope for you.

Supported numeric types are constrained to `IFloatingPointIeee754<T>` — `float`, `double`, and `Half`. AutoDiff is a non-nullable domain per [ADR-001](docs/adr/001-autodiff-nonnullable-domain.md): convert nullable columns with `FillNull`/`DropNulls` before crossing the AutoDiff boundary.

### Converting to Gradient Tensors

You can convert Columns, Series, or entire DataFrames into `ReverseGradTensor<T>` objects:

```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;

var df = NivaraFrame.Create(
    ("x", NivaraColumn<float>.Create(new[] { 1.0f, 2.0f, 3.0f })),
    ("y", NivaraColumn<float>.Create(new[] { 2.0f, 4.0f, 6.0f }))
);

// Convert to dictionary of ReverseGradTensors.
// requiresGrad marks trainable inputs, but graph tracking only happens inside GradientUtils.Grad().
var tensors = df.ToReverseGradTensors<float>(new[] { "x", "y" }, requiresGrad: true);

var x = tensors["x"];
var y = tensors["y"];
```

### Computing Gradients

`ReverseGradTensor<T>` supports `+`, `-`, `*`, `/` and unary `-` operator overloads that delegate to `ReverseGradOperations`:

```csharp
// Simple linear model: z = w * x + b using operator overloads.
var w = ReverseGradTensor<float>.FromArray(new[] { 0.5f, 0.5f, 0.5f }, requiresGrad: true);
var b = ReverseGradTensor<float>.FromArray(new[] { 0.1f, 0.1f, 0.1f }, requiresGrad: true);

using (GradientUtils.Grad())
{
    var z = w * x + b;

    // MSE loss: mean((z - y)^2)
    var diff = z - y;
    var loss = ReverseGradOperations.Mean(diff * diff);

    // Backward pass — computes gradients for w and b
    loss.Backward();
}

// Access gradients directly on the tensor
Console.WriteLine($"w gradient at index 0: {w.Grad![0]:F4}");
Console.WriteLine($"b gradient at index 0: {b.Grad![0]:F4}");
```

### Retrieving Gradients

Extract gradients back into a Nivara DataFrame for analysis:

```csharp
// Get a DataFrame containing gradients for all tracking tensors
var gradFrame = tensors.ToGradientFrame();
```

### Zeroing Gradients

When training in a loop, zero out gradients before the next pass:

```csharp
tensors.BatchZeroGrad();
```

> For module-based models (`Linear`, `Sequential`), optimizers (`SGD`, `Adam`, `AdamW`), training loops, model serialization, and data-parallel training, see [AUTODIFF.md](docs/AUTODIFF.md) and the Act 7b / Act 8 examples in [EXAMPLES.md](EXAMPLES.md). The `samples/` directory includes character-level GPT, a neural chess evaluator, a hybrid Nivara+LLM agent workflow, a VAE for synthetic pattern generation, PyTorch parity benchmarks, a MiniLM inference pipeline, DistilBERT fine-tuning (SST-2), and MobileNetV2/ResNet-18 inference.

AutoDiff correctness is validated against PyTorch: the `NivaraTorch` suite runs **55 functional tests** whose forward/backward values are compared against tensors generated by `gen_reference.py` (21+ layer types). A cross-framework parity exercise trained an identical 3-layer MLP in both Nivara and PyTorch with **<0.04% loss-curve divergence over 50 epochs**.

Module models expose `StateDict()` and `LoadStateDict()` for in-memory
save/restore, transfer learning, and partial loading. Use
`ModelSerializer.StateDictToJson(state)` / `JsonToStateDict<T>(json)` when you
want the same state dictionary on disk or over the wire.

### Embedding Layers

Nivara includes differentiable embedding layers for mapping discrete token IDs to dense vectors:

```csharp
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;

// Dense embedding: vocabSize → embeddingDim
var embedding = new Embedding<float>(vocabSize: 1000, embeddingDim: 64);

// Forward pass with token IDs (flattened batch)
var tokenIds = ReverseGradTensor<float>.FromMatrix(
    new[] { 1.0f, 5.0f, 12.0f, 3.0f }, batchSize: 2, seqLen: 2, requiresGrad: false);
var embedded = embedding.Forward(tokenIds);
// embedded.Shape = [2, 2, 64]

// Sparse embedding bag: sums active feature rows per batch, ignores padding index
var sparseEmb = new SparseEmbedding<float>(numEmbeddings: 500, embeddingDim: 32, paddingIndex: -1);
// Input shape: [batchSize, maxActiveFeatures] — padding index entries are skipped
```

### Transformer Blocks

The `TransformerBlock` implements a pre-norm transformer with causal self-attention. Norm type is configurable via the `NormType` enum — `RMSNorm` (default) or `LayerNorm`:

```csharp
// RMSNorm (default)
var block = new TransformerBlock<float>(
    nEmbd: 128, nHead: 4, dropout: 0.1, maxSeqLen: 256);

// LayerNorm variant
var lnBlock = new TransformerBlock<float>(
    nEmbd: 128, nHead: 4, dropout: 0.1, maxSeqLen: 256, normType: NormType.LayerNorm);

// Forward: [seqLen, nEmbd] → [seqLen, nEmbd]
var output = block.Forward(inputTensor);
```

Architecture per block: `RMSNorm → Multi-Head Causal Attention → Residual → RMSNorm → Gated MLP (GELU) → Residual`. The FFN uses GELU instead of ReLU² — matching PyTorch's `GELU()` activation.

### Convolution Layers

Nivara includes 1D and 2D convolution layers with full autograd support. Both use tiled im2col → `TensorPrimitives.Dot` kernels; the Conv1d weight layout is PyTorch-compatible `[outChannels, inChannels, kernelSize]`:

```csharp
using Nivara.AutoDiff.Nn;

// 1D convolution: [batch, inChannels, length] → [batch, outChannels, outLength]
var conv1d = new Conv1d<float>(inChannels: 64, outChannels: 128, kernelSize: 3, stride: 1, padding: 1);
var output1d = conv1d.Forward(inputTensor);  // [B, 128, L]

// 2D convolution: [batch, inChannels, H, W] → [batch, outChannels, H', W']
// Uses PatchLocation lookup + 1×1 fast path + InputGrad specializations
var conv2d = new Conv2d<float>(inChannels: 3, outChannels: 16, kernelSize: 3, stride: 1, padding: 1);
var output2d = conv2d.Forward(inputTensor);  // [B, 16, H, W]

// Grouped convolution (depthwise)
var depthwise = new Conv2d<float>(inChannels: 64, outChannels: 64, kernelSize: 3, groups: 64);

// Transposed convolution (decoder upsampling)
var deconv = new ConvTranspose2d<float>(inChannels: 32, outChannels: 16, kernelSize: 4, stride: 2, padding: 1);
var upsampled = deconv.Forward(latentTensor);  // [B, 16, H*2, W*2]

// Depthwise separable convolution (MobileNet-style)
var separable = new DepthwiseSeparableConv2d<float>(inChannels: 64, outChannels: 128, kernelSize: 3);
```

### Normalization Layers

```csharp
// Batch normalization (train/eval modes, running statistics)
var bn1d = new BatchNorm1d<float>(numFeatures: 128);
var bn2d = new BatchNorm2d<float>(numFeatures: 64);

// BatchNorm1d also accepts 3D [B, C, L] input for Conv1d pipelines
var bn1d_3d = new BatchNorm1d<float>(numFeatures: 64);  // feed with tensor of shape [batch, channels, length]

// Layer normalization (no running stats, normalizes over last dimension)
var ln = new LayerNorm<float>(normalizedShape: 128);
```

### Activations

```csharp
// Activations are functional — call them via the Activation helper class
var h = Activation.Gelu(x);        // tanh approximation, PyTorch-compatible
var r = Activation.Relu(x);
var t = Activation.Tanh(x);
var s = Activation.Sigmoid(x);
var l = Activation.LeakyRelu(x, negativeSlope: 0.01f);

// Or call the raw operation directly
var h2 = ReverseGradOperations.Gelu(x);
```

### Pooling Layers

```csharp
// Max pooling (2D, [B, C, H, W])
var pool = new MaxPool2d<float>(kernelSize: 2, stride: 2, padding: 0);
var pooled = pool.Forward(x);  // [B, C, H/2, W/2]

// Adaptive average pooling (global → [B, C, 1, 1])
var gap = new AdaptiveAvgPool2d<float>(outputSize: 1);
var flat = gap.Forward(x);  // used by classifier heads
```

### Variational Autoencoders

```csharp
// Standard VAE (MLP-based)
var vae = new VAE<float>(inputDim: 784, latentDim: 32, hiddenDim: 256);
var (recon, mu, logVar) = vae.EncodeDecode(input);

// Convolutional VAE (spatial latent representations)
var convVae = new ConvVAE<float>(
    inputChannels: 1,
    encoderChannels: new[] { 32, 64 },
    latentChannels: 16,
    spatialSize: 28,
    kernelSize: 3, stride: 2, padding: 1);
var recon = convVae.Forward(imageTensor);
```

### Attention

```csharp
// Standalone multi-head attention (self-attention or cross-attention)
var mha = new MultiheadAttention<float>(embedDim: 128, numHeads: 4);
var selfAttn = mha.Forward(input);                          // self-attention
var crossAttn = mha.Forward(query, key, value);             // cross-attention
var causalAttn = mha.Forward(input, causal: true);          // with causal mask
```

### NLP Models

Two ready-to-use differentiable NLP models ship as application-level sample code in `samples/Nivara.Samples/` (namespace `Nivara.AutoDiff.Nn`), built from core primitives (`Embedding`, `Linear`, `MeanPool`):

#### Text Classifier (sequence → single label)

```csharp
var classifier = new TextClassifierModel<float>(
    vocabSize: 5000, embeddingDim: 64, hiddenDim: 128, numClasses: 3, maxSeqLen: 50);

// Training
using (GradientUtils.Grad())
{
    var logits = classifier.Forward(inputTokens);
    var loss = new CrossEntropyLoss<float>().Forward(logits, labels);
    loss.Backward();
    optimizer.Step();
    classifier.ZeroGrad();
}

// Inference
int[] predictedClasses = classifier.Predict(tokenIds);
```

#### Token Classifier (sequence → per-token labels)

```csharp
var tokenClassifier = new TokenClassifierModel<float>(
    vocabSize: 5000, embeddingDim: 64, hiddenDim: 128, numClasses: 9, maxSeqLen: 50);

// Forward: [batchSize, maxSeqLen] → [batchSize * maxSeqLen, numClasses]
var logits = tokenClassifier.Forward(inputTokens);

// Inference — returns one label per token position
int[] tokenLabels = tokenClassifier.Predict(tokenIds);
```

### Tokenization

`TextTokenizer` builds a word-level vocabulary from documents and encodes/decodes text:

```csharp
var tokenizer = TextTokenizer.FromDocuments(
    trainingDocuments, maxVocabSize: 10000, minFreq: 2);

int[] tokenIds = tokenizer.Encode("hello world");
string text = tokenizer.Decode(tokenIds);

// Special tokens
Console.WriteLine(tokenizer.PadToken);  // <PAD> index
Console.WriteLine(tokenizer.VocabSize);

// Serialize/deserialize
string json = tokenizer.ToJson();
var restored = TextTokenizer.FromJson(json);
```

### Sampling

`Sampler` provides temperature-scaled categorical sampling with optional top-K filtering:

```csharp
var sampler = new Sampler<float>(seed: 42);

// logits: [vocabSize] raw model output
int nextToken = sampler.Sample(logits, temperature: 0.8, topK: 50);
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
var parquetOptions = ParquetWriteOptions.Default.With(
    compression: ParquetCompression.Snappy,
    rowGroupSize: 10000,
    validateSchema: true);

frame.ToParquet("employees_custom.parquet", parquetOptions);

// Batch operations
var frames = new[] { frame1, frame2, frame3 };
NivaraFrameExtensions.WriteParquetBatch("batch.parquet", frames, parquetOptions);
```

### Configuration and Performance Tuning

Nivara's performance tuning is built into the core APIs:

```csharp
// Queries are lazy and optimized automatically (predicate pushdown,
// operation fusion, projection pushdown)
var query = frame.Query<Person>()
    .Where(p => p.Age > 30)
    .Select(p => new { p.Name, p.Salary });

// Inspect the optimized plan
Console.WriteLine(query.ExplainPlan());

// Execute
var result = query.Collect();
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
