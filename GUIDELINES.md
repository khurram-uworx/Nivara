# Nivara Development Guidelines

- Please go through the [CONTRIBUTING.md](CONTRIBUTING.md) first

This document captures lessons learned, architectural decisions, and non-obvious gotchas discovered during Nivara development. It serves as living knowledge for LLMs and maintainers to avoid repeating mistakes and understand the rationale behind design decisions.

## Architecture Decisions & Rationale

### Storage Strategy Evolution
**Decision**: Started with MemoryStorage for all types, added TensorStorage optimization later
**Rationale**: Generic constraints on static methods in generic classes are problematic (CS0080). Runtime type checking with factory pattern proved more flexible than compile-time constraints.

### Vectorization Approach
**Decision**: Use `ColumnStorageFactory.IsVectorizable<T>()` for type checking, not `storage.IsVectorizable`
**Rationale**: Storage selection and arithmetic vectorization have different requirements. Factory method provides consistent type checking across the codebase.

### Null Handling Strategy
**Decision**: Explicit null masks rather than NaN-based semantics
**Rationale**: More predictable behavior across different data types. NaN only works for floating-point types.

### Comparison Operations Design
**Decision**: All comparisons return `NivaraColumn<bool>` with null propagation
**Rationale**: Null compared to anything yields null. Consistent with SQL semantics and prevents unexpected boolean results.

### Series Indexing Strategy
**Decision**: NivaraSeries<T> has separate indexers for `this[int position]` and `this[object label]`
**Rationale**: Provides both position-based and label-based access. Integer literals get routed to position indexer, while boxed integers and other objects go to label indexer.

## Critical Implementation Gotchas

### ReadOnlyMemory<T>? Null Detection
**Problem**: Empty ReadOnlyMemory<T> still has HasValue = true
**Solution**: Always check both HasValue AND Length > 0
```csharp
// WRONG - empty ReadOnlyMemory still has HasValue = true
public bool HasNulls => _nullMask.HasValue;

// CORRECT - check both HasValue and Length
public bool HasNulls => _nullMask.HasValue && _nullMask.Value.Length > 0;
```

### Slicing Empty Collections
**Problem**: Slicing empty ReadOnlyMemory throws ArgumentOutOfRangeException
**Solution**: Always check length before slicing
```csharp
// WRONG - will fail if null mask is empty
if (_nullMask.HasValue)
    slicedNullMask = _nullMask.Value.Slice(start, length);

// CORRECT - check length before slicing  
if (_nullMask.HasValue && _nullMask.Value.Length > 0)
    slicedNullMask = _nullMask.Value.Slice(start, length);
```

### Generic Constraints Limitations
**Problem**: Cannot add `where T : struct` to static methods in generic classes (CS0080)
**Solution**: Use runtime type checking instead of compile-time constraints
```csharp
// WRONG - compiler error CS0080
public static NivaraColumn<T> CreateFromNullable(T?[] values) where T : struct

// CORRECT - use runtime checking
public static NivaraColumn<T> CreateFromNullable(T?[] values)
{
    if (!typeof(T).IsValueType)
        throw new InvalidOperationException("Method only supports value types");
}
```

### Nullable Generics Without Constraints
**Problem**: Cannot use `T?` without `where T : struct` constraint
**Solution**: Use runtime type handling or avoid nullable generic parameters
```csharp
// WRONG - compiler error without struct constraint
public static void Method<T>(T? value) { }

// CORRECT - use runtime type checking
public static void Method(Array nullableValues) { /* runtime type checking */ }
```

### Series Indexer Ambiguity
**Problem**: Integer literals can be ambiguous between position and label access in series
**Solution**: Use explicit casting or GetByLabel method for clarity
```csharp
// AMBIGUOUS - could be position 42 or label 42
var value = series[42];

// CLEAR - explicitly position-based access
var value = series[42]; // int literal goes to position indexer

// CLEAR - explicitly label-based access  
var value = series[(object)42]; // boxed int goes to label indexer
var value = series.GetByLabel(42); // explicit method call
```

### Diagnostic Information Architecture
**Decision**: Separate diagnostic classes for column metadata and operation tracking
**Rationale**: Column diagnostics provide static information about storage and performance characteristics, while operation diagnostics track runtime kernel selection and performance metrics.

**Pattern**: Use `ColumnDiagnostics` for storage analysis and `DiagnosticsTracker` for operation monitoring
```csharp
// Column-level diagnostics
var diagnostics = column.Diagnostics;
var storageType = diagnostics.StorageType;
var recommendedKernel = diagnostics.RecommendedKernel;

// Operation-level diagnostics
DiagnosticsTracker.IsEnabled = true;
var result = column1.Add(column2); // Automatically tracked
var operations = DiagnosticsTracker.GetRecordedOperations();
```

### Kernel Selection Logic
**Decision**: Centralized kernel selection in `DetermineKernelType()` method
**Rationale**: Consistent logic across all operations. Considers vectorizability, hardware acceleration, and data size thresholds.

**Implementation**: Check vectorizability → hardware acceleration → size threshold
```csharp
private KernelType DetermineKernelType()
{
    if (!storage.IsVectorizable) return KernelType.Scalar;
    if (!Vector.IsHardwareAccelerated) return KernelType.Scalar;
    if (Length < vectorSize * 4) return KernelType.Scalar; // Overhead threshold
    return KernelType.Vectorized;
}
```

### MemoryMarshal with Unconstrained Generics
**Problem**: MemoryMarshal.Cast requires unmanaged constraint but generic T doesn't have it
**Solution**: Use runtime type checking and safe conversion
```csharp
// WRONG - compiler error CS0453
var converted = MemoryMarshal.Cast<T, int>(values);

// CORRECT - use runtime type checking
if (typeof(T) == typeof(int))
{
    var intValues = values.ToArray();
    var intStorage = new TensorStorage<int>((ReadOnlySpan<int>)(object)intValues.AsSpan());
    return (IColumnStorage<T>)(object)intStorage;
}
```

## Testing Lessons Learned

### TestCase Limitations with Nulls
**Problem**: Cannot use null values in TestCase attribute arrays
**Solution**: Use regular Test methods with inline test data
```csharp
// WRONG - compiler error with null in array
[TestCase(new string[] { "a", null, "c" })]

// CORRECT - use regular test with inline data
[Test]
public void TestMethod()
{
    var testCases = new[] { new string[] { "a", null!, "c" } };
}
```

### TestCase Attribute Array Creation Limitations
**Problem**: TestCase attributes require constant expressions, cannot use `new string[] { ... }` syntax
**Error**: CS0182 - An attribute argument must be a constant expression, typeof expression or array creation expression
**Solution**: Convert parameterized tests to regular Test methods with inline test data
```csharp
// WRONG - compiler error CS0182
[TestCase(new string[] { "apple", "banana", "cherry" })]
[TestCase(new string[] { "hello", "world" })]
public void TestMethod(string[] values) { }

// CORRECT - use regular test with inline test cases
[Test]
public void TestMethod()
{
    var testCases = new[]
    {
        new string[] { "apple", "banana", "cherry" },
        new string[] { "hello", "world" }
    };
    
    foreach (var values in testCases)
    {
        // Test logic here
    }
}
```

### Complex Anonymous Types in Tests
**Problem**: Compiler cannot infer type for complex anonymous type arrays
**Solution**: Use explicit typing or separate focused tests
```csharp
// WRONG - compiler error CS0826
var testCases = new[] {
    new { Type = typeof(int), Values = new int[] { 1, 2, 3 } },
    new { Type = typeof(float), Values = new float[] { 1.0f, 2.0f } }
};

// CORRECT - separate focused tests per type
[Test] public void TestInt() { /* test int specific behavior */ }
[Test] public void TestFloat() { /* test float specific behavior */ }
```

### Reflection with Span Parameters
**Problem**: Cannot pass Span<T> as object parameter to reflection calls
**Solution**: Convert to array first
```csharp
// WRONG - compiler error CS0030
var result = method.Invoke(null, new object[] { spanValue });

// CORRECT - convert to array first
var result = method.Invoke(null, new object[] { spanValue.ToArray() });
```

### ReadOnlySpan<T>? Nullable Issues
**Problem**: Cannot make ReadOnlySpan<T> nullable (CS9244)
**Solution**: Use array parameters instead of nullable spans
```csharp
// WRONG - compiler error CS9244
public static void Method(ReadOnlySpan<T>? values) { }

// CORRECT - use nullable array parameter
public static void Method(T[]? values) { }
```

### Nullable Generic Method Constraints
**Problem**: Cannot add constraints to static methods in generic classes (CS0080)
**Error**: CS0080 - Constraints are not allowed on non-generic declarations
**Solution**: Use runtime type checking with Array parameter and manual processing
```csharp
// WRONG - compiler error CS0080
public static NivaraColumn<T> CreateFromNullable(T?[] values) where T : struct

// CORRECT - use Array parameter with runtime type checking
public static NivaraColumn<T> CreateFromNullable(Array values)
{
    if (!typeof(T).IsValueType)
        throw new InvalidOperationException("Method only supports value types");
    
    // Validate array element type
    var expectedNullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
    var actualElementType = values.GetType().GetElementType();
    if (actualElementType != expectedNullableType)
        throw new ArgumentException($"Array element type must be {expectedNullableType.Name}");
    
    // Process manually using GetValue()
    for (int i = 0; i < values.Length; i++)
    {
        var value = values.GetValue(i);
        // Handle null and non-null values...
    }
}
```

### Over-Engineering Generic Solutions
**Problem**: Attempting complex pattern matching or reflection when simple solutions exist
**Lesson**: Always check existing patterns in codebase first. Manual processing with GetValue() is often simpler than complex generic solutions.
**Solution**: Use the simplest approach that works - manual array processing with runtime type checking

## Type System Discoveries

### Vectorizable Types
**Current list**: `int`, `float`, `double`, `bool` (confirmed working with TensorPrimitives)
**Note**: `bool` is vectorizable despite being non-numeric. Always include in vectorizable type checks.

### INumber<T> Limitations
**Problem**: Not all numeric types implement INumber<T> consistently
**Solution**: Maintain explicit lists of supported types rather than relying on interface constraints

### Comparison Type Support
**All types**: Support equality comparison
**IComparable<T> types only**: Support ordering comparisons (>, <, >=, <=)
**Pattern**: Use `IsComparableType()` helper to check ordering support at runtime

## Arithmetic Operations Insights

### Dynamic Dispatch Strategy
**Decision**: Use dynamic dispatch with runtime type checking for arithmetic operations
**Rationale**: Simpler than complex generic constraints. Allows clean separation of concerns.

### Operator Overload Testing
**Gotcha**: Can't use operators directly in Assert.Throws
**Solution**: Wrap in lambda expressions
```csharp
// WRONG
Assert.Throws<ArgumentException>(() => column1 + column2);

// CORRECT  
Assert.Throws<ArgumentException>(() => { var result = column1 + column2; });
```

### Null Propagation in Operations
**Rule**: Any operation involving null should propagate null to the result
**Implementation**: Merge null masks using OR operation, then apply to result

## Performance Patterns That Work

### Vectorization Detection
- Check `Vector.IsHardwareAccelerated` for SIMD availability
- Use `TensorPrimitives` methods when available
- Always provide scalar fallbacks

### Memory Efficiency
- Use `ReadOnlySpan<T>` and `ReadOnlyMemory<T>` for zero-copy operations
- Implement proper IDisposable patterns for tensor resources
- Prefer slicing over copying for data views

## What We Tried That Didn't Work

### Compile-Time Generic Constraints Everywhere
**Tried**: Using `where T : struct, INumber<T>` on all arithmetic methods
**Failed**: Static methods in generic classes can't have additional constraints (CS0080)
**Learned**: Runtime type checking is more flexible for complex generic scenarios

### Reflection-Based Generic Testing
**Tried**: Single parameterized test method using reflection to test all types
**Failed**: Span<T> parameters don't work with reflection, complex type inference issues
**Learned**: Separate focused tests per type are clearer and more maintainable

### NaN-Based Null Semantics
**Tried**: Using NaN values to represent nulls in floating-point columns
**Failed**: Doesn't work for integer types, inconsistent behavior across types
**Learned**: Explicit null masks provide predictable behavior across all data types

---

**Note**: This document captures living knowledge and should be updated as new insights are discovered. Focus on "why" decisions were made and "what" didn't work, not "how" to implement (that belongs in CONTRIBUTING.md).

## Query Engine Foundation Implementation

### Interface Design Decisions
**Decision**: Created both generic `IColumn<T>` and non-generic `IColumn` interfaces
**Rationale**: Query engine needs to work with columns of unknown types at runtime. Non-generic interface provides type-erased access while maintaining type safety through generic interface for user code.

**Pattern**: Non-generic interface includes `Type ElementType` and `object? GetValue(int index)` for runtime type handling
```csharp
// Non-generic for query engine
public interface IColumn : IDisposable
{
    Type ElementType { get; }
    object? GetValue(int index);
}

// Generic for user code
public interface IColumn<T> : IColumn
{
    T this[int index] { get; }
}
```

### Schema System Architecture
**Decision**: Immutable schema with transformation methods (WithColumn, WithoutColumn, SelectColumns)
**Rationale**: Immutable design prevents accidental schema corruption during query transformations. Transformation methods enable functional-style schema evolution.

**Pattern**: Case-insensitive column name handling throughout schema system
```csharp
// Use StringComparer.OrdinalIgnoreCase for all column name dictionaries
var typeDict = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
```

### Exception Hierarchy Design
**Decision**: Specific exception types for different failure modes (ColumnNotFoundException, ColumnTypeMismatchException, etc.)
**Rationale**: Enables precise error handling and provides context-specific error messages. Helps users understand exactly what went wrong.

**Pattern**: Include available options in error messages when possible
```csharp
throw new ColumnNotFoundException(columnName, availableColumns);
// Results in: "Column 'foo' not found. Available columns: bar, baz, qux"
```

### Column Expression System
**Decision**: Operator overloading with both expression-to-expression and expression-to-scalar operations
**Rationale**: Provides natural syntax for query building while maintaining type safety. Supports both `Col("A") + Col("B")` and `Col("A") + 5` patterns.

**Gotcha**: Required `Equals` and `GetHashCode` overrides when implementing equality operators
**Solution**: Override both methods to avoid compiler warnings and maintain consistency
```csharp
public override bool Equals(object? obj) => ReferenceEquals(this, obj);
public override int GetHashCode() => base.GetHashCode();
```

### Package Dependencies Strategy
**Decision**: Added CsvHelper for CSV processing capabilities in Nivara.Extensions
**Rationale**: Mature, well-tested library with good performance characteristics. Avoids reinventing CSV parsing which has many edge cases. Kept in Extensions project to maintain core library independence from third-party dependencies.

**Pattern**: Third-party dependencies go in Nivara.Extensions, core interfaces made public when needed
```csharp
// Core library defines public interfaces
public interface IQuerySource { ... }

// Extensions library implements with third-party dependencies
internal sealed class CsvLazySource : IQuerySource { ... }
```

**Version**: CsvHelper 33.0.1 - latest stable version compatible with .NET 10.0

### Query Plan Infrastructure
**Decision**: Immutable QueryPlan with schema transformation validation
**Rationale**: Immutable design prevents plan corruption. Schema validation catches errors early in query building phase rather than at execution time.

**Pattern**: Compute result schema during plan construction
```csharp
private Schema ComputeResultSchema()
{
    var schema = Source.Schema;
    foreach (var operation in Operations)
        schema = operation.TransformSchema(schema);
    return schema;
}
```

### Extensions Architecture Pattern
**Decision**: Third-party dependencies and implementations go in Nivara.Extensions project
**Rationale**: Keeps core library lightweight and free of external dependencies. Allows users to opt-in to specific functionality without bloating the core package.

**Pattern**: Core defines public interfaces, Extensions implements with third-party libraries
```csharp
// Core: src/Nivara/IQuerySource.cs
public interface IQuerySource { ... }

// Extensions: src/Nivara.Extensions/IO/CsvDataSource.cs  
internal sealed class CsvLazySource : IQuerySource
{
    // Uses CsvHelper internally
}

// Extensions: src/Nivara.Extensions/IO/CsvExtensions.cs
public static class CsvExtensions
{
    public static IQuerySource ScanCsv(string filePath) { ... }
}
```

**Benefits**: 
- Core library stays dependency-free
- Extensions can use specialized libraries (CsvHelper, Parquet.Net, etc.)
- Users only pay for what they use
- Easier to maintain and test core functionality

### Namespace and Folder Organization
**Decision**: Use consistent `Nivara` namespace across all projects, organize IO functionality in dedicated folders
**Rationale**: C# namespace flexibility allows clean organization without Java-style rigid package structure. Dedicated IO folders improve maintainability and discoverability.

**Pattern**: Organize by functionality, not by project boundaries
```csharp
// Core JSON support (no third-party dependencies)
namespace Nivara.IO;
// File: src/Nivara/IO/JsonDataSource.cs

// Extensions CSV support (uses CsvHelper)
namespace Nivara.IO; 
// File: src/Nivara.Extensions/IO/CsvDataSource.cs
// File: src/Nivara.Extensions/IO/CsvExtensions.cs
```

**Structure**:
- `src/Nivara/IO/` - Core IO functionality using only .NET built-ins (JSON, etc.)
- `src/Nivara.Extensions/IO/` - IO functionality requiring third-party libraries (CSV, Parquet, etc.)
- Consistent `Nivara.IO` namespace across both projects
- Static factory classes: `Csv.Scan()`, `Json.Scan()`, etc.

## Schema System Implementation

### Schema Class Design Decisions
**Decision**: Immutable Schema class with transformation methods (WithColumn, WithoutColumn, SelectColumns)
**Rationale**: Immutable design prevents accidental schema corruption during query transformations. Transformation methods enable functional-style schema evolution without side effects.

**Pattern**: Case-insensitive column name handling throughout schema system
```csharp
// Use StringComparer.OrdinalIgnoreCase for all column name dictionaries
var typeDict = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
var metadataDict = new Dictionary<string, ColumnMetadata>(StringComparer.OrdinalIgnoreCase);
```

### Exception Hierarchy for Schema Operations
**Decision**: Specific exception types for different failure modes (ColumnNotFoundException, ColumnTypeMismatchException, SchemaValidationException)
**Rationale**: Enables precise error handling and provides context-specific error messages. Helps users understand exactly what went wrong.

**Pattern**: Include available options in error messages when possible
```csharp
throw new ColumnNotFoundException(columnName, availableColumns);
// Results in: "Column 'foo' not found. Available columns: bar, baz, qux"
```

### Schema Compatibility Validation
**Decision**: Separate exact match and compatible type checking in IsCompatibleWith method
**Rationale**: Query engine needs both strict validation (for type safety) and flexible validation (for numeric type coercion). Single method with parameter provides both capabilities.

**Pattern**: Numeric type compatibility matrix for operations
```csharp
private static bool AreTypesCompatible(Type type1, Type type2)
{
    if (type1 == type2) return true;
    
    // Numeric type compatibility
    var numericTypes = new[]
    {
        typeof(byte), typeof(sbyte), typeof(short), typeof(ushort),
        typeof(int), typeof(uint), typeof(long), typeof(ulong),
        typeof(float), typeof(double), typeof(decimal)
    };
    
    return numericTypes.Contains(type1) && numericTypes.Contains(type2);
}
```

### ColumnMetadata Design
**Decision**: Immutable ColumnMetadata with With() method for updates
**Rationale**: Consistent with Schema immutability pattern. Provides type-safe way to update specific properties without affecting others.

**Pattern**: Builder-style With() method for partial updates
```csharp
var updated = original.With(isNullable: false, defaultValue: "New Default");
// Only specified properties are updated, others remain unchanged
```

### Schema Testing Strategy
**Decision**: Comprehensive unit tests covering all transformation methods and edge cases
**Rationale**: Schema is foundational to query engine correctness. Thorough testing prevents runtime errors in complex query scenarios.

**Coverage**: 30 test cases covering:
- Constructor validation (null checks, duplicate names, invalid types)
- Column access methods (case-insensitive, error handling)
- Transformation methods (WithColumn, WithoutColumn, SelectColumns)
- Compatibility validation (exact and flexible matching)
- Metadata operations and immutability
- Edge cases (empty schemas, single columns, case sensitivity)

### Schema Validation Gotchas
**Problem**: Case sensitivity can cause subtle bugs in column lookups
**Solution**: Always use StringComparer.OrdinalIgnoreCase for column name dictionaries and comparisons
```csharp
// WRONG - case sensitive comparison
if (columnName == storedName) { ... }

// CORRECT - case insensitive comparison
if (string.Equals(columnName, storedName, StringComparison.OrdinalIgnoreCase)) { ... }
```

**Problem**: Schema transformation methods can accidentally mutate original schema
**Solution**: Always create new instances, never modify existing dictionaries
```csharp
// WRONG - modifies original dictionary
columnTypes[newName] = newType;

// CORRECT - creates new dictionary
var newTypeDict = new Dictionary<string, Type>(columnTypes, StringComparer.OrdinalIgnoreCase);
newTypeDict[newName] = newType;
```
## Project Organization and Namespace Structure

### Folder and Namespace Organization
**Decision**: Organize code by functional area using dedicated folders and namespaces
**Rationale**: Improves discoverability, maintainability, and follows .NET best practices. Makes it easier for LLMs and developers to quickly locate related functionality.

### Core Project Structure (`src/Nivara/`)

```
src/Nivara/
├── Diagnostics/           # Performance analysis and diagnostic tools
│   ├── ColumnDiagnostics.cs      # Column-level performance diagnostics
│   └── OperationDiagnostics.cs   # Operation tracking and kernel selection
├── Exceptions/            # Custom exception hierarchy
│   └── QueryEngineExceptions.cs  # All query engine exceptions
├── Expressions/           # Query expression system
│   └── ColumnExpression.cs       # Column expressions and operators
├── IO/                    # Built-in data source functionality
│   ├── JsonDataSource.cs         # JSON reading (no third-party deps)
│   └── JsonExtensions.cs         # JSON factory methods
├── Memory/                # Memory-based storage implementations
│   ├── MemoryStorage.cs           # Non-vectorizable type storage
│   └── NullableStorageHelper.cs  # Nullable value type utilities
├── Tensors/               # Tensor-based storage implementations
│   └── TensorStorage.cs           # Vectorizable type storage
└── [Root Files]           # Core interfaces and main classes
    ├── IColumn.cs                 # Column interfaces
    ├── IColumnStorage.cs          # Storage abstraction
    ├── IFrame.cs                  # Frame interface
    ├── NivaraColumn.cs           # Main column implementation
    ├── NivaraSeries.cs           # Series implementation
    ├── Schema.cs                  # Schema management
    └── [Query Engine Files]       # Query planning and execution
```

### Namespace Mapping

| Folder | Namespace | Purpose |
|--------|-----------|---------|
| `Diagnostics/` | `Nivara.Diagnostics` | Performance analysis, kernel selection, operation tracking |
| `Exceptions/` | `Nivara.Exceptions` | Custom exception types with context-specific error messages |
| `Expressions/` | `Nivara.Expressions` | Column expressions, operators, query building |
| `IO/` | `Nivara.IO` | Data source abstractions and built-in implementations |
| `Memory/` | `Nivara.Memory` | Memory-based storage for non-vectorizable types |
| `Tensors/` | `Nivara.Tensors` | Tensor-based storage for vectorizable types |
| `[Root]` | `Nivara` | Core interfaces, main classes, schema management |

### Test Project Structure (`tests/Nivara.Tests/`)

Tests follow the same organizational structure as the source code:

```
tests/Nivara.Tests/
├── Diagnostics/
│   └── DiagnosticsTests.cs       # Tests for diagnostic functionality
├── Memory/
│   └── MemoryStorageTests.cs     # Tests for memory storage
├── Tensors/
│   └── TensorStorageTests.cs     # Tests for tensor storage
└── [Root Files]                  # Tests for core functionality
    ├── SchemaTests.cs             # Schema system tests
    ├── NivaraColumnTests.cs      # Column implementation tests
    ├── NivaraSeriesTests.cs      # Series implementation tests
    └── ColumnStorageFactoryTests.cs
```

### Using Statement Patterns

**Pattern**: Always include necessary using statements for cross-namespace references
```csharp
// In core files that use moved classes
using Nivara.Diagnostics;    // For ColumnDiagnostics, StorageType, etc.
using Nivara.Exceptions;     // For custom exceptions
using Nivara.Memory;         // For MemoryStorage
using Nivara.Tensors;        // For TensorStorage

// In moved files that reference core classes
using Nivara;                // For core interfaces and classes
```

### Extensions Project Organization

The Extensions project follows the same pattern but focuses on third-party integrations:

```
src/Nivara.Extensions/
└── IO/                    # Third-party data source functionality
    ├── CsvDataSource.cs           # CSV reading (uses CsvHelper)
    └── CsvExtensions.cs           # CSV factory methods
```

### Quick Reference for LLMs

When working with Nivara code:

1. **Diagnostics**: Look in `Nivara.Diagnostics` namespace
   - `ColumnDiagnostics` - Column performance info
   - `DiagnosticsTracker` - Operation tracking
   - `StorageType`, `KernelType` enums

2. **Exceptions**: Look in `Nivara.Exceptions` namespace
   - `ColumnNotFoundException` - Missing column errors
   - `ColumnTypeMismatchException` - Type mismatch errors
   - `SchemaValidationException` - Schema validation errors
   - `DataSourceException` - Data source errors
   - `QueryExecutionException` - Query execution errors

3. **Expressions**: Look in `Nivara.Expressions` namespace
   - `ColumnExpression` - Base expression class
   - `ColumnReference` - Column references
   - `BinaryExpression`, `ComparisonExpression` - Operators

4. **Storage**: Look in appropriate namespace
   - `Nivara.Memory.MemoryStorage<T>` - Non-vectorizable types
   - `Nivara.Tensors.TensorStorage<T>` - Vectorizable types
   - `Nivara.Memory.NullableStorageHelper` - Nullable utilities

5. **Core**: Look in `Nivara` namespace
   - `NivaraColumn<T>`, `NivaraSeries<T>` - Main classes
   - `Schema`, `ColumnMetadata` - Schema management
   - `IColumn<T>`, `IFrame` - Core interfaces

### Benefits of This Organization

- **Discoverability**: Related functionality is grouped together
- **Maintainability**: Changes to specific areas are isolated
- **Testing**: Test structure mirrors source structure
- **Namespace Clarity**: Clear separation of concerns
- **IDE Support**: Better IntelliSense and navigation
- **Documentation**: Easier to document and understand architecture

## NivaraFrame Implementation

### Frame Design Decisions
**Decision**: Immutable NivaraFrame with transformation methods (WithColumn, WithoutColumn, SelectColumns)
**Rationale**: Consistent with Schema immutability pattern. Prevents accidental frame corruption during transformations. Enables functional-style frame evolution.

**Pattern**: Case-insensitive column name handling throughout frame operations
```csharp
// Use StringComparer.OrdinalIgnoreCase for all column name dictionaries
var columnDict = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
```

### Column Length Validation Strategy
**Decision**: Validate all columns have the same length during frame creation
**Rationale**: Prevents runtime errors and ensures data integrity. All columns in a frame must represent the same number of rows.

**Pattern**: Comprehensive validation with detailed error messages
```csharp
if (column.Length != expectedLength.Value)
{
    var existingColumns = string.Join(", ", names.Where(n => !string.Equals(n, name, StringComparison.OrdinalIgnoreCase)));
    throw new ArgumentException(
        $"Column length mismatch: Column '{name}' has length {column.Length}, but expected {expectedLength.Value} " +
        $"to match existing columns [{existingColumns}]. All columns in a frame must have the same length.",
        nameof(namedColumns));
}
```

### Type Safety in Frame Operations
**Decision**: Use both generic `GetColumn<T>()` and non-generic `GetColumn()` methods
**Rationale**: Generic method provides compile-time type safety for user code. Non-generic method enables runtime type handling for query engine operations.

**Pattern**: Clear error messages for type mismatches
```csharp
if (column is not NivaraColumn<T> typedColumn)
{
    var actualType = column.ElementType;
    var expectedType = typeof(T);
    throw new ColumnTypeMismatchException(name, expectedType, actualType);
}
```

### Frame Transformation Immutability
**Decision**: All frame transformation methods return new instances
**Rationale**: Prevents accidental mutation of existing frames. Enables safe sharing of frame instances across different parts of the application.

**Pattern**: Functional-style transformations
```csharp
// WithColumn creates new frame with additional column
public NivaraFrame WithColumn(string name, IColumn column)
{
    // Validation...
    var newColumns = columns.Concat(new[] { new KeyValuePair<string, IColumn>(name, column) });
    var namedColumns = newColumns.Select(kvp => (kvp.Key, kvp.Value));
    return new NivaraFrame(namedColumns);
}
```

### Frame Testing Gotchas
**Problem**: Compiler cannot infer tuple types when one element is null
**Solution**: Use explicit tuple type declarations
```csharp
// WRONG - compiler error CS0826
var columns = new[] { (null!, (IColumn)col) };

// CORRECT - explicit tuple type
var columns = new (string, IColumn)[] { (null!, col) };
```

**Problem**: NivaraColumn<T> doesn't implicitly convert to IColumn in array literals
**Solution**: Cast to IColumn when needed or use explicit tuple types
```csharp
// WRONG - type inference issues
var columns = new[] { ("Test", col1), ("test", col2) };

// CORRECT - explicit casting or tuple types
var columns = new[] { ("Test", (IColumn)col1), ("test", (IColumn)col2) };
```

## Column Expression System Implementation

### Expression Hierarchy Design Decisions
**Decision**: Created comprehensive ColumnExpression hierarchy with operator overloading for both expression-to-expression and expression-to-scalar operations
**Rationale**: Provides natural syntax for query building while maintaining type safety. Supports both `Col("A") + Col("B")` and `Col("A") + 5` patterns through different expression types.

**Pattern**: Separate expression types for different operation categories
```csharp
// Base abstract class with operator overloads
public abstract class ColumnExpression { /* operators */ }

// Specific expression types
public sealed class ColumnReference : ColumnExpression { /* column references */ }
public sealed class BinaryExpression : ColumnExpression { /* column-to-column operations */ }
public sealed class ScalarExpression : ColumnExpression { /* column-to-scalar operations */ }
public sealed class ComparisonExpression : ColumnExpression { /* comparison operations */ }
public sealed class LiteralExpression : ColumnExpression { /* literal values */ }
```

### Operator Overloading Implementation
**Decision**: Comprehensive operator overloading for arithmetic (+, -, *, /) and comparison (>, <, >=, <=, ==, !=) operations
**Rationale**: Enables natural mathematical syntax in query expressions. Separate overloads for expression-to-expression and expression-to-scalar operations provide flexibility.

**Gotcha**: Required `Equals` and `GetHashCode` overrides when implementing equality operators
**Solution**: Override both methods to avoid compiler warnings and maintain consistency
```csharp
public override bool Equals(object? obj) => ReferenceEquals(this, obj);
public override int GetHashCode() => base.GetHashCode();
```

### Static Factory Class Pattern
**Decision**: Use `ColumnExpressions` static class instead of global functions for creating column expressions
**Rationale**: More OOP-compliant than global functions. Provides clear namespace organization and follows .NET conventions.

**Pattern**: Static factory methods in dedicated class
```csharp
public static class ColumnExpressions
{
    public static ColumnExpression Col(string name) => new ColumnReference(name);
    public static ColumnExpression Col<T>(string name) => new ColumnReference(name, typeof(T));
    public static ColumnExpression Lit(object? value) => new LiteralExpression(value);
}
```

**Anti-Pattern**: Global functions in root namespace
```csharp
// WRONG - not OOP compliant
public static ColumnExpression Col(string name) { ... } // in global namespace

// CORRECT - use static factory class
ColumnExpressions.Col("Name") // clear, organized, OOP-compliant
```

### Expression Validation Strategy
**Decision**: Schema validation with detailed error messages including available alternatives
**Rationale**: Helps users quickly identify and fix column reference errors. Provides context about what columns are actually available.

**Pattern**: Enhanced error messages with suggestions
```csharp
if (!schema.HasColumn(ColumnName))
{
    var availableColumns = string.Join(", ", schema.ColumnNames);
    throw new SchemaValidationException($"Column '{ColumnName}' not found in schema. Available columns: {availableColumns}");
}
```

### Type System Integration
**Decision**: Automatic type promotion for binary operations with fallback to `object` type
**Rationale**: Provides reasonable type inference for numeric operations while handling edge cases gracefully.

**Pattern**: Numeric type promotion hierarchy
```csharp
private static Type DetermineResultType(Type leftType, Type rightType)
{
    if (leftType == rightType) return leftType;
    
    // Numeric type promotion (double > float > long > int > short > byte)
    var numericTypes = new[] { typeof(double), typeof(float), typeof(long), typeof(int), typeof(short), typeof(byte) };
    var leftIndex = Array.IndexOf(numericTypes, leftType);
    var rightIndex = Array.IndexOf(numericTypes, rightType);
    
    if (leftIndex >= 0 && rightIndex >= 0)
        return numericTypes[Math.Min(leftIndex, rightIndex)]; // Higher precision wins
    
    return typeof(object); // Fallback for non-numeric types
}
```

### Expression Composition Support
**Decision**: Full support for complex expression composition through operator chaining
**Rationale**: Enables building sophisticated query conditions like `(Col("Age") + 5) * Col("Salary") > 50000`. Each operation returns a new expression that can be further composed.

**Pattern**: Immutable expression trees
```csharp
// Each operation creates new expression node
var ageExpr = ColumnExpressions.Col("Age");           // ColumnReference
var agePlus5 = ageExpr + 5;                          // ScalarExpression
var salaryExpr = ColumnExpressions.Col("Salary");    // ColumnReference  
var product = agePlus5 * salaryExpr;                 // BinaryExpression
var condition = product > 50000;                     // ComparisonExpression
```

### Testing Strategy for Expression System
**Decision**: Comprehensive unit tests covering all operator combinations, type scenarios, and validation cases
**Rationale**: Expression system is foundational to query correctness. Thorough testing prevents runtime errors in complex query scenarios.

**Coverage**: 20 test cases covering:
- Expression creation and basic properties
- All arithmetic operators (binary and scalar)
- All comparison operators (binary and scalar)
- Expression composition and complex expressions
- Schema validation (valid and invalid cases)
- Type promotion and inference
- Error handling and edge cases

### Expression System Gotchas
**Problem**: Complex expression composition can create deeply nested expression trees
**Solution**: Each expression type implements proper `ToString()` methods for debugging and provides clear `Name` properties for display

**Problem**: Type inference for complex expressions can be ambiguous
**Solution**: Conservative type promotion with fallback to `object` type ensures operations don't fail unexpectedly

**Problem**: Schema validation needs to be performed at query execution time, not expression creation time
**Solution**: Expressions store validation logic but defer execution until `Validate(Schema)` is called during query planning

### Design Document Consistency
**Lesson**: Keep design documents synchronized with actual implementation class names and patterns
**Solution**: Regular review and update of design documents to match implemented code. Use actual class names (`ColumnExpression`, `Schema`) rather than conceptual names (`NivaraColumnExpression`, `NivaraSchema`)

**Pattern**: Design document should reflect actual implementation
```csharp
// Design document should show actual class names
public abstract class ColumnExpression { ... }  // Not NivaraColumnExpression
public sealed class Schema { ... }              // Not NivaraSchema
```

## Query Engine Implementation

### QueryFrame and Query Operations Design
**Decision**: Created QueryFrame class with fluent API for lazy query construction and separate operation classes (FilterOperation, SelectOperation, GroupByOperation)
**Rationale**: Provides natural LINQ-like syntax while maintaining clear separation between query building and execution. Each operation is responsible for schema transformation and execution logic.

**Pattern**: Immutable query building with method chaining
```csharp
var result = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name", "Salary")
    .GroupBy("Department")
    .Collect(); // Execution barrier
```

### Dynamic Dispatch for Column Creation
**Decision**: Use type switching with dynamic dispatch instead of reflection for creating columns from unknown types
**Rationale**: Avoids reflection overhead and complexity while maintaining type safety. Follows the same pattern as arithmetic operations in NivaraColumn.

**Pattern**: Type switching for column creation
```csharp
// WRONG - using reflection
var columnType = typeof(NivaraColumn<>).MakeGenericType(elementType);
var createMethod = columnType.GetMethod("Create", new[] { elementType.MakeArrayType() });
var result = createMethod.Invoke(null, new object[] { array });

// CORRECT - using dynamic dispatch
return elementType switch
{
    Type t when t == typeof(int) => CreateColumnTyped<int>(values),
    Type t when t == typeof(string) => CreateColumnTyped<string>(values),
    _ => CreateColumnGeneric(values)
};

private static IColumn CreateColumnTyped<T>(List<object> values)
{
    var array = values.Cast<T>().ToArray();
    return NivaraColumn<T>.Create(array);
}
```

### Expression Evaluation Architecture
**Decision**: Created ExpressionEvaluator class that handles all column expression types with dynamic dispatch for arithmetic and comparison operations
**Rationale**: Centralizes expression evaluation logic and provides consistent error handling. Uses the same dynamic dispatch pattern as NivaraColumn arithmetic operations.

**Pattern**: Expression evaluation with dynamic dispatch
```csharp
// Arithmetic operations use dynamic dispatch like NivaraColumn
private static object? AddValues(object? left, object? right)
{
    if (left == null || right == null) return null;
    
    return (left, right) switch
    {
        (int l, int r) => l + r,
        (double l, double r) => l + r,
        _ => Convert.ToDouble(left) + Convert.ToDouble(right)
    };
}
```

### Query Plan and Execution Design
**Decision**: Separate QueryPlan, QueryExecutor, and QueryOptimizer classes with clear responsibilities
**Rationale**: Enables future optimization passes and provides clear separation between query representation, execution, and optimization logic.

**Pattern**: Query execution pipeline
```csharp
// Query building (lazy)
var queryFrame = source.Filter(condition).Select(columns);

// Query execution (eager)
var plan = new QueryPlan(source, operations);
var executor = new QueryExecutor();
var result = executor.Execute(plan);
```

### Memory vs Lazy Query Sources
**Decision**: MemoryQuerySource for in-memory data (IsLazy = false) and separate lazy sources for file-based data
**Rationale**: Clear distinction between materialized data (frames) and unmaterialized data (files). Memory sources have no IO cost, while lazy sources defer IO until execution.

**Pattern**: Query source abstraction
```csharp
// Memory source (already materialized)
public bool IsLazy => false;
public IReadOnlyDictionary<string, IColumn> Execute() => columns;

// File source (lazy evaluation)
public bool IsLazy => true;
public IReadOnlyDictionary<string, IColumn> Execute() => ReadFromFile();
```

### Query Operation Interface Design
**Decision**: IQueryOperation interface with TransformSchema and Execute methods
**Rationale**: Enables schema validation during query building and consistent execution interface. Schema transformation allows early error detection.

**Pattern**: Operation interface implementation
```csharp
public Schema TransformSchema(Schema inputSchema)
{
    // Validate operation can be applied to input schema
    // Return transformed schema (may be same for Filter, different for Select)
}

public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
{
    // Apply operation to input columns and return result columns
}
```

### Testing Strategy for Query Engine
**Decision**: Comprehensive unit tests covering query building, execution, and error conditions
**Rationale**: Query engine is complex with many interaction points. Tests verify both lazy query building and eager execution work correctly.

**Coverage**: 13 test cases covering:
- Query plan building (Filter, Select, GroupBy)
- Method chaining and fluent API
- Query execution with Collect()
- Schema transformation validation
- Error handling (null conditions, empty operations)
- Diagnostic and optimization analysis

### Query Engine Gotchas
**Problem**: Method overload ambiguity between `Select(params ColumnExpression[])` and `Select(params string[])`
**Solution**: Provide both overloads but be explicit in tests to avoid compiler ambiguity
```csharp
// AMBIGUOUS in tests
queryFrame.Select(); // Compiler can't choose overload

// CLEAR in tests
queryFrame.Select(new string[0]); // Explicit array type
```

**Problem**: Reflection for column creation is slow and complex
**Solution**: Use type switching with dynamic dispatch following existing NivaraColumn patterns
```csharp
// Follow the same pattern as AddTensorPrimitive in NivaraColumn
destination[i] = (T)(object)((dynamic)x[i]! + (dynamic)y[i]!)!;
```

**Problem**: Expression evaluation needs to handle all column types dynamically
**Solution**: Use switch expressions on Type with fallback to generic object columns for unknown types