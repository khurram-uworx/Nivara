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

## Arrow/Parquet I/O Implementation

### Project Structure and Namespace Organization
**Decision**: Keep Arrow and Parquet implementations in Nivara.Extensions project, core I/O abstractions in Nivara project
**Rationale**: Maintains separation between core library (dependency-free) and extensions (third-party dependencies). Apache.Arrow and Parquet.Net packages are already referenced in Nivara.Extensions.

**Pattern**: Core abstractions in Nivara, implementations in Extensions
```csharp
// Core abstractions (Nivara project)
namespace Nivara.IO;
- NivaraIOException           # Base I/O exception
- UnsupportedTypeException    # Type conversion errors  
- DataCorruptionException     # Data integrity errors

// Implementations (Nivara.Extensions project)
namespace Nivara.IO;
- ArrowConversionOptions      # Arrow conversion configuration
- ParquetReadOptions          # Parquet reading configuration
- ParquetWriteOptions         # Parquet writing configuration
- ArrowInterop               # Arrow conversion implementation
- ParquetReader              # Parquet reading implementation
- ParquetWriter              # Parquet writing implementation
```

### Exception Hierarchy Design
**Decision**: Created comprehensive I/O exception hierarchy in core Nivara project
**Rationale**: Provides specific error types for different I/O failure modes while maintaining consistency with existing exception patterns. Core exceptions can be used by any I/O implementation.

**Pattern**: Hierarchical exceptions with context information
```csharp
// Base exception with file path and operation context
public class NivaraIOException : Exception
{
    public string? FilePath { get; init; }
    public string? OperationContext { get; init; }
}

// Specific exception types inherit from base
public sealed class UnsupportedTypeException : NivaraIOException
{
    public Type UnsupportedType { get; init; }
    public IReadOnlyList<string> SuggestedAlternatives { get; init; }
}
```

### Configuration Options Architecture
**Decision**: Separate configuration classes for different I/O operations (ArrowConversionOptions, ParquetReadOptions, ParquetWriteOptions)
**Rationale**: Provides fine-grained control over I/O behavior while maintaining clear separation of concerns. Each operation type has its own specific configuration needs.

**Pattern**: Options classes with sensible defaults and comprehensive documentation
```csharp
public class ArrowConversionOptions
{
    public bool UseZeroCopy { get; set; } = true;           // Performance optimization
    public TimeZoneInfo TimeZone { get; set; } = TimeZoneInfo.Utc;  // DateTime handling
    public bool ValidateTypes { get; set; } = true;        // Type safety
    public Encoding StringEncoding { get; set; } = Encoding.UTF8;   // Text encoding
}
```

### Namespace Structure for I/O Operations
**Decision**: Use `Nivara.IO` namespace for all Arrow/Parquet implementations
**Rationale**: Consistent with existing CSV implementation pattern. Keeps all I/O extensions in the same namespace while separating from core functionality.

**Structure**:
```
src/Nivara/IO/                    # Core I/O abstractions
├── IOExceptions.cs               # Exception hierarchy

src/Nivara.Extensions/IO/         # Third-party I/O implementations  
├── ArrowConversionOptions.cs     # Arrow configuration
├── ParquetReadOptions.cs         # Parquet read configuration
├── ParquetWriteOptions.cs        # Parquet write configuration
├── ArrowInterop.cs              # Arrow conversion logic
├── ParquetReader.cs             # Parquet reading logic
├── ParquetWriter.cs             # Parquet writing logic
├── CsvDataSource.cs             # Existing CSV implementation
└── CsvExtensions.cs             # Existing CSV extensions
```

### I/O Implementation Patterns
**Decision**: Static classes for stateless I/O operations with options pattern for configuration
**Rationale**: Follows existing CSV implementation pattern. Static methods are appropriate for stateless operations, while options classes provide flexible configuration.

**Pattern**: Static classes with options parameters
```csharp
public static class ArrowInterop
{
    public static Table ToArrowTable(NivaraFrame frame, ArrowConversionOptions? options = null);
    public static NivaraFrame FromArrowTable(Table arrowTable, ArrowConversionOptions? options = null);
}

public static class ParquetReader
{
    public static Task<NivaraFrame> ReadParquetAsync(string filePath, ParquetReadOptions? options = null);
    public static NivaraFrame ReadParquet(string filePath, ParquetReadOptions? options = null);
}
```

### Package Dependencies Strategy
**Decision**: Apache.Arrow and Parquet.Net packages are already referenced in Nivara.Extensions project
**Rationale**: Maintains core library independence from third-party dependencies. Extensions project can use specialized libraries for specific I/O formats.

**Current Dependencies** (already in Nivara.Extensions.csproj):
- Apache.Arrow 22.1.0 - Arrow format support
- Parquet.Net 5.4.0 - Parquet format support
- CsvHelper 33.0.1 - CSV format support (existing)

### Error Handling Strategy
**Decision**: Comprehensive error context with file paths, operation types, and suggested alternatives
**Rationale**: Helps users quickly identify and resolve I/O issues. Provides actionable error messages with context about what went wrong and how to fix it.

**Pattern**: Context-rich error messages
```csharp
// Include file path and operation context
throw new NivaraIOException("Failed to read Parquet file", filePath, "ParquetReader.ReadParquet");

// Include suggested alternatives for unsupported types
throw new UnsupportedTypeException(typeof(Guid), new[] { "string", "byte[]" });

// Include affected data ranges for corruption errors
throw new DataCorruptionException("Invalid data detected", filePath, "ParquetReader", 
    new[] { "Column1", "Column2" }, new Range(100, 200));
```

### Testing Strategy for I/O Operations
**Decision**: Unit tests for configuration classes and exception types, integration tests for actual I/O operations
**Rationale**: Configuration and exception classes can be tested in isolation. I/O operations will require integration tests with actual Arrow/Parquet data in subsequent tasks.

**Pattern**: Test configuration and error handling first, then integration
```csharp
// Unit tests for configuration (current task)
[Test] public void ArrowConversionOptions_DefaultValues() { /* test defaults */ }
[Test] public void ParquetWriteOptions_CompressionValidation() { /* test validation */ }

// Integration tests for I/O operations (future tasks)  
[Test] public void ArrowInterop_RoundTripPreservesData() { /* test with real data */ }
[Test] public void ParquetReader_HandlesLargeFiles() { /* test with real files */ }
```

### Build Verification Strategy
**Decision**: Verify both Nivara and Nivara.Extensions projects build successfully after structural changes
**Rationale**: Ensures new namespace structure doesn't break existing functionality. Both projects must compile cleanly before proceeding to implementation tasks.

**Pattern**: Build verification after structural changes
```bash
# Verify core project builds
dotnet build src/Nivara/Nivara.csproj

# Verify extensions project builds  
dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj
```

### Future Implementation Notes
**Note**: This task only sets up the project structure and core interfaces. Actual Arrow/Parquet conversion logic will be implemented in subsequent tasks following the established patterns.

**Next Steps**: 
- Task 2: Implement type mapping system
- Task 3: Implement Arrow interoperability  
- Task 4: Implement Parquet I/O operations
- Task 5+: Add streaming, batching, and optimization features

### Type Mapping System Implementation
**Decision**: Created comprehensive TypeMapper class with CLR ↔ Arrow ↔ Parquet type mappings using static dictionaries and dynamic dispatch
**Rationale**: Provides fast, consistent type conversion across all I/O operations. Static dictionaries avoid reflection overhead while dynamic dispatch handles runtime type scenarios.

**Pattern**: Dictionary-based type mapping with fallback error handling
```csharp
// Static type mapping dictionaries
private static readonly Dictionary<Type, IArrowType> ClrToArrowMap = new()
{
    { typeof(bool), BooleanType.Default },
    { typeof(int), Int32Type.Default },
    // ... other mappings
};

// Dynamic dispatch for type conversion
return elementType switch
{
    Type t when t == typeof(int) => CreateArrowArray<int>(values),
    Type t when t == typeof(string) => CreateArrowArray<string>(values),
    _ => throw new UnsupportedTypeException(elementType, GetTypeSuggestions(elementType))
};
```

### Nullable Type Handling in I/O Operations
**Decision**: Extract underlying type using `Nullable.GetUnderlyingType()` for nullable value types, treat reference types as inherently nullable
**Rationale**: Consistent with .NET nullable semantics. Arrow and Parquet handle nullability at the column level, not the type level.

**Pattern**: Nullable type extraction and handling
```csharp
// Extract underlying type for nullable value types
var actualType = Nullable.GetUnderlyingType(clrType) ?? clrType;
var isNullable = Nullable.GetUnderlyingType(clrType) != null || !actualType.IsValueType;

// Use actual type for mapping, preserve nullability information
var arrowType = ClrToArrowMap[actualType];
var parquetField = new DataField<T>(name, isNullable);
```

### Error Handling with Type Suggestions
**Decision**: Provide context-specific suggestions for unsupported types based on common conversion patterns
**Rationale**: Helps users quickly identify appropriate alternative types. Reduces trial-and-error when working with unsupported types.

**Pattern**: Context-aware error messages with suggestions
```csharp
private static List<string> GetTypeSuggestions(Type unsupportedType)
{
    return unsupportedType switch
    {
        Type t when t == typeof(Guid) => new List<string> { "string", "byte[]" },
        Type t when t == typeof(TimeSpan) => new List<string> { "long (ticks)", "double (seconds)" },
        Type t when t.IsEnum => new List<string> { "int", "string" },
        _ => new List<string> { "bool", "int", "long", "float", "double", "DateTime", "string" }
    };
}
```

### InternalsVisibleTo Pattern for Extensions Testing
**Decision**: Added InternalsVisibleTo attribute to Nivara.Extensions project for test access to internal classes
**Rationale**: Allows comprehensive testing of internal implementation classes while keeping them hidden from public API. Follows same pattern as core Nivara project.

**Pattern**: Assembly attribute in project file
```xml
<ItemGroup>
  <AssemblyAttribute Include="System.Runtime.CompilerServices.InternalsVisibleTo">
    <_Parameter1>Nivara.Tests</_Parameter1>
  </AssemblyAttribute>
</ItemGroup>
```

### Type Mapping Testing Strategy
**Decision**: Comprehensive unit tests covering all supported types, edge cases, and error conditions
**Rationale**: Type mapping is foundational to I/O correctness. Tests verify bidirectional mapping consistency and proper error handling.

**Coverage**: 15 test cases covering:
- All supported primitive type mappings (CLR ↔ Arrow ↔ Parquet)
- Nullable type handling and nullability preservation
- Unsupported type error handling with suggestions
- Round-trip mapping consistency
- Special cases (DateTime timezone handling, enum suggestions)
- Type support validation methods

### Type Mapping Gotchas
**Problem**: Arrow TimestampType timezone format varies between "UTC" and "+00:00"
**Solution**: Use flexible assertions that accept both formats
```csharp
// WRONG - assumes specific timezone format
Assert.That(timestampType.Timezone, Is.EqualTo("UTC"));

// CORRECT - accepts both common UTC representations
Assert.That(timestampType.Timezone, Is.EqualTo("+00:00").Or.EqualTo("UTC"));
```

**Problem**: Parquet.Net HasNulls property is obsolete, use IsNullable instead
**Solution**: Update test assertions to use current API
```csharp
// WRONG - obsolete property
Assert.That(field.HasNulls, Is.True);

// CORRECT - current property
Assert.That(field.IsNullable, Is.True);
```

**Problem**: Complex tuple type inference in test arrays
**Solution**: Use explicit tuple type declarations
```csharp
// WRONG - compiler cannot infer complex tuple types
var testCases = new[] { (BooleanType.Default, typeof(bool)) };

// CORRECT - explicit tuple type declaration
var testCases = new (IArrowType ArrowType, Type ExpectedClrType)[]
{
    (BooleanType.Default, typeof(bool))
};
```

## DataFrame Operations Infrastructure Implementation

### Core Query Operation Infrastructure Design
**Decision**: Created comprehensive query operation infrastructure with ExecutionStrategy enumeration, ExecutionContext class, generic IQueryOperation<T> interface, DataFrameOperation base class, and QueryNode hierarchy
**Rationale**: Provides foundation for advanced DataFrame operations with different execution strategies (Lazy, Eager, Streaming, Parallel) and proper query plan representation for optimization.

**Pattern**: Layered architecture with clear separation of concerns
```csharp
// Execution strategies for different use cases
public enum ExecutionStrategy { Lazy, Eager, Streaming, Parallel }

// Execution context with configuration and progress reporting
public sealed class ExecutionContext
{
    public ExecutionStrategy Strategy { get; set; }
    public int MaxDegreeOfParallelism { get; set; }
    public long MemoryBudget { get; set; }
    public CancellationToken CancellationToken { get; set; }
    public IProgress<ExecutionProgress>? Progress { get; set; }
}

// Generic query operation interface
public interface IQueryOperation<T>
{
    QueryPlan Plan { get; }
    ExecutionStrategy Strategy { get; }
    IQueryOperation<TResult> Transform<TResult>(Func<T, TResult> transform);
    Task<T> ExecuteAsync(CancellationToken cancellationToken = default);
}

// Base class for DataFrame operations
public abstract class DataFrameOperation : IQueryOperation<NivaraFrame>
{
    protected virtual NivaraFrame ExecuteLazy(ExecutionContext context);
    protected virtual NivaraFrame ExecuteEager(ExecutionContext context);
    protected virtual NivaraFrame ExecuteStreaming(ExecutionContext context);
    protected virtual NivaraFrame ExecuteParallel(ExecutionContext context);
}
```

### QueryNode Hierarchy for Query Plans
**Decision**: Created comprehensive QueryNode hierarchy with visitor pattern support for query plan representation and traversal
**Rationale**: Enables sophisticated query optimization by providing structured representation of query operations. Visitor pattern allows for easy extension of query analysis and transformation capabilities.

**Pattern**: Abstract base class with concrete node types and visitor support
```csharp
// Base query node with common properties
public abstract class QueryNode
{
    public List<QueryNode> Children { get; }
    public Schema OutputSchema { get; }
    public long EstimatedRowCount { get; set; }
    public TimeSpan EstimatedExecutionTime { get; set; }
    public abstract void Accept(IQueryNodeVisitor visitor);
    public abstract T Accept<T>(IQueryNodeVisitor<T> visitor);
}

// Concrete node types for different operations
public sealed class SourceNode : QueryNode { /* data source */ }
public sealed class FilterNode : QueryNode { /* filter operations */ }
public sealed class ProjectionNode : QueryNode { /* column selection */ }
public sealed class GroupByNode : QueryNode { /* grouping operations */ }

// Visitor interfaces for traversal and transformation
public interface IQueryNodeVisitor { /* void visitor */ }
public interface IQueryNodeVisitor<T> { /* transforming visitor */ }
```

### Execution Context and Progress Reporting
**Decision**: Comprehensive ExecutionContext with progress reporting, cancellation support, and resource management
**Rationale**: Provides fine-grained control over query execution behavior. Progress reporting enables user feedback for long-running operations. Resource management prevents memory issues with large datasets.

**Pattern**: Builder-style factory methods for common configurations
```csharp
// Factory methods for common execution contexts
var parallelContext = ExecutionContext.WithParallelism(8);
var streamingContext = ExecutionContext.WithStrategy(ExecutionStrategy.Streaming);
var budgetContext = ExecutionContext.WithMemoryBudget(512 * 1024 * 1024);
var cancellableContext = ExecutionContext.WithCancellation(cancellationToken);

// Progress reporting for long-running operations
context.Progress?.Report(new ExecutionProgress("FilterOperation", 500, 1000));
```

### Public API Design Decisions
**Decision**: Made QueryPlan, IQueryOperation, and related infrastructure public to support the generic IQueryOperation<T> interface
**Rationale**: Generic interface requires public access to QueryPlan for type safety. Public API enables advanced users to create custom operations and execution strategies.

**Pattern**: Public interfaces with internal implementation details
```csharp
// Public interfaces for extensibility
public interface IQueryOperation<T> { /* generic interface */ }
public interface IQueryOperation { /* non-generic interface */ }
public sealed class QueryPlan { /* public for interface requirements */ }
public abstract class DataFrameOperation { /* public base class */ }

// Internal implementation details remain internal
internal sealed class FilterOperation : IQueryOperation { /* internal implementations */ }
internal sealed class SelectOperation : IQueryOperation { /* internal implementations */ }
```

### Testing Strategy for Query Infrastructure
**Decision**: Comprehensive unit tests covering ExecutionContext, ExecutionProgress, and QueryNode hierarchy with visitor pattern testing
**Rationale**: Query infrastructure is foundational to DataFrame operations correctness. Tests verify configuration handling, progress reporting, and query plan representation work correctly.

**Coverage**: 20 test cases covering:
- ExecutionContext creation, cloning, and factory methods
- ExecutionProgress calculation and formatting
- QueryNode hierarchy creation and property management
- Visitor pattern implementation and traversal
- ToString formatting and error handling
- WithChildren immutability and node copying

### Infrastructure Implementation Gotchas
**Problem**: Generic IQueryOperation<T> interface requires public QueryPlan class
**Solution**: Made QueryPlan and related classes public to support generic interface requirements
```csharp
// WRONG - internal QueryPlan with public generic interface
internal sealed class QueryPlan { }
public interface IQueryOperation<T> { QueryPlan Plan { get; } } // Compiler error

// CORRECT - public QueryPlan for public interface
public sealed class QueryPlan { }
public interface IQueryOperation<T> { QueryPlan Plan { get; } } // Compiles correctly
```

**Problem**: QueryNode ToString methods need to include row count and schema information for debugging
**Solution**: Override ToString in each concrete node type to include relevant debugging information
```csharp
// Base pattern for QueryNode ToString
public override string ToString()
{
    var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
    return $"{NodeType} [Specific Info, Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
}
```

**Problem**: ExecutionContext needs sensible defaults for all properties
**Solution**: Provide reasonable defaults based on system capabilities
```csharp
// Sensible defaults based on system
Strategy = ExecutionStrategy.Lazy; // Safe default
MaxDegreeOfParallelism = Environment.ProcessorCount; // Use all cores
MemoryBudget = 1024 * 1024 * 1024; // 1GB default budget
CancellationToken = CancellationToken.None; // No cancellation by default
```

## Arrow Interoperability Implementation

### Apache Arrow API Usage Patterns
**Decision**: Use `Table.TableFromRecordBatches()` for creating Arrow tables, individual `Append()` calls for builders instead of `AppendRange()`
**Rationale**: Follows Apache Arrow best practices and avoids API compatibility issues. Individual append calls provide better null handling control.

**Pattern**: Proper Arrow table and array creation
```csharp
// Table creation with single RecordBatch
var schema = new Apache.Arrow.Schema(fields, null);
var recordBatch = new RecordBatch(schema, arrowArrays, frame.RowCount);
return Table.TableFromRecordBatches(schema, new[] { recordBatch });

// Array building with individual appends
var builder = new Int32Array.Builder();
for (int i = 0; i < column.Length; i++)
{
    if (column.IsNull(i))
        builder.AppendNull();
    else
        builder.Append(column[i]);
}
return builder.Build();
```

### Dynamic Dispatch for Arrow Conversion
**Decision**: Use type switching with dynamic dispatch for converting between Nivara columns and Arrow arrays
**Rationale**: Avoids reflection overhead while maintaining type safety. Follows established patterns from NivaraColumn arithmetic operations.

**Pattern**: Type switching for Arrow conversion
```csharp
// Convert column to Arrow array using dynamic dispatch
return elementType switch
{
    Type t when t == typeof(bool) => ConvertColumnToArrowArrayTyped<bool>(column, options),
    Type t when t == typeof(int) => ConvertColumnToArrowArrayTyped<int>(column, options),
    Type t when t == typeof(string) => ConvertColumnToArrowArrayTyped<string>(column, options),
    _ => throw new UnsupportedTypeException(elementType, TypeMapper.GetTypeSuggestions(elementType))
};
```

### Nullable Type Handling in Arrow Conversion
**Decision**: Use reflection and `Activator.CreateInstance` for creating nullable value type instances during Arrow-to-Nivara conversion
**Rationale**: Arrow arrays store nullability information separately from values. Need to reconstruct nullable types for value types while preserving null semantics.

**Pattern**: Nullable type reconstruction
```csharp
// For value types, create nullable array with proper type
if (typeof(T).IsValueType)
{
    var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
    var nullableArray = System.Array.CreateInstance(nullableType, values.Count);
    
    for (int i = 0; i < values.Count; i++)
    {
        if (values[i] != null)
        {
            var nullableInstance = Activator.CreateInstance(nullableType, values[i]);
            nullableArray.SetValue(nullableInstance, i);
        }
    }
    
    return NivaraColumn<T>.CreateFromNullable(nullableArray);
}
```

### DateTime and Timezone Handling
**Decision**: Convert all DateTime values to UTC for Arrow storage, handle timezone conversion during Arrow-to-DateTime conversion
**Rationale**: Arrow TimestampType expects UTC timestamps. Provides consistent timezone handling across different data sources.

**Pattern**: DateTime timezone normalization
```csharp
// Convert DateTime to UTC for Arrow storage
var dateTime = column[i];
if (dateTime.Kind == DateTimeKind.Unspecified)
    dateTime = DateTime.SpecifyKind(dateTime, DateTimeKind.Utc);
else if (dateTime.Kind == DateTimeKind.Local)
    dateTime = dateTime.ToUniversalTime();

// Convert from Arrow timestamp to DateTime with timezone
var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
var dateTime = unixEpoch.AddMicroseconds(timestampValue);
if (options.TimeZone != TimeZoneInfo.Utc)
    dateTime = TimeZoneInfo.ConvertTimeFromUtc(dateTime, options.TimeZone);
```

### Arrow ChunkedArray Access Pattern
**Decision**: Use `ArrayCount` property and `Array(index)` method to access chunks in Arrow columns
**Rationale**: Arrow columns can contain multiple chunks. Must iterate through all chunks to extract complete data.

**Pattern**: Multi-chunk Arrow column processing
```csharp
// Get the ChunkedArray and process all chunks
var chunkedArray = arrowColumn.Data;
int chunkCount = chunkedArray.ArrayCount;

for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
{
    var chunk = chunkedArray.Array(chunkIndex);
    
    for (int i = 0; i < chunk.Length; i++)
    {
        if (chunk.IsNull(i))
            values.Add(default(T?));
        else
            values.Add(ExtractValueFromArrowArray<T>(chunk, i, options));
    }
}
```

### Arrow Interop Testing Strategy
**Decision**: Comprehensive unit tests covering all supported types, null handling, empty data, and round-trip conversion
**Rationale**: Arrow interoperability is foundational to I/O correctness. Tests verify data preservation across conversion boundaries.

**Coverage**: 13 test cases covering:
- Empty frame and series conversion
- All supported primitive types (bool, int, long, float, double, string, DateTime, byte, short, uint, ulong, ushort, sbyte)
- Null value handling and preservation
- Multi-column frame conversion
- Round-trip data preservation (Frame → Arrow → Frame, Series → Arrow → Series)
- Error handling (null arguments)
- Schema preservation and validation

### Arrow Interop Gotchas
**Problem**: Apache Arrow Schema constructor requires explicit null parameter for metadata
**Solution**: Always pass `null` as second parameter when creating schemas without metadata
```csharp
// WRONG - missing metadata parameter
var schema = new Apache.Arrow.Schema(fields);

// CORRECT - explicit null metadata
var schema = new Apache.Arrow.Schema(fields, null);
```

**Problem**: Arrow TimestampArray builder requires DateTimeOffset, not DateTime
**Solution**: Convert DateTime to DateTimeOffset using Unix epoch calculation
```csharp
// WRONG - passing DateTime directly
builder.Append(dateTime);

// CORRECT - convert to DateTimeOffset
var unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
var microseconds = (long)(dateTime - unixEpoch).TotalMicroseconds;
builder.Append(DateTimeOffset.FromUnixTimeMilliseconds(microseconds / 1000));
```

**Problem**: Empty Arrow tables still need valid schema and empty RecordBatch
**Solution**: Create proper empty structures instead of returning null
```csharp
// WRONG - returning null or invalid empty table
if (frame.ColumnCount == 0) return null;

// CORRECT - create valid empty table structure
var emptySchema = new Apache.Arrow.Schema(new Field[0], null);
var emptyBatch = new RecordBatch(emptySchema, new IArrowArray[0], 0);
return Table.TableFromRecordBatches(emptySchema, new[] { emptyBatch });
```

**Problem**: NivaraFrame requires at least one column, but Arrow tables can be truly empty
**Solution**: Handle empty Arrow tables by creating a placeholder column
```csharp
// Handle truly empty Arrow tables
if (arrowTable.ColumnCount == 0)
{
    var emptyColumn = NivaraColumn<object>.Create(System.Array.Empty<object>());
    return NivaraFrame.Create(("EmptyColumn", emptyColumn));
}
```

## Parquet I/O Implementation

### Parquet.Net API Usage Patterns
**Decision**: Use `Parquet.ParquetReader.CreateAsync()` for reading and `Parquet.ParquetWriter.CreateAsync()` for writing
**Rationale**: Follows Parquet.Net best practices for async I/O operations. Provides proper resource management and streaming support.

**Pattern**: Proper Parquet file reading
```csharp
// Reading with proper resource management
using var parquetReader = await Parquet.ParquetReader.CreateAsync(stream);
using var rowGroupReader = parquetReader.OpenRowGroupReader(rowGroupIndex);
var columnData = await rowGroupReader.ReadColumnAsync(field);
```

### Parquet Schema Validation Strategy
**Decision**: Validate Parquet schema against supported types before processing data
**Rationale**: Provides early error detection and clear error messages for unsupported types. Prevents runtime failures during data conversion.

**Pattern**: Schema validation with detailed error messages
```csharp
private static void ValidateParquetSchema(ParquetSchema schema)
{
    var dataFields = schema.GetDataFields();
    var unsupportedFields = new List<string>();

    foreach (var field in dataFields)
    {
        if (!IsTypeSupported(field.ClrType))
        {
            unsupportedFields.Add($"{field.Name} ({field.ClrType.Name})");
        }
    }

    if (unsupportedFields.Count > 0)
    {
        var supportedTypes = string.Join(", ", TypeMapper.GetSupportedTypes().Select(t => t.Name));
        throw new SchemaValidationException($"Unsupported field types found: {string.Join(", ", unsupportedFields)}. Supported types: {supportedTypes}");
    }
}
```

### Null Handling in Parquet Conversion
**Decision**: Use `CreateFromNullable()` for value types and `CreateForReferenceType()` for reference types when creating NivaraColumns from Parquet data
**Rationale**: Leverages existing NivaraColumn API for proper null handling. Avoids direct access to internal storage classes.

**Pattern**: Type-appropriate column creation
```csharp
// For value types - use nullable arrays
private static NivaraColumn<T> CreateNivaraColumn<T>(DataColumn columnData) where T : struct
{
    var nullableArray = new T?[columnData.Data.Length];
    
    for (int i = 0; i < columnData.Data.Length; i++)
    {
        var value = columnData.Data.GetValue(i);
        nullableArray[i] = value != null ? (T)Convert.ChangeType(value, typeof(T)) : null;
    }
    
    return NivaraColumn<T>.CreateFromNullable(nullableArray);
}

// For reference types - use null-aware creation
private static NivaraColumn<string> CreateStringColumn(DataColumn columnData)
{
    var values = new string[columnData.Data.Length];
    
    for (int i = 0; i < columnData.Data.Length; i++)
    {
        var value = columnData.Data.GetValue(i);
        values[i] = value?.ToString(); // null stays null
    }
    
    return NivaraColumn<string>.CreateForReferenceType(values);
}
```

### Empty Frame Handling in Parquet I/O
**Decision**: Create dummy columns with "_empty" name for empty frames since NivaraFrame requires at least one column
**Rationale**: NivaraFrame constructor requires at least one column, but Parquet files can be truly empty. Dummy column approach maintains API consistency.

**Pattern**: Empty frame handling
```csharp
// Handle empty Parquet files
if (parquetReader.RowGroupCount == 0)
{
    var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
    return NivaraFrame.Create(("_empty", emptyColumn));
}

// Detect and handle dummy columns when reading
if (dataFields.Length == 1 && dataFields[0].Name == "_empty")
{
    var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
    return NivaraFrame.Create(("_empty", emptyColumn));
}
```

### Error Handling in Parquet Operations
**Decision**: Use specific exception types (NivaraIOException, SchemaValidationException, DataCorruptionException) with detailed context
**Rationale**: Provides precise error information for debugging and user feedback. Includes file paths, operation context, and affected data ranges.

**Pattern**: Context-rich error handling
```csharp
try
{
    var columnData = await rowGroupReader.ReadColumnAsync(field);
    var column = CreateNivaraColumnFromParquetData(columnData, field);
}
catch (Exception ex)
{
    throw new DataCorruptionException($"Failed to read column '{columnName}': {ex.Message}", ex)
    {
        AffectedColumns = new[] { columnName },
        AffectedRowRange = new Range(0, 1000) // Reasonable default range
    };
}
```

### Parquet I/O Gotchas
**Problem**: Cannot access internal MemoryStorage constructor from Extensions project
**Solution**: Use public NivaraColumn API methods (CreateFromNullable, CreateForReferenceType) instead of direct storage access
```csharp
// WRONG - trying to access internal storage
var storage = new MemoryStorage<T>(data, nullMask); // Compiler error - internal class

// CORRECT - use public API
var nullableArray = new T?[length];
// ... populate nullableArray ...
return NivaraColumn<T>.CreateFromNullable(nullableArray);
```

**Problem**: Parquet.Net ParquetReader doesn't have ThriftMetadata property in newer versions
**Solution**: Use reasonable default values for error context instead of accessing unavailable metadata
```csharp
// WRONG - accessing unavailable property
var range = new Range(0, (int)parquetReader.ThriftMetadata.RowGroups[index].TotalByteSize);

// CORRECT - use reasonable default
var range = new Range(0, 1000); // Use default range for error context
```

**Problem**: NivaraFrame.Create expects array parameter, not List
**Solution**: Convert List to array before passing to Create method
```csharp
// WRONG - passing List directly
return NivaraFrame.Create(columns); // Compiler error if columns is List

// CORRECT - convert to array
return NivaraFrame.Create(columns.ToArray());
```

### Parquet Testing Strategy
**Decision**: Focus on schema validation and error handling in initial implementation, defer integration tests to later tasks
**Rationale**: Core functionality needs to be solid before adding comprehensive integration tests. Schema validation catches most issues early.

**Coverage**: Initial implementation covers:
- File and stream reading methods (async and sync)
- Schema validation with detailed error messages
- Empty file handling with dummy columns
- Type conversion for all supported types
- Error handling with proper exception types
- Streaming interface (simplified implementation)

