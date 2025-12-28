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