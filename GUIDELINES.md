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

### Safe Type Conversion Pattern for TensorPrimitives
**Problem**: MemoryMarshal.Cast requires unmanaged constraints that generic methods can't provide
**Solution**: Use `(T)(object)` casting pattern for safe type conversion in generic contexts
```csharp
// WRONG - MemoryMarshal.Cast requires unmanaged constraint
var xFloat = MemoryMarshal.Cast<T, float>(x); // Compiler error CS0453

// CORRECT - safe type conversion pattern
if (type == typeof(float))
{
    var yFloat = (float)(object)y!;
    for (int i = 0; i < x.Length; i++)
    {
        destination[i] = (float)(object)x[i]! > yFloat;
    }
}
```

**Rationale**: The `(T)(object)` pattern provides safe type conversion without requiring generic constraints. While it involves boxing/unboxing, it's more reliable than MemoryMarshal.Cast which requires unmanaged constraints that can't be applied to static methods in generic classes (CS0080).

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

## System.Numerics.Tensors API Usage Patterns

### Tensor Creation and Data Access
**Decision**: Use `Tensor.Create(T[], ReadOnlySpan<IntPtr>)` for creating tensors from arrays, `FlattenTo(Span<T>)` for accessing tensor data
**Rationale**: System.Numerics.Tensors API requires arrays for tensor creation and provides `FlattenTo` method for efficient data extraction. Direct span access methods like `AsReadOnlySpan()` are not available.

**Pattern**: Proper tensor creation and data access
```csharp
// Tensor creation from array with dimensions
var dataArray = values.ToArray();
var tensor = Tensor.Create(dataArray, [values.Length]);

// Data access using FlattenTo
var buffer = new T[tensor.FlattenedLength];
tensor.FlattenTo(buffer);
return buffer[index]; // Access specific element

// Span access for operations
var span = buffer.AsSpan();
```

### Empty Tensor Handling
**Decision**: Use `Tensor.Create<T>(Array.Empty<T>())` for creating empty tensors
**Rationale**: Empty ReadOnlySpan<int> cannot be used directly with Tensor.Create. Array.Empty<T>() provides the correct empty array for tensor creation.

**Pattern**: Empty tensor creation
```csharp
// WRONG - ReadOnlySpan<int>.Empty causes compilation errors
_data = Tensor.Create<T>(ReadOnlySpan<int>.Empty);

// CORRECT - use Array.Empty<T>() for empty tensors
_data = Tensor.Create<T>(Array.Empty<T>());
```

### Nullable Value Type Handling with Tensors
**Decision**: Extract values from nullable types before creating tensors, create separate null mask tensor
**Rationale**: Tensors work with concrete types, not nullable types. Null information is stored in a separate boolean tensor for efficient vectorized operations.

**Pattern**: Nullable type processing for tensors
```csharp
// Process nullable values into separate data and null mask arrays
var dataArray = new T[values.Length];
var nullMaskArray = new bool[values.Length];
bool hasAnyNulls = false;

for (int i = 0; i < values.Length; i++)
{
    if (values[i].HasValue)
    {
        dataArray[i] = values[i]!.Value; // Use null-forgiving operator
        nullMaskArray[i] = false;
    }
    else
    {
        dataArray[i] = default(T);
        nullMaskArray[i] = true;
        hasAnyNulls = true;
    }
}

// Create tensors from processed arrays
_data = Tensor.Create(dataArray, [values.Length]);
_nullMask = hasAnyNulls ? Tensor.Create(nullMaskArray, [values.Length]) : null;
```

### Tensor Slicing Operations
**Decision**: Use `FlattenTo` to extract data, slice as span, then create new tensor from sliced array
**Rationale**: Direct tensor slicing APIs are not available. FlattenTo provides efficient data extraction for slicing operations.

**Pattern**: Tensor slicing implementation
```csharp
// Extract data using FlattenTo
var dataBuffer = new T[_data.FlattenedLength];
_data.FlattenTo(dataBuffer);

// Slice as span and convert back to array
var slicedDataArray = dataBuffer.AsSpan(start, length).ToArray();
var slicedData = Tensor.Create(slicedDataArray, [length]);

// Handle null mask slicing similarly
if (_nullMask != null)
{
    var nullMaskBuffer = new bool[_nullMask.FlattenedLength];
    _nullMask.FlattenTo(nullMaskBuffer);
    var slicedNullMaskArray = nullMaskBuffer.AsSpan(start, length).ToArray();
    slicedNullMask = Tensor.Create(slicedNullMaskArray, [length]);
}
```

### Cross-Project Tensor Usage
**Decision**: Add `using System.Numerics.Tensors;` to any file that creates or manipulates tensors
**Rationale**: Tensor creation and manipulation requires the System.Numerics.Tensors namespace. Helper classes in other projects (like NullableStorageHelper) need this import when working with tensors.

**Pattern**: Consistent using statements for tensor operations
```csharp
// Always include when working with tensors
using System.Numerics.Tensors;
using Nivara.Storage; // For TensorStorage<T>

// Create tensors in helper methods
return new TensorStorage<T>(dataArray, hasNulls ? Tensor.Create(nullMaskArray, [values.Length]) : null);
```

### Performance Considerations with FlattenTo
**Decision**: Cache flattened data when multiple accesses are needed, use direct FlattenTo for single access
**Rationale**: FlattenTo creates a copy of tensor data. For single element access, the copy overhead is acceptable. For multiple accesses, caching the flattened data improves performance.

**Pattern**: Efficient tensor data access
```csharp
// Single element access - direct FlattenTo
public T this[int index]
{
    get
    {
        var buffer = new T[_data.FlattenedLength];
        _data.FlattenTo(buffer);
        return buffer[index];
    }
}

// Multiple element access - consider caching (future optimization)
// Could cache flattened data for repeated access patterns
```

### Tensor Disposal and Resource Management
**Decision**: Tensors in System.Numerics.Tensors don't require explicit disposal
**Rationale**: Unlike some tensor libraries, System.Numerics.Tensors manages memory automatically. No explicit cleanup is needed in Dispose methods.

**Pattern**: Tensor disposal handling
```csharp
public void Dispose()
{
    if (!_disposed)
    {
        // Tensors don't require explicit disposal in System.Numerics.Tensors
        _disposed = true;
    }
}
```

## TensorStorage Implementation Lessons Learned

### API Evolution from Arrays to Tensors
**Problem**: Original implementation used arrays with comment "will be optimized with System.Numerics.Tensors later"
**Solution**: Proper tensor implementation using System.Numerics.Tensors API with FlattenTo for data access
**Lesson**: Follow through on architectural decisions. Placeholder implementations should be replaced with proper implementations as soon as the required APIs are understood.

### Tensor API Learning Curve
**Problem**: Initial attempts used non-existent methods like `AsReadOnlySpan()` on tensors
**Solution**: Research actual API through documentation and use `FlattenTo` for data extraction
**Lesson**: When working with new APIs, always verify method availability through official documentation rather than assuming based on similar APIs.

### Null-Forgiving Operator Usage
**Problem**: Compiler warnings about nullable value types even when null checks are present
**Solution**: Use null-forgiving operator (`!`) when you know the value is not null after explicit checks
**Lesson**: In nullable contexts, use null-forgiving operator judiciously after explicit null checks to satisfy compiler analysis.

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
public static class Csv
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
| `Storage/` | `Nivara.Storage` | Tensor-based and Memory-based storage |
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
using Nivara.Storage;         // For TensorStorage and MemoryStorage

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
   - `Nivara.Storage.MemoryStorage<T>` - Non-vectorizable types
   - `Nivara.Storage.TensorStorage<T>` - Vectorizable types

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

## Arrow/Parquet I/O Performance Optimizations

### Async Operations with Cancellation Support
**Decision**: Added comprehensive cancellation token support to all async I/O operations
**Rationale**: Enables proper cancellation of long-running I/O operations and prevents resource leaks. All async methods now accept CancellationToken parameters and check for cancellation at appropriate points.

**Pattern**: Cancellation token propagation through all async operations
```csharp
// All async I/O methods now support cancellation
public static async Task<NivaraFrame> ReadParquetAsync(string filePath, ParquetReadOptions? options = null, CancellationToken cancellationToken = default)
{
    // Check for cancellation before processing each column
    cancellationToken.ThrowIfCancellationRequested();
    
    var columnData = await rowGroupReader.ReadColumnAsync(field);
    // ... processing
}
```

### Buffer Pooling for Memory Optimization
**Decision**: Implemented BufferPool class for reusing byte, int, and double arrays during I/O operations
**Rationale**: Reduces memory allocations and garbage collection pressure when processing large datasets. Particularly effective for columns with more than 1024 elements.

**Pattern**: Buffer rental and return for large data processing
```csharp
// Use buffer pool for large arrays to reduce allocations
if (length > 1024)
{
    var buffer = BufferPool.RentIntBuffer(length);
    try
    {
        // Process data using buffer
        for (int i = 0; i < length; i++)
        {
            // Processing logic
        }
    }
    finally
    {
        BufferPool.ReturnIntBuffer(buffer);
    }
}
```

### Streaming Buffer Management
**Decision**: Created StreamingBufferManager for bounded memory usage during large dataset processing
**Rationale**: Ensures memory usage remains bounded regardless of dataset size. Provides configurable memory budgets and automatic garbage collection triggers.

**Pattern**: Memory-bounded streaming operations
```csharp
// Streaming with memory budget management
using var bufferManager = new StreamingBufferManager(memoryBudget: 256L * 1024 * 1024);

// Rent buffers within memory budget
var buffer = bufferManager.RentBuffer(chunkSize);
try
{
    // Process data chunk
}
finally
{
    bufferManager.ReturnBuffer(buffer);
}

// Automatic garbage collection when memory usage is high
bufferManager.TryCollectGarbage();
```

### Memory Usage Thresholds
**Decision**: Use 1024-element threshold for enabling buffer pooling optimizations
**Rationale**: Small arrays (< 1024 elements) have minimal allocation overhead, while larger arrays benefit significantly from pooling. This threshold balances performance gains against pooling overhead.

**Thresholds**:
- **Buffer Pooling**: Enabled for arrays > 1024 elements
- **String Processing**: 32-byte estimate per string for buffer sizing
- **Memory Budget**: Default 256MB for streaming operations
- **Garbage Collection**: Triggered at 80% of memory budget

### Performance Monitoring Integration
**Decision**: Integrated buffer pool statistics and memory usage tracking
**Rationale**: Enables monitoring of memory optimization effectiveness and debugging of memory-related issues.

**Pattern**: Performance monitoring and diagnostics
```csharp
// Get buffer pool statistics
var (byteBuffers, intBuffers, doubleBuffers) = BufferPool.GetPoolStatistics();

// Monitor streaming buffer manager usage
var currentUsage = bufferManager.CurrentMemoryUsage;
var isOverBudget = bufferManager.IsMemoryBudgetExceeded;
```

### I/O Operation Optimizations
**Decision**: Applied buffer pooling to all major I/O operations (Parquet reading/writing, column extraction, frame concatenation)
**Rationale**: Consistent memory optimization across all I/O paths ensures predictable performance characteristics regardless of operation type.

**Optimized Operations**:
- Parquet column data extraction and creation
- String column processing with estimated buffer sizing
- Frame concatenation for batch operations
- Arrow array conversion (future enhancement)

### Error Handling with Resource Cleanup
**Decision**: Proper resource cleanup in all buffer pooling operations using try/finally blocks
**Rationale**: Ensures buffers are returned to pools even when exceptions occur, preventing memory leaks and pool exhaustion.

**Pattern**: Exception-safe buffer management
```csharp
var buffer = BufferPool.RentIntBuffer(length);
try
{
    // Risky operations that might throw
    ProcessData(buffer);
}
finally
{
    // Always return buffer to pool
    BufferPool.ReturnIntBuffer(buffer);
}
```

### Future Performance Enhancements
**Note**: Current implementation provides foundation for additional optimizations:
- Zero-copy Arrow array creation using memory-mapped files
- Parallel column processing for multi-core systems
- Adaptive buffer sizing based on dataset characteristics
- Memory-mapped Parquet reading for very large files

### Performance Testing Recommendations
**Testing Strategy**: Performance optimizations should be validated with:
- Large dataset processing (> 1GB Parquet files)
- Memory usage monitoring during long-running operations
- Cancellation token responsiveness testing
- Buffer pool effectiveness measurement
- Garbage collection frequency analysis

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

## Extension Methods Implementation

### Extension Method Design Patterns
**Decision**: Follow existing CSV/JSON extension method patterns with both instance methods and static factory methods
**Rationale**: Provides consistent API experience across all I/O operations. Instance extension methods enable fluent chaining while static methods provide clear entry points.

**Pattern**: Dual extension method approach
```csharp
// Instance extension methods for fluent API
public static void ToParquet(this NivaraFrame frame, string filePath, ParquetWriteOptions? options = null)
public static Task ToParquetAsync(this NivaraFrame frame, string filePath, ParquetWriteOptions? options = null)

// Static methods for loading (following CSV/JSON pattern)
public static NivaraFrame LoadParquet(string filePath, ParquetReadOptions? options = null)
public static Task<NivaraFrame> LoadParquetAsync(string filePath, ParquetReadOptions? options = null)
```

### Arrow Extension Method Design
**Decision**: Make `FromArrowTable` an extension method on `Table` rather than a static method
**Rationale**: Enables fluent method chaining and follows the same pattern as `ToArrowTable`. Provides more natural API for round-trip operations.

**Pattern**: Bidirectional extension methods
```csharp
// NivaraFrame to Arrow
public static Table ToArrowTable(this NivaraFrame frame, ArrowConversionOptions? options = null)

// Arrow to NivaraFrame (extension method on Table)
public static NivaraFrame FromArrowTable(this Table arrowTable, ArrowConversionOptions? options = null)

// Enables fluent chaining
var roundTrip = frame.ToArrowTable().FromArrowTable();
```

### Stream-based Extension Methods
**Decision**: Provide separate stream-based methods with clear naming (`ToParquetStream`, `LoadParquetFromStream`)
**Rationale**: Distinguishes between file-based and stream-based operations. Prevents method overload ambiguity and makes resource management expectations clear.

**Pattern**: Explicit stream method naming
```csharp
// File-based methods
public static void ToParquet(this NivaraFrame frame, string filePath, ...)
public static NivaraFrame LoadParquet(string filePath, ...)

// Stream-based methods with explicit naming
public static void ToParquetStream(this NivaraFrame frame, Stream stream, ...)
public static NivaraFrame LoadParquetFromStream(Stream stream, ...)
```

### Extension Method Testing Strategy
**Decision**: Test both API design aspects (method chaining, overloads, async variants) and basic functionality
**Rationale**: Extension methods are primarily about API design, so tests should verify the API works as intended rather than deep functionality (which is tested in the underlying classes).

**Coverage**: Extension method tests focus on:
- Method availability and overloads
- Null parameter validation
- Method chaining capabilities
- Sync/async variant availability
- Default parameter handling
- Generic type constraint behavior

### Extension Method Gotchas
**Problem**: Extension methods on null objects still throw ArgumentNullException
**Solution**: Test null parameter scenarios explicitly to ensure proper error handling
```csharp
// Test null extension method calls
NivaraFrame frame = null!;
Assert.Throws<ArgumentNullException>(() => frame.ToArrowTable());

Table table = null!;
Assert.Throws<ArgumentNullException>(() => table.FromArrowTable());
```

**Problem**: Static methods in extension classes need explicit class qualification in tests
**Solution**: Use full class name when calling static methods in tests
```csharp
// WRONG - compiler error if method is static
Assert.Throws<ArgumentNullException>(() => LoadParquet(null!));

// CORRECT - explicit class qualification
Assert.Throws<ArgumentNullException>(() => NivaraFrameExtensions.LoadParquet(null!));
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


## Arrow/Parquet I/O Advanced Configuration Implementation

### Zero-Copy Optimization Strategy
**Decision**: Implemented UseZeroCopy option with fallback mechanism to copying when zero-copy is not feasible
**Rationale**: Zero-copy operations can significantly improve performance by avoiding memory duplication, but not all scenarios support it (e.g., columns with nulls, incompatible memory layouts).

**Pattern**: Try zero-copy first, fallback to copying
```csharp
// Try zero-copy optimization first if enabled
if (options.UseZeroCopy)
{
    var zeroCopyArray = TryCreateZeroCopyInt32Array(column);
    if (zeroCopyArray != null)
        return zeroCopyArray;
}

// Fallback to copying approach
var builder = new Int32Array.Builder();
// ... standard copying implementation
```

**Implementation Notes**:
- Zero-copy is currently implemented as placeholder methods that return null to fallback to copying
- Real zero-copy implementation would require:
  1. Access to underlying memory buffers from NivaraColumn storage
  2. Arrow array creation from existing buffers
  3. Proper memory ownership and lifecycle management
- Zero-copy is more complex with null values due to separate null mask handling

### Validation Configuration Implementation
**Decision**: Implemented ValidateTypes option for Arrow conversions with comprehensive type checking
**Rationale**: Type validation can be expensive for large datasets, so providing an option to disable it improves performance for trusted data sources.

**Pattern**: Conditional validation based on options
```csharp
// Validate types if requested
if (options.ValidateTypes)
{
    ValidateFrameTypesForArrowConversion(frame);
}
```

**Validation Coverage**:
- Frame-level validation: Checks all column types are supported for Arrow conversion
- Table-level validation: Validates Arrow types can be converted to Nivara types
- Series-level validation: Ensures series type is supported
- Array-level validation: Checks Arrow array type compatibility with target Nivara type

### Configuration Options Testing Strategy
**Decision**: Comprehensive unit tests covering all configuration scenarios including default values, enable/disable behavior, and custom settings
**Rationale**: Configuration options affect core functionality, so thorough testing ensures they work correctly and don't break existing behavior.

**Coverage**: 12 additional test cases covering:
- ValidateTypes enable/disable for all conversion methods
- UseZeroCopy enable/disable behavior
- Default configuration values validation
- Custom timezone handling for DateTime conversions
- Custom string encoding options
- Configuration option combinations

### Arrow Timezone Handling Gotcha
**Problem**: Arrow TimestampType in field definition must match TimestampType used in array creation
**Solution**: Ensure consistent timezone usage between field creation and array building
```csharp
// WRONG - timezone mismatch between field and array
var arrowType = TypeMapper.MapClrToArrow(typeof(DateTime)); // Uses UTC
var timestampType = new TimestampType(TimeUnit.Microsecond, options.TimeZone); // Uses custom timezone

// CORRECT - use same timezone for both field and array
if (column.ElementType == typeof(DateTime) && arrowType is TimestampType)
{
    arrowType = new TimestampType(TimeUnit.Microsecond, options.TimeZone);
}
```

### Performance Optimization Patterns
**Decision**: Implemented configuration-driven performance optimizations with graceful degradation
**Rationale**: Different use cases have different performance requirements. Configuration options allow users to optimize for their specific scenarios.

**Optimization Strategies**:
1. **Zero-Copy Operations**: Avoid memory duplication when possible
2. **Validation Bypass**: Skip expensive type checking for trusted data
3. **Custom Encoding**: Allow optimization for specific character sets
4. **Timezone Optimization**: Use UTC by default to avoid conversion overhead

### Configuration Testing Gotchas
**Problem**: Testing timezone-specific functionality can cause Arrow type mismatches
**Solution**: Use UTC timezone in tests to avoid compatibility issues, or ensure consistent timezone usage throughout the conversion pipeline

## ML.NET Integration Implementation

### VBuffer Conversion Strategy
**Decision**: Implemented bidirectional conversion between NivaraSeries and ML.NET VBuffer format with support for both dense and sparse tensors
**Rationale**: VBuffer is ML.NET's standard format for feature vectors. Supporting both dense and sparse formats enables efficient memory usage for different data patterns.

**Pattern**: Handle both dense and sparse VBuffers transparently
```csharp
// Dense VBuffer conversion
var vbuffers = seriesList.ToBatchTensors(); // Creates dense VBuffers
var reconstructedSeries = TensorConversions.FromBatchTensors(vbuffers);

// Sparse VBuffer handling
if (tensor.IsDense)
{
    var denseValues = tensor.GetValues();
    // Process all values
}
else
{
    // Sparse tensor - fill with defaults and set non-zero values
    Array.Fill(values, T.Zero);
    var sparseValues = tensor.GetValues();
    var sparseIndices = tensor.GetIndices();
    // Process only non-zero values
}
```

### Tensor Reshaping Implementation
**Decision**: Implemented multi-dimensional tensor reshaping with row-major order indexing
**Rationale**: Consistent with .NET Array conventions and most ML frameworks. Provides efficient conversion between linear series data and multi-dimensional tensor representations.

**Pattern**: Linear index to multi-dimensional index conversion
```csharp
// Convert linear index to multi-dimensional indices
var temp = i;
for (int dim = dimensions.Length - 1; dim >= 0; dim--)
{
    indices[dim] = temp % dimensions[dim];
    temp /= dimensions[dim];
}
tensor.SetValue(column[i], indices);
```

### ML.NET Integration Testing Strategy
**Decision**: Comprehensive round-trip testing for all conversion methods with multiple numeric types
**Rationale**: ML.NET integration is critical for machine learning workflows. Round-trip tests ensure data integrity across conversion boundaries.

**Coverage**: 13 additional test cases covering:
- VBuffer batch conversions (dense and sparse)
- Tensor reshaping and flattening
- Multiple numeric types (float, double, int)
- Empty collection handling
- Error conditions and edge cases
- Round-trip data preservation
- Dimension validation

### ML.NET Integration Gotchas
**Problem**: VBuffer constructor parameter order differs between dense and sparse formats
**Solution**: Use appropriate constructor overloads for each format
```csharp
// Dense VBuffer
var denseVBuffer = new VBuffer<T>(values.Length, values);

// Sparse VBuffer  
var sparseVBuffer = new VBuffer<T>(totalLength, nonZeroCount, values, indices);
```

**Problem**: NivaraSeries.Create parameter order is values first, then index
**Solution**: Always pass values as first parameter, index as second
```csharp
// WRONG - index first
NivaraSeries<T>.Create(indices, values);

// CORRECT - values first
NivaraSeries<T>.Create(values, indices);
```

**Problem**: Configuration options need to be tested for both enabled and disabled states
**Solution**: Create separate test methods for each configuration state to ensure comprehensive coverage

## Parquet Writing Implementation

### Parquet.Net API Usage Patterns
**Decision**: Use non-nullable arrays for Parquet.Net DataColumn constructor, even for nullable fields
**Rationale**: Parquet.Net expects array types to match the DataField generic type exactly. For `DataField<int>` marked as nullable, pass `int[]` not `int?[]`. Nullability is handled internally by Parquet.Net.

**Pattern**: Extract data as non-nullable arrays, let Parquet.Net handle null semantics
```csharp
// WRONG - passing nullable array to DataColumn
var nullableArray = new T?[column.Length];
for (int i = 0; i < column.Length; i++)
    nullableArray[i] = column.IsNull(i) ? null : column[i];
var dataColumn = new DataColumn(field, nullableArray); // ArgumentException

// CORRECT - pass non-nullable array, Parquet.Net handles nulls
var values = new T[column.Length];
for (int i = 0; i < column.Length; i++)
    values[i] = column.IsNull(i) ? default(T) : column[i];
var dataColumn = new DataColumn(field, values); // Works correctly
```

### Schema Access in NivaraFrame
**Decision**: Use `frame.Schema.GetColumnType(columnName)` instead of non-existent `frame.ColumnTypes[i]`
**Rationale**: NivaraFrame doesn't expose a direct ColumnTypes indexer. Schema provides the correct API for accessing column type information.

**Pattern**: Access column types through Schema
```csharp
// WRONG - ColumnTypes property doesn't exist
var columnType = frame.ColumnTypes[i]; // Compiler error

// CORRECT - use Schema to get column type
var columnName = frame.ColumnNames[i];
var columnType = frame.Schema.GetColumnType(columnName);
```

### Parquet Empty Frame Handling
**Decision**: Create dummy "_empty" column for empty frames since Parquet requires at least one field
**Rationale**: Parquet format specification requires at least one field. NivaraFrame also requires at least one column, so this approach maintains consistency.

**Pattern**: Handle empty frames with dummy column
```csharp
if (frame.ColumnCount == 0)
{
    var dummyField = new DataField<int>("_empty");
    var emptySchema = new ParquetSchema(dummyField);
    using var emptyWriter = await Parquet.ParquetWriter.CreateAsync(emptySchema, stream);
    using var emptyRowGroup = emptyWriter.CreateRowGroup();
    var emptyColumn = new DataColumn(dummyField, Array.Empty<int>());
    await emptyRowGroup.WriteColumnAsync(emptyColumn);
    return;
}
```

### String Null Handling in Parquet
**Decision**: Pass `null` values directly for string columns, not empty strings
**Rationale**: Parquet.Net can handle null string values correctly. Converting nulls to empty strings loses semantic meaning.

**Pattern**: Preserve null semantics for reference types
```csharp
// WRONG - converts nulls to empty strings
stringArray[i] = column.IsNull(i) ? string.Empty : column[i];

// CORRECT - preserve null values
stringArray[i] = column.IsNull(i) ? null! : column[i];
```

### Parquet Batch Writing Strategy
**Decision**: Concatenate multiple frames into single frame before writing
**Rationale**: Parquet.Net doesn't support multiple datasets in one file. Concatenation approach is simpler than managing multiple row groups.

**Pattern**: Schema validation before concatenation
```csharp
// Validate schema compatibility first
if (options.ValidateSchema)
{
    ValidateFrameSchemaCompatibility(frameList);
}

// Concatenate frames and write as single frame
var concatenatedFrame = ConcatenateFrames(frameList);
try
{
    await WriteParquetAsync(concatenatedFrame, stream, options);
}
finally
{
    concatenatedFrame.Dispose(); // Important: dispose concatenated frame
}
```

### Error Context in I/O Operations
**Decision**: Wrap low-level exceptions with context-rich NivaraIOException types
**Rationale**: Provides better debugging experience with file paths, operation context, and affected data ranges.

**Pattern**: Context-rich exception wrapping
```csharp
try
{
    var columnData = ExtractColumnDataArray(frame, columnName, columnType);
    var dataColumn = new DataColumn(field, columnData);
    await rowGroupWriter.WriteColumnAsync(dataColumn);
}
catch (Exception ex)
{
    throw new DataCorruptionException($"Failed to write column '{columnName}': {ex.Message}", ex)
    {
        AffectedColumns = new[] { columnName },
        AffectedRowRange = new Range(0, frame.RowCount)
    };
}
```

### Parquet Writer Testing Strategy
**Decision**: Test with various data types, null values, empty frames, and error conditions
**Rationale**: Parquet writing involves complex type mapping and null handling. Comprehensive tests catch edge cases early.

**Coverage**: 10 test cases covering:
- Basic data type writing (int, string, double, bool, long, DateTime)
- Null value handling for both value types and reference types
- Empty frame handling
- Stream vs file writing
- Batch writing with multiple frames
- Schema validation errors
- Argument validation (null parameters)

### Parquet Writer Gotchas
**Problem**: Method overload ambiguity with null parameters
**Solution**: Use explicit type casting for null parameters in tests
```csharp
// WRONG - ambiguous between string and Stream overloads
Assert.Throws<ArgumentNullException>(() => ParquetWriter.WriteParquet(frame, null!));

// CORRECT - explicit type casting
Assert.Throws<ArgumentNullException>(() => ParquetWriter.WriteParquet(frame, (string)null!));
```

**Problem**: Parquet.Net DataColumn constructor expects exact type match
**Solution**: Always pass arrays with types matching the DataField generic parameter, not nullable versions

**Problem**: Frame concatenation requires proper resource disposal
**Solution**: Always dispose concatenated frames in finally blocks to prevent memory leaks

## Comprehensive Error Handling Implementation

### Parameter Validation Strategy
**Decision**: Use `ArgumentNullException.ThrowIfNull()` consistently across all public methods with additional validation for specific parameter types
**Rationale**: Provides consistent error handling and clear error messages. Modern .NET approach is more concise than manual null checks.

**Pattern**: Comprehensive parameter validation
```csharp
// Basic null validation
ArgumentNullException.ThrowIfNull(frame);
ArgumentNullException.ThrowIfNull(filePath);

// Additional validation for specific types
if (string.IsNullOrWhiteSpace(filePath))
    throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

if (!stream.CanWrite)
    throw new ArgumentException("Stream must be writable", nameof(stream));
```

### Exception Context Enhancement
**Decision**: Wrap all low-level exceptions with context-rich NivaraIOException types that include operation context, file paths, and affected data ranges
**Rationale**: Provides better debugging experience and helps users quickly identify the source of issues.

**Pattern**: Context-rich exception wrapping with operation context
```csharp
try
{
    // Core operation logic
    var result = PerformOperation();
    return result;
}
catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException and not DataCorruptionException)
{
    throw new NivaraIOException($"Failed to perform operation: {ex.Message}", ex)
    {
        OperationContext = "ClassName.MethodName",
        FilePath = filePath // when applicable
    };
}
```

### Column-Level Error Handling
**Decision**: Provide detailed error context for column-specific failures including column names and affected row ranges
**Rationale**: Helps users identify exactly which data is causing issues, especially important for large datasets.

**Pattern**: Column-specific error context
```csharp
try
{
    var column = ProcessColumn(columnName, columnData);
    return column;
}
catch (Exception ex) when (ex is not UnsupportedTypeException)
{
    throw new DataCorruptionException($"Failed to process column '{columnName}': {ex.Message}", ex)
    {
        AffectedColumns = new[] { columnName },
        AffectedRowRange = new Range(0, rowCount),
        OperationContext = "ClassName.MethodName"
    };
}
```

### Extension Method Parameter Validation
**Decision**: Add parameter validation to all extension methods to provide early error detection and consistent behavior
**Rationale**: Extension methods are often the first point of contact for users, so they should provide clear error messages for invalid inputs.

**Pattern**: Extension method validation
```csharp
public static void ToParquet(this NivaraFrame frame, string filePath, ParquetWriteOptions? options = null)
{
    ArgumentNullException.ThrowIfNull(frame);
    ArgumentNullException.ThrowIfNull(filePath);
    
    if (string.IsNullOrWhiteSpace(filePath))
        throw new ArgumentException("File path cannot be empty or whitespace", nameof(filePath));

    // Delegate to underlying implementation
    ParquetWriter.WriteParquet(frame, filePath, options);
}
```

### Error Handling Testing Strategy
**Decision**: Test both successful operations and all error conditions to ensure proper exception types and messages
**Rationale**: Error handling is critical for user experience. Tests verify that users get helpful error messages rather than cryptic internal exceptions.

**Coverage**: Error handling tests should cover:
- Null parameter validation for all public methods
- Invalid parameter validation (empty strings, non-readable streams, etc.)
- Type conversion errors with proper suggestions
- File I/O errors with proper context
- Data corruption scenarios with affected column/row information

### Error Handling Gotchas
**Problem**: Duplicate try blocks can cause compilation errors
**Solution**: Carefully review try-catch structure when refactoring error handling code
```csharp
// WRONG - duplicate try blocks
try
{
try
{
    // operation
}

// CORRECT - single try block
try
{
    // operation
}
catch (Exception ex)
{
    // error handling
}
```

**Problem**: Exception filtering can become complex with multiple exception types
**Solution**: Use clear when clauses and document which exceptions are allowed to bubble up
```csharp
// Clear exception filtering
catch (Exception ex) when (ex is not ArgumentNullException and not UnsupportedTypeException and not DataCorruptionException)
{
    // Wrap only unexpected exceptions
}
```

## Lazy Data Sources Implementation

### Data Source Architecture Decisions
**Decision**: Created separate lazy and eager data sources with IQuerySource abstraction and factory methods in extension classes
**Rationale**: Provides clear separation between lazy evaluation (scanning) and eager evaluation (reading). IQuerySource abstraction enables consistent query engine integration regardless of data source type.

**Pattern**: Dual data source approach with factory methods
```csharp
// Lazy data sources (scanning)
public static IQuerySource ScanCsv(string filePath) => new CsvLazySource(filePath);
public static QueryFrame ScanCsvAsQueryFrame(string filePath) => new QueryFrame(ScanCsv(filePath));

// Eager data sources (reading)  
public static NivaraFrame ReadCsvAsFrame(string filePath) => new CsvEagerSource(filePath).Execute().ToNivaraFrame();

// Static factory classes for convenience
public static class Csv
{
    public static QueryFrame ScanAsQueryFrame(string filePath) => Csv.ScanCsvAsQueryFrame(filePath);
    public static NivaraFrame ReadAsFrame(string filePath) => Csv.ReadCsvAsFrame(filePath);
}
```

### Schema Inference Strategy
**Decision**: Implement schema inference by reading sample rows and analyzing data types with fallback to string type
**Rationale**: Provides automatic type detection while maintaining predictable behavior. Different formats have different type inference characteristics (CSV: string/int, JSON: string/double).

**Pattern**: Sample-based schema inference with type-specific defaults
```csharp
// CSV schema inference (conservative, prefers integers)
private Schema InferSchemaFromSample()
{
    var sampleRows = ReadSampleRows(Math.Min(100, totalRows));
    var columnTypes = new Dictionary<string, Type>();
    
    foreach (var header in headers)
    {
        var columnValues = sampleRows.Select(row => row[header]).Where(v => !string.IsNullOrEmpty(v));
        
        // Try int first, then string
        if (columnValues.All(v => int.TryParse(v, out _)))
            columnTypes[header] = typeof(int);
        else
            columnTypes[header] = typeof(string);
    }
    
    return new Schema(columnTypes.Select(kvp => (kvp.Key, kvp.Value)));
}

// JSON schema inference (uses double for numbers)
private Schema InferSchemaFromSample()
{
    var sampleObjects = ReadSampleObjects(Math.Min(100, totalObjects));
    var columnTypes = new Dictionary<string, Type>();
    
    foreach (var property in GetAllProperties(sampleObjects))
    {
        var values = sampleObjects.Select(obj => obj[property]).Where(v => v != null);
        
        // JSON numbers default to double
        if (values.All(v => v is JsonElement elem && elem.ValueKind == JsonValueKind.Number))
            columnTypes[property] = typeof(double);
        else
            columnTypes[property] = typeof(string);
    }
    
    return new Schema(columnTypes.Select(kvp => (kvp.Key, kvp.Value)));
}
```

### Column Creation Bug Fixes
**Problem**: `CreateColumn` and `CreateEmptyColumn` methods in data sources were causing null reference exceptions
**Root Cause**: Methods were not properly handling the dynamic dispatch pattern for creating columns from unknown types at runtime
**Solution**: Implemented proper type switching with fallback to generic object columns

**Pattern**: Fixed dynamic column creation
```csharp
// WRONG - causes null reference exceptions
private static IColumn CreateColumn(Type elementType, List<object> values)
{
    // Missing proper type handling
    return null; // This was causing the exceptions
}

// CORRECT - proper type switching with dynamic dispatch
private static IColumn CreateColumn(Type elementType, List<object> values)
{
    return elementType switch
    {
        Type t when t == typeof(int) => CreateColumnTyped<int>(values),
        Type t when t == typeof(double) => CreateColumnTyped<double>(values),
        Type t when t == typeof(string) => CreateColumnTyped<string>(values),
        Type t when t == typeof(bool) => CreateColumnTyped<bool>(values),
        _ => CreateColumnGeneric(elementType, values)
    };
}

private static IColumn CreateColumnTyped<T>(List<object> values)
{
    var array = values.Cast<T>().ToArray();
    return NivaraColumn<T>.Create(array);
}

private static IColumn CreateColumnGeneric(Type elementType, List<object> values)
{
    var array = Array.CreateInstance(elementType, values.Count);
    for (int i = 0; i < values.Count; i++)
        array.SetValue(values[i], i);
    
    // Use reflection for unknown types
    var createMethod = typeof(NivaraColumn<>).MakeGenericType(elementType)
        .GetMethod("Create", new[] { elementType.MakeArrayType() });
    return (IColumn)createMethod!.Invoke(null, new object[] { array })!;
}
```

### Cross-Project Access Pattern
**Problem**: Extensions project needed access to internal QueryFrame constructor
**Solution**: Added `InternalsVisibleTo` attribute to core project to allow Extensions project access to internal APIs

**Pattern**: Controlled internal access for extensions
```csharp
// In src/Nivara/Nivara.csproj
<ItemGroup>
    <InternalsVisibleTo Include="Nivara.Extensions" />
</ItemGroup>

// Enables Extensions project to use internal constructors
internal QueryFrame(IQuerySource source) { ... } // Can be used from Extensions
```

### Factory Method Integration
**Decision**: Added factory methods to extension classes that create QueryFrames directly from data sources
**Rationale**: Provides convenient API for users who want to start with lazy queries immediately without manually creating QueryFrame instances.

**Pattern**: Extension class factory methods
```csharp
// Class provides both source and QueryFrame factory methods
public static class Csv
{
    // Low-level source creation
    public static IQuerySource ScanCsv(string filePath) => new CsvLazySource(filePath);
    
    // High-level QueryFrame creation (most common use case)
    public static QueryFrame ScanCsvAsQueryFrame(string filePath) => new QueryFrame(ScanCsv(filePath));
    
    // Eager reading for immediate materialization
    public static NivaraFrame ReadCsvAsFrame(string filePath) => /* implementation */;
}
```

### Error Handling for Data Sources
**Decision**: Use specific exception types (DataSourceException, FileNotFoundException) with detailed error messages
**Rationale**: Provides clear error context for data source issues. Helps users distinguish between file system errors and data format errors.

**Pattern**: Context-specific error handling
```csharp
// File system errors
if (!File.Exists(filePath))
    throw new FileNotFoundException($"CSV file not found: {filePath}");

// Data format errors
if (headers.Length == 0)
    throw new DataSourceException($"CSV file has no headers: {filePath}");

// Schema inference errors (JSON specific)
if (sampleObjects.Count == 0)
    throw new DataSourceException($"Cannot infer schema from empty JSON array: {filePath}");
```

### Testing Strategy for Lazy Data Sources
**Decision**: Comprehensive test suite covering lazy evaluation, schema inference, error conditions, and property-based testing
**Rationale**: Lazy data sources are foundational to query engine correctness. Tests verify both lazy behavior and data correctness.

**Coverage**: 27 test cases covering:
- Lazy source creation and schema inference (CSV and JSON)
- QueryFrame factory methods and lazy evaluation verification
- Schema inference correctness (CSV: string/int types, JSON: string/double types)
- Lazy evaluation behavior (no execution until Collect())
- Query execution with filtering and result validation
- Empty file handling (CSV: empty columns, JSON: schema inference error)
- Error conditions (null parameters, non-existent files)
- Eager vs lazy comparison (same results, different execution timing)
- Static factory class methods (Csv.ScanAsQueryFrame, Json.ReadAsFrame, etc.)
- Property-based testing for lazy evaluation and schema inference

### Lazy Data Sources Gotchas
**Problem**: Empty JSON arrays cannot have schema inferred because there are no sample objects
**Solution**: Throw DataSourceException for empty JSON arrays since schema inference is impossible
```csharp
// JSON empty file handling
if (jsonArray.GetArrayLength() == 0)
    throw new DataSourceException($"Cannot infer schema from empty JSON array: {filePath}");
```

**Problem**: CSV empty files still have headers, so they can have valid schemas with zero-length columns
**Solution**: Handle empty CSV files gracefully by creating columns with zero length
```csharp
// CSV empty file handling - create valid empty columns
var columns = new Dictionary<string, IColumn>();
foreach (var header in headers)
{
    var columnType = schema.GetColumnType(header);
    columns[header] = CreateEmptyColumn(columnType);
}
```

**Problem**: QueryFrame constructor is internal but needed by Extensions project
**Solution**: Use InternalsVisibleTo attribute to allow controlled access from Extensions project

**Problem**: Dynamic column creation requires careful type handling to avoid null reference exceptions
**Solution**: Use comprehensive type switching with proper fallback to reflection for unknown types

### Performance Characteristics
**Lazy Sources**: 
- Schema inference requires reading sample data (minimal I/O)
- Query building has no I/O cost
- Execution cost deferred until Collect()

**Eager Sources**:
- Immediate I/O cost during creation
- No additional cost during query operations
- Better for small datasets or when data will definitely be used

**Schema Inference**:
- CSV: Reads up to 100 sample rows for type detection
- JSON: Reads up to 100 sample objects for type detection
- Fallback to string type for ambiguous data
- Conservative approach prioritizes correctness over performance

### Integration with Query Engine
**Decision**: Lazy data sources integrate seamlessly with QueryFrame operations through IQuerySource interface
**Rationale**: Provides consistent API regardless of data source type. Query operations work the same whether source is lazy file-based or eager memory-based.

**Pattern**: Transparent query engine integration
```csharp
// Same query operations work for both lazy and eager sources
var csvQuery = Csv.ScanCsvAsQueryFrame("data.csv")
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name", "Salary");

var memoryQuery = frame.AsQueryFrame()
    .Filter(ColumnExpressions.Col("Age") > 30)
    .Select("Name", "Salary");

// Both execute the same way
var csvResult = csvQuery.Collect();
var memoryResult = memoryQuery.Collect();
```
## Eager Data Sources Implementation

### Eager vs Lazy Architecture Decisions
**Decision**: Implemented eager data sources as wrappers around lazy sources using `Lazy<T>` for immediate execution
**Rationale**: Avoids code duplication by reusing lazy source logic while providing different execution semantics. Eager sources execute immediately upon first access but still benefit from lazy initialization patterns.

**Pattern**: Wrapper pattern with immediate execution semantics
```csharp
// Eager source wraps lazy source with immediate execution
internal sealed class CsvEagerSource : IQuerySource
{
    private readonly CsvLazySource lazySource;
    private readonly Lazy<IReadOnlyDictionary<string, IColumn>> lazyColumns;

    public CsvEagerSource(string filePath, CsvOptions options)
    {
        lazySource = new CsvLazySource(filePath, options);
        lazyColumns = new Lazy<IReadOnlyDictionary<string, IColumn>>(lazySource.Execute);
    }

    public bool IsLazy => false; // Key difference from lazy sources
    public IReadOnlyDictionary<string, IColumn> Execute() => lazyColumns.Value;
}
```

### Data Consistency Validation Strategy
**Decision**: Implement graceful degradation for data inconsistencies with fallback to string types
**Rationale**: Real-world data often has inconsistencies. Rather than failing completely, the system falls back to more general types (string) when mixed types are detected during schema inference.

**Pattern**: Type inference with graceful fallback
```csharp
// CSV type inference with fallback
private static Type InferColumnType(List<dynamic> sampleRecords, string columnName)
{
    var values = ExtractNonNullValues(sampleRecords, columnName);
    
    if (values.Count == 0)
        return typeof(string); // Default fallback
    
    // Try specific types in order of preference
    if (values.All(v => int.TryParse(v, out _)))
        return typeof(int);
    
    if (values.All(v => double.TryParse(v, out _)))
        return typeof(double);
    
    return typeof(string); // Fallback for mixed or unparseable types
}
```

### Error Handling for Severe Data Corruption
**Decision**: Use specific exception types (DataSourceException) with detailed context for severe errors
**Rationale**: Distinguishes between recoverable inconsistencies (handled gracefully) and severe corruption (requires user intervention). Provides clear error messages with file paths and operation context.

**Pattern**: Context-rich error reporting for severe issues
```csharp
try
{
    var records = csv.GetRecords<dynamic>().ToList();
    return ProcessRecords(records);
}
catch (Exception ex)
{
    throw new DataSourceException($"Failed to read CSV file '{filePath}': {ex.Message}", ex);
}
```

### Immediate Execution Validation
**Decision**: Eager sources provide immediate data availability without additional I/O during data access
**Rationale**: Eager sources should front-load all I/O costs and make data immediately available. This provides predictable performance characteristics and simpler memory models.

**Pattern**: Immediate execution with no lazy behavior
```csharp
// Eager source executes immediately and caches results
public IReadOnlyDictionary<string, IColumn> Execute()
{
    return lazyColumns.Value; // Executes once, then cached
}

// Subsequent data access requires no additional I/O
var frame = Csv.ReadCsvAsFrame("data.csv"); // I/O happens here
var value1 = frame.GetColumn<string>("Name")[0];      // No I/O, data already loaded
var value2 = frame.GetColumn<string>("Name")[1];      // No I/O, data already loaded
```

### Property-Based Testing for Eager Sources
**Decision**: Comprehensive property tests covering immediate execution behavior and data consistency validation
**Rationale**: Eager sources have different performance characteristics than lazy sources. Tests verify immediate execution semantics and robust error handling.

**Coverage**: 4 new property tests covering:
- **Property 6: Eager loading behavior** - Immediate execution and data availability
- **Property 17: Data consistency validation** - Graceful handling of inconsistent data
- **Error reporting** - Clear error messages for severe corruption
- **Empty file handling** - Appropriate behavior for edge cases

### Testing Strategy for Data Consistency
**Decision**: Test both graceful degradation (mixed types) and error reporting (severe corruption)
**Rationale**: Real-world data sources have varying levels of issues. System should handle minor inconsistencies gracefully while reporting severe problems clearly.

**Pattern**: Multi-level error handling validation
```csharp
// Test graceful degradation for mixed types
var inconsistentData = "Name,Age\nAlice,30\nBob,not_a_number";
var frame = Csv.ReadCsvAsFrame(inconsistentData);
Assert.That(frame.Schema.GetColumnType("Age"), Is.EqualTo(typeof(string))); // Falls back to string

// Test error reporting for severe corruption
var corruptData = ""; // Empty file with no headers
Assert.Throws<DataSourceException>(() => Csv.ReadCsvAsFrame(corruptData));
```

### Performance Characteristics Documentation
**Eager Sources**:
- **Upfront Cost**: All I/O and parsing happens during source creation/first access
- **Memory Usage**: Full dataset loaded into memory immediately
- **Access Pattern**: O(1) data access after initial load
- **Best For**: Small datasets, guaranteed data usage, simple memory models

**Lazy Sources**:
- **Deferred Cost**: I/O and parsing deferred until Collect()
- **Memory Usage**: Minimal until execution, then full dataset
- **Access Pattern**: No cost until Collect(), then full processing
- **Best For**: Large datasets, conditional processing, filtered queries

### Integration with Query Engine
**Decision**: Eager sources integrate seamlessly with QueryFrame operations but lose lazy evaluation benefits
**Rationale**: Provides consistent API regardless of source type. Users can choose execution strategy (eager vs lazy) based on their specific needs.

**Pattern**: Transparent integration with different execution semantics
```csharp
// Eager source - data loaded immediately
var eagerFrame = Csv.ReadCsvAsFrame("data.csv");  // I/O happens here
var eagerQuery = eagerFrame.AsQueryFrame()                  // No additional I/O
    .Filter(ColumnExpressions.Col("Age") > 30)              // No I/O, data in memory
    .Collect();                                             // No I/O, just processing

// Lazy source - data loaded on demand
var lazyQuery = Csv.ScanCsvAsQueryFrame("data.csv") // Minimal I/O (schema only)
    .Filter(ColumnExpressions.Col("Age") > 30)                // No I/O, building query
    .Collect();                                               // I/O happens here
```

### Factory Method Consistency
**Decision**: Provide both extension methods and static factory classes for eager operations
**Rationale**: Maintains API consistency with lazy sources. Users can choose their preferred calling style while getting the same functionality.

**Pattern**: Dual API approach for eager operations
```csharp
// Extension methods on static classes
var frame1 = Csv.ReadCsvAsFrame("data.csv");
var frame2 = Json.ReadJsonAsFrame("data.json");

// Static factory classes
var frame3 = Csv.ReadAsFrame("data.csv");
var frame4 = Json.ReadAsFrame("data.json");

// Both approaches provide identical functionality
Assert.That(frame1.RowCount, Is.EqualTo(frame3.RowCount));
Assert.That(frame2.RowCount, Is.EqualTo(frame4.RowCount));
```

### Eager Data Sources Gotchas
**Problem**: Eager sources still use lazy initialization internally, which can be confusing
**Solution**: Document that "eager" refers to execution semantics (immediate vs deferred), not implementation details

**Problem**: Data consistency validation needs to balance robustness with usability
**Solution**: Use graceful degradation for common issues (mixed types) but clear errors for severe problems (corruption, missing files)

**Problem**: Empty files need different handling between CSV (headers available) and JSON (no schema inference possible)
**Solution**: CSV empty files return valid frames with zero-length columns, JSON empty arrays throw DataSourceException

**Problem**: File access time testing can be unreliable due to OS caching and file system behavior
**Solution**: Focus on testing data availability and execution semantics rather than precise I/O timing

### Memory Management Considerations
**Decision**: Eager sources hold full dataset in memory until disposal
**Rationale**: Provides predictable memory usage patterns. Users choosing eager sources accept the memory trade-off for simpler access patterns.

**Pattern**: Full dataset retention with proper disposal
```csharp
// Eager source keeps all data in memory
using var frame = Csv.ReadCsvAsFrame("large_file.csv"); // Full file loaded
// Data remains accessible throughout frame lifetime
var column = frame.GetColumn<string>("Name"); // No additional I/O
// Memory released when frame is disposed
```

### Error Context Enhancement
**Decision**: Include file paths and operation context in all error messages
**Rationale**: Helps users quickly identify problematic data sources and understand what operation failed.

**Pattern**: Enhanced error context for debugging
```csharp
catch (Exception ex)
{
    throw new DataSourceException($"Failed to read CSV file '{filePath}': {ex.Message}", ex);
    // Includes: operation type, file path, underlying error details
}
```

## Query Execution Engine and Optimization Implementation

### Query Optimization Architecture
**Decision**: Implement optimization as separate passes in QueryOptimizer class
**Rationale**: Modular design allows individual optimizations to be enabled/disabled and makes the system easier to test and maintain. Each optimization pass is independent and can be applied in sequence.

**Pattern**: Sequential optimization passes with fallback safety
```csharp
public QueryPlan Optimize(QueryPlan plan)
{
    try
    {
        var optimizedPlan = plan;
        optimizedPlan = ApplyPredicatePushdown(optimizedPlan);
        optimizedPlan = ApplyOperationFusion(optimizedPlan);
        optimizedPlan = ApplyColumnElimination(optimizedPlan);
        optimizedPlan = ApplyOperationReordering(optimizedPlan);
        return optimizedPlan;
    }
    catch (Exception)
    {
        // If optimization fails, return the original plan
        // Optimization should never break a valid query
        return plan;
    }
}
```

### Optimization Pass Implementations

#### 1. Predicate Pushdown Optimization
**Decision**: Move filter operations as early as possible in the query pipeline
**Rationale**: Filtering data early reduces the amount of data processed by subsequent operations, improving performance significantly.

**Implementation Strategy**:
- Collect all filter operations from the pipeline
- Analyze dependencies between filters and other operations
- Move filters before operations that don't depend on them
- Preserve semantic correctness by checking column availability

**Key Considerations**:
- Select operations may eliminate columns needed by filters
- GroupBy operations change the data structure
- Conservative approach: only push down when safe

#### 2. Operation Fusion Optimization
**Decision**: Combine compatible operations into single operations
**Rationale**: Reduces the number of passes over data and can enable more efficient execution.

**Fusion Rules**:
- **Multiple Filters**: Combine using AND logic with BinaryExpression
- **Multiple Selects**: Use the final select operation (later selects override earlier ones)
- **Conservative Approach**: Only fuse when semantically safe

**Pattern**: Type-based fusion logic
```csharp
private static IQueryOperation? FuseOperations(IQueryOperation first, IQueryOperation second)
{
    // Fuse multiple filter operations
    if (first is FilterOperation filter1 && second is FilterOperation filter2)
    {
        var combinedCondition = new BinaryExpression(BinaryOperator.And, filter1.Condition, filter2.Condition);
        return new FilterOperation(combinedCondition);
    }
    
    // Fuse multiple select operations (take the final projection)
    if (first is SelectOperation select1 && second is SelectOperation select2)
    {
        // Validate that select2 columns are available from select1
        return CanFuseSelects(select1, select2) ? select2 : null;
    }
    
    return null; // Cannot fuse these operations
}
```

#### 3. Column Elimination Optimization
**Decision**: Remove unused columns early in the pipeline
**Rationale**: Reduces memory usage and I/O overhead by eliminating columns that aren't referenced in the query.

**Implementation Strategy**:
- Analyze all operations to determine which columns are actually used
- Add a Select operation at the beginning if unused columns are detected
- Preserve all columns if no explicit column usage is found (conservative)

**Column Usage Analysis**:
```csharp
private static HashSet<string> AnalyzeColumnUsage(QueryPlan plan)
{
    var usedColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    
    foreach (var operation in plan.Operations)
    {
        if (operation is FilterOperation filter)
            usedColumns.UnionWith(GetReferencedColumns(filter.Condition));
        else if (operation is SelectOperation select)
            foreach (var column in select.Columns)
                usedColumns.UnionWith(GetReferencedColumns(column));
        // ... handle other operation types
    }
    
    // If no operations use columns explicitly, assume all columns are used
    if (usedColumns.Count == 0)
        usedColumns.UnionWith(plan.Source.Schema.ColumnNames);
    
    return usedColumns;
}
```

#### 4. Operation Reordering Optimization
**Decision**: Reorder operations for better performance while preserving semantics
**Rationale**: Some operation orders are more efficient than others (e.g., filters before selects).

**Reordering Rules**:
- **Filters First**: Move filter operations early to reduce data volume
- **Selects After Filters**: Column selection after filtering is more efficient
- **GroupBy Last**: Grouping operations should come after filtering and selection
- **Conservative Safety**: Only reorder when semantically safe

**Safety Validation**:
```csharp
private static bool IsReorderingSafe(List<IQueryOperation> original, List<IQueryOperation> reordered, Schema sourceSchema)
{
    // Conservative approach - only allow reordering of simple operations
    var allowedTypes = new[] { "Filter", "Select" };
    var originalTypes = original.Select(op => op.OperationType).ToList();
    var reorderedTypes = reordered.Select(op => op.OperationType).ToList();
    
    return originalTypes.All(t => allowedTypes.Contains(t)) && 
           reorderedTypes.All(t => allowedTypes.Contains(t));
}
```

### Expression Analysis for Optimization
**Decision**: Recursive expression analysis to extract column references
**Rationale**: Optimizations need to understand which columns are referenced by expressions to make safe transformations.

**Pattern**: Visitor pattern for expression traversal
```csharp
private static HashSet<string> GetReferencedColumns(ColumnExpression expression)
{
    var columns = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    
    if (expression is ColumnReference columnRef)
    {
        columns.Add(columnRef.ColumnName);
    }
    else if (expression is BinaryExpression binary)
    {
        columns.UnionWith(GetReferencedColumns(binary.Left));
        columns.UnionWith(GetReferencedColumns(binary.Right));
    }
    else if (expression is ComparisonExpression comparison)
    {
        columns.UnionWith(GetReferencedColumns(comparison.Left));
        if (comparison.Right is ColumnExpression rightExpr)
            columns.UnionWith(GetReferencedColumns(rightExpr));
    }
    // ... handle other expression types
    
    return columns;
}
```

### Query Optimization Diagnostics
**Decision**: Provide optimization analysis and suggestions to users
**Rationale**: Helps users understand query performance characteristics and identify optimization opportunities.

**Implementation**: Static analysis method that examines query plans
```csharp
public static IReadOnlyList<string> AnalyzeOptimizationOpportunities(QueryPlan plan)
{
    var suggestions = new List<string>();
    
    // Check for multiple filter operations
    var filterCount = plan.Operations.Count(op => op.OperationType == "Filter");
    if (filterCount > 1)
        suggestions.Add($"Found {filterCount} filter operations - consider combining them for better performance");
    
    // Check for predicate pushdown opportunities
    if (plan.Source.IsLazy && filterCount > 0)
        suggestions.Add("Filter operations on lazy source detected - predicate pushdown optimization available");
    
    // Check for column elimination opportunities
    var sourceColumnCount = plan.Source.Schema.ColumnNames.Count;
    var resultColumnCount = plan.ResultSchema.ColumnNames.Count;
    if (resultColumnCount < sourceColumnCount)
        suggestions.Add($"Query uses {resultColumnCount} of {sourceColumnCount} columns - consider adding explicit column selection");
    
    return suggestions;
}
```

### Property-Based Testing for Query Optimization
**Decision**: Implement comprehensive property tests to validate optimization correctness
**Rationale**: Query optimization is complex and error-prone. Property tests ensure that optimizations preserve query semantics across all scenarios.

**Key Properties Tested**:
- **Property 8: Collect execution barrier** - Multiple Collect() calls produce identical results
- **Property 9: Query optimization during execution** - Optimization preserves query semantics
- **Property 16: Comprehensive query optimization** - All optimization passes preserve correctness

**Testing Strategy**:
```csharp
// Test that optimization preserves semantics
var baselineQuery = source.Filter(condition);
var optimizableQuery = source.Select(allColumns).Filter(condition); // Suboptimal order

var baselineResult = baselineQuery.Collect();
var optimizedResult = optimizableQuery.Collect();

// Results should be identical despite different optimization opportunities
Assert.That(optimizedResult.RowCount, Is.EqualTo(baselineResult.RowCount));
// ... verify data equality
```

### Optimization Safety Principles
**Decision**: Always prioritize correctness over performance
**Rationale**: A correct but slow query is better than a fast but incorrect query.

**Safety Guidelines**:
1. **Conservative Analysis**: When in doubt, don't optimize
2. **Fallback Safety**: If optimization fails, return original plan
3. **Semantic Preservation**: Never change query results
4. **Dependency Checking**: Validate column availability before transformations
5. **Type Safety**: Ensure schema transformations are valid

### Future Optimization Opportunities
**Identified Areas for Enhancement**:
- **Cost-Based Optimization**: Use statistics to choose optimal execution plans
- **Join Reordering**: Optimize join order based on selectivity estimates
- **Index Utilization**: Push predicates to data sources that support indexing
- **Parallel Execution**: Identify operations that can be parallelized
- **Materialization Points**: Choose optimal points to materialize intermediate results

### Lessons Learned from Implementation
**Expression System Integration**: The optimization system relies heavily on the expression system for analyzing column dependencies. The recursive expression analysis pattern proved essential for safe optimization.

**Schema Transformation Validation**: Every optimization must validate that schema transformations are valid. The immutable schema design made this validation straightforward and safe.

**Testing Complexity**: Property-based testing was crucial for validating optimization correctness. Simple unit tests were insufficient to catch edge cases in optimization logic.

**Performance vs. Correctness Trade-offs**: The conservative approach to optimization ensures correctness but may miss some optimization opportunities. This trade-off was intentional to prioritize reliability.

## Error Handling and Diagnostics Implementation

### Diagnostic Mode Architecture
**Decision**: Separate diagnostic modes (None, Basic, Detailed, Performance, Comprehensive) with static QueryDiagnostics class
**Rationale**: Provides flexible diagnostic information without performance overhead when not needed. Static class allows global configuration while instance methods provide query-specific analysis.

**Pattern**: Use enum for diagnostic levels and static methods for analysis
```csharp
// Global diagnostic mode setting
QueryDiagnostics.GlobalMode = QueryDiagnosticMode.Performance;

// Query-specific diagnostic information
var diagnostics = queryFrame.GetDiagnosticInfo(QueryDiagnosticMode.Comprehensive);
var recommendations = queryFrame.AnalyzeQueryPlan();
```

### Deferred Error Handling Strategy
**Decision**: DeferredErrorHandler class for lazy operations with error collection and reporting at execution time
**Rationale**: Lazy operations should defer all errors until Collect() to maintain lazy semantics. Allows query building to continue even with file access issues.

**Pattern**: Collect errors during lazy operations, report during execution
```csharp
// In lazy data sources
try {
    return InferSchema();
} catch (Exception ex) {
    errorHandler.AddFileAccessError(filePath, ex, "ScanCsv");
    return new Schema(new[] { ("placeholder", typeof(string)) }); // Minimal schema
}

// At execution time
errorHandler.ThrowIfHasDeferredErrors("CSV data source execution");
```

### File IO Error Handling Enhancement
**Decision**: Comprehensive file access validation with specific exception types for different failure modes
**Rationale**: File operations have many failure modes (permissions, missing files, corruption). Specific error messages help users diagnose and fix issues quickly.

**Pattern**: Check file existence, permissions, and content before processing
```csharp
// File existence and accessibility checks
if (!File.Exists(filePath))
    throw new DataSourceException($"File not found: '{filePath}'");

try {
    fileContent = File.ReadAllText(filePath);
} catch (UnauthorizedAccessException ex) {
    throw new DataSourceException($"Access denied to file '{filePath}'. Check permissions.", ex);
} catch (IOException ex) {
    throw new DataSourceException($"IO error reading file '{filePath}': {ex.Message}", ex);
}
```

### CsvHelper Exception Handling Gotcha
**Problem**: CsvHelper.Exceptions namespace doesn't exist in newer versions
**Solution**: Use generic exception catching with type name checking for CSV-specific errors
```csharp
// WRONG - CsvHelper.Exceptions namespace doesn't exist
using CsvHelper.Exceptions;
catch (CsvHelperException ex) { ... }

// CORRECT - use generic exception catching
catch (Exception ex) when (ex.GetType().Name.Contains("Csv")) {
    throw new DataSourceException($"CSV parsing error: {ex.Message}", ex);
}
```

### Deferred Error Handling Challenges
**Problem**: Returning placeholder schemas during error deferral can cause downstream schema validation failures
**Lesson**: Deferred error handling requires careful balance between lazy semantics and schema consistency. Placeholder schemas should be designed to pass basic validation while still allowing error reporting at execution time.

**Consideration**: May need to revisit deferred error strategy to avoid schema validation cascading failures in tests. Alternative approaches:
- Fail fast on schema inference errors even in lazy mode
- Use more sophisticated placeholder schemas that match expected column types
- Implement schema validation bypass for placeholder schemas

### Diagnostic Integration Benefits
**Achievement**: Successfully integrated diagnostic modes with QueryFrame providing multiple levels of analysis
- Basic: Simple operation count and schema information
- Detailed: Complete query plan explanation with schema transformations
- Performance: Execution cost estimates and optimization opportunities
- Comprehensive: All diagnostic information plus recommendations

**Usage**: Diagnostic information helps users understand query performance characteristics and optimization opportunities without requiring deep knowledge of the query engine internals.

## Expression Validation and Type Resolution

### ColumnReference Type Resolution Issue
**Problem**: ColumnReference expressions created with `Col("columnName")` defaulted to `typeof(object)` for ResultType, causing type compatibility validation failures during schema validation.
**Root Cause**: The `ColumnReference.Validate()` method checked column existence and type compatibility but didn't update the `ResultType` property to match the actual schema type.
**Solution**: Made `ResultType` property settable with protected setter and updated `ColumnReference.Validate()` to set `ResultType` to the actual schema type when it was initially `typeof(object)`.

**Pattern**: Expression validation should update type information based on schema
```csharp
public override void Validate(Schema schema)
{
    // Validate column exists
    if (!schema.HasColumn(ColumnName))
        throw new SchemaValidationException($"Column '{ColumnName}' not found...");

    var actualType = schema.GetColumnType(ColumnName);
    
    // Update ResultType if it wasn't explicitly set
    if (ResultType == typeof(object))
    {
        ResultType = actualType;
    }
    else if (ResultType != actualType)
    {
        throw new SchemaValidationException($"Type mismatch...");
    }
}
```

### Expression Validation Error Messages
**Problem**: `TypeCompatibilityValidator.ValidateComparisonCompatibility()` was using operation names as column names in `ColumnTypeMismatchException`, causing confusing error messages.
**Solution**: Replaced calls to validator methods with direct type checking and `SchemaValidationException` with descriptive messages.

**Pattern**: Use appropriate exception types for different validation contexts
```csharp
// WRONG - uses ColumnTypeMismatchException for expression validation
TypeCompatibilityValidator.ValidateComparisonCompatibility(leftType, rightType, operationName);

// CORRECT - use SchemaValidationException with descriptive message
if (!TypeCompatibilityValidator.AreComparisonCompatible(Left.ResultType, Right.ResultType))
{
    throw new SchemaValidationException(
        $"Comparison {Operator} operation requires compatible types. " +
        $"Left operand type: {Left.ResultType.Name}, Right operand type: {Right.ResultType.Name}");
}
```

### Virtual vs Abstract Properties with Setters
**Decision**: Changed `ResultType` from abstract to virtual property with protected setter
**Rationale**: Allows base class to provide default implementation while enabling derived classes to update the property during validation. Abstract properties with setters require all derived classes to implement both getter and setter.

**Pattern**: Use virtual properties with protected setters for properties that need runtime updates
```csharp
// Base class
public virtual Type ResultType { get; protected set; } = typeof(object);

// Derived class can override if needed
public override Type ResultType { get; protected set; }

// Or use inherited implementation and update during validation
public override void Validate(Schema schema)
{
    // ... validation logic ...
    ResultType = actualType; // Update inherited property
}
```
