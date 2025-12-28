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

// Access elements
Console.WriteLine(intColumn[0]);     // 1
Console.WriteLine(intColumn.Length); // 5

// Arithmetic operations (vectorizable types only)
var doubled = intColumn * 2;                    // Scalar multiplication
var sum = intColumn + intColumn;                // Element-wise addition
var result = intColumn.Multiply(3);             // Method syntax

// Working with nulls (reference types)
var stringWithNulls = NivaraColumn<string>.CreateForReferenceType(
    new[] { "hello", null, "world" });
    
Console.WriteLine(stringWithNulls.HasNulls);    // True
Console.WriteLine(stringWithNulls.IsNull(1));   // True
```

### Supported Operations

#### Arithmetic Operations
- Scalar multiplication: `column * scalar` or `column.Multiply(scalar)`
- Element-wise addition: `left + right` or `left.Add(right)`
- Proper null propagation in all operations

#### Column Operations
- Indexer access: `column[index]`
- Length property: `column.Length`
- Null checking: `column.IsNull(index)`, `column.HasNulls`
- Slicing: `column.Slice(start, length)`

## Architecture

### Storage Types

Nivara automatically selects the appropriate storage implementation:

- **TensorStorage**: For vectorizable types (int, float, double, long, short, byte, bool)
  - Uses System.Numerics.Tensors for SIMD acceleration
  - Optimized for numerical computations
  
- **MemoryStorage**: For non-vectorizable types (string, Guid, DateTime, custom objects)
  - Uses Memory<T> for efficient memory management
  - Supports null detection for reference types

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
```

### Slicing and Views

```csharp
var data = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

// Create a view of elements 2-5 (zero-based indexing)
var slice = data.Slice(2, 4); // Contains [3, 4, 5, 6]

Console.WriteLine(slice.Length); // 4
Console.WriteLine(slice[0]);     // 3
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
├── Nivara/                 # Core library
├── Nivara.Extensions/      # Extension methods and utilities
samples/
├── Nivara.SampleApp/       # Sample applications
tests/
├── Nivara.Tests/           # Unit and property tests
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
- ✅ Null handling for reference types
- ✅ Comprehensive test suite

### Upcoming Features
- 🔄 Series with indexing (`NivaraSeries<T>`)
- 🔄 Comparison operations
- 🔄 Advanced null handling
- 🔄 Performance optimizations
- 📋 DataFrame operations
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