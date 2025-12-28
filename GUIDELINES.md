# Nivara Development Guidelines

This document captures C# development patterns, common issues, and best practices discovered during Nivara development. It serves as a reference to avoid repeating mistakes and improve development efficiency.

## Project Structure

### Core Architecture
- **src/Nivara**: Main library containing core column types and interfaces
- **src/Nivara.Extensions**: Extension methods and additional functionality
- **tests/Nivara.Tests**: Unit tests and property-based tests
- **samples/Nivara.SampleApp**: Sample applications demonstrating usage

### Key Dependencies
- **System.Numerics.Tensors**: For vectorized operations and tensor storage
- **NUnit**: Unit testing framework for comprehensive test coverage

## C# Development Patterns

### Generic Type Constraints
When working with vectorizable types, use appropriate constraints:
```csharp
// For arithmetic operations
public static NivaraColumn<T> Add<T>(NivaraColumn<T> left, NivaraColumn<T> right)
    where T : struct, INumber<T>

// For unmanaged types (tensor storage)
internal class TensorStorage<T> : IColumnStorage<T>
    where T : unmanaged
```

### Memory Management
- Implement `IDisposable` for types that manage unmanaged resources
- Use `ReadOnlySpan<T>` and `ReadOnlyMemory<T>` for efficient data access
- Prefer views and slicing over copying when possible
- Always dispose of tensors and memory resources properly

### Null Handling
- Use explicit null masks rather than NaN-based semantics
- Implement nullable reference types consistently
- Handle both value types (`T?`) and reference types appropriately

### Performance Considerations
- Use `System.Numerics.Tensors.TensorPrimitives` for vectorized operations
- Batch scalar operations for non-vectorizable types
- Avoid unnecessary allocations in hot paths
- Use `unsafe` code judiciously for performance-critical sections

## Common Issues and Solutions

### Issue: Generic Type Resolution
**Problem**: Compiler cannot infer generic types in complex scenarios
**Solution**: Use explicit type parameters or helper methods with constraints

### Issue: Tensor Disposal
**Problem**: Memory leaks when tensors are not properly disposed
**Solution**: Always implement IDisposable and use using statements

### Issue: Null Mask Synchronization
**Problem**: Null masks getting out of sync with data during operations
**Solution**: Always update null masks atomically with data operations

### Issue: SIMD Availability
**Problem**: Vectorized operations not available on all platforms
**Solution**: Implement fallback scalar operations and runtime detection

## Testing Patterns

### Unit Testing
- Use NUnit for comprehensive test coverage
- Generate realistic test data for columnar operations
- Test with various column sizes (small: 1-10, medium: 100-1000, large: 10K+)
- Include null patterns in test data
- Focus on specific edge cases and integration points
- Test error conditions and boundary values
- Validate diagnostic information and performance characteristics

### Test Organization
```csharp
[TestFixture]
public class NivaraColumnTests
{
    [Test]
    public void VectorizableTypes_ShouldUseTensorStorage()
    {
        // Test with various vectorizable types
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        Assert.That(intColumn.Storage.IsVectorizable, Is.True);
    }
    
    [Test]
    public void EmptyColumn_ShouldHaveZeroLength()
    {
        // Unit test for specific edge case
        var column = NivaraColumn<int>.Create(Array.Empty<int>());
        Assert.That(column.Length, Is.EqualTo(0));
    }
}
```

## Code Style Guidelines

### Naming Conventions
- Use `PascalCase` for public members
- Use `camelCase` for private fields with underscore prefix (`_field`)
- Use descriptive names for generic type parameters (`T` for element type, `TIndex` for index type)

### Documentation
- Use XML documentation comments for all public APIs
- Include parameter validation and exception documentation
- Provide usage examples for complex APIs

### Error Handling
- Use specific exception types (`ArgumentOutOfRangeException`, `InvalidOperationException`)
- Provide clear, actionable error messages
- Validate parameters early in public methods

## Performance Optimization Notes

### Vectorization
- Check `Vector.IsHardwareAccelerated` for SIMD availability
- Use `TensorPrimitives` methods for optimal performance
- Profile vectorized vs scalar performance on target platforms

### Memory Allocation
- Minimize allocations in hot paths
- Use object pooling for frequently created objects
- Consider `ArrayPool<T>` for temporary arrays

### Benchmarking
- Use BenchmarkDotNet for performance measurements
- Test with realistic data sizes and patterns
- Compare against baseline implementations

## Common Mistakes to Avoid

1. **Not disposing tensors**: Always implement IDisposable properly
2. **Ignoring null masks**: Keep null masks synchronized with data
3. **Assuming vectorization**: Always provide scalar fallbacks
4. **Over-allocating**: Use views and slices instead of copying
5. **Inconsistent error handling**: Use appropriate exception types
6. **Missing bounds checking**: Validate indices and ranges
7. **Forgetting generic constraints**: Use appropriate type constraints

## Future Considerations

### Extensibility
- Design interfaces to support future storage backends
- Consider plugin architecture for custom operations
- Plan for distributed/parallel processing scenarios

### Compatibility
- Maintain backward compatibility in public APIs
- Consider serialization and versioning requirements
- Plan for cross-platform deployment scenarios

---

**Note**: This document should be updated as new patterns and issues are discovered during development. Always consult this guide before implementing new features to avoid known pitfalls.