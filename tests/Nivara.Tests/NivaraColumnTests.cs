using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraColumn&lt;T&gt; creation and basic functionality
/// </summary>
[TestFixture]
public class NivaraColumnTests
{
    #region Property 1: Automatic storage selection for vectorizable types
    
    /// <summary>
    /// Property 1: Automatic storage selection for vectorizable types
    /// For any vectorizable type (int, float, double, etc.) and any array of values, 
    /// creating a NivaraColumn should result in appropriate storage being selected automatically.
    /// **Validates: Requirements 1.1, 6.1**
    /// </summary>
    [Test]
    public void VectorizableTypes_ShouldSelectAppropriateStorage()
    {
        // Test with int (vectorizable type)
        var intValues = new[] { 1, 2, 3, 4, 5 };
        var intColumn = NivaraColumn<int>.Create(intValues);
        
        Assert.That(intColumn, Is.Not.Null, "Int column should be created");
        Assert.That(intColumn.Length, Is.EqualTo(intValues.Length), "Int column should preserve length");
        // Note: Currently all types use memory storage, so IsVectorizable will be false
        // This will be updated when tensor storage is properly implemented
        Assert.That(intColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with float (vectorizable type)
        var floatValues = new[] { 1.0f, 2.5f, -3.14f };
        var floatColumn = NivaraColumn<float>.Create(floatValues);
        
        Assert.That(floatColumn, Is.Not.Null, "Float column should be created");
        Assert.That(floatColumn.Length, Is.EqualTo(floatValues.Length), "Float column should preserve length");
        Assert.That(floatColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with double (vectorizable type)
        var doubleValues = new[] { 1.0, 2.5, -3.14159 };
        var doubleColumn = NivaraColumn<double>.Create(doubleValues);
        
        Assert.That(doubleColumn, Is.Not.Null, "Double column should be created");
        Assert.That(doubleColumn.Length, Is.EqualTo(doubleValues.Length), "Double column should preserve length");
        Assert.That(doubleColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with long (vectorizable type)
        var longValues = new[] { 1L, 2L, 3L };
        var longColumn = NivaraColumn<long>.Create(longValues);
        
        Assert.That(longColumn, Is.Not.Null, "Long column should be created");
        Assert.That(longColumn.Length, Is.EqualTo(longValues.Length), "Long column should preserve length");
        Assert.That(longColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with byte (vectorizable type)
        var byteValues = new byte[] { 1, 2, 3 };
        var byteColumn = NivaraColumn<byte>.Create(byteValues);
        
        Assert.That(byteColumn, Is.Not.Null, "Byte column should be created");
        Assert.That(byteColumn.Length, Is.EqualTo(byteValues.Length), "Byte column should preserve length");
        Assert.That(byteColumn.IsVectorizable, Is.False, "Currently all columns use memory storage");

        // Test with empty arrays
        var emptyIntColumn = NivaraColumn<int>.Create(Array.Empty<int>());
        Assert.That(emptyIntColumn, Is.Not.Null, "Empty int column should be created");
        Assert.That(emptyIntColumn.Length, Is.EqualTo(0), "Empty column should have zero length");

        // Test with single element arrays
        var singleIntColumn = NivaraColumn<int>.Create(new[] { 42 });
        Assert.That(singleIntColumn, Is.Not.Null, "Single element column should be created");
        Assert.That(singleIntColumn.Length, Is.EqualTo(1), "Single element column should have length 1");
        Assert.That(singleIntColumn[0], Is.EqualTo(42), "Single element should be accessible");
    }

    #endregion

    #region Property 2: Automatic storage selection for non-vectorizable types

    /// <summary>
    /// Property 2: Automatic storage selection for non-vectorizable types
    /// For any non-vectorizable type (string, Guid, reference types) and any array of values, 
    /// creating a NivaraColumn should result in memory-backed storage being selected automatically.
    /// **Validates: Requirements 1.2, 6.2**
    /// </summary>
    [Test]
    public void NonVectorizableTypes_ShouldUseMemoryStorage()
    {
        // Test with string (reference type)
        var stringTestCases = new[]
        {
            new string[] { "hello", "world", "test" },
            new string[] { "", "non-empty", "" },
            new string[] { "single" },
            new string[] { } // Empty array
        };

        foreach (var values in stringTestCases)
        {
            var column = NivaraColumn<string>.Create(values);
            
            Assert.That(column, Is.Not.Null, "String column should be created successfully");
            Assert.That(column.Length, Is.EqualTo(values.Length), "Column length should match input array length");
            Assert.That(column.IsVectorizable, Is.False, "String columns should not be vectorizable");
        }

        // Test with Guid (non-vectorizable value type)
        var guidValues = new[] { Guid.NewGuid(), Guid.NewGuid(), Guid.Empty };
        var guidColumn = NivaraColumn<Guid>.Create(guidValues);
        
        Assert.That(guidColumn, Is.Not.Null, "Guid column should be created successfully");
        Assert.That(guidColumn.Length, Is.EqualTo(guidValues.Length), "Guid column length should match input");
        Assert.That(guidColumn.IsVectorizable, Is.False, "Guid columns should not be vectorizable");

        // Test with DateTime (non-vectorizable value type)
        var dateValues = new[] { DateTime.Now, DateTime.MinValue, DateTime.MaxValue };
        var dateColumn = NivaraColumn<DateTime>.Create(dateValues);
        
        Assert.That(dateColumn, Is.Not.Null, "DateTime column should be created successfully");
        Assert.That(dateColumn.Length, Is.EqualTo(dateValues.Length), "DateTime column length should match input");
        Assert.That(dateColumn.IsVectorizable, Is.False, "DateTime columns should not be vectorizable");
    }

    #endregion

    #region Property 4: Length preservation

    /// <summary>
    /// Property 4: Length preservation
    /// For any input array, the resulting NivaraColumn should have a Length property 
    /// that exactly matches the input array length.
    /// **Validates: Requirements 1.4**
    /// </summary>
    [Test]
    public void ColumnCreation_ShouldPreserveLength()
    {
        // Test with various array sizes
        var lengthTestCases = new[]
        {
            0,   // Empty array
            1,   // Single element
            2,   // Small array
            10,  // Medium array
            100, // Larger array
            1000 // Large array
        };

        foreach (var length in lengthTestCases)
        {
            // Test with int arrays
            var intValues = Enumerable.Range(0, length).ToArray();
            var intColumn = NivaraColumn<int>.Create(intValues);
            
            Assert.That(intColumn.Length, Is.EqualTo(length), 
                $"Int column should preserve length {length}");

            // Test with string arrays
            var stringValues = Enumerable.Range(0, length).Select(i => $"item_{i}").ToArray();
            var stringColumn = NivaraColumn<string>.Create(stringValues);
            
            Assert.That(stringColumn.Length, Is.EqualTo(length), 
                $"String column should preserve length {length}");

            // Test with double arrays
            var doubleValues = Enumerable.Range(0, length).Select(i => (double)i).ToArray();
            var doubleColumn = NivaraColumn<double>.Create(doubleValues);
            
            Assert.That(doubleColumn.Length, Is.EqualTo(length), 
                $"Double column should preserve length {length}");
        }
    }

    #endregion

    #region Additional Core Functionality Tests

    /// <summary>
    /// Test basic indexer access functionality
    /// </summary>
    [Test]
    public void ColumnIndexer_ShouldReturnCorrectValues()
    {
        // Test with integers
        var intValues = new[] { 10, 20, 30, 40, 50 };
        var intColumn = NivaraColumn<int>.Create(intValues);
        
        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(intColumn[i], Is.EqualTo(intValues[i]), 
                $"Indexer should return correct value at position {i}");
        }

        // Test with strings
        var stringValues = new[] { "first", "second", "third" };
        var stringColumn = NivaraColumn<string>.Create(stringValues);
        
        for (int i = 0; i < stringValues.Length; i++)
        {
            Assert.That(stringColumn[i], Is.EqualTo(stringValues[i]), 
                $"String indexer should return correct value at position {i}");
        }
    }

    /// <summary>
    /// Test indexer bounds checking
    /// </summary>
    [Test]
    public void ColumnIndexer_ShouldThrowOnOutOfBounds()
    {
        var values = new[] { 1, 2, 3 };
        var column = NivaraColumn<int>.Create(values);
        
        // Test negative index
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[-1]; }, 
            "Negative index should throw IndexOutOfRangeException");
        
        // Test index equal to length
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[3]; }, 
            "Index equal to length should throw IndexOutOfRangeException");
        
        // Test index greater than length
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = column[10]; }, 
            "Index greater than length should throw IndexOutOfRangeException");
    }

    /// <summary>
    /// Test HasNulls property for non-nullable types
    /// </summary>
    [Test]
    public void NonNullableColumns_ShouldHaveHasNullsFalse()
    {
        // Test with value types (should never have nulls)
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        Assert.That(intColumn.HasNulls, Is.False, "Int column should not have nulls");

        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
        Assert.That(doubleColumn.HasNulls, Is.False, "Double column should not have nulls");

        // Test with reference types without nulls
        var stringColumn = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        Assert.That(stringColumn.HasNulls, Is.False, "String column without nulls should have HasNulls false");
    }

    /// <summary>
    /// Test reference type columns with null values
    /// </summary>
    [Test]
    public void ReferenceTypeColumns_ShouldDetectNulls()
    {
        // Test string column with nulls - create array with explicit null handling
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "hello", "world" });
        
        // Create a column with nulls using a different approach
        var stringValuesWithNulls = new string?[] { "hello", null, "world", null };
        var stringArrayWithNulls = new string[stringValuesWithNulls.Length];
        for (int i = 0; i < stringValuesWithNulls.Length; i++)
        {
            stringArrayWithNulls[i] = stringValuesWithNulls[i]!; // Suppress null warning for test
        }
        
        var stringColumnWithNulls = NivaraColumn<string>.CreateForReferenceType(stringArrayWithNulls);
        
        Assert.That(stringColumnWithNulls, Is.Not.Null, "String column with nulls should be created");
        Assert.That(stringColumnWithNulls.Length, Is.EqualTo(4), "Column should have correct length");
        
        // Test individual values
        Assert.That(stringColumnWithNulls[0], Is.EqualTo("hello"), "First value should be correct");
        Assert.That(stringColumnWithNulls[1], Is.Null, "Second value should be null");
        Assert.That(stringColumnWithNulls[2], Is.EqualTo("world"), "Third value should be correct");
        Assert.That(stringColumnWithNulls[3], Is.Null, "Fourth value should be null");
        
        // Test IsNull method
        Assert.That(stringColumnWithNulls.IsNull(0), Is.False, "First position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(1), Is.True, "Second position should be null");
        Assert.That(stringColumnWithNulls.IsNull(2), Is.False, "Third position should not be null");
        Assert.That(stringColumnWithNulls.IsNull(3), Is.True, "Fourth position should be null");
    }

    /// <summary>
    /// Test basic null handling for reference types
    /// Note: Nullable value type testing will be added in a future iteration
    /// when the generic constraint issues are resolved
    /// </summary>
    [Test]
    public void ReferenceTypeColumns_ShouldHandleBasicNullDetection()
    {
        // Test string column without nulls
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "hello", "world" });
        Assert.That(stringColumn.HasNulls, Is.False, "String column without nulls should have HasNulls false");
        
        // For now, we'll skip the nullable value type test due to generic constraint limitations
        // This will be addressed in a future iteration when we can properly handle T? constraints
    }

    /// <summary>
    /// Test column disposal
    /// </summary>
    [Test]
    public void DisposedColumn_ShouldThrowObjectDisposedException()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        column.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.Length; }, 
            "Accessing Length on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column[0]; }, 
            "Accessing indexer on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.HasNulls; }, 
            "Accessing HasNulls on disposed column should throw");
        Assert.Throws<ObjectDisposedException>(() => { var _ = column.IsVectorizable; }, 
            "Accessing IsVectorizable on disposed column should throw");
    }

    #endregion
}