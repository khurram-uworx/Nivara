using NUnit.Framework;
using Nivara;

namespace Nivara.Tests;

/// <summary>
/// Tests for TensorStorage implementation covering null mask tracking, length preservation, and indexer access
/// </summary>
[TestFixture]
public class TensorStorageTests
{
    #region Property 3: Null mask tracking
    
    /// <summary>
    /// Feature: core-column-types, Property 3: Null mask tracking
    /// For any array containing null values, creating a TensorStorage should maintain a null mask that correctly tracks all null positions.
    /// Validates: Requirements 1.3
    /// </summary>
    [TestCase(new int[] { 1, 2, 3 }, new bool[] { false, false, false })]
    [TestCase(new int[] { 1, 0, 3 }, new bool[] { false, true, false })] // 0 treated as null for testing
    [TestCase(new int[] { 0, 0, 0 }, new bool[] { true, true, true })]
    [TestCase(new int[] { 1 }, new bool[] { false })]
    [TestCase(new int[] { 0 }, new bool[] { true })]
    public void TensorStorage_WithNullMask_TracksNullPositionsCorrectly(int[] values, bool[] expectedNullMask)
    {
        // Create storage with explicit null mask (simulating null tracking)
        var storage = new TensorStorage<int>(values, expectedNullMask);
        
        Assert.That(storage.HasNulls, Is.True, "Storage should indicate it has nulls when null mask is provided");
        Assert.That(storage.NullMask.Length, Is.EqualTo(expectedNullMask.Length), "Null mask length should match data length");
        
        for (int i = 0; i < expectedNullMask.Length; i++)
        {
            Assert.That(storage.NullMask[i], Is.EqualTo(expectedNullMask[i]), 
                $"Null mask at position {i} should be {expectedNullMask[i]}");
        }
    }
    
    [TestCase(new double[] { 1.0, 2.0, 3.0 })]
    [TestCase(new float[] { 1.0f, 2.0f, 3.0f })]
    [TestCase(new bool[] { true, false, true })]
    public void TensorStorage_WithoutNulls_HasEmptyNullMask<T>(T[] values) where T : unmanaged
    {
        var storage = new TensorStorage<T>(values);
        
        Assert.That(storage.HasNulls, Is.False, "Storage should not indicate nulls when no null mask provided");
        Assert.That(storage.NullMask.Length, Is.EqualTo(0), "Null mask should be empty when no nulls present");
    }
    
    #endregion
    
    #region Property 4: Length preservation
    
    /// <summary>
    /// Feature: core-column-types, Property 4: Length preservation
    /// For any input array, the resulting TensorStorage should have a Length property that exactly matches the input array length.
    /// Validates: Requirements 1.4
    /// </summary>
    [TestCase(new int[] { })]
    [TestCase(new int[] { 1 })]
    [TestCase(new int[] { 1, 2 })]
    [TestCase(new int[] { 1, 2, 3, 4, 5 })]
    [TestCase(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })]
    public void TensorStorage_PreservesInputLength(int[] values)
    {
        var storage = new TensorStorage<int>(values);
        
        Assert.That(storage.Length, Is.EqualTo(values.Length), 
            $"Storage length should match input array length of {values.Length}");
    }
    
    [TestCase(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9 })]
    [TestCase(new float[] { 1.0f, 2.0f, 3.0f, 4.0f })]
    [TestCase(new bool[] { true, false, true, false, true, false })]
    public void TensorStorage_PreservesInputLength_DifferentTypes<T>(T[] values) where T : unmanaged
    {
        var storage = new TensorStorage<T>(values);
        
        Assert.That(storage.Length, Is.EqualTo(values.Length), 
            $"Storage length should match input array length of {values.Length} for type {typeof(T).Name}");
    }
    
    #endregion
    
    #region Property 5: Indexer access with null semantics
    
    /// <summary>
    /// Feature: core-column-types, Property 5: Indexer access with null semantics
    /// For any TensorStorage and any valid index, accessing the value should return the correct value while preserving null semantics.
    /// Validates: Requirements 1.5
    /// </summary>
    [TestCase(new int[] { 10, 20, 30, 40, 50 }, 0, 10)]
    [TestCase(new int[] { 10, 20, 30, 40, 50 }, 2, 30)]
    [TestCase(new int[] { 10, 20, 30, 40, 50 }, 4, 50)]
    [TestCase(new int[] { -1, 0, 1 }, 0, -1)]
    [TestCase(new int[] { -1, 0, 1 }, 1, 0)]
    [TestCase(new int[] { -1, 0, 1 }, 2, 1)]
    public void TensorStorage_IndexerAccess_ReturnsCorrectValues(int[] values, int index, int expectedValue)
    {
        var storage = new TensorStorage<int>(values);
        
        Assert.That(storage[index], Is.EqualTo(expectedValue), 
            $"Value at index {index} should be {expectedValue}");
    }
    
    [TestCase(new double[] { 1.1, 2.2, 3.3 }, 1, 2.2)]
    [TestCase(new float[] { 1.0f, 2.0f }, 0, 1.0f)]
    [TestCase(new bool[] { true, false, true }, 2, true)]
    public void TensorStorage_IndexerAccess_DifferentTypes<T>(T[] values, int index, T expectedValue) where T : unmanaged
    {
        var storage = new TensorStorage<T>(values);
        
        Assert.That(storage[index], Is.EqualTo(expectedValue), 
            $"Value at index {index} should be {expectedValue} for type {typeof(T).Name}");
    }
    
    [TestCase(new int[] { 1, 2, 3 }, -1)]
    [TestCase(new int[] { 1, 2, 3 }, 3)]
    [TestCase(new int[] { 1, 2, 3 }, 10)]
    [TestCase(new int[] { }, 0)]
    public void TensorStorage_IndexerAccess_ThrowsForInvalidIndex(int[] values, int invalidIndex)
    {
        var storage = new TensorStorage<int>(values);
        
        Assert.Throws<IndexOutOfRangeException>(() => _ = storage[invalidIndex], 
            $"Accessing index {invalidIndex} should throw IndexOutOfRangeException");
    }
    
    #endregion
    
    #region Additional Tests for TensorStorage Specific Functionality
    
    [Test]
    public void TensorStorage_IsVectorizable_ReturnsTrue()
    {
        var storage = new TensorStorage<int>(new[] { 1, 2, 3 });
        
        Assert.That(storage.IsVectorizable, Is.True, "TensorStorage should always be vectorizable");
    }
    
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 1, 3)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 0, 2)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 2, 3)]
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 0, 5)]
    public void TensorStorage_Slice_ReturnsCorrectSubset(int[] values, int start, int length)
    {
        var storage = new TensorStorage<int>(values);
        
        var sliced = storage.Slice(start, length);
        
        Assert.That(sliced.Length, Is.EqualTo(length), "Sliced storage should have correct length");
        
        for (int i = 0; i < length; i++)
        {
            Assert.That(sliced[i], Is.EqualTo(values[start + i]), 
                $"Sliced value at index {i} should match original value at index {start + i}");
        }
    }
    
    [TestCase(new int[] { 1, 2, 3 }, -1, 1)]
    [TestCase(new int[] { 1, 2, 3 }, 0, -1)]
    [TestCase(new int[] { 1, 2, 3 }, 2, 3)]
    [TestCase(new int[] { 1, 2, 3 }, 4, 1)]
    public void TensorStorage_Slice_ThrowsForInvalidParameters(int[] values, int start, int length)
    {
        var storage = new TensorStorage<int>(values);
        
        Assert.Throws<ArgumentOutOfRangeException>(() => storage.Slice(start, length), 
            $"Slice with start={start}, length={length} should throw ArgumentOutOfRangeException");
    }
    
    [Test]
    public void TensorStorage_Dispose_PreventsAccess()
    {
        var storage = new TensorStorage<int>(new[] { 1, 2, 3 });
        
        storage.Dispose();
        
        Assert.Throws<ObjectDisposedException>(() => _ = storage[0], 
            "Accessing disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => _ = storage.NullMask, 
            "Accessing null mask of disposed storage should throw ObjectDisposedException");
        Assert.Throws<ObjectDisposedException>(() => storage.Slice(0, 1), 
            "Slicing disposed storage should throw ObjectDisposedException");
    }
    
    #endregion
}