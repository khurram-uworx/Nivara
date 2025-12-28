using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraSeries&lt;T&gt; creation and functionality
/// </summary>
[TestFixture]
public class NivaraSeriesTests
{
    #region Property 10: Series creation with optional index
    
    /// <summary>
    /// Property 10: Series creation with optional index
    /// For any value array and any optional index array, creating a NivaraSeries should accept both 
    /// and maintain the correct value-index relationships.
    /// **Validates: Requirements 4.1**
    /// </summary>
    [Test]
    public void SeriesCreation_WithOptionalIndex_ShouldMaintainValueIndexRelationships()
    {
        // Test series creation without explicit index (should use default integer index)
        var values1 = new[] { 10, 20, 30 };
        var series1 = NivaraSeries<int>.Create(values1);
        
        Assert.That(series1, Is.Not.Null, "Series should be created without explicit index");
        Assert.That(series1.Length, Is.EqualTo(values1.Length), "Series should preserve length");
        Assert.That(series1[0], Is.EqualTo(10), "First value should be accessible by position");
        Assert.That(series1[1], Is.EqualTo(20), "Second value should be accessible by position");
        Assert.That(series1[2], Is.EqualTo(30), "Third value should be accessible by position");
        
        // Verify default integer index
        Assert.That(series1.GetLabel(0), Is.EqualTo(0), "Default index should be integer positions");
        Assert.That(series1.GetLabel(1), Is.EqualTo(1), "Default index should be integer positions");
        Assert.That(series1.GetLabel(2), Is.EqualTo(2), "Default index should be integer positions");

        // Test series creation with explicit string index
        var values2 = new[] { "apple", "banana", "cherry" };
        var index2 = new[] { "a", "b", "c" };
        var series2 = NivaraSeries<string>.Create(values2, index2);
        
        Assert.That(series2, Is.Not.Null, "Series should be created with explicit index");
        Assert.That(series2.Length, Is.EqualTo(values2.Length), "Series should preserve length");
        Assert.That(series2[0], Is.EqualTo("apple"), "First value should be accessible by position");
        Assert.That(series2["a"], Is.EqualTo("apple"), "First value should be accessible by label");
        Assert.That(series2["b"], Is.EqualTo("banana"), "Second value should be accessible by label");
        Assert.That(series2["c"], Is.EqualTo("cherry"), "Third value should be accessible by label");

        // Test series creation with mixed object index
        var values3 = new[] { 1.0, 2.0, 3.0 };
        var index3 = new object[] { "first", 42, DateTime.Today };
        var series3 = NivaraSeries<double>.Create(values3, index3);
        
        Assert.That(series3, Is.Not.Null, "Series should be created with mixed object index");
        Assert.That(series3.Length, Is.EqualTo(values3.Length), "Series should preserve length");
        Assert.That(series3["first"], Is.EqualTo(1.0), "Value should be accessible by string label");
        Assert.That(series3[(object)42], Is.EqualTo(2.0), "Value should be accessible by integer label");
        Assert.That(series3[DateTime.Today], Is.EqualTo(3.0), "Value should be accessible by DateTime label");
    }

    [Test]
    public void SeriesCreation_WithMismatchedIndexLength_ShouldThrowException()
    {
        var values = new[] { 1, 2, 3 };
        var index = new[] { "a", "b" }; // Different length
        
        Assert.Throws<ArgumentException>(() => NivaraSeries<int>.Create(values, index),
            "Should throw when index length doesn't match values length");
    }

    [Test]
    public void SeriesCreation_WithNullValues_ShouldThrowException()
    {
        Assert.Throws<ArgumentNullException>(() => NivaraSeries<int>.Create(null!),
            "Should throw when values array is null");
    }

    #endregion

    #region Property 11: Series label-based access
    
    /// <summary>
    /// Property 11: Series label-based access
    /// For any NivaraSeries and any label, accessing by label should return the corresponding value 
    /// if the label exists, or appropriate null/default if not found.
    /// **Validates: Requirements 4.2**
    /// </summary>
    [Test]
    public void SeriesLabelAccess_WithValidLabels_ShouldReturnCorrectValues()
    {
        var values = new[] { 100, 200, 300, 400 };
        var index = new[] { "first", "second", "third", "fourth" };
        var series = NivaraSeries<int>.Create(values, index);
        
        // Test direct label access
        Assert.That(series["first"], Is.EqualTo(100), "Should return correct value for first label");
        Assert.That(series["second"], Is.EqualTo(200), "Should return correct value for second label");
        Assert.That(series["third"], Is.EqualTo(300), "Should return correct value for third label");
        Assert.That(series["fourth"], Is.EqualTo(400), "Should return correct value for fourth label");
        
        // Test GetByLabel method
        Assert.That(series.GetByLabel("first"), Is.EqualTo(100), "GetByLabel should return correct value");
        Assert.That(series.GetByLabel("third"), Is.EqualTo(300), "GetByLabel should return correct value");
        
        // Test TryGetByLabel with existing labels
        Assert.That(series.TryGetByLabel("second", out var value1), Is.True, "TryGetByLabel should succeed for existing label");
        Assert.That(value1, Is.EqualTo(200), "TryGetByLabel should return correct value");
        
        // Test ContainsLabel
        Assert.That(series.ContainsLabel("first"), Is.True, "Should contain existing label");
        Assert.That(series.ContainsLabel("third"), Is.True, "Should contain existing label");
    }

    [Test]
    public void SeriesLabelAccess_WithInvalidLabels_ShouldHandleAppropriately()
    {
        var values = new[] { 100, 200, 300 };
        var index = new[] { "a", "b", "c" };
        var series = NivaraSeries<int>.Create(values, index);
        
        // Test direct access with non-existent label should throw
        Assert.Throws<KeyNotFoundException>(() => { var _ = series["nonexistent"]; },
            "Should throw KeyNotFoundException for non-existent label");
        
        // Test GetByLabel with non-existent label should throw
        Assert.Throws<KeyNotFoundException>(() => series.GetByLabel("missing"),
            "GetByLabel should throw KeyNotFoundException for non-existent label");
        
        // Test TryGetByLabel with non-existent label should return false
        Assert.That(series.TryGetByLabel("missing", out var value), Is.False, 
            "TryGetByLabel should return false for non-existent label");
        Assert.That(value, Is.EqualTo(default(int)), "TryGetByLabel should return default value when not found");
        
        // Test ContainsLabel with non-existent label
        Assert.That(series.ContainsLabel("missing"), Is.False, "Should not contain non-existent label");
    }

    [Test]
    public void SeriesLabelAccess_WithDifferentLabelTypes_ShouldWork()
    {
        var values = new[] { "value1", "value2", "value3" };
        var index = new object[] { 1, "string", DateTime.Today };
        var series = NivaraSeries<string>.Create(values, index);
        
        Assert.That(series.GetByLabel(1), Is.EqualTo("value1"), "Should access by integer label using GetByLabel");
        Assert.That(series.GetByLabel("string"), Is.EqualTo("value2"), "Should access by string label using GetByLabel");
        Assert.That(series.GetByLabel(DateTime.Today), Is.EqualTo("value3"), "Should access by DateTime label using GetByLabel");
        
        // Test using indexer with explicit object casting to avoid ambiguity
        Assert.That(series[(object)1], Is.EqualTo("value1"), "Should access by integer label using indexer");
        Assert.That(series[(object)"string"], Is.EqualTo("value2"), "Should access by string label using indexer");
        Assert.That(series[(object)DateTime.Today], Is.EqualTo("value3"), "Should access by DateTime label using indexer");
    }

    #endregion

    #region Property 12: Default integer indexing
    
    /// <summary>
    /// Property 12: Default integer indexing
    /// For any NivaraSeries created without explicit index, the series should use integer positions 
    /// (0, 1, 2, ...) as the default index.
    /// **Validates: Requirements 4.3**
    /// </summary>
    [Test]
    public void SeriesDefaultIndexing_ShouldUseIntegerPositions()
    {
        // Test with different value types
        var intValues = new[] { 10, 20, 30, 40, 50 };
        var intSeries = NivaraSeries<int>.Create(intValues);
        
        // Verify default integer indexing
        for (int i = 0; i < intValues.Length; i++)
        {
            Assert.That(intSeries.GetLabel(i), Is.EqualTo(i), 
                $"Default index at position {i} should be {i}");
            Assert.That(intSeries[i], Is.EqualTo(intValues[i]), 
                $"Value at position {i} should be accessible by integer index");
        }
        
        // Test with string values
        var stringValues = new[] { "a", "b", "c" };
        var stringSeries = NivaraSeries<string>.Create(stringValues);
        
        for (int i = 0; i < stringValues.Length; i++)
        {
            Assert.That(stringSeries.GetLabel(i), Is.EqualTo(i), 
                $"Default index at position {i} should be {i}");
            Assert.That(stringSeries[i], Is.EqualTo(stringValues[i]), 
                $"Value at position {i} should be accessible by integer index");
        }
        
        // Test accessing by default integer labels
        Assert.That(intSeries[0], Is.EqualTo(10), "Should access first element by label 0");
        Assert.That(intSeries[1], Is.EqualTo(20), "Should access second element by label 1");
        Assert.That(intSeries[4], Is.EqualTo(50), "Should access last element by label 4");
    }

    [Test]
    public void SeriesDefaultIndexing_EmptySeries_ShouldWork()
    {
        var emptySeries = NivaraSeries<int>.Create(Array.Empty<int>());
        
        Assert.That(emptySeries.Length, Is.EqualTo(0), "Empty series should have length 0");
        Assert.That(emptySeries.Index.Length, Is.EqualTo(0), "Empty series index should have length 0");
    }

    [Test]
    public void SeriesDefaultIndexing_SingleElement_ShouldWork()
    {
        var singleSeries = NivaraSeries<string>.Create(new[] { "single" });
        
        Assert.That(singleSeries.Length, Is.EqualTo(1), "Single element series should have length 1");
        Assert.That(singleSeries.GetLabel(0), Is.EqualTo(0), "Single element should have index 0");
        Assert.That(singleSeries[0], Is.EqualTo("single"), "Should access single element by position");
        Assert.That(singleSeries[0], Is.EqualTo("single"), "Should access single element by label 0");
    }

    #endregion

    #region Additional Series Functionality Tests

    [Test]
    public void SeriesSlicing_ShouldPreserveIndexValueRelationships()
    {
        var values = new[] { 10, 20, 30, 40, 50 };
        var index = new[] { "a", "b", "c", "d", "e" };
        var series = NivaraSeries<int>.Create(values, index);
        
        // Test slicing
        var sliced = series.Slice(1, 3); // Should get elements at positions 1, 2, 3
        
        Assert.That(sliced.Length, Is.EqualTo(3), "Sliced series should have correct length");
        Assert.That(sliced[0], Is.EqualTo(20), "First element of slice should be correct");
        Assert.That(sliced[1], Is.EqualTo(30), "Second element of slice should be correct");
        Assert.That(sliced[2], Is.EqualTo(40), "Third element of slice should be correct");
        
        // Verify index relationships are preserved
        Assert.That(sliced.GetLabel(0), Is.EqualTo("b"), "First label of slice should be correct");
        Assert.That(sliced.GetLabel(1), Is.EqualTo("c"), "Second label of slice should be correct");
        Assert.That(sliced.GetLabel(2), Is.EqualTo("d"), "Third label of slice should be correct");
        
        // Test label-based access on sliced series
        Assert.That(sliced["b"], Is.EqualTo(20), "Should access sliced element by original label");
        Assert.That(sliced["c"], Is.EqualTo(30), "Should access sliced element by original label");
        Assert.That(sliced["d"], Is.EqualTo(40), "Should access sliced element by original label");
    }

    [Test]
    public void SeriesNullHandling_ShouldWorkCorrectly()
    {
        // Create series with null values (using reference type)
        var values = new[] { "apple", null!, "cherry" };
        var index = new[] { "a", "b", "c" };
        var series = NivaraSeries<string>.Create(values, index);
        
        Assert.That(series.HasNulls, Is.True, "Series should report having nulls");
        Assert.That(series.IsNull(1), Is.True, "Second element should be null");
        Assert.That(series.IsNull(0), Is.False, "First element should not be null");
        Assert.That(series.IsNull(2), Is.False, "Third element should not be null");
        
        // Test accessing null value by position and label
        Assert.That(series[1], Is.Null, "Should return null for null element by position");
        Assert.That(series["b"], Is.Null, "Should return null for null element by label");
    }

    [Test]
    public void SeriesDisposal_ShouldDisposeUnderlyingColumns()
    {
        var values = new[] { 1, 2, 3 };
        var series = NivaraSeries<int>.Create(values);
        
        // Verify series works before disposal
        Assert.That(series.Length, Is.EqualTo(3), "Series should work before disposal");
        
        // Dispose the series
        series.Dispose();
        
        // Verify operations throw after disposal
        Assert.Throws<ObjectDisposedException>(() => { var _ = series.Length; },
            "Should throw ObjectDisposedException after disposal");
        Assert.Throws<ObjectDisposedException>(() => { var _ = series[0]; },
            "Should throw ObjectDisposedException after disposal");
        Assert.Throws<ObjectDisposedException>(() => { var _ = series["test"]; },
            "Should throw ObjectDisposedException after disposal");
    }

    [Test]
    public void SeriesIndexBounds_ShouldThrowForInvalidPositions()
    {
        var series = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = series[-1]; },
            "Should throw for negative index");
        Assert.Throws<IndexOutOfRangeException>(() => { var _ = series[3]; },
            "Should throw for index beyond length");
        Assert.Throws<IndexOutOfRangeException>(() => series.IsNull(-1),
            "IsNull should throw for negative index");
        Assert.Throws<IndexOutOfRangeException>(() => series.IsNull(3),
            "IsNull should throw for index beyond length");
    }

    #endregion
}