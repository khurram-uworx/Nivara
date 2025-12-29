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

    #region Property 13: Series alignment

    /// <summary>
    /// Property 13: Series alignment
    /// For any two NivaraSeries with different indices, alignment operations should correctly 
    /// match values based on index labels and handle missing indices appropriately.
    /// **Validates: Requirements 4.4**
    /// </summary>
    [Test]
    public void SeriesAlignment_WithDifferentIndices_ShouldMatchCorrectly()
    {
        // Test basic alignment with some matching indices
        var values1 = new[] { 10, 20, 30, 40 };
        var index1 = new[] { "a", "b", "c", "d" };
        var series1 = NivaraSeries<int>.Create(values1, index1);

        var values2 = new[] { 100, 200, 300 };
        var index2 = new[] { "b", "d", "e" }; // "b" and "d" match, "e" doesn't
        var series2 = NivaraSeries<int>.Create(values2, index2);

        // Test Align method (returns aligned version of first series)
        var aligned = series1.Align(series2);

        Assert.That(aligned.Length, Is.EqualTo(2), "Aligned series should contain only matching indices");
        Assert.That(aligned["b"], Is.EqualTo(20), "Value for 'b' should come from first series");
        Assert.That(aligned["d"], Is.EqualTo(40), "Value for 'd' should come from first series");
        Assert.That(aligned.ContainsLabel("a"), Is.False, "Non-matching index 'a' should not be in aligned series");
        Assert.That(aligned.ContainsLabel("c"), Is.False, "Non-matching index 'c' should not be in aligned series");
        Assert.That(aligned.ContainsLabel("e"), Is.False, "Non-matching index 'e' should not be in aligned series");

        // Test AlignBoth method (returns aligned versions of both series)
        var (alignedLeft, alignedRight) = series1.AlignBoth(series2);

        Assert.That(alignedLeft.Length, Is.EqualTo(2), "Left aligned series should contain only matching indices");
        Assert.That(alignedRight.Length, Is.EqualTo(2), "Right aligned series should contain only matching indices");

        // Both should have same index labels in same order
        Assert.That(alignedLeft.GetLabel(0), Is.EqualTo(alignedRight.GetLabel(0)), "Both series should have same index labels");
        Assert.That(alignedLeft.GetLabel(1), Is.EqualTo(alignedRight.GetLabel(1)), "Both series should have same index labels");

        // Values should come from respective original series
        Assert.That(alignedLeft["b"], Is.EqualTo(20), "Left aligned value for 'b' should come from first series");
        Assert.That(alignedRight["b"], Is.EqualTo(100), "Right aligned value for 'b' should come from second series");
        Assert.That(alignedLeft["d"], Is.EqualTo(40), "Left aligned value for 'd' should come from first series");
        Assert.That(alignedRight["d"], Is.EqualTo(200), "Right aligned value for 'd' should come from second series");
    }

    [Test]
    public void SeriesAlignment_WithNoMatchingIndices_ShouldReturnEmptySeries()
    {
        var values1 = new[] { 10, 20, 30 };
        var index1 = new[] { "a", "b", "c" };
        var series1 = NivaraSeries<int>.Create(values1, index1);

        var values2 = new[] { 100, 200 };
        var index2 = new[] { "x", "y" }; // No matching indices
        var series2 = NivaraSeries<int>.Create(values2, index2);

        var aligned = series1.Align(series2);
        Assert.That(aligned.Length, Is.EqualTo(0), "Alignment with no matching indices should return empty series");

        var (alignedLeft, alignedRight) = series1.AlignBoth(series2);
        Assert.That(alignedLeft.Length, Is.EqualTo(0), "Left aligned series should be empty when no indices match");
        Assert.That(alignedRight.Length, Is.EqualTo(0), "Right aligned series should be empty when no indices match");
    }

    [Test]
    public void SeriesAlignment_WithIdenticalIndices_ShouldPreserveAllElements()
    {
        var values1 = new[] { 10, 20, 30 };
        var index1 = new[] { "a", "b", "c" };
        var series1 = NivaraSeries<int>.Create(values1, index1);

        var values2 = new[] { 100, 200, 300 };
        var index2 = new[] { "a", "b", "c" }; // Identical indices
        var series2 = NivaraSeries<int>.Create(values2, index2);

        var aligned = series1.Align(series2);
        Assert.That(aligned.Length, Is.EqualTo(3), "Alignment with identical indices should preserve all elements");
        Assert.That(aligned["a"], Is.EqualTo(10), "All values should be preserved");
        Assert.That(aligned["b"], Is.EqualTo(20), "All values should be preserved");
        Assert.That(aligned["c"], Is.EqualTo(30), "All values should be preserved");
    }

    [Test]
    public void SeriesAlignment_WithDifferentIndexTypes_ShouldWork()
    {
        var values1 = new[] { 10, 20, 30 };
        var index1 = new object[] { 1, "b", DateTime.Today };
        var series1 = NivaraSeries<int>.Create(values1, index1);

        var values2 = new[] { 100, 200 };
        var index2 = new object[] { "b", DateTime.Today }; // Mixed types, some matching
        var series2 = NivaraSeries<int>.Create(values2, index2);

        var aligned = series1.Align(series2);
        Assert.That(aligned.Length, Is.EqualTo(2), "Should align based on object equality regardless of type");
        Assert.That(aligned["b"], Is.EqualTo(20), "String index should match");
        Assert.That(aligned[DateTime.Today], Is.EqualTo(30), "DateTime index should match");
    }

    #endregion

    #region Property 14: Index-value relationship preservation

    /// <summary>
    /// Property 14: Index-value relationship preservation
    /// For any NivaraSeries operation, the relationship between index labels and their 
    /// corresponding values should be maintained in the result.
    /// **Validates: Requirements 4.5**
    /// </summary>
    [Test]
    public void SeriesOperations_ShouldPreserveIndexValueRelationships()
    {
        // Test scalar multiplication preserves relationships
        var values = new[] { 10, 20, 30 };
        var index = new[] { "first", "second", "third" };
        var series = NivaraSeries<int>.Create(values, index);

        var multiplied = series.Multiply(2);

        Assert.That(multiplied.Length, Is.EqualTo(3), "Scalar operation should preserve length");
        Assert.That(multiplied["first"], Is.EqualTo(20), "Index-value relationship should be preserved");
        Assert.That(multiplied["second"], Is.EqualTo(40), "Index-value relationship should be preserved");
        Assert.That(multiplied["third"], Is.EqualTo(60), "Index-value relationship should be preserved");

        // Verify original series is unchanged (immutability)
        Assert.That(series["first"], Is.EqualTo(10), "Original series should be unchanged");
        Assert.That(series["second"], Is.EqualTo(20), "Original series should be unchanged");
        Assert.That(series["third"], Is.EqualTo(30), "Original series should be unchanged");

        // Test element-wise operations preserve relationships
        var values2 = new[] { 1, 2, 3 };
        var index2 = new[] { "first", "second", "third" }; // Same indices
        var series2 = NivaraSeries<int>.Create(values2, index2);

        var added = series.Add(series2);

        Assert.That(added.Length, Is.EqualTo(3), "Element-wise operation should preserve matching indices");
        Assert.That(added["first"], Is.EqualTo(11), "Addition should preserve index-value relationships");
        Assert.That(added["second"], Is.EqualTo(22), "Addition should preserve index-value relationships");
        Assert.That(added["third"], Is.EqualTo(33), "Addition should preserve index-value relationships");

        var multipliedSeries = series.Multiply(series2);

        Assert.That(multipliedSeries.Length, Is.EqualTo(3), "Element-wise operation should preserve matching indices");
        Assert.That(multipliedSeries["first"], Is.EqualTo(10), "Multiplication should preserve index-value relationships");
        Assert.That(multipliedSeries["second"], Is.EqualTo(40), "Multiplication should preserve index-value relationships");
        Assert.That(multipliedSeries["third"], Is.EqualTo(90), "Multiplication should preserve index-value relationships");
    }

    [Test]
    public void SeriesOperations_WithPartialAlignment_ShouldPreserveOnlyMatchingRelationships()
    {
        var values1 = new[] { 10, 20, 30, 40 };
        var index1 = new[] { "a", "b", "c", "d" };
        var series1 = NivaraSeries<int>.Create(values1, index1);

        var values2 = new[] { 1, 2 };
        var index2 = new[] { "b", "d" }; // Only partial overlap
        var series2 = NivaraSeries<int>.Create(values2, index2);

        var added = series1.Add(series2);

        Assert.That(added.Length, Is.EqualTo(2), "Result should contain only matching indices");
        Assert.That(added["b"], Is.EqualTo(21), "Index-value relationship should be preserved for 'b'");
        Assert.That(added["d"], Is.EqualTo(42), "Index-value relationship should be preserved for 'd'");
        Assert.That(added.ContainsLabel("a"), Is.False, "Non-matching indices should not be in result");
        Assert.That(added.ContainsLabel("c"), Is.False, "Non-matching indices should not be in result");
    }

    [Test]
    public void SeriesSlicing_ShouldPreserveIndexValueRelationships()
    {
        var values = new[] { 10, 20, 30, 40, 50 };
        var index = new[] { "a", "b", "c", "d", "e" };
        var series = NivaraSeries<int>.Create(values, index);

        var sliced = series.Slice(1, 3); // Get elements at positions 1, 2, 3

        Assert.That(sliced.Length, Is.EqualTo(3), "Sliced series should have correct length");
        Assert.That(sliced[0], Is.EqualTo(20), "Position-based access should work on sliced series");
        Assert.That(sliced[1], Is.EqualTo(30), "Position-based access should work on sliced series");
        Assert.That(sliced[2], Is.EqualTo(40), "Position-based access should work on sliced series");

        // Verify index-value relationships are preserved
        Assert.That(sliced["b"], Is.EqualTo(20), "Index-value relationship should be preserved in slice");
        Assert.That(sliced["c"], Is.EqualTo(30), "Index-value relationship should be preserved in slice");
        Assert.That(sliced["d"], Is.EqualTo(40), "Index-value relationship should be preserved in slice");

        // Verify original indices not in slice are not accessible
        Assert.That(sliced.ContainsLabel("a"), Is.False, "Indices not in slice should not be accessible");
        Assert.That(sliced.ContainsLabel("e"), Is.False, "Indices not in slice should not be accessible");
    }

    [Test]
    public void SeriesOperations_WithNullValues_ShouldPreserveRelationships()
    {
        // Test with nullable reference type
        var values = new[] { "apple", null!, "cherry" };
        var index = new[] { "a", "b", "c" };
        var series = NivaraSeries<string>.Create(values, index);

        var sliced = series.Slice(0, 3);

        Assert.That(sliced["a"], Is.EqualTo("apple"), "Non-null value relationship should be preserved");
        Assert.That(sliced["b"], Is.Null, "Null value relationship should be preserved");
        Assert.That(sliced["c"], Is.EqualTo("cherry"), "Non-null value relationship should be preserved");
        Assert.That(sliced.IsNull(1), Is.True, "Null status should be preserved");
    }

    #endregion

    #region Additional Series Functionality Tests

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