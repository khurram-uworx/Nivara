using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraSeries aggregate functions (Sum, Average, Min, Max).
/// Covers vectorized operations, null handling, and type support validation.
/// </summary>
[TestFixture]
public class NivaraSeriesAggregateTests
{
    #region Sum Tests

    /// <summary>
    /// Feature: nivara-series, Property: Sum computation
    /// For any series of numeric values, Sum should return the correct arithmetic sum.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 15)]
    [TestCase(new int[] { -1, -2, -3 }, -6)]
    [TestCase(new int[] { 0, 0, 0 }, 0)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, -1 }, int.MaxValue - 1)]
    [Category("Feature: nivara-series, Property: Sum computation")]
    public void Sum_IntegerValues_ReturnsCorrectSum(int[] values, int expected)
    {
        // Arrange
        var series = NivaraSeries<int>.Create(values);

        // Act
        var result = series.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum computation with vectorization
    /// For any series of float values, Sum should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new float[] { 1.5f, 2.5f, 3.0f }, 7.0f)]
    [TestCase(new float[] { -1.1f, 1.1f }, 0.0f)]
    [TestCase(new float[] { float.MaxValue, -float.MaxValue }, 0.0f)]
    [Category("Feature: nivara-series, Property: Sum computation with vectorization")]
    public void Sum_FloatValues_ReturnsCorrectSum(float[] values, float expected)
    {
        // Arrange
        var series = NivaraSeries<float>.Create(values);

        // Act
        var result = series.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum computation with vectorization
    /// For any series of double values, Sum should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new double[] { 1.5, 2.5, 3.0 }, 7.0)]
    [TestCase(new double[] { -1.1, 1.1 }, 0.0)]
    [TestCase(new double[] { Math.PI, Math.E }, Math.PI + Math.E)]
    [Category("Feature: nivara-series, Property: Sum computation with vectorization")]
    public void Sum_DoubleValues_ReturnsCorrectSum(double[] values, double expected)
    {
        // Arrange
        var series = NivaraSeries<double>.Create(values);

        // Act
        var result = series.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum with null handling
    /// For any series with null values, Sum should compute sum of valid values only.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum with null handling")]
    public void Sum_WithNullValues_ReturnsValidSum()
    {
        // Arrange
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act
        var result = series.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(9)); // 1 + 3 + 5
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum error handling
    /// For any empty series, Sum should throw InvalidOperationException.
    /// Validates: Error handling for edge cases
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum error handling")]
    public void Sum_EmptySeries_ThrowsInvalidOperationException()
    {
        // Arrange
        var series = NivaraSeries<int>.Create(Array.Empty<int>());

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => series.Sum());
        Assert.That(ex.Message, Does.Contain("Cannot compute sum of empty series"));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum error handling
    /// For any series with all null values, Sum should throw InvalidOperationException.
    /// Validates: Error handling for null-only series
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum error handling")]
    public void Sum_AllNullValues_ThrowsInvalidOperationException()
    {
        // Arrange
        var nullableData = new int?[] { null, null, null };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => series.Sum());
        Assert.That(ex.Message, Does.Contain("all values are null"));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum type validation
    /// For any non-numeric type, Sum should throw InvalidOperationException.
    /// Validates: Type safety for aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum type validation")]
    public void Sum_NonNumericType_ThrowsInvalidOperationException()
    {
        // Arrange
        var series = NivaraSeries<string>.Create(new[] { "a", "b", "c" });

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => series.Sum());
        Assert.That(ex.Message, Does.Contain("Sum operation is not supported for type String"));
    }

    #endregion

    #region Average Tests

    /// <summary>
    /// Feature: nivara-series, Property: Average computation
    /// For any series of numeric values, Average should return the correct arithmetic mean.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 3)]
    [TestCase(new int[] { 10, 20, 30 }, 20)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { -5, 5 }, 0)]
    [Category("Feature: nivara-series, Property: Average computation")]
    public void Average_IntegerValues_ReturnsCorrectAverage(int[] values, int expected)
    {
        // Arrange
        var series = NivaraSeries<int>.Create(values);

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Average computation with vectorization
    /// For any series of float values, Average should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new float[] { 1.0f, 2.0f, 3.0f }, 2.0f)]
    [TestCase(new float[] { 1.5f, 2.5f }, 2.0f)]
    [Category("Feature: nivara-series, Property: Average computation with vectorization")]
    public void Average_FloatValues_ReturnsCorrectAverage(float[] values, float expected)
    {
        // Arrange
        var series = NivaraSeries<float>.Create(values);

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Average with null handling
    /// For any series with null values, Average should compute average of valid values only.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Average with null handling")]
    public void Average_WithNullValues_ReturnsValidAverage()
    {
        // Arrange
        var nullableData = new int?[] { 2, null, 4, null, 6 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo(4)); // (2 + 4 + 6) / 3
    }

    #endregion

    #region Min Tests

    /// <summary>
    /// Feature: nivara-series, Property: Min computation
    /// For any series of comparable values, Min should return the smallest value.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 5, 2, 8, 1, 9 }, 1)]
    [TestCase(new int[] { -1, -5, -2 }, -5)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, int.MinValue }, int.MinValue)]
    [Category("Feature: nivara-series, Property: Min computation")]
    public void Min_IntegerValues_ReturnsCorrectMin(int[] values, int expected)
    {
        // Arrange
        var series = NivaraSeries<int>.Create(values);

        // Act
        var result = series.Min();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Min computation with vectorization
    /// For any series of float values, Min should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new float[] { 3.5f, 1.2f, 4.8f }, 1.2f)]
    [TestCase(new float[] { -1.5f, -2.5f }, -2.5f)]
    [Category("Feature: nivara-series, Property: Min computation with vectorization")]
    public void Min_FloatValues_ReturnsCorrectMin(float[] values, float expected)
    {
        // Arrange
        var series = NivaraSeries<float>.Create(values);

        // Act
        var result = series.Min();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Min with string values
    /// For any series of string values, Min should return lexicographically smallest value.
    /// Validates: Comparison operations for non-numeric types
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Min with string values")]
    public void Min_StringValues_ReturnsCorrectMin()
    {
        // Arrange
        var series = NivaraSeries<string>.Create(new[] { "zebra", "apple", "banana" });

        // Act
        var result = series.Min();

        // Assert
        Assert.That(result, Is.EqualTo("apple"));
    }

    #endregion

    #region Max Tests

    /// <summary>
    /// Feature: nivara-series, Property: Max computation
    /// For any series of comparable values, Max should return the largest value.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 5, 2, 8, 1, 9 }, 9)]
    [TestCase(new int[] { -1, -5, -2 }, -1)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, int.MinValue }, int.MaxValue)]
    [Category("Feature: nivara-series, Property: Max computation")]
    public void Max_IntegerValues_ReturnsCorrectMax(int[] values, int expected)
    {
        // Arrange
        var series = NivaraSeries<int>.Create(values);

        // Act
        var result = series.Max();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Max computation with vectorization
    /// For any series of double values, Max should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new double[] { 3.5, 1.2, 4.8 }, 4.8)]
    [TestCase(new double[] { -1.5, -2.5 }, -1.5)]
    [Category("Feature: nivara-series, Property: Max computation with vectorization")]
    public void Max_DoubleValues_ReturnsCorrectMax(double[] values, double expected)
    {
        // Arrange
        var series = NivaraSeries<double>.Create(values);

        // Act
        var result = series.Max();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Max with null handling
    /// For any series with null values, Max should find maximum among valid values only.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Max with null handling")]
    public void Max_WithNullValues_ReturnsValidMax()
    {
        // Arrange
        var nullableData = new int?[] { 2, null, 8, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act
        var result = series.Max();

        // Assert
        Assert.That(result, Is.EqualTo(8));
    }

    #endregion

    #region Edge Cases and Error Handling

    /// <summary>
    /// Feature: nivara-series, Property: Aggregate error handling
    /// For any disposed series, aggregate functions should throw ObjectDisposedException.
    /// Validates: Resource management and disposal patterns
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Aggregate error handling")]
    public void AggregateFunction_DisposedSeries_ThrowsObjectDisposedException()
    {
        // Arrange
        var series = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        series.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => series.Sum());
        Assert.Throws<ObjectDisposedException>(() => series.Average());
        Assert.Throws<ObjectDisposedException>(() => series.Min());
        Assert.Throws<ObjectDisposedException>(() => series.Max());
    }

    /// <summary>
    /// Feature: nivara-series, Property: Aggregate type validation
    /// For any non-comparable type, Min/Max should throw InvalidOperationException.
    /// Validates: Type safety for comparison operations
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Aggregate type validation")]
    public void MinMax_NonComparableType_ThrowsInvalidOperationException()
    {
        // Arrange - using object type which doesn't implement IComparable
        var series = NivaraSeries<object>.Create(new object[] { new object(), new object() });

        // Act & Assert
        var minEx = Assert.Throws<InvalidOperationException>(() => series.Min());
        var maxEx = Assert.Throws<InvalidOperationException>(() => series.Max());

        Assert.That(minEx.Message, Does.Contain("Min operation is not supported"));
        Assert.That(maxEx.Message, Does.Contain("Max operation is not supported"));
    }

    #endregion
}