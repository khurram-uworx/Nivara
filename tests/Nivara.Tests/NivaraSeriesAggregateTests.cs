using Nivara.Tensors;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraSeries aggregate functions (Average, Sum, Min, Max) and the
/// column-level reductions (Sum, Min, Max) that share the numeric kernel domain.
/// Series aggregates are null-aware and vectorized via TensorPrimitives.
/// </summary>
[TestFixture]
public class NivaraSeriesAggregateTests
{
    #region Sum Tests

    /// <summary>
    /// Feature: core-column-types, Property: Sum computation
    /// For any column of numeric values, Sum should return the correct arithmetic sum.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 1, 2, 3, 4, 5 }, 15)]
    [TestCase(new int[] { -1, -2, -3 }, -6)]
    [TestCase(new int[] { 0, 0, 0 }, 0)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, -1 }, int.MaxValue - 1)]
    [Category("Feature: core-column-types, Property: Sum computation")]
    public void Sum_IntegerValues_ReturnsCorrectSum(int[] values, int expected)
    {
        // Arrange
        var column = NivaraColumn<int>.Create(values);

        // Act
        var result = column.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Sum computation with vectorization
    /// For any column of float values, Sum should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new float[] { 1.5f, 2.5f, 3.0f }, 7.0f)]
    [TestCase(new float[] { -1.1f, 1.1f }, 0.0f)]
    [TestCase(new float[] { float.MaxValue, -float.MaxValue }, 0.0f)]
    [Category("Feature: core-column-types, Property: Sum computation with vectorization")]
    public void Sum_FloatValues_ReturnsCorrectSum(float[] values, float expected)
    {
        // Arrange
        var column = NivaraColumn<float>.Create(values);

        // Act
        var result = column.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Sum computation with vectorization
    /// For any column of double values, Sum should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new double[] { 1.5, 2.5, 3.0 }, 7.0)]
    [TestCase(new double[] { -1.1, 1.1 }, 0.0)]
    [TestCase(new double[] { Math.PI, Math.E }, Math.PI + Math.E)]
    [Category("Feature: core-column-types, Property: Sum computation with vectorization")]
    public void Sum_DoubleValues_ReturnsCorrectSum(double[] values, double expected)
    {
        // Arrange
        var column = NivaraColumn<double>.Create(values);

        // Act
        var result = column.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Sum with null handling
    /// For any column with null values, Sum should compute sum of valid values only.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Sum with null handling")]
    public void Sum_WithNullValues_ReturnsValidSum()
    {
        // Arrange
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn.CreateFromNullable(nullableData);

        // Act
        var result = column.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(9)); // 1 + 3 + 5
    }

    /// <summary>
    /// Feature: core-column-types, Property: Sum error handling
    /// For any empty column, Sum should throw InvalidOperationException.
    /// Validates: Error handling for edge cases
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Sum error handling")]
    public void Sum_EmptyColumn_ThrowsInvalidOperationException()
    {
        // Arrange
        var column = NivaraColumn<int>.Create(Array.Empty<int>());

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => column.Sum());
        Assert.That(ex.Message, Does.Contain("Cannot compute Sum on an empty column"));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Sum with null handling
    /// For any column with all null values, Sum should return zero.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Sum with null handling")]
    public void Sum_AllNullValues_ReturnsZero()
    {
        // Arrange
        var nullableData = new int?[] { null, null, null };
        var column = NivaraColumn.CreateFromNullable(nullableData);

        // Act
        var result = column.Sum();

        // Assert
        Assert.That(result, Is.EqualTo(0));
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
        var column = NivaraColumn.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo(4)); // (2 + 4 + 6) / 3
    }

    /// <summary>
    /// Feature: nivara-series, Property: Average computation on char
    /// For a char series, Average should compute the mean using char division semantics.
    /// Validates: char support in the small integral type domain (issue #164)
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Average computation on char")]
    public void Average_CharValues_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<char>.Create(new[] { (char)2, (char)4, (char)6 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((char)4)); // (2 + 4 + 6) / 3
    }

    /// <summary>
    /// Feature: nivara-series, Property: Average on extended numeric domain
    /// For Half/nint/nuint/Int128/UInt128 series, Average should compute the mean instead of
    /// throwing NotSupportedException after the SIMD sum succeeds (issue #172).
    /// Validates: full numeric domain for average
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_HalfValues_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<Half>.Create(new[] { (Half)1.5, (Half)2.5, (Half)3.5 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((Half)2.5));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_NIntValues_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<nint>.Create(new nint[] { 2, 4, 6 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((nint)4)); // (2 + 4 + 6) / 3
    }

    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_NUIntValues_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<nuint>.Create(new nuint[] { 2, 4, 6 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((nuint)4)); // (2 + 4 + 6) / 3
    }

    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_Int128Values_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<Int128>.Create(new Int128[] { 2, 4, 6 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((Int128)4)); // (2 + 4 + 6) / 3
    }

    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_UInt128Values_ReturnsCorrectAverage()
    {
        // Arrange
        var series = NivaraSeries<UInt128>.Create(new UInt128[] { 2, 4, 6 });

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((UInt128)4)); // (2 + 4 + 6) / 3
    }

    [Test]
    [Category("Feature: nivara-series, Property: Average on extended numeric domain")]
    public void Average_Int128WithNulls_ReturnsValidAverage()
    {
        // Arrange
        var column = NivaraColumn.CreateFromNullable(new Int128?[] { 2, null, 4, null, 6 });
        var series = new NivaraSeries<Int128>(column);

        // Act
        var result = series.Average();

        // Assert
        Assert.That(result, Is.EqualTo((Int128)4)); // (2 + 4 + 6) / 3
    }

    #endregion

    #region Min Tests

    /// <summary>
    /// Feature: core-column-types, Property: Min computation
    /// For any column of numeric values, Min should return the smallest value.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 5, 2, 8, 1, 9 }, 1)]
    [TestCase(new int[] { -1, -5, -2 }, -5)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, int.MinValue }, int.MinValue)]
    [Category("Feature: core-column-types, Property: Min computation")]
    public void Min_IntegerValues_ReturnsCorrectMin(int[] values, int expected)
    {
        // Arrange
        var column = NivaraColumn<int>.Create(values);

        // Act
        var result = column.Min();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Min computation with vectorization
    /// For any column of float values, Min should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new float[] { 3.5f, 1.2f, 4.8f }, 1.2f)]
    [TestCase(new float[] { -1.5f, -2.5f }, -2.5f)]
    [Category("Feature: core-column-types, Property: Min computation with vectorization")]
    public void Min_FloatValues_ReturnsCorrectMin(float[] values, float expected)
    {
        // Arrange
        var column = NivaraColumn<float>.Create(values);

        // Act
        var result = column.Min();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001f));
    }

    #endregion

    #region Max Tests

    /// <summary>
    /// Feature: core-column-types, Property: Max computation
    /// For any column of numeric values, Max should return the largest value.
    /// Validates: Requirements for aggregate functions
    /// </summary>
    [TestCase(new int[] { 5, 2, 8, 1, 9 }, 9)]
    [TestCase(new int[] { -1, -5, -2 }, -1)]
    [TestCase(new int[] { 42 }, 42)]
    [TestCase(new int[] { int.MaxValue, int.MinValue }, int.MaxValue)]
    [Category("Feature: core-column-types, Property: Max computation")]
    public void Max_IntegerValues_ReturnsCorrectMax(int[] values, int expected)
    {
        // Arrange
        var column = NivaraColumn<int>.Create(values);

        // Act
        var result = column.Max();

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Max computation with vectorization
    /// For any column of double values, Max should use TensorPrimitives when possible.
    /// Validates: Vectorized operations for performance
    /// </summary>
    [TestCase(new double[] { 3.5, 1.2, 4.8 }, 4.8)]
    [TestCase(new double[] { -1.5, -2.5 }, -1.5)]
    [Category("Feature: core-column-types, Property: Max computation with vectorization")]
    public void Max_DoubleValues_ReturnsCorrectMax(double[] values, double expected)
    {
        // Arrange
        var column = NivaraColumn<double>.Create(values);

        // Act
        var result = column.Max();

        // Assert
        Assert.That(result, Is.EqualTo(expected).Within(0.0001));
    }

    /// <summary>
    /// Feature: core-column-types, Property: Max with null handling
    /// For any column with null values, Max should find maximum among valid values only.
    /// Validates: Null-aware aggregate operations
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Max with null handling")]
    public void Max_WithNullValues_ReturnsValidMax()
    {
        // Arrange
        var nullableData = new int?[] { 2, null, 8, null, 5 };
        var column = NivaraColumn.CreateFromNullable(nullableData);

        // Act
        var result = column.Max();

        // Assert
        Assert.That(result, Is.EqualTo(8));
    }

    #endregion

    #region All Integer Primitives

    /// <summary>
    /// Feature: core-column-types, Property: Integer primitive vectorization
    /// For every integer primitive, Sum/Min/Max should route through the generic
    /// TensorPrimitives kernel and produce the correct result.
    /// Validates: Vectorized operations for integer types
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Integer primitive vectorization")]
    public void Sum_AllIntegerPrimitives_ReturnsCorrectSum()
    {
        verifyIntegerAggregate(new long[] { 3, 1, 2 }, 6L, 1L, 3L);
        verifyIntegerAggregate(new short[] { 3, 1, 2 }, (short)6, (short)1, (short)3);
        verifyIntegerAggregate(new byte[] { 3, 1, 2 }, (byte)6, (byte)1, (byte)3);
        verifyIntegerAggregate(new ushort[] { 3, 1, 2 }, (ushort)6, (ushort)1, (ushort)3);
        verifyIntegerAggregate(new uint[] { 3, 1, 2 }, 6u, 1u, 3u);
        verifyIntegerAggregate(new ulong[] { 3, 1, 2 }, 6ul, 1ul, 3ul);
        verifyIntegerAggregate(new sbyte[] { 3, 1, 2 }, (sbyte)6, (sbyte)1, (sbyte)3);
        verifyIntegerAggregate(new char[] { (char)3, (char)1, (char)2 }, (char)6, (char)1, (char)3);
    }

    static void verifyIntegerAggregate<T>(T[] values, T expectedSum, T expectedMin, T expectedMax)
        where T : unmanaged, INumber<T>
    {
        Assert.That(NivaraColumn<T>.Create(values).Sum(), Is.EqualTo(expectedSum));
        Assert.That(NivaraColumn<T>.Create(values).Min(), Is.EqualTo(expectedMin));
        Assert.That(NivaraColumn<T>.Create(values).Max(), Is.EqualTo(expectedMax));
    }

    #endregion

    #region Series-Level Aggregates

    /// <summary>
    /// Feature: nivara-series, Property: Sum computation
    /// For any series of numeric values, Sum should return the correct arithmetic sum.
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum computation")]
    public void Sum_IntegerSeries_ReturnsCorrectSum()
    {
        var series = NivaraSeries<int>.Create(new[] { 1, 2, 3, 4, 5 });
        Assert.That(series.Sum(), Is.EqualTo(15));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Sum computation with vectorization")]
    public void Sum_FloatSeries_ReturnsCorrectSum()
    {
        var series = NivaraSeries<float>.Create(new[] { 1.5f, 2.5f, 3.0f });
        Assert.That(series.Sum(), Is.EqualTo(7.0f).Within(0.0001f));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Sum with null handling")]
    public void Sum_Series_WithNullValues_ReturnsValidSum()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { 1, null, 3, null, 5 });
        using var series = new NivaraSeries<int>(column);

        Assert.That(series.Sum(), Is.EqualTo(9)); // 1 + 3 + 5
    }

    [Test]
    [Category("Feature: nivara-series, Property: Sum error handling")]
    public void Sum_EmptySeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<int>.Create(Array.Empty<int>());

        var ex = Assert.Throws<InvalidOperationException>(() => series.Sum());
        Assert.That(ex.Message, Does.Contain("Cannot compute Sum of empty series"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Sum with null handling")]
    public void Sum_AllNullSeries_ReturnsZero()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { null, null, null });
        using var series = new NivaraSeries<int>(column);

        Assert.That(series.Sum(), Is.EqualTo(0));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Sum on extended numeric domain
    /// For Half/nint/nuint/Int128/UInt128 series, Sum should dispatch through the
    /// extended numeric kernel domain.
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Sum on extended numeric domain")]
    public void Sum_ExtendedDomain_ReturnsCorrectSum()
    {
        Assert.That(NivaraSeries<Half>.Create(new[] { (Half)1.5, (Half)2.5 }).Sum(), Is.EqualTo((Half)4.0));
        Assert.That(NivaraSeries<nint>.Create(new nint[] { 2, 4, 6 }).Sum(), Is.EqualTo((nint)12));
        Assert.That(NivaraSeries<nuint>.Create(new nuint[] { 2, 4, 6 }).Sum(), Is.EqualTo((nuint)12));
        Assert.That(NivaraSeries<Int128>.Create(new Int128[] { 2, 4, 6 }).Sum(), Is.EqualTo((Int128)12));
        Assert.That(NivaraSeries<UInt128>.Create(new UInt128[] { 2, 4, 6 }).Sum(), Is.EqualTo((UInt128)12));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Min/Max computation
    /// For any series of numeric values, Min/Max should return the smallest/largest value.
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Min computation")]
    public void Min_IntegerSeries_ReturnsCorrectMin()
    {
        var series = NivaraSeries<int>.Create(new[] { 5, 2, 8, 1, 9 });
        Assert.That(series.Min(), Is.EqualTo(1));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Max computation")]
    public void Max_IntegerSeries_ReturnsCorrectMax()
    {
        var series = NivaraSeries<int>.Create(new[] { 5, 2, 8, 1, 9 });
        Assert.That(series.Max(), Is.EqualTo(9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Min/Max with null handling")]
    public void MinMax_Series_WithNullValues_ReturnsValidExtremes()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { 2, null, 8, null, 5 });
        using var series = new NivaraSeries<int>(column);

        Assert.That(series.Min(), Is.EqualTo(2));
        Assert.That(series.Max(), Is.EqualTo(8));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Min/Max error handling")]
    public void MinMax_EmptySeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<int>.Create(Array.Empty<int>());

        var minEx = Assert.Throws<InvalidOperationException>(() => series.Min());
        var maxEx = Assert.Throws<InvalidOperationException>(() => series.Max());
        Assert.That(minEx.Message, Does.Contain("Cannot compute Min of empty series"));
        Assert.That(maxEx.Message, Does.Contain("Cannot compute Max of empty series"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Min/Max error handling")]
    public void MinMax_AllNullSeries_ThrowsInvalidOperationException()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { null, null, null });
        using var series = new NivaraSeries<int>(column);

        var minEx = Assert.Throws<InvalidOperationException>(() => series.Min());
        var maxEx = Assert.Throws<InvalidOperationException>(() => series.Max());
        Assert.That(minEx.Message, Does.Contain("all values are null"));
        Assert.That(maxEx.Message, Does.Contain("all values are null"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Sum/Min/Max on extended numeric domain")]
    public void MinMax_ExtendedDomain_ReturnsCorrectExtremes()
    {
        Assert.That(NivaraSeries<Half>.Create(new[] { (Half)3.5, (Half)1.5 }).Min(), Is.EqualTo((Half)1.5));
        Assert.That(NivaraSeries<Half>.Create(new[] { (Half)3.5, (Half)1.5 }).Max(), Is.EqualTo((Half)3.5));
        Assert.That(NivaraSeries<Int128>.Create(new Int128[] { 3, 1, 2 }).Min(), Is.EqualTo((Int128)1));
        Assert.That(NivaraSeries<UInt128>.Create(new UInt128[] { 3, 1, 2 }).Max(), Is.EqualTo((UInt128)3));
    }

    /// <summary>
    /// Feature: nivara-series, Property: Aggregate type validation
    /// For non-numeric series, Sum/Min/Max should throw a clear InvalidOperationException.
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Aggregate type validation")]
    public void Aggregates_NonNumericSeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<string>.Create(new[] { "a", "b" });

        var sumEx = Assert.Throws<InvalidOperationException>(() => series.Sum());
        var minEx = Assert.Throws<InvalidOperationException>(() => series.Min());
        var maxEx = Assert.Throws<InvalidOperationException>(() => series.Max());

        Assert.That(sumEx.Message, Does.Contain("Sum operation is not supported"));
        Assert.That(minEx.Message, Does.Contain("Min operation is not supported"));
        Assert.That(maxEx.Message, Does.Contain("Max operation is not supported"));
    }

    #endregion

    #region Quantile Tests

    /// <summary>
    /// Feature: nivara-series, Property: Quantile computation
    /// For any series of numeric values, Quantile(q) should compute the q-th quantile with linear
    /// interpolation (numpy default / polars interpolation="linear", Hyndman-Fan type 7).
    /// </summary>
    [Test]
    [Category("Feature: nivara-series, Property: Quantile computation")]
    public void Quantile_IntegerSeries_ReturnsCorrectQuantile()
    {
        using var series = NivaraSeries<int>.Create(new[] { 1, 2, 3, 4 });

        Assert.That(series.Quantile(0.5), Is.EqualTo(2.5).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile computation")]
    public void Quantile_HighQuantile_InterpolatesLinearly()
    {
        using var series = NivaraSeries<int>.Create(new[] { 10, 20, 30, 40, 50 });

        Assert.That(series.Quantile(0.9), Is.EqualTo(46.0).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile computation")]
    public void Quantile_MinMaxQuantile_ReturnsExtremes()
    {
        using var series = NivaraSeries<int>.Create(new[] { 3, 9, 6 });

        Assert.That(series.Quantile(0.0), Is.EqualTo(3.0));
        Assert.That(series.Quantile(1.0), Is.EqualTo(9.0));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile with null handling")]
    public void Quantile_WithNullValues_IgnoresNulls()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
        using var series = new NivaraSeries<double>(column);

        Assert.That(series.Quantile(0.25), Is.EqualTo(2.5).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile error handling")]
    public void Quantile_InvalidQuantile_ThrowsArgumentOutOfRangeException()
    {
        using var series = NivaraSeries<int>.Create(new[] { 1, 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() => series.Quantile(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => series.Quantile(1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => series.Quantile(double.NaN));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile error handling")]
    public void Quantile_EmptySeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<int>.Create(Array.Empty<int>());

        var ex = Assert.Throws<InvalidOperationException>(() => series.Quantile(0.5));
        Assert.That(ex.Message, Does.Contain("Cannot compute Quantile of empty series"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile error handling")]
    public void Quantile_AllNullSeries_ThrowsInvalidOperationException()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { null, null });
        using var series = new NivaraSeries<int>(column);

        var ex = Assert.Throws<InvalidOperationException>(() => series.Quantile(0.5));
        Assert.That(ex.Message, Does.Contain("all values are null"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile on extended numeric domain")]
    public void Quantile_ExtendedDomain_ReturnsCorrectQuantile()
    {
        using var halfSeries = NivaraSeries<Half>.Create(new[] { (Half)1, (Half)2, (Half)3, (Half)4 });
        Assert.That(halfSeries.Quantile(0.5), Is.EqualTo(2.5).Within(1e-3));

        using var int128Series = NivaraSeries<Int128>.Create(new Int128[] { 1, 2, 3, 4 });
        Assert.That(int128Series.Quantile(0.5), Is.EqualTo(2.5).Within(1e-9));

        using var decimalSeries = NivaraSeries<decimal>.Create(new decimal[] { 1, 2, 3, 4 });
        Assert.That(decimalSeries.Quantile(0.5), Is.EqualTo(2.5).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Quantile type validation")]
    public void Quantile_NonNumericSeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<string>.Create(new[] { "a", "b" });

        var ex = Assert.Throws<InvalidOperationException>(() => series.Quantile(0.5));
        Assert.That(ex.Message, Does.Contain("Quantile operation is not supported"));
    }

    #endregion

    #region Median Tests

    [Test]
    [Category("Feature: nivara-series, Property: Median computation")]
    public void Median_OddLengthSeries_ReturnsMiddleValue()
    {
        using var series = NivaraSeries<int>.Create(new[] { 3, 1, 2 });

        Assert.That(series.Median(), Is.EqualTo(2.0));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median computation")]
    public void Median_EvenLengthSeries_AveragesMiddleTwo()
    {
        using var series = NivaraSeries<int>.Create(new[] { 1, 3, 2, 4 });

        Assert.That(series.Median(), Is.EqualTo(2.5).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median computation")]
    public void Median_SingleElementSeries_ReturnsThatElement()
    {
        using var series = NivaraSeries<int>.Create(new[] { 7 });

        Assert.That(series.Median(), Is.EqualTo(7.0));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median with null handling")]
    public void Median_WithNullValues_IgnoresNulls()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { 5, null, 3, 1, 4 });
        using var series = new NivaraSeries<double>(column);

        Assert.That(series.Median(), Is.EqualTo(3.5).Within(1e-9));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median error handling")]
    public void Median_EmptySeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<int>.Create(Array.Empty<int>());

        var ex = Assert.Throws<InvalidOperationException>(() => series.Median());
        Assert.That(ex.Message, Does.Contain("Cannot compute Median of empty series"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median error handling")]
    public void Median_AllNullSeries_ThrowsInvalidOperationException()
    {
        var column = NivaraColumn.CreateFromNullable(new int?[] { null, null });
        using var series = new NivaraSeries<int>(column);

        var ex = Assert.Throws<InvalidOperationException>(() => series.Median());
        Assert.That(ex.Message, Does.Contain("all values are null"));
    }

    [Test]
    [Category("Feature: nivara-series, Property: Median type validation")]
    public void Median_NonNumericSeries_ThrowsInvalidOperationException()
    {
        using var series = NivaraSeries<string>.Create(new[] { "a", "b" });

        var ex = Assert.Throws<InvalidOperationException>(() => series.Median());
        Assert.That(ex.Message, Does.Contain("Median operation is not supported"));
    }

    #endregion

    #region Edge Cases and Error Handling

    /// <summary>
    /// Feature: core-column-types, Property: Aggregate error handling
    /// For any disposed column/series, aggregate functions should throw ObjectDisposedException.
    /// Validates: Resource management and disposal patterns
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Aggregate error handling")]
    public void AggregateFunction_Disposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var series = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        series.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => series.Average());
        Assert.Throws<ObjectDisposedException>(() => series.Sum());
        Assert.Throws<ObjectDisposedException>(() => series.Min());
        Assert.Throws<ObjectDisposedException>(() => series.Max());

        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        column.Dispose();
        Assert.Throws<ObjectDisposedException>(() => column.Sum());
        Assert.Throws<ObjectDisposedException>(() => column.Min());
        Assert.Throws<ObjectDisposedException>(() => column.Max());
    }

    /// <summary>
    /// Feature: core-column-types, Property: Min/Max error handling
    /// For any column where all values are null, Min/Max should throw InvalidOperationException.
    /// Validates: Error handling for null-only columns
    /// </summary>
    [Test]
    [Category("Feature: core-column-types, Property: Min/Max error handling")]
    public void MinMax_AllNullValues_ThrowsInvalidOperationException()
    {
        // Arrange
        var nullableData = new int?[] { null, null, null };
        var column = NivaraColumn.CreateFromNullable(nullableData);

        // Act & Assert
        var minEx = Assert.Throws<InvalidOperationException>(() => column.Min());
        var maxEx = Assert.Throws<InvalidOperationException>(() => column.Max());

        Assert.That(minEx.Message, Does.Contain("all values are null"));
        Assert.That(maxEx.Message, Does.Contain("all values are null"));
    }

    #endregion
}
