using Nivara.Tensors;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraSeries aggregate functions and the column-level reductions
/// (Sum, Min, Max) that back the series. NivaraSeries keeps Average; Sum/Min/Max
/// live on NivaraColumn&lt;T&gt; (null-aware, TensorPrimitives).
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
