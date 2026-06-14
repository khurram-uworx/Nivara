#pragma warning disable CS0618 // Tests exercise deprecated APIs during migration

using Nivara.Tensors;
using NUnit.Framework;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for tensor interoperability between Nivara and System.Numerics.Tensors.
/// Covers round-trip conversions, zero-copy operations, and data integrity.
/// </summary>
[TestFixture]
public class TensorInteropTests
{
    #region Series to Tensor Conversions

    [Test]
    public void ToTensor_WithValidSeries_ReturnsCorrectTensor()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        using var series = NivaraSeries<float>.Create(data);

        // Act
        var tensor = series.ToTensor();

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(1));
        Assert.That(tensor.Lengths[0], Is.EqualTo((nint)5));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(tensor.AsTensorSpan()[i], Is.EqualTo(data[i]).Within(1e-6f));
        }
    }

    [Test]
    public void ToTensor_WithEmptySeries_ReturnsEmptyTensor()
    {
        // Arrange
        using var series = NivaraSeries<int>.Create(Array.Empty<int>());

        // Act
        var tensor = series.ToTensor();

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(1));
        Assert.That(tensor.Lengths[0], Is.EqualTo((nint)0));
    }

    [Test]
    public void ToTensor_WithNullValues_Throws()
    {
        // Arrange
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });
        using var series = new NivaraSeries<float>(column);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => series.ToTensor());
        Assert.That(ex.Message, Does.Contain("Cannot convert column with null values"));
    }

    [Test]
    public void ToTensor_WithNullReplacement_ReplacesOnlyNullPositions()
    {
        // Arrange
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });
        using var series = new NivaraSeries<float>(column);

        // Act
        var tensor = series.ToTensor(-1.0f);

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(1));
        Assert.That(tensor.Lengths[0], Is.EqualTo((nint)3));
        Assert.That(tensor.AsTensorSpan()[0], Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(tensor.AsTensorSpan()[1], Is.EqualTo(-1.0f).Within(1e-6f));
        Assert.That(tensor.AsTensorSpan()[2], Is.EqualTo(3.0f).Within(1e-6f));
    }

    [Test]
    public void ColumnToNullableTensor_WithNoNulls_ReturnsNullMaskAsNull()
    {
        var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });

        var nullableTensor = column.ToNullableTensor();

        Assert.That(nullableTensor.Data.Rank, Is.EqualTo(1));
        Assert.That(nullableTensor.Data.Lengths[0], Is.EqualTo((nint)3));
        Assert.That(nullableTensor.NullMask, Is.Null);
        Assert.That(nullableTensor.Data.AsTensorSpan()[0], Is.EqualTo(1));
        Assert.That(nullableTensor.Data.AsTensorSpan()[2], Is.EqualTo(3));
    }

    [Test]
    public void SeriesToNullableTensor_WithNulls_PreservesMask()
    {
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 1.0f, null, 3.0f });
        using var series = new NivaraSeries<float>(column);

        var nullableTensor = series.ToNullableTensor();

        Assert.That(nullableTensor.Data.Rank, Is.EqualTo(1));
        Assert.That(nullableTensor.Data.Lengths[0], Is.EqualTo((nint)3));
        Assert.That(nullableTensor.NullMask, Is.Not.Null);
        Assert.That(nullableTensor.NullMask!.Lengths[0], Is.EqualTo((nint)3));
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[0], Is.False);
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[1], Is.True);
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[2], Is.False);
    }

    [Test]
    public void FromTensor_WithValidTensor_ReturnsCorrectSeries()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0, 4.0 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 4 }));

        // Act
        using var series = TensorInteropExtensions.FromTensor(tensor);

        // Assert
        Assert.That(series.Length, Is.EqualTo(4));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(series[i], Is.EqualTo(data[i]).Within(1e-10));
            Assert.That(series.IsNull(i), Is.False);
        }
    }

    [Test]
    public void FromTensor_WithEmptyTensor_ReturnsEmptySeries()
    {
        // Arrange
        var tensor = Tensor.Create<float>(Array.Empty<float>(), new ReadOnlySpan<nint>(new nint[] { 0 }));

        // Act
        using var series = TensorInteropExtensions.FromTensor(tensor);

        // Assert
        Assert.That(series.Length, Is.EqualTo(0));
    }

    [Test]
    public void FromTensor_WithMultiDimensionalTensor_ThrowsException()
    {
        // Arrange
        var data = new int[] { 1, 2, 3, 4 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorInteropExtensions.FromTensor(tensor));
        Assert.That(ex.Message, Does.Contain("Only 1D tensors are supported"));
    }

    #endregion

    #region Series to TensorSpan Conversions

    [Test]
    public void ToTensorSpan_WithValidSeries_ReturnsCorrectTensorSpan()
    {
        // Arrange
        var data = new int[] { 10, 20, 30, 40 };
        using var series = NivaraSeries<int>.Create(data);

        // Act
        var tensorSpan = series.ToTensorSpan();

        // Assert
        Assert.That(tensorSpan.Rank, Is.EqualTo(1));
        Assert.That(tensorSpan.Lengths[0], Is.EqualTo((nint)4));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(tensorSpan[i], Is.EqualTo(data[i]));
        }
    }

    [Test]
    public void ToTensorSpan_WithEmptySeries_ReturnsEmptyTensorSpan()
    {
        // Arrange
        using var series = NivaraSeries<double>.Create(Array.Empty<double>());

        // Act
        var tensorSpan = series.ToTensorSpan();

        // Assert
        Assert.That(tensorSpan.Rank, Is.EqualTo(0));
    }

    [Test]
    public void FromTensorSpan_WithValidTensorSpan_ReturnsCorrectSeries()
    {
        // Arrange
        var data = new float[] { 1.5f, 2.5f, 3.5f };
        var tensorSpan = new ReadOnlyTensorSpan<float>(data, new ReadOnlySpan<nint>(new nint[] { 3 }), default);

        // Act
        using var series = TensorInteropExtensions.FromTensorSpan(tensorSpan);

        // Assert
        Assert.That(series.Length, Is.EqualTo(3));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(series[i], Is.EqualTo(data[i]).Within(1e-6f));
            Assert.That(series.IsNull(i), Is.False);
        }
    }

    [Test]
    public void FromTensorSpan_WithMultiDimensionalTensorSpan_ThrowsException()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0, 4.0 };
        var tensorSpan = new ReadOnlyTensorSpan<double>(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }), default);

        // Act & Assert
        try
        {
            TensorInteropExtensions.FromTensorSpan(tensorSpan);
            Assert.Fail("Expected ArgumentException was not thrown");
        }
        catch (ArgumentException ex)
        {
            Assert.That(ex.Message, Does.Contain("Only 1D tensor spans are supported"));
        }
    }

    #endregion

    #region Round-Trip Tests

    [Test]
    public void SeriesRoundTrip_ThroughTensor_PreservesData()
    {
        // Arrange
        var originalData = new float[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };
        using var originalSeries = NivaraSeries<float>.Create(originalData);

        // Act
        var tensor = originalSeries.ToTensor();
        using var roundTripSeries = TensorInteropExtensions.FromTensor(tensor);

        // Assert
        Assert.That(roundTripSeries.Length, Is.EqualTo(originalSeries.Length));

        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(roundTripSeries[i], Is.EqualTo(originalSeries[i]).Within(1e-6f));
            Assert.That(roundTripSeries.IsNull(i), Is.EqualTo(originalSeries.IsNull(i)));
        }
    }

    [Test]
    public void SeriesRoundTrip_ThroughTensorSpan_PreservesData()
    {
        // Arrange
        var originalData = new double[] { 1.0, 2.0, 3.0, 4.0 };
        using var originalSeries = NivaraSeries<double>.Create(originalData);

        // Act
        var tensorSpan = originalSeries.ToTensorSpan();
        using var roundTripSeries = TensorInteropExtensions.FromTensorSpan(tensorSpan);

        // Assert
        Assert.That(roundTripSeries.Length, Is.EqualTo(originalSeries.Length));

        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(roundTripSeries[i], Is.EqualTo(originalSeries[i]).Within(1e-10));
            Assert.That(roundTripSeries.IsNull(i), Is.EqualTo(originalSeries.IsNull(i)));
        }
    }

    [TestCase(typeof(int))]
    [TestCase(typeof(float))]
    [TestCase(typeof(double))]
    [TestCase(typeof(long))]
    public void RoundTripPreservesData_ForDifferentNumericTypes(Type numericType)
    {
        // This test uses reflection to test different numeric types
        var method = GetType().GetMethod(nameof(RoundTripTest_Generic),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var genericMethod = method!.MakeGenericMethod(numericType);

        genericMethod.Invoke(this, null);
    }

    private void RoundTripTest_Generic<T>() where T : unmanaged, INumber<T>
    {
        // Create test data based on type
        var testData = CreateTestData<T>();
        using var originalSeries = NivaraSeries<T>.Create(testData);

        // Test tensor round-trip
        var tensor = originalSeries.ToTensor();
        using var roundTripSeries = TensorInteropExtensions.FromTensor(tensor);

        Assert.That(roundTripSeries.Length, Is.EqualTo(originalSeries.Length));

        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(roundTripSeries[i], Is.EqualTo(originalSeries[i]));
        }
    }

    private T[] CreateTestData<T>() where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(int))
            return (T[])(object)new int[] { 1, 2, 3, 4, 5 };
        if (typeof(T) == typeof(float))
            return (T[])(object)new float[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f };
        if (typeof(T) == typeof(double))
            return (T[])(object)new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 };
        if (typeof(T) == typeof(long))
            return (T[])(object)new long[] { 1L, 2L, 3L, 4L, 5L };

        throw new NotSupportedException($"Type {typeof(T)} not supported in test");
    }

    #endregion

    #region Frame to Tensor Conversions

    [Test]
    public void FrameToTensor_WithValidFrame_ReturnsCorrect2DTensor()
    {
        // Arrange
        var col1Data = new float[] { 1.0f, 4.0f };
        var col2Data = new float[] { 2.0f, 5.0f };
        var col3Data = new float[] { 3.0f, 6.0f };

        var columns = new[]
        {
            ("Col1", (IColumn)NivaraColumn<float>.Create(col1Data)),
            ("Col2", (IColumn)NivaraColumn<float>.Create(col2Data)),
            ("Col3", (IColumn)NivaraColumn<float>.Create(col3Data))
        };

        using var frame = new NivaraFrame(columns);

        // Act
        var tensor = frame.ToTensor<float>();

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.Lengths[0], Is.EqualTo((nint)2)); // rows
        Assert.That(tensor.Lengths[1], Is.EqualTo((nint)3)); // columns

        // Verify data in row-major order
        var tensorSpan = tensor.AsTensorSpan();
        Assert.That(tensorSpan[0, 0], Is.EqualTo(1.0f).Within(1e-6f)); // Row 0, Col 0
        Assert.That(tensorSpan[0, 1], Is.EqualTo(2.0f).Within(1e-6f)); // Row 0, Col 1
        Assert.That(tensorSpan[0, 2], Is.EqualTo(3.0f).Within(1e-6f)); // Row 0, Col 2
        Assert.That(tensorSpan[1, 0], Is.EqualTo(4.0f).Within(1e-6f)); // Row 1, Col 0
        Assert.That(tensorSpan[1, 1], Is.EqualTo(5.0f).Within(1e-6f)); // Row 1, Col 1
        Assert.That(tensorSpan[1, 2], Is.EqualTo(6.0f).Within(1e-6f)); // Row 1, Col 2
    }

    [Test]
    public void FromTensor_With2DTensor_ReturnsCorrectFrame()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 3 }));
        var columnNames = new[] { "A", "B", "C" };

        // Act
        using var frame = TensorInteropExtensions.FromTensor(tensor, columnNames);

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(2));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));
        Assert.That(frame.ColumnNames.ToArray(), Is.EqualTo(columnNames));

        // Verify data
        var colA = frame.GetColumn<double>("A");
        var colB = frame.GetColumn<double>("B");
        var colC = frame.GetColumn<double>("C");

        Assert.That(colA[0], Is.EqualTo(1.0).Within(1e-10));
        Assert.That(colB[0], Is.EqualTo(2.0).Within(1e-10));
        Assert.That(colC[0], Is.EqualTo(3.0).Within(1e-10));
        Assert.That(colA[1], Is.EqualTo(4.0).Within(1e-10));
        Assert.That(colB[1], Is.EqualTo(5.0).Within(1e-10));
        Assert.That(colC[1], Is.EqualTo(6.0).Within(1e-10));
    }

    [Test]
    public void FromTensor_WithDefaultColumnNames_GeneratesCorrectNames()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        // Act
        using var frame = TensorInteropExtensions.FromTensor<float>(tensor, null);

        // Assert
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.ColumnNames.ToArray(), Is.EqualTo(new[] { "Col_0", "Col_1" }));
    }

    [Test]
    public void FromTensor_WithMismatchedColumnNames_ThrowsException()
    {
        // Arrange
        var data = new int[] { 1, 2, 3, 4 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));
        var wrongColumnNames = new[] { "A" }; // Should be 2 names, not 1

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorInteropExtensions.FromTensor(tensor, wrongColumnNames));
        Assert.That(ex.Message, Does.Contain("Column names count"));
    }

    [Test]
    public void FrameToNullableTensor_WithNulls_UsesTwoDimensionalMask()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 1.0f, 3.0f })),
            ("B", (IColumn)NivaraColumn<float>.CreateFromNullable(new float?[] { null, 4.0f })),
        });

        var nullableTensor = frame.ToNullableTensor<float>();

        Assert.That(nullableTensor.Data.Rank, Is.EqualTo(2));
        Assert.That(nullableTensor.Data.Lengths[0], Is.EqualTo((nint)2));
        Assert.That(nullableTensor.Data.Lengths[1], Is.EqualTo((nint)2));
        Assert.That(nullableTensor.NullMask, Is.Not.Null);
        Assert.That(nullableTensor.NullMask!.Rank, Is.EqualTo(2));
        Assert.That(nullableTensor.NullMask.Lengths[0], Is.EqualTo((nint)2));
        Assert.That(nullableTensor.NullMask.Lengths[1], Is.EqualTo((nint)2));
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[0, 0], Is.False);
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[0, 1], Is.True);
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[1, 0], Is.False);
        Assert.That(nullableTensor.NullMask.AsTensorSpan()[1, 1], Is.False);
    }

    [Test]
    public void FrameToNullableTensor_WithoutNulls_ReturnsNullMaskAsNull()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<int>.Create(new[] { 1, 3 })),
            ("B", (IColumn)NivaraColumn<int>.Create(new[] { 2, 4 })),
        });

        var nullableTensor = frame.ToNullableTensor<int>();

        Assert.That(nullableTensor.NullMask, Is.Null);
        Assert.That(nullableTensor.Data.AsTensorSpan()[0, 0], Is.EqualTo(1));
        Assert.That(nullableTensor.Data.AsTensorSpan()[0, 1], Is.EqualTo(2));
        Assert.That(nullableTensor.Data.AsTensorSpan()[1, 0], Is.EqualTo(3));
        Assert.That(nullableTensor.Data.AsTensorSpan()[1, 1], Is.EqualTo(4));
    }

    [Test]
    public void FromTensor_WithRowLabels_AddsLabelColumn()
    {
        var data = new double[] { 1.0, 2.0, 3.0, 4.0 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        using var frame = TensorInteropExtensions.FromTensor(
            tensor,
            new[] { "X", "Y" },
            new object[] { "row-1", "row-2" },
            "Id");

        Assert.That(frame.ColumnNames.ToArray(), Is.EqualTo(new[] { "Id", "X", "Y" }));
        Assert.That(frame.GetColumn<object>("Id")[0], Is.EqualTo("row-1"));
        Assert.That(frame.GetColumn<object>("Id")[1], Is.EqualTo("row-2"));
        Assert.That(frame.GetColumn<double>("X")[1], Is.EqualTo(3.0));
        Assert.That(frame.GetColumn<double>("Y")[1], Is.EqualTo(4.0));
    }

    [Test]
    public void FromTensor_WithMismatchedRowLabels_ThrowsException()
    {
        var tensor = Tensor.Create(new[] { 1, 2, 3, 4 }, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        var ex = Assert.Throws<ArgumentException>(() =>
            TensorInteropExtensions.FromTensor(tensor, new[] { "A", "B" }, new object[] { "row-1" }));

        Assert.That(ex!.Message, Does.Contain("Row labels count"));
    }

    [Test]
    public void FromTensor_WithRowLabelColumnCollision_ThrowsException()
    {
        var tensor = Tensor.Create(new[] { 1, 2, 3, 4 }, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        var ex = Assert.Throws<ArgumentException>(() =>
            TensorInteropExtensions.FromTensor(tensor, new[] { "Label", "B" }, new object[] { "row-1", "row-2" }));

        Assert.That(ex!.Message, Does.Contain("Duplicate column name"));
    }

    [Test]
    public void FromMatrix_DelegatesToTensorConversion()
    {
        var tensor = Tensor.Create(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new ReadOnlySpan<nint>(new nint[] { 2, 2 }));

        using var frame = NivaraFrame.FromMatrix(
            tensor,
            new[] { "Left", "Right" },
            new object[] { "r1", "r2" },
            "RowId");

        Assert.That(frame.ColumnNames.ToArray(), Is.EqualTo(new[] { "RowId", "Left", "Right" }));
        Assert.That(frame.GetColumn<object>("RowId")[0], Is.EqualTo("r1"));
        Assert.That(frame.GetColumn<float>("Left")[0], Is.EqualTo(1.0f));
        Assert.That(frame.GetColumn<float>("Right")[1], Is.EqualTo(4.0f));
    }

    [Test]
    public void FromRows_CreatesLabelAndFeatureColumns()
    {
        using var frame = NivaraFrame.FromRows(
            new[]
            {
                ("A", new[] { 0.1f, 0.2f }),
                ("B", new[] { 0.3f, 0.4f }),
            },
            new[] { "e0", "e1" },
            "DocumentId");

        Assert.That(frame.ColumnNames.ToArray(), Is.EqualTo(new[] { "DocumentId", "e0", "e1" }));
        Assert.That(frame.GetColumn<string>("DocumentId")[0], Is.EqualTo("A"));
        Assert.That(frame.GetColumn<float>("e0")[1], Is.EqualTo(0.3f));
        Assert.That(frame.GetColumn<float>("e1")[1], Is.EqualTo(0.4f));
    }

    [Test]
    public void FromRows_WithMismatchedVectorWidths_ThrowsException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraFrame.FromRows(new[]
            {
                ("A", new[] { 1, 2 }),
                ("B", new[] { 3 }),
            }));

        Assert.That(ex!.Message, Does.Contain("same length"));
    }

    [Test]
    public void FromRows_WithEmptyRows_ThrowsException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraFrame.FromRows<int>(Array.Empty<(string Label, int[] Vector)>()));

        Assert.That(ex!.Message, Does.Contain("At least one row"));
    }

    #endregion

    #region Reshape Operations

    [Test]
    public void ReshapeToTensor_WithValidDimensions_ReturnsCorrectTensor()
    {
        // Arrange
        var data = new int[] { 1, 2, 3, 4, 5, 6 };
        using var series = NivaraSeries<int>.Create(data);

        // Act
        var tensor = series.ReshapeToTensor(2, 3);

        // Assert
        Assert.That(tensor.Rank, Is.EqualTo(2));
        Assert.That(tensor.Lengths[0], Is.EqualTo((nint)2));
        Assert.That(tensor.Lengths[1], Is.EqualTo((nint)3));

        var tensorSpan = tensor.AsTensorSpan();
        Assert.That(tensorSpan[0, 0], Is.EqualTo(1));
        Assert.That(tensorSpan[0, 1], Is.EqualTo(2));
        Assert.That(tensorSpan[0, 2], Is.EqualTo(3));
        Assert.That(tensorSpan[1, 0], Is.EqualTo(4));
        Assert.That(tensorSpan[1, 1], Is.EqualTo(5));
        Assert.That(tensorSpan[1, 2], Is.EqualTo(6));
    }

    [Test]
    public void ReshapeToTensor_WithInvalidDimensions_ThrowsException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        using var series = NivaraSeries<float>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => series.ReshapeToTensor(2, 2)); // 4 elements needed, only 3 available
        Assert.That(ex.Message, Does.Contain("Total elements in dimensions"));
    }

    #endregion

    #region Flatten Operations

    [Test]
    public void FlattenFromTensor_WithMultiDimensionalTensor_ReturnsCorrectSeries()
    {
        // Arrange
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var tensor = Tensor.Create(data, new ReadOnlySpan<nint>(new nint[] { 2, 3 }));

        // Act
        using var series = TensorInteropExtensions.FlattenFromTensor(tensor);

        // Assert
        Assert.That(series.Length, Is.EqualTo(6));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(series[i], Is.EqualTo(data[i]).Within(1e-10));
        }
    }

    [Test]
    public void FlattenFromTensorSpan_WithMultiDimensionalTensorSpan_ReturnsCorrectSeries()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var tensorSpan = new ReadOnlyTensorSpan<float>(data, new ReadOnlySpan<nint>(new nint[] { 2, 2 }), default);

        // Act
        using var series = TensorInteropExtensions.FlattenFromTensorSpan<float>(tensorSpan);

        // Assert
        Assert.That(series.Length, Is.EqualTo(4));

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(series[i], Is.EqualTo(data[i]).Within(1e-6f));
        }
    }

    #endregion

    #region Batch Operations

    [Test]
    public void ToBatchTensors_WithMultipleSeries_ReturnsCorrectTensors()
    {
        // Arrange
        var series1 = NivaraSeries<float>.Create(new float[] { 1.0f, 2.0f });
        var series2 = NivaraSeries<float>.Create(new float[] { 3.0f, 4.0f });
        var series3 = NivaraSeries<float>.Create(new float[] { 5.0f, 6.0f });
        var seriesArray = new[] { series1, series2, series3 };

        try
        {
            // Act
            var tensors = seriesArray.ToBatchTensors();

            // Assert
            Assert.That(tensors.Length, Is.EqualTo(3));

            for (int i = 0; i < tensors.Length; i++)
            {
                Assert.That(tensors[i].Rank, Is.EqualTo(1));
                Assert.That(tensors[i].Lengths[0], Is.EqualTo((nint)2));
            }

            // Verify data
            Assert.That(tensors[0].AsTensorSpan()[0], Is.EqualTo(1.0f).Within(1e-6f));
            Assert.That(tensors[0].AsTensorSpan()[1], Is.EqualTo(2.0f).Within(1e-6f));
            Assert.That(tensors[1].AsTensorSpan()[0], Is.EqualTo(3.0f).Within(1e-6f));
            Assert.That(tensors[1].AsTensorSpan()[1], Is.EqualTo(4.0f).Within(1e-6f));
            Assert.That(tensors[2].AsTensorSpan()[0], Is.EqualTo(5.0f).Within(1e-6f));
            Assert.That(tensors[2].AsTensorSpan()[1], Is.EqualTo(6.0f).Within(1e-6f));
        }
        finally
        {
            // Cleanup
            series1.Dispose();
            series2.Dispose();
            series3.Dispose();
        }
    }

    [Test]
    public void FromBatchTensors_WithMultipleTensors_ReturnsCorrectSeries()
    {
        // Arrange
        var tensor1 = Tensor.Create(new int[] { 1, 2 }, new ReadOnlySpan<nint>(new nint[] { 2 }));
        var tensor2 = Tensor.Create(new int[] { 3, 4 }, new ReadOnlySpan<nint>(new nint[] { 2 }));
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var seriesArray = TensorInteropExtensions.FromBatchTensors(tensors);

        try
        {
            // Assert
            Assert.That(seriesArray.Length, Is.EqualTo(2));

            Assert.That(seriesArray[0].Length, Is.EqualTo(2));
            Assert.That(seriesArray[0][0], Is.EqualTo(1));
            Assert.That(seriesArray[0][1], Is.EqualTo(2));

            Assert.That(seriesArray[1].Length, Is.EqualTo(2));
            Assert.That(seriesArray[1][0], Is.EqualTo(3));
            Assert.That(seriesArray[1][1], Is.EqualTo(4));
        }
        finally
        {
            // Cleanup
            foreach (var series in seriesArray)
            {
                series.Dispose();
            }
        }
    }

    [Test]
    public void CollectionExpressions_CreateColumnAndSeries()
    {
        NivaraColumn<int> column = [1, 2, 3];
        NivaraSeries<float> series = [1.0f, 2.0f, 3.0f];

        Assert.That(column.Length, Is.EqualTo(3));
        Assert.That(column[0], Is.EqualTo(1));
        Assert.That(column[2], Is.EqualTo(3));
        Assert.That(series.Length, Is.EqualTo(3));
        Assert.That(series[0], Is.EqualTo(1.0f));
        Assert.That(series[2], Is.EqualTo(3.0f));
    }

    [Test]
    public void FrameDot_WithProductColumns_ReturnsOneScorePerColumn()
    {
        // Arrange
        using var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
            ("C", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
        });

        // Act
        using var scores = products.Dot(user);

        // Assert
        Assert.That(scores.Length, Is.EqualTo(3));
        Assert.That(scores[0], Is.EqualTo(1.16f).Within(1e-6f));
        Assert.That(scores[1], Is.EqualTo(0.50f).Within(1e-6f));
        Assert.That(scores[2], Is.EqualTo(1.11f).Within(1e-6f));
    }

    [Test]
    public void FrameCosineSimilarity_WithProductColumns_MatchesTensorPrimitives()
    {
        // Arrange
        using var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
            ("C", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
        });

        // Act
        using var scores = products.CosineSimilarity(user);

        // Assert
        Assert.That(scores.Length, Is.EqualTo(3));
        Assert.That(scores[0], Is.EqualTo(TensorPrimitives.CosineSimilarity(
            new[] { 0.9f, 0.2f, 0.5f, 0.4f },
            new[] { 0.8f, 0.1f, 0.6f, 0.3f })).Within(1e-6f));
        Assert.That(scores[1], Is.EqualTo(TensorPrimitives.CosineSimilarity(
            new[] { 0.1f, 0.9f, 0.2f, 0.7f },
            new[] { 0.8f, 0.1f, 0.6f, 0.3f })).Within(1e-6f));
        Assert.That(scores[2], Is.EqualTo(TensorPrimitives.CosineSimilarity(
            new[] { 0.7f, 0.1f, 0.8f, 0.2f },
            new[] { 0.8f, 0.1f, 0.6f, 0.3f })).Within(1e-6f));
    }

    [Test]
    public void ArgSortDescending_ReturnsStableIndicesWithNullsLast()
    {
        // Arrange
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 0.5f, null, 0.9f, 0.9f, 0.1f });
        using var series = new NivaraSeries<float>(column);

        // Act
        var indices = series.ArgSortDescending();

        // Assert
        Assert.That(indices, Is.EqualTo(new[] { 2, 3, 0, 4, 1 }));
    }

    #endregion

    #region Ranking Workflow

    [Test]
    public void TopKDescending_WithStringLabels_ReturnsTopKInOrder()
    {
        var data = new float[] { 0.5f, 0.9f, 0.1f, 0.7f };
        var labels = new[] { "A", "B", "C", "D" };
        using var series = NivaraSeries<float>.Create(data, labels);

        var result = series.TopKDescending(2);

        Assert.That(result.Length, Is.EqualTo(2));
        Assert.That(result[0].Label, Is.EqualTo("B"));
        Assert.That(result[0].Score, Is.EqualTo(0.9f));
        Assert.That(result[1].Label, Is.EqualTo("D"));
        Assert.That(result[1].Score, Is.EqualTo(0.7f));
    }

    [Test]
    public void TopKDescending_WithPositionalIndex_ReturnsNullLabels()
    {
        var data = new float[] { 0.5f, 0.9f, 0.1f, 0.7f };
        using var series = NivaraSeries<float>.Create(data);

        var result = series.TopKDescending(2);

        Assert.That(result.Length, Is.EqualTo(2));
        Assert.That(result[0].Label, Is.Null);
        Assert.That(result[1].Label, Is.Null);
    }

    [Test]
    public void TopKDescending_WithCountZero_ReturnsEmpty()
    {
        using var series = NivaraSeries<float>.Create(new[] { 1.0f, 2.0f });

        var result = series.TopKDescending(0);

        Assert.That(result, Is.Empty);
    }

    [Test]
    public void TopKDescending_WithNegativeCount_Throws()
    {
        using var series = NivaraSeries<float>.Create(new[] { 1.0f, 2.0f });

        Assert.Throws<ArgumentOutOfRangeException>(() => series.TopKDescending(-1));
    }

    [Test]
    public void TopKDescending_WithCountLargerThanLength_ReturnsAll()
    {
        var data = new float[] { 0.5f, 0.9f };
        using var series = NivaraSeries<float>.Create(data);

        var result = series.TopKDescending(10);

        Assert.That(result.Length, Is.EqualTo(2));
    }

    [Test]
    public void TopKDescending_WithCountLargerThanNonNullCount_ExcludesNulls()
    {
        var column = NivaraColumn<float>.CreateFromNullable(new float?[] { 0.5f, null, 0.9f, null, 0.7f });
        using var series = new NivaraSeries<float>(column);

        var result = series.TopKDescending(10);

        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result.Select(item => item.Score), Is.EqualTo(new[] { 0.9f, 0.7f, 0.5f }));
    }

    [Test]
    public void TopKDescending_HeapPath_ExcludesNullsAndPreservesStableTieOrder()
    {
        var column = NivaraColumn<float>.CreateFromNullable(new float?[]
        {
            0.1f, 0.2f, 0.3f, 0.4f, 0.95f,
            null, 0.5f, 0.6f, 0.7f, 0.8f,
            0.95f, null, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f, 0.9f
        });
        var labels = Enumerable.Range(0, 20).Select(i => $"L{i}").Cast<object>().ToArray();
        using var series = new NivaraSeries<float>(column, NivaraColumn<object>.CreateForReferenceType(labels));

        var result = series.TopKDescending(2);

        Assert.That(result.Length, Is.EqualTo(2));
        Assert.That(result[0].Label, Is.EqualTo("L4"));
        Assert.That(result[0].Score, Is.EqualTo(0.95f));
        Assert.That(result[1].Label, Is.EqualTo("L10"));
        Assert.That(result[1].Score, Is.EqualTo(0.95f));
    }

    [Test]
    public void TopKDescending_WithEmptySeries_ReturnsEmpty()
    {
        using var series = new NivaraSeries<float>();

        var result = series.TopKDescending(5);

        Assert.That(result, Is.Empty);
    }

    [Test]
    public void RankingWorkflow_CosineSimilarityThenArgSort_ReturnsCorrectOrder()
    {
        using var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
            ("C", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
        });

        using var scores = products.CosineSimilarity(user);
        var ranking = scores.ArgSortDescending();

        Assert.That(ranking.Length, Is.EqualTo(3));
        Assert.That(ranking[0], Is.EqualTo(0));
        Assert.That(ranking[1], Is.EqualTo(2));
        Assert.That(ranking[2], Is.EqualTo(1));
    }

    [Test]
    public void RankingWorkflow_CosineSimilarityThenTopK_ReturnsLabeledResults()
    {
        using var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
            ("C", (IColumn)NivaraColumn<float>.Create(new[] { 0.7f, 0.1f, 0.8f, 0.2f })),
        });

        using var scores = products.CosineSimilarity(user);
        var top2 = scores.TopKDescending(2);

        Assert.That(top2.Length, Is.EqualTo(2));
        Assert.That(top2[0].Label, Is.EqualTo("A"));
        Assert.That(top2[1].Label, Is.EqualTo("C"));
    }

    [Test]
    public void ColumnNorms_WithFloatColumns_MatchesTensorPrimitives()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(new[] { 3.0f, 4.0f })),
            ("Y", (IColumn)NivaraColumn<float>.Create(new[] { 0.0f, 1.0f })),
        });

        using var norms = frame.ColumnNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms[0], Is.EqualTo(TensorPrimitives.Norm(new[] { 3.0f, 4.0f })).Within(1e-6f));
        Assert.That(norms[1], Is.EqualTo(TensorPrimitives.Norm(new[] { 0.0f, 1.0f })).Within(1e-6f));
        Assert.That(norms.IsNull(0), Is.False);
        Assert.That(norms.IsNull(1), Is.False);
    }

    [Test]
    public void ColumnNorms_WithNullColumn_ReturnsNullForThatColumn()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(new[] { 3.0f, 4.0f })),
            ("Y", (IColumn)NivaraColumn<float>.CreateFromNullable(new float?[] { null, 1.0f })),
        });

        using var norms = frame.ColumnNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms.IsNull(0), Is.False);
        Assert.That(norms.IsNull(1), Is.True);
    }

    [Test]
    public void RowNorms_WithFloatRows_MatchesTensorPrimitives()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(new[] { 3.0f, 0.0f })),
            ("Y", (IColumn)NivaraColumn<float>.Create(new[] { 4.0f, 1.0f })),
        });

        using var norms = frame.RowNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms[0], Is.EqualTo(TensorPrimitives.Norm(new[] { 3.0f, 4.0f })).Within(1e-6f));
        Assert.That(norms[1], Is.EqualTo(TensorPrimitives.Norm(new[] { 0.0f, 1.0f })).Within(1e-6f));
        Assert.That(norms.IsNull(0), Is.False);
        Assert.That(norms.IsNull(1), Is.False);
    }

    [Test]
    public void RowNorms_WithNullInRow_ReturnsNullForThatRow()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(new[] { 3.0f, 0.0f })),
            ("Y", (IColumn)NivaraColumn<float>.CreateFromNullable(new float?[] { null, 1.0f })),
        });

        using var norms = frame.RowNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms.IsNull(0), Is.True);
        Assert.That(norms.IsNull(1), Is.False);
    }

    [Test]
    public void RowNorms_WithWideFrame_UsesPooledBufferAndPreservesNulls()
    {
        var columns = Enumerable.Range(0, 1024)
            .Select(i => (
                $"C{i}",
                (IColumn)(i == 100
                    ? NivaraColumn<float>.CreateFromNullable(new float?[] { null, 2.0f })
                    : NivaraColumn<float>.Create(new[] { 1.0f, 2.0f }))))
            .ToArray();
        using var frame = new NivaraFrame(columns);

        using var norms = frame.RowNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms.IsNull(0), Is.True);
        Assert.That(norms.IsNull(1), Is.False);
        Assert.That(norms[1], Is.EqualTo(64.0f).Within(1e-6f));
    }

    [Test]
    public void FrameDot_WithEmptyColumnsAndEmptyVector_ReturnsZeroScorePerColumn()
    {
        using var vector = NivaraSeries<float>.Create(Array.Empty<float>());
        using var frame = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
            ("B", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
        });

        using var scores = frame.Dot(vector);

        Assert.That(scores.Length, Is.EqualTo(2));
        Assert.That(scores.GetLabel(0), Is.EqualTo("A"));
        Assert.That(scores.GetLabel(1), Is.EqualTo("B"));
        Assert.That(scores[0], Is.EqualTo(0.0f));
        Assert.That(scores[1], Is.EqualTo(0.0f));
        Assert.That(scores.IsNull(0), Is.False);
        Assert.That(scores.IsNull(1), Is.False);
    }

    [Test]
    public void ColumnNorms_WithEmptyColumns_ReturnsZeroNormPerColumn()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
            ("Y", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
        });

        using var norms = frame.ColumnNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(2));
        Assert.That(norms.GetLabel(0), Is.EqualTo("X"));
        Assert.That(norms.GetLabel(1), Is.EqualTo("Y"));
        Assert.That(norms[0], Is.EqualTo(0.0f));
        Assert.That(norms[1], Is.EqualTo(0.0f));
    }

    [Test]
    public void RowNorms_WithZeroRowFrame_ReturnsEmptySeries()
    {
        using var frame = new NivaraFrame(new[]
        {
            ("X", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
            ("Y", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
        });

        using var norms = frame.RowNorms<float>();

        Assert.That(norms.Length, Is.EqualTo(0));
    }

    [Test]
    public void CosineSimilarity_WithEmptyColumnsAndEmptyVector_ThrowsClearException()
    {
        using var vector = NivaraSeries<float>.Create(Array.Empty<float>());
        using var frame = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
            ("B", (IColumn)NivaraColumn<float>.Create(Array.Empty<float>())),
        });

        var ex = Assert.Throws<ArgumentException>(() => frame.CosineSimilarity(vector));
        Assert.That(ex!.Message, Does.Contain("Cannot compute cosine similarity for empty vectors"));
    }

    [Test]
    public void CosineSimilarity_WithLengthMismatch_ThrowsArgumentException()
    {
        using var vector = NivaraSeries<float>.Create(new[] { 1.0f });
        using var frame = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 1.0f, 2.0f })),
        });

        Assert.Throws<ArgumentException>(() => frame.CosineSimilarity(vector));
    }

    [Test]
    public void Dot_WithNullVector_ReturnsNullForAllColumns()
    {
        using var vector = new NivaraSeries<float>(NivaraColumn<float>.CreateFromNullable(new float?[] { null, 0.1f, 0.6f, 0.3f }));
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.Create(new[] { 0.1f, 0.9f, 0.2f, 0.7f })),
        });

        using var scores = products.Dot(vector);

        Assert.That(scores.Length, Is.EqualTo(2));
        Assert.That(scores.IsNull(0), Is.True);
        Assert.That(scores.IsNull(1), Is.True);
    }

    [Test]
    public void CosineSimilarity_WithNullInOneColumn_ReturnsNullOnlyForThatColumn()
    {
        using var user = NivaraSeries<float>.Create(new[] { 0.8f, 0.1f, 0.6f, 0.3f });
        using var products = new NivaraFrame(new[]
        {
            ("A", (IColumn)NivaraColumn<float>.Create(new[] { 0.9f, 0.2f, 0.5f, 0.4f })),
            ("B", (IColumn)NivaraColumn<float>.CreateFromNullable(new float?[] { 0.1f, null, 0.2f, 0.7f })),
        });

        using var scores = products.CosineSimilarity(user);

        Assert.That(scores.Length, Is.EqualTo(2));
        Assert.That(scores.IsNull(0), Is.False);
        Assert.That(scores.IsNull(1), Is.True);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Test]
    public void FromTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        Tensor<double> nullTensor = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TensorInteropExtensions.FromTensor(nullTensor));
    }

    [Test]
    public void ReshapeToTensor_WithEmptyDimensions_ThrowsException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f };
        using var series = NivaraSeries<float>.Create(data);

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => series.ReshapeToTensor());
        Assert.That(ex.Message, Does.Contain("At least one dimension must be specified"));
    }

    [Test]
    public void ToTensorSpan_WithDisposedSeries_ThrowsObjectDisposedException()
    {
        // Arrange
        var data = new int[] { 1, 2, 3 };
        var series = NivaraSeries<int>.Create(data);
        series.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => series.ToTensorSpan());
    }

    #endregion

    #region Performance and Memory Tests

    [Test]
    public void TensorSpanConversion_IsZeroCopy_ForTensorStorage()
    {
        // This test verifies that TensorSpan conversion is efficient
        // We can't easily test true zero-copy without internal access,
        // but we can verify the operation completes quickly for large data

        // Arrange
        var largeData = Enumerable.Range(0, 100000).Select(i => (float)i).ToArray();
        using var series = NivaraSeries<float>.Create(largeData);

        // Act & Assert - Should complete quickly without throwing
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var tensorSpan = series.ToTensorSpan();
        stopwatch.Stop();

        // Verify the operation was fast (less than 100ms for 100k elements)
        Assert.That(stopwatch.ElapsedMilliseconds, Is.LessThan(100));

        // Verify data integrity for a few sample points
        Assert.That(tensorSpan[0], Is.EqualTo(0.0f));
        Assert.That(tensorSpan[50000], Is.EqualTo(50000.0f));
        Assert.That(tensorSpan[99999], Is.EqualTo(99999.0f));
    }

    #endregion
}
