using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraSeriesIsValidTests
{
    [Test]
    public void IsValid_WithValidData_ReturnsTrue()
    {
        // Arrange
        var data = new int[] { 1, 2, 3, 4, 5 };
        var series = NivaraSeries<int>.Create(data);

        // Act & Assert
        for (int i = 0; i < series.Length; i++)
        {
            Assert.That(series.IsValid(i), Is.True, $"Index {i} should be valid");
            Assert.That(series.IsNull(i), Is.False, $"Index {i} should not be null");
        }
    }

    [Test]
    public void IsValid_WithNullableData_ReturnsCorrectValues()
    {
        // Arrange
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<int>(column);

        // Act & Assert
        Assert.That(series.IsValid(0), Is.True, "Index 0 should be valid");
        Assert.That(series.IsNull(0), Is.False, "Index 0 should not be null");

        Assert.That(series.IsValid(1), Is.False, "Index 1 should not be valid");
        Assert.That(series.IsNull(1), Is.True, "Index 1 should be null");

        Assert.That(series.IsValid(2), Is.True, "Index 2 should be valid");
        Assert.That(series.IsNull(2), Is.False, "Index 2 should not be null");

        Assert.That(series.IsValid(3), Is.False, "Index 3 should not be valid");
        Assert.That(series.IsNull(3), Is.True, "Index 3 should be null");

        Assert.That(series.IsValid(4), Is.True, "Index 4 should be valid");
        Assert.That(series.IsNull(4), Is.False, "Index 4 should not be null");
    }

    [Test]
    public void IsValid_OutOfBounds_ThrowsIndexOutOfRangeException()
    {
        // Arrange
        var data = new int[] { 1, 2, 3 };
        var series = NivaraSeries<int>.Create(data);

        // Act & Assert
        Assert.Throws<IndexOutOfRangeException>(() => series.IsValid(-1));
        Assert.Throws<IndexOutOfRangeException>(() => series.IsValid(3));
        Assert.Throws<IndexOutOfRangeException>(() => series.IsValid(10));
    }

    [Test]
    public void IsValid_EmptySeries_ThrowsIndexOutOfRangeException()
    {
        // Arrange
        var series = new NivaraSeries<int>();

        // Act & Assert
        Assert.Throws<IndexOutOfRangeException>(() => series.IsValid(0));
    }

    [Test]
    public void TensorExtensions_WithValidData_WorksCorrectly()
    {
        // Arrange
        var series1 = NivaraSeries<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var series2 = NivaraSeries<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });

        // Act
        var sum = series1.AddTensor(series2);
        var product = series1.MultiplyTensor(series2);
        var dotProduct = series1.DotProduct(series2);
        var norm = series1.Norm();
        var tensorSum = series1.SumTensor();

        // Assert
        Assert.That(sum.Length, Is.EqualTo(3));
        Assert.That(sum[0], Is.EqualTo(5.0f).Within(0.001f));
        Assert.That(sum[1], Is.EqualTo(7.0f).Within(0.001f));
        Assert.That(sum[2], Is.EqualTo(9.0f).Within(0.001f));

        Assert.That(product.Length, Is.EqualTo(3));
        Assert.That(product[0], Is.EqualTo(4.0f).Within(0.001f));
        Assert.That(product[1], Is.EqualTo(10.0f).Within(0.001f));
        Assert.That(product[2], Is.EqualTo(18.0f).Within(0.001f));

        Assert.That(dotProduct, Is.EqualTo(32.0f).Within(0.001f)); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.That(norm, Is.EqualTo(3.7416573f).Within(0.001f)); // sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.742
        Assert.That(tensorSum, Is.EqualTo(6.0f).Within(0.001f)); // 1 + 2 + 3 = 6
    }

    [Test]
    public void TensorExtensions_WithNullData_ElementwiseHelpersPropagateNullsAndReducersThrow()
    {
        // Arrange
        var nullableData1 = new float?[] { 1.0f, null, 3.0f };
        var column1 = NivaraColumn<float>.CreateFromNullable(nullableData1);
        var series1 = new NivaraSeries<float>(column1);
        var series2 = NivaraSeries<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });

        // Act
        var sum = series1.AddTensor(series2);
        var product = series1.MultiplyTensor(series2);

        // Assert
        Assert.That(sum.Length, Is.EqualTo(3));
        Assert.That(sum[0], Is.EqualTo(5.0f).Within(0.001f));
        Assert.That(sum.IsNull(1), Is.True);
        Assert.That(sum[2], Is.EqualTo(9.0f).Within(0.001f));

        Assert.That(product.Length, Is.EqualTo(3));
        Assert.That(product[0], Is.EqualTo(4.0f).Within(0.001f));
        Assert.That(product.IsNull(1), Is.True);
        Assert.That(product[2], Is.EqualTo(18.0f).Within(0.001f));

        var dotProductException = Assert.Throws<InvalidOperationException>(() => series1.DotProduct(series2));
        var normException = Assert.Throws<InvalidOperationException>(() => series1.Norm());
        var sumException = Assert.Throws<InvalidOperationException>(() => series1.SumTensor());

        Assert.That(dotProductException!.Message, Does.Contain("null values").And.Contain("index 1"));
        Assert.That(normException!.Message, Does.Contain("null values").And.Contain("index 1"));
        Assert.That(sumException!.Message, Does.Contain("null values").And.Contain("index 1"));
    }

    [Test]
    public void TensorExtensions_WithDifferentLengths_ThrowsArgumentException()
    {
        // Arrange
        var series1 = NivaraSeries<float>.Create(new float[] { 1.0f, 2.0f });
        var series2 = NivaraSeries<float>.Create(new float[] { 4.0f, 5.0f, 6.0f });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => series1.AddTensor(series2));
        Assert.Throws<ArgumentException>(() => series1.MultiplyTensor(series2));
        Assert.Throws<ArgumentException>(() => series1.DotProduct(series2));
    }

    [Test]
    public void TensorExtensions_WithEmptySeries_ReturnsExpectedResults()
    {
        // Arrange
        var emptySeries1 = new NivaraSeries<float>();
        var emptySeries2 = new NivaraSeries<float>();

        // Act
        var sum = emptySeries1.AddTensor(emptySeries2);
        var product = emptySeries1.MultiplyTensor(emptySeries2);
        var dotProduct = emptySeries1.DotProduct(emptySeries2);
        var norm = emptySeries1.Norm();
        var tensorSum = emptySeries1.SumTensor();

        // Assert
        Assert.That(sum.Length, Is.EqualTo(0));
        Assert.That(product.Length, Is.EqualTo(0));
        Assert.That(dotProduct, Is.EqualTo(0.0f));
        Assert.That(norm, Is.EqualTo(0.0f));
        Assert.That(tensorSum, Is.EqualTo(0.0f));
    }

    [Test]
    public void TensorExtensions_WithDoubleType_WorksCorrectly()
    {
        // Arrange
        var series1 = NivaraSeries<double>.Create(new double[] { 1.0, 2.0, 3.0 });
        var series2 = NivaraSeries<double>.Create(new double[] { 4.0, 5.0, 6.0 });

        // Act
        var sum = series1.AddTensor(series2);
        var product = series1.MultiplyTensor(series2);
        var dotProduct = series1.DotProduct(series2);
        var norm = series1.Norm();

        // Assert
        Assert.That(sum[0], Is.EqualTo(5.0).Within(0.001));
        Assert.That(sum[1], Is.EqualTo(7.0).Within(0.001));
        Assert.That(sum[2], Is.EqualTo(9.0).Within(0.001));

        Assert.That(product[0], Is.EqualTo(4.0).Within(0.001));
        Assert.That(product[1], Is.EqualTo(10.0).Within(0.001));
        Assert.That(product[2], Is.EqualTo(18.0).Within(0.001));

        Assert.That(dotProduct, Is.EqualTo(32.0).Within(0.001));
        Assert.That(norm, Is.EqualTo(3.7416573867739413).Within(0.001));
    }

    [Test]
    public void TensorExtensions_WithIntegerType_UsesFallback()
    {
        // Arrange
        var series1 = NivaraSeries<int>.Create(new int[] { 1, 2, 3 });
        var series2 = NivaraSeries<int>.Create(new int[] { 4, 5, 6 });

        // Act
        var sum = series1.AddTensor(series2);
        var product = series1.MultiplyTensor(series2);
        var dotProduct = series1.DotProduct(series2);
        var norm = series1.Norm(); // For integers, this returns sum of squares
        var tensorSum = series1.SumTensor();

        // Assert
        Assert.That(sum[0], Is.EqualTo(5));
        Assert.That(sum[1], Is.EqualTo(7));
        Assert.That(sum[2], Is.EqualTo(9));

        Assert.That(product[0], Is.EqualTo(4));
        Assert.That(product[1], Is.EqualTo(10));
        Assert.That(product[2], Is.EqualTo(18));

        Assert.That(dotProduct, Is.EqualTo(32)); // 1*4 + 2*5 + 3*6 = 32
        Assert.That(norm, Is.EqualTo(14)); // For integers: 1^2 + 2^2 + 3^2 = 14 (sum of squares)
        Assert.That(tensorSum, Is.EqualTo(6)); // 1 + 2 + 3 = 6
    }

    [Test]
    public void TransformTensor_WithValidData_WorksCorrectly()
    {
        // Arrange
        var series = NivaraSeries<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });

        // Act
        var transformed = series.TransformTensor(x => x * 2.0f);

        // Assert
        Assert.That(transformed.Length, Is.EqualTo(3));
        Assert.That(transformed[0], Is.EqualTo(2.0f).Within(0.001f));
        Assert.That(transformed[1], Is.EqualTo(4.0f).Within(0.001f));
        Assert.That(transformed[2], Is.EqualTo(6.0f).Within(0.001f));
    }

    [Test]
    public void TransformTensor_WithNullData_ThrowsInvalidOperationException()
    {
        // Arrange
        var nullableData = new float?[] { 1.0f, null, 3.0f };
        var column = NivaraColumn<float>.CreateFromNullable(nullableData);
        var series = new NivaraSeries<float>(column);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => series.TransformTensor(x => x * 2.0f));
    }
}
