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
}
