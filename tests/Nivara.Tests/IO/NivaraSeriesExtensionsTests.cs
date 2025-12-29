using NUnit.Framework;
using Nivara.IO;

namespace Nivara.Tests.IO;

[TestFixture]
public class NivaraSeriesExtensionsTests
{
    [Test]
    public void ToArrowArray_WithValidSeries_ShouldNotThrow()
    {
        // Arrange
        var series = CreateTestSeries();

        // Act & Assert - should not throw
        Assert.DoesNotThrow(() => series.ToArrowArray());
    }

    [Test]
    public void ToArrowArray_WithNullSeries_ShouldThrowArgumentNullException()
    {
        // Arrange
        NivaraSeries<int> series = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => series.ToArrowArray());
    }

    [Test]
    public void ToNivaraSeries_WithNullArray_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => NivaraSeriesExtensions.ToNivaraSeries<int>(null!));
    }

    [Test]
    public void ExtensionMethods_ShouldSupportGenericTypeConstraints()
    {
        // Arrange
        var intSeries = NivaraSeries<int>.Create(new[] { 1, 2, 3 });
        var stringSeries = NivaraSeries<string>.Create(new[] { "a", "b", "c" });

        // Act & Assert - should work with different types
        Assert.DoesNotThrow(() => intSeries.ToArrowArray());
        Assert.DoesNotThrow(() => stringSeries.ToArrowArray());
    }

    [Test]
    public void ExtensionMethods_ShouldSupportRoundTripConversion()
    {
        // Arrange
        var originalSeries = CreateTestSeries();

        // Act
        var arrowArray = originalSeries.ToArrowArray();
        var roundTripSeries = arrowArray.ToNivaraSeries<int>();

        // Assert
        Assert.That(roundTripSeries, Is.Not.Null);
        Assert.That(roundTripSeries.Length, Is.EqualTo(originalSeries.Length));
        
        for (int i = 0; i < originalSeries.Length; i++)
        {
            Assert.That(roundTripSeries[i], Is.EqualTo(originalSeries[i]));
        }
    }

    [Test]
    public void ExtensionMethods_ShouldProvideDefaultParameterOverloads()
    {
        // Arrange
        var series = CreateTestSeries();

        // Act & Assert - methods should work without options parameter
        Assert.DoesNotThrow(() => series.ToArrowArray());
        
        // And with options parameter
        var options = new ArrowConversionOptions();
        Assert.DoesNotThrow(() => series.ToArrowArray(options));
    }

    [Test]
    public void ExtensionMethods_ShouldHandleEmptySeries()
    {
        // Arrange
        var emptySeries = NivaraSeries<int>.Create(Array.Empty<int>());

        // Act & Assert - should handle empty series gracefully
        Assert.DoesNotThrow(() => emptySeries.ToArrowArray());
        
        var arrowArray = emptySeries.ToArrowArray();
        Assert.That(arrowArray.Length, Is.EqualTo(0));
    }

    private static NivaraSeries<int> CreateTestSeries()
    {
        return NivaraSeries<int>.Create(new[] { 1, 2, 3, 4, 5 });
    }
}