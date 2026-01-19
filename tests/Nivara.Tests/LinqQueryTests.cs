using Nivara.Expressions;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class LinqQueryTests
{
    [Test]
    public void Where_WithLambda_FiltersCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Where(x => x["Numbers"] > 3)
            .ToNivaraFrame();

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(4));
        Assert.That(numbers[1], Is.EqualTo(5));
    }

    [Test]
    public void Select_WithLambda_ProjectsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Select(x => x["Letters"])
            .ToNivaraFrame();

        // Assert
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Letters" }));
        Assert.That(result.RowCount, Is.EqualTo(5));
    }

    [Test]
    public void Select_WithMultipleLambdas_ProjectsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Select(x => x["Letters"], x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        Assert.That(result.ColumnNames, Is.EqualTo(new[] { "Letters", "Numbers" }));
    }

    [Test]
    public void OrderBy_WithLambda_SortsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderBy(x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(1));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(3));
    }

    [Test]
    public void OrderByDescending_WithLambda_SortsCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 3, 2 });
        var frame = NivaraFrame.Create(("Numbers", col1));

        // Act
        var result = frame.AsQueryFrame()
            .OrderByDescending(x => x["Numbers"])
            .ToNivaraFrame();

        // Assert
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(3));
        Assert.That(numbers[1], Is.EqualTo(2));
        Assert.That(numbers[2], Is.EqualTo(1));
    }

    [Test]
    public void ChainedLinqOperations_RunCorrectly()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 5, 1, 4, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "e", "a", "d", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.AsQueryFrame()
            .Where(x => x["Numbers"] > 2)
            .OrderBy(x => x["Numbers"])
            .Select(x => x["Letters"])
            .ToList();

        // Assert
        // Expected: 5, 4, 3 -> Sorted: 3, 4, 5 -> Letters: c, d, e
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnNames.Count, Is.EqualTo(1));
        
        var letters = result.GetColumn<string>("Letters");
        Assert.That(letters[0], Is.EqualTo("c")); // 3
        Assert.That(letters[1], Is.EqualTo("d")); // 4
        Assert.That(letters[2], Is.EqualTo("e")); // 5
    }
}
