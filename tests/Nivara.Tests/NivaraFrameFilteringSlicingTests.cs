using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraFrameFilteringSlicingTests
{
    [Test]
    public void FilterByMask_WithValidMask_ReturnsFilteredFrame()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d", "e" });
        var frame = NivaraFrame.Create(
            ("Numbers", intColumn),
            ("Letters", stringColumn)
        );

        var mask = NivaraColumn<bool>.Create(new[] { true, false, true, false, true });

        // Act
        var filtered = frame.FilterByMask(mask);

        // Assert
        Assert.That(filtered.RowCount, Is.EqualTo(3), "Filtered frame should have 3 rows");
        Assert.That(filtered.ColumnCount, Is.EqualTo(2), "Filtered frame should preserve column count");
        
        var filteredNumbers = filtered.GetColumn<int>("Numbers");
        var filteredLetters = filtered.GetColumn<string>("Letters");
        
        Assert.That(filteredNumbers[0], Is.EqualTo(1), "First filtered number should be 1");
        Assert.That(filteredNumbers[1], Is.EqualTo(3), "Second filtered number should be 3");
        Assert.That(filteredNumbers[2], Is.EqualTo(5), "Third filtered number should be 5");
        
        Assert.That(filteredLetters[0], Is.EqualTo("a"), "First filtered letter should be 'a'");
        Assert.That(filteredLetters[1], Is.EqualTo("c"), "Second filtered letter should be 'c'");
        Assert.That(filteredLetters[2], Is.EqualTo("e"), "Third filtered letter should be 'e'");
    }

    [Test]
    public void FilterByMask_WithAllFalseMask_ReturnsEmptyFrame()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));
        var mask = NivaraColumn<bool>.Create(new[] { false, false, false });

        // Act
        var filtered = frame.FilterByMask(mask);

        // Assert
        Assert.That(filtered.RowCount, Is.EqualTo(0), "Filtered frame should be empty");
        Assert.That(filtered.ColumnCount, Is.EqualTo(1), "Filtered frame should preserve column count");
        Assert.That(filtered.HasColumn("Numbers"), Is.True, "Filtered frame should preserve column names");
    }

    [Test]
    public void FilterByMask_WithMismatchedLength_ThrowsArgumentException()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));
        var mask = NivaraColumn<bool>.Create(new[] { true, false }); // Wrong length

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.FilterByMask(mask));
        Assert.That(ex!.Message, Does.Contain("Mask length (2) must match frame row count (3)"));
    }

    [Test]
    public void Take_WithValidCount_ReturnsFirstNRows()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var taken = frame.Take(3);

        // Assert
        Assert.That(taken.RowCount, Is.EqualTo(3), "Taken frame should have 3 rows");
        var numbers = taken.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(1), "First number should be 1");
        Assert.That(numbers[1], Is.EqualTo(2), "Second number should be 2");
        Assert.That(numbers[2], Is.EqualTo(3), "Third number should be 3");
    }

    [Test]
    public void Take_WithCountGreaterThanRowCount_ReturnsAllRows()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var taken = frame.Take(10);

        // Assert
        Assert.That(taken.RowCount, Is.EqualTo(3), "Taken frame should have all 3 rows");
    }

    [Test]
    public void Take_WithZeroCount_ReturnsEmptyFrame()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var taken = frame.Take(0);

        // Assert
        Assert.That(taken.RowCount, Is.EqualTo(0), "Taken frame should be empty");
        Assert.That(taken.ColumnCount, Is.EqualTo(1), "Taken frame should preserve column count");
    }

    [Test]
    public void Skip_WithValidCount_ReturnsRemainingRows()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var skipped = frame.Skip(2);

        // Assert
        Assert.That(skipped.RowCount, Is.EqualTo(3), "Skipped frame should have 3 rows");
        var numbers = skipped.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(3), "First number should be 3");
        Assert.That(numbers[1], Is.EqualTo(4), "Second number should be 4");
        Assert.That(numbers[2], Is.EqualTo(5), "Third number should be 5");
    }

    [Test]
    public void Skip_WithCountGreaterThanRowCount_ReturnsEmptyFrame()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var skipped = frame.Skip(10);

        // Assert
        Assert.That(skipped.RowCount, Is.EqualTo(0), "Skipped frame should be empty");
        Assert.That(skipped.ColumnCount, Is.EqualTo(1), "Skipped frame should preserve column count");
    }

    [Test]
    public void SkipThenTake_AppliesOperationsInCorrectOrder()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act - Skip first 3, then take next 4
        var result = frame.Skip(3).Take(4);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(4), "Result should have 4 rows");
        var numbers = result.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(4), "First number should be 4");
        Assert.That(numbers[1], Is.EqualTo(5), "Second number should be 5");
        Assert.That(numbers[2], Is.EqualTo(6), "Third number should be 6");
        Assert.That(numbers[3], Is.EqualTo(7), "Fourth number should be 7");
    }

    [Test]
    public void Slice_WithValidParameters_ReturnsCorrectSubset()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act
        var sliced = frame.Slice(2, 4); // Start at index 2, take 4 elements

        // Assert
        Assert.That(sliced.RowCount, Is.EqualTo(4), "Sliced frame should have 4 rows");
        var numbers = sliced.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(3), "First number should be 3");
        Assert.That(numbers[1], Is.EqualTo(4), "Second number should be 4");
        Assert.That(numbers[2], Is.EqualTo(5), "Third number should be 5");
        Assert.That(numbers[3], Is.EqualTo(6), "Fourth number should be 6");
    }

    [Test]
    public void Slice_WithInvalidParameters_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.Slice(-1, 2), "Negative start should throw");
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.Slice(0, -1), "Negative length should throw");
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.Slice(2, 3), "Start + length > row count should throw");
    }

    [Test]
    public void FilteringAndSlicing_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange
        var nullableInts = new int?[] { 1, null, 3, null, 5 };
        var intColumn = NivaraColumn<int>.CreateFromNullable(nullableInts);
        var frame = NivaraFrame.Create(("Numbers", intColumn));

        // Act - Filter to keep only non-null values
        var mask = NivaraColumn<bool>.Create(new[] { true, false, true, false, true });
        var filtered = frame.FilterByMask(mask);

        // Assert
        Assert.That(filtered.RowCount, Is.EqualTo(3), "Filtered frame should have 3 rows");
        var numbers = filtered.GetColumn<int>("Numbers");
        Assert.That(numbers[0], Is.EqualTo(1), "First number should be 1");
        Assert.That(numbers[1], Is.EqualTo(3), "Second number should be 3");
        Assert.That(numbers[2], Is.EqualTo(5), "Third number should be 5");
    }
}