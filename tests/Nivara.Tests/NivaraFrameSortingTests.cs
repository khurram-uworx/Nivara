using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraFrame sorting and reordering functionality
/// </summary>
[TestFixture]
public class NivaraFrameSortingTests
{
    [Test]
    public void ReorderByIndices_WithValidIndices_ShouldReorderRows()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var letters = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", numbers), ("Letters", letters));

        var indices = new[] { 2, 0, 1 }; // Reorder to: 30,c -> 10,a -> 20,b

        // Act
        var reordered = frame.ReorderByIndices(indices);

        // Assert
        Assert.That(reordered.RowCount, Is.EqualTo(3));
        Assert.That(reordered.ColumnCount, Is.EqualTo(2));

        var reorderedNumbers = reordered.GetColumn<int>("Numbers");
        var reorderedLetters = reordered.GetColumn<string>("Letters");

        Assert.That(reorderedNumbers.ToArray(), Is.EqualTo(new[] { 30, 10, 20 }));
        Assert.That(reorderedLetters.ToArray(), Is.EqualTo(new[] { "c", "a", "b" }));
    }

    [Test]
    public void ReorderByIndices_WithIdentityIndices_ShouldReturnEquivalentFrame()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var letters = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", numbers), ("Letters", letters));

        var indices = new[] { 0, 1, 2 }; // Identity order

        // Act
        var reordered = frame.ReorderByIndices(indices);

        // Assert
        Assert.That(reordered.RowCount, Is.EqualTo(3));
        Assert.That(reordered.ColumnCount, Is.EqualTo(2));

        var reorderedNumbers = reordered.GetColumn<int>("Numbers");
        var reorderedLetters = reordered.GetColumn<string>("Letters");

        Assert.That(reorderedNumbers.ToArray(), Is.EqualTo(new[] { 10, 20, 30 }));
        Assert.That(reorderedLetters.ToArray(), Is.EqualTo(new[] { "a", "b", "c" }));
    }

    [Test]
    public void ReorderByIndices_WithNullIndices_ShouldThrowArgumentNullException()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var frame = NivaraFrame.Create(("Numbers", numbers));

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => frame.ReorderByIndices(null!));
    }

    [Test]
    public void ReorderByIndices_WithWrongLength_ShouldThrowArgumentException()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var frame = NivaraFrame.Create(("Numbers", numbers));

        var wrongLengthIndices = new[] { 0, 1 }; // Should be length 3

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.ReorderByIndices(wrongLengthIndices));
        Assert.That(ex.Message, Contains.Substring("length"));
    }

    [Test]
    public void ReorderByIndices_WithOutOfBoundsIndices_ShouldThrowArgumentException()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var frame = NivaraFrame.Create(("Numbers", numbers));

        var outOfBoundsIndices = new[] { 0, 1, 5 }; // Index 5 is out of bounds

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.ReorderByIndices(outOfBoundsIndices));
        Assert.That(ex.Message, Contains.Substring("out of bounds"));
    }

    [Test]
    public void ReorderByIndices_WithNegativeIndices_ShouldThrowArgumentException()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var frame = NivaraFrame.Create(("Numbers", numbers));

        var negativeIndices = new[] { 0, -1, 2 }; // Index -1 is invalid

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.ReorderByIndices(negativeIndices));
        Assert.That(ex.Message, Contains.Substring("out of bounds"));
    }

    [Test]
    public void ReorderByIndices_WithDuplicateIndices_ShouldDuplicateRows()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var letters = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", numbers), ("Letters", letters));

        var duplicateIndices = new[] { 0, 0, 1 }; // Duplicate first row

        // Act
        var reordered = frame.ReorderByIndices(duplicateIndices);

        // Assert
        Assert.That(reordered.RowCount, Is.EqualTo(3));

        var reorderedNumbers = reordered.GetColumn<int>("Numbers");
        var reorderedLetters = reordered.GetColumn<string>("Letters");

        Assert.That(reorderedNumbers.ToArray(), Is.EqualTo(new[] { 10, 10, 20 }));
        Assert.That(reorderedLetters.ToArray(), Is.EqualTo(new[] { "a", "a", "b" }));
    }

    [Test]
    public void ReorderByIndices_WithNullableColumns_ShouldPreserveNulls()
    {
        // Arrange
        var nullableNumbers = NivaraColumn<int>.CreateFromNullable(new int?[] { 10, null, 30 });
        var letters = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", nullableNumbers), ("Letters", letters));

        var indices = new[] { 1, 2, 0 }; // Reorder to: null,b -> 30,c -> 10,a

        // Act
        var reordered = frame.ReorderByIndices(indices);

        // Assert
        var reorderedNumbers = reordered.GetColumn<int>("Numbers");
        var reorderedLetters = reordered.GetColumn<string>("Letters");

        Assert.That(reorderedNumbers.GetValue(0), Is.Null);
        Assert.That(reorderedNumbers.GetValue(1), Is.EqualTo(30));
        Assert.That(reorderedNumbers.GetValue(2), Is.EqualTo(10));

        Assert.That(reorderedLetters.ToArray(), Is.EqualTo(new[] { "b", "c", "a" }));
    }

    [Test]
    public void ReorderByIndices_WithMixedTypes_ShouldHandleAllTypes()
    {
        // Arrange
        var integers = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var doubles = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3 });
        var strings = NivaraColumn<string>.Create(new[] { "one", "two", "three" });
        var booleans = NivaraColumn<bool>.Create(new[] { true, false, true });
        var dates = NivaraColumn<DateTime>.Create(new[] {
            new DateTime(2023, 1, 1),
            new DateTime(2023, 2, 1),
            new DateTime(2023, 3, 1)
        });

        var frame = NivaraFrame.Create(
            ("Integers", integers),
            ("Doubles", doubles),
            ("Strings", strings),
            ("Booleans", booleans),
            ("Dates", dates)
        );

        var indices = new[] { 2, 0, 1 }; // Reverse order: 3rd, 1st, 2nd

        // Act
        var reordered = frame.ReorderByIndices(indices);

        // Assert
        Assert.That(reordered.GetColumn<int>("Integers").ToArray(), Is.EqualTo(new[] { 3, 1, 2 }));
        Assert.That(reordered.GetColumn<double>("Doubles").ToArray(), Is.EqualTo(new[] { 3.3, 1.1, 2.2 }));
        Assert.That(reordered.GetColumn<string>("Strings").ToArray(), Is.EqualTo(new[] { "three", "one", "two" }));
        Assert.That(reordered.GetColumn<bool>("Booleans").ToArray(), Is.EqualTo(new[] { true, true, false }));
        Assert.That(reordered.GetColumn<DateTime>("Dates").ToArray(), Is.EqualTo(new[] {
            new DateTime(2023, 3, 1),
            new DateTime(2023, 1, 1),
            new DateTime(2023, 2, 1)
        }));
    }

    [Test]
    public void ReorderByIndices_WithEmptyFrame_ShouldHandleGracefully()
    {
        // Arrange
        var emptyNumbers = NivaraColumn<int>.Create(Array.Empty<int>());
        var frame = NivaraFrame.Create(("Numbers", emptyNumbers));

        var emptyIndices = Array.Empty<int>();

        // Act
        var reordered = frame.ReorderByIndices(emptyIndices);

        // Assert
        Assert.That(reordered.RowCount, Is.EqualTo(0));
        Assert.That(reordered.ColumnCount, Is.EqualTo(1));
        Assert.That(reordered.ColumnNames, Contains.Item("Numbers"));
    }

    [Test]
    public void ReorderByIndices_WithSingleRow_ShouldReturnSameData()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 42 });
        var letters = NivaraColumn<string>.Create(new[] { "answer" });
        var frame = NivaraFrame.Create(("Numbers", numbers), ("Letters", letters));

        var indices = new[] { 0 };

        // Act
        var reordered = frame.ReorderByIndices(indices);

        // Assert
        Assert.That(reordered.RowCount, Is.EqualTo(1));
        Assert.That(reordered.GetColumn<int>("Numbers").ToArray(), Is.EqualTo(new[] { 42 }));
        Assert.That(reordered.GetColumn<string>("Letters").ToArray(), Is.EqualTo(new[] { "answer" }));
    }
}