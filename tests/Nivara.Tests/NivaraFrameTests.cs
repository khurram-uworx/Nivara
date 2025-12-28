using NUnit.Framework;
using Nivara.Exceptions;

namespace Nivara.Tests;

/// <summary>
/// Tests for NivaraFrame core functionality including creation, column management, and validation
/// </summary>
[TestFixture]
public class NivaraFrameTests
{
    [Test]
    public void Create_WithValidColumns_ShouldCreateFrame()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });

        // Act
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.ColumnNames, Is.EqualTo(new[] { "Numbers", "Letters" }));
    }

    [Test]
    public void Create_WithDictionary_ShouldCreateFrame()
    {
        // Arrange
        var columns = new Dictionary<string, IColumn>
        {
            ["Numbers"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }),
            ["Letters"] = NivaraColumn<string>.Create(new[] { "a", "b", "c" })
        };

        // Act
        var frame = NivaraFrame.Create(columns);

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.ColumnNames, Contains.Item("Numbers"));
        Assert.That(frame.ColumnNames, Contains.Item("Letters"));
    }

    [Test]
    public void Constructor_WithNullColumns_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new NivaraFrame(null!));
    }

    [Test]
    public void Constructor_WithEmptyColumns_ShouldThrowArgumentException()
    {
        // Arrange
        var emptyColumns = Array.Empty<(string, IColumn)>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(emptyColumns));
        Assert.That(ex.Message, Contains.Substring("must contain at least one column"));
    }

    [Test]
    public void Constructor_WithNullColumnName_ShouldThrowArgumentException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var columns = new (string, IColumn)[] { (null!, col) };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(columns));
        Assert.That(ex.Message, Contains.Substring("Column names cannot be null or whitespace"));
    }

    [Test]
    public void Constructor_WithWhitespaceColumnName_ShouldThrowArgumentException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var columns = new[] { ("   ", (IColumn)col) };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(columns));
        Assert.That(ex.Message, Contains.Substring("Column names cannot be null or whitespace"));
    }

    [Test]
    public void Constructor_WithNullColumn_ShouldThrowArgumentException()
    {
        // Arrange
        var columns = new (string, IColumn)[] { ("Test", null!) };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(columns));
        Assert.That(ex.Message, Contains.Substring("Column 'Test' cannot be null"));
    }

    [Test]
    public void Constructor_WithDuplicateColumnNames_ShouldThrowArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var columns = new[] { ("Test", (IColumn)col1), ("test", (IColumn)col2) }; // Case-insensitive duplicate

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(columns));
        Assert.That(ex.Message, Contains.Substring("Duplicate column name 'test' found"));
    }

    [Test]
    public void Constructor_WithDifferentColumnLengths_ShouldThrowArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b" }); // Different length
        var columns = new[] { ("Numbers", (IColumn)col1), ("Letters", (IColumn)col2) };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new NivaraFrame(columns));
        Assert.That(ex.Message, Contains.Substring("Column length mismatch"));
        Assert.That(ex.Message, Contains.Substring("Column 'Letters' has length 2, but expected 3"));
    }

    [Test]
    public void GetColumn_WithValidName_ShouldReturnTypedColumn()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var retrievedCol = frame.GetColumn<int>("Numbers");

        // Assert
        Assert.That(retrievedCol, Is.Not.Null);
        Assert.That(retrievedCol[0], Is.EqualTo(1));
        Assert.That(retrievedCol[1], Is.EqualTo(2));
        Assert.That(retrievedCol[2], Is.EqualTo(3));
    }

    [Test]
    public void GetColumn_WithCaseInsensitiveName_ShouldReturnColumn()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act
        var retrievedCol = frame.GetColumn<int>("NUMBERS");

        // Assert
        Assert.That(retrievedCol, Is.Not.Null);
        Assert.That(retrievedCol[0], Is.EqualTo(1));
    }

    [Test]
    public void GetColumn_WithNonExistentName_ShouldThrowColumnNotFoundException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => frame.GetColumn<int>("NonExistent"));
        Assert.That(ex.ColumnName, Is.EqualTo("NonExistent"));
        Assert.That(ex.AvailableColumns, Contains.Item("Numbers"));
    }

    [Test]
    public void GetColumn_WithWrongType_ShouldThrowColumnTypeMismatchException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<ColumnTypeMismatchException>(() => frame.GetColumn<string>("Numbers"));
        Assert.That(ex.ColumnName, Is.EqualTo("Numbers"));
        Assert.That(ex.ExpectedType, Is.EqualTo(typeof(string)));
        Assert.That(ex.ActualType, Is.EqualTo(typeof(int)));
    }

    [Test]
    public void GetColumn_WithNullOrWhitespaceName_ShouldThrowArgumentException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        Assert.Throws<ArgumentException>(() => frame.GetColumn<int>(null!));
        Assert.Throws<ArgumentException>(() => frame.GetColumn<int>("   "));
    }

    [Test]
    public void GetColumn_Untyped_WithValidName_ShouldReturnColumn()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act
        var retrievedCol = frame.GetColumn("Numbers");

        // Assert
        Assert.That(retrievedCol, Is.Not.Null);
        Assert.That(retrievedCol.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(retrievedCol.Length, Is.EqualTo(3));
    }

    [Test]
    public void HasColumn_WithExistingName_ShouldReturnTrue()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        Assert.That(frame.HasColumn("Numbers"), Is.True);
        Assert.That(frame.HasColumn("NUMBERS"), Is.True); // Case-insensitive
    }

    [Test]
    public void HasColumn_WithNonExistentName_ShouldReturnFalse()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        Assert.That(frame.HasColumn("NonExistent"), Is.False);
    }

    [Test]
    public void HasColumn_WithNullOrWhitespaceName_ShouldReturnFalse()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        Assert.That(frame.HasColumn(null!), Is.False);
        Assert.That(frame.HasColumn("   "), Is.False);
    }

    [Test]
    public void Schema_ShouldReturnCorrectSchema()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var schema = frame.Schema;

        // Assert
        Assert.That(schema.ColumnNames, Is.EqualTo(new[] { "Numbers", "Letters" }));
        Assert.That(schema.GetColumnType("Numbers"), Is.EqualTo(typeof(int)));
        Assert.That(schema.GetColumnType("Letters"), Is.EqualTo(typeof(string)));
    }

    [Test]
    public void WithColumn_WithValidColumn_ShouldReturnNewFrame()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var newCol = NivaraColumn<string>.Create(new[] { "a", "b", "c" });

        // Act
        var newFrame = frame.WithColumn("Letters", newCol);

        // Assert
        Assert.That(newFrame.ColumnCount, Is.EqualTo(2));
        Assert.That(newFrame.HasColumn("Numbers"), Is.True);
        Assert.That(newFrame.HasColumn("Letters"), Is.True);
        
        // Original frame should be unchanged
        Assert.That(frame.ColumnCount, Is.EqualTo(1));
    }

    [Test]
    public void WithColumn_WithExistingName_ShouldThrowArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var newCol = NivaraColumn<string>.Create(new[] { "a", "b", "c" });

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.WithColumn("Numbers", newCol));
        Assert.That(ex.Message, Contains.Substring("Column 'Numbers' already exists"));
    }

    [Test]
    public void WithColumn_WithDifferentLength_ShouldThrowArgumentException()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col1));
        var newCol = NivaraColumn<string>.Create(new[] { "a", "b" }); // Different length

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.WithColumn("Letters", newCol));
        Assert.That(ex.Message, Contains.Substring("Column length mismatch"));
    }

    [Test]
    public void WithoutColumn_WithValidName_ShouldReturnNewFrame()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var newFrame = frame.WithoutColumn("Letters");

        // Assert
        Assert.That(newFrame.ColumnCount, Is.EqualTo(1));
        Assert.That(newFrame.HasColumn("Numbers"), Is.True);
        Assert.That(newFrame.HasColumn("Letters"), Is.False);
        
        // Original frame should be unchanged
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
    }

    [Test]
    public void WithoutColumn_WithNonExistentName_ShouldThrowColumnNotFoundException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => frame.WithoutColumn("NonExistent"));
        Assert.That(ex.ColumnName, Is.EqualTo("NonExistent"));
    }

    [Test]
    public void WithoutColumn_WithLastColumn_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => frame.WithoutColumn("Numbers"));
        Assert.That(ex.Message, Contains.Substring("Cannot remove the last column"));
    }

    [Test]
    public void SelectColumns_WithValidNames_ShouldReturnNewFrame()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var col3 = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3 });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2), ("Decimals", col3));

        // Act
        var newFrame = frame.SelectColumns("Letters", "Numbers");

        // Assert
        Assert.That(newFrame.ColumnCount, Is.EqualTo(2));
        Assert.That(newFrame.ColumnNames, Is.EqualTo(new[] { "Letters", "Numbers" }));
        Assert.That(newFrame.HasColumn("Decimals"), Is.False);
    }

    [Test]
    public void SelectColumns_WithArray_ShouldReturnNewFrame()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var newFrame = frame.SelectColumns(new[] { "Letters" });

        // Assert
        Assert.That(newFrame.ColumnCount, Is.EqualTo(1));
        Assert.That(newFrame.HasColumn("Letters"), Is.True);
        Assert.That(newFrame.HasColumn("Numbers"), Is.False);
    }

    [Test]
    public void SelectColumns_WithEmptyList_ShouldThrowArgumentException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => frame.SelectColumns());
        Assert.That(ex.Message, Contains.Substring("Must specify at least one column name"));
    }

    [Test]
    public void SelectColumns_WithNonExistentName_ShouldThrowColumnNotFoundException()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => frame.SelectColumns("NonExistent"));
        Assert.That(ex.ColumnName, Is.EqualTo("NonExistent"));
    }

    [Test]
    public void ToString_ShouldReturnFormattedString()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        var result = frame.ToString();

        // Assert
        Assert.That(result, Contains.Substring("NivaraFrame"));
        Assert.That(result, Contains.Substring("3 rows"));
        Assert.That(result, Contains.Substring("2 columns"));
        Assert.That(result, Contains.Substring("Numbers"));
        Assert.That(result, Contains.Substring("Letters"));
    }

    [Test]
    public void Dispose_ShouldDisposeAllColumns()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var col2 = NivaraColumn<string>.Create(new[] { "a", "b", "c" });
        var frame = NivaraFrame.Create(("Numbers", col1), ("Letters", col2));

        // Act
        frame.Dispose();

        // Assert - accessing properties should throw ObjectDisposedException
        Assert.Throws<ObjectDisposedException>(() => { var _ = frame.RowCount; });
        Assert.Throws<ObjectDisposedException>(() => { var _ = frame.ColumnCount; });
        Assert.Throws<ObjectDisposedException>(() => { var _ = frame.ColumnNames; });
        Assert.Throws<ObjectDisposedException>(() => { var _ = frame.Schema; });
        Assert.Throws<ObjectDisposedException>(() => frame.GetColumn<int>("Numbers"));
        Assert.Throws<ObjectDisposedException>(() => frame.HasColumn("Numbers"));
    }

    [Test]
    public void ToString_AfterDispose_ShouldReturnDisposedMessage()
    {
        // Arrange
        var col = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var frame = NivaraFrame.Create(("Numbers", col));

        // Act
        frame.Dispose();
        var result = frame.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("NivaraFrame [Disposed]"));
    }

    [Test]
    public void EmptyFrame_WithSingleEmptyColumn_ShouldWork()
    {
        // Arrange
        var emptyCol = NivaraColumn<int>.Create(Array.Empty<int>());

        // Act
        var frame = NivaraFrame.Create(("Empty", emptyCol));

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(0));
        Assert.That(frame.ColumnCount, Is.EqualTo(1));
        Assert.That(frame.HasColumn("Empty"), Is.True);
    }

    [Test]
    public void SingleRowFrame_ShouldWork()
    {
        // Arrange
        var col1 = NivaraColumn<int>.Create(new[] { 42 });
        var col2 = NivaraColumn<string>.Create(new[] { "test" });

        // Act
        var frame = NivaraFrame.Create(("Number", col1), ("Text", col2));

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(1));
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.GetColumn<int>("Number")[0], Is.EqualTo(42));
        Assert.That(frame.GetColumn<string>("Text")[0], Is.EqualTo("test"));
    }

    [Test]
    public void LargeFrame_ShouldWork()
    {
        // Arrange
        var size = 10000;
        var numbers = Enumerable.Range(1, size).ToArray();
        var strings = Enumerable.Range(1, size).Select(i => $"item_{i}").ToArray();
        
        var col1 = NivaraColumn<int>.Create(numbers);
        var col2 = NivaraColumn<string>.Create(strings);

        // Act
        var frame = NivaraFrame.Create(("Numbers", col1), ("Strings", col2));

        // Assert
        Assert.That(frame.RowCount, Is.EqualTo(size));
        Assert.That(frame.ColumnCount, Is.EqualTo(2));
        Assert.That(frame.GetColumn<int>("Numbers")[0], Is.EqualTo(1));
        Assert.That(frame.GetColumn<int>("Numbers")[size - 1], Is.EqualTo(size));
        Assert.That(frame.GetColumn<string>("Strings")[0], Is.EqualTo("item_1"));
        Assert.That(frame.GetColumn<string>("Strings")[size - 1], Is.EqualTo($"item_{size}"));
    }
}