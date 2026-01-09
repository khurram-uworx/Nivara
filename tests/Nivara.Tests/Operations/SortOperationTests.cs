using Nivara.Exceptions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Operations;

/// <summary>
/// Tests for SortOperation functionality including single-column and multi-column sorting
/// </summary>
[TestFixture]
public class SortOperationTests
{
    [Test]
    public void SortKey_Constructor_WithValidParameters_ShouldCreateSortKey()
    {
        // Act
        var sortKey = new SortKey("TestColumn", SortDirection.Descending, NullOrdering.NullsFirst);

        // Assert
        Assert.That(sortKey.ColumnName, Is.EqualTo("TestColumn"));
        Assert.That(sortKey.Direction, Is.EqualTo(SortDirection.Descending));
        Assert.That(sortKey.NullOrdering, Is.EqualTo(NullOrdering.NullsFirst));
    }

    [Test]
    public void SortKey_Constructor_WithDefaults_ShouldUseDefaultValues()
    {
        // Act
        var sortKey = new SortKey("TestColumn");

        // Assert
        Assert.That(sortKey.ColumnName, Is.EqualTo("TestColumn"));
        Assert.That(sortKey.Direction, Is.EqualTo(SortDirection.Ascending));
        Assert.That(sortKey.NullOrdering, Is.EqualTo(NullOrdering.NullsLast));
    }

    [Test]
    public void SortKey_Constructor_WithNullColumnName_ShouldThrowArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SortKey(null!));
        Assert.Throws<ArgumentException>(() => new SortKey(""));
        Assert.Throws<ArgumentException>(() => new SortKey("   "));
    }

    [Test]
    public void SortOperation_Constructor_WithValidSortKeys_ShouldCreateOperation()
    {
        // Arrange
        var sortKeys = new[]
        {
            new SortKey("Column1", SortDirection.Ascending),
            new SortKey("Column2", SortDirection.Descending)
        };

        // Act
        var operation = new SortOperation(sortKeys);

        // Assert
        Assert.That(operation.SortKeys.Count, Is.EqualTo(2));
        Assert.That(operation.IsStable, Is.True);
        Assert.That(operation.OperationType, Is.EqualTo("Sort"));
    }

    [Test]
    public void SortOperation_Constructor_WithSingleColumn_ShouldCreateOperation()
    {
        // Act
        var operation = new SortOperation("TestColumn", SortDirection.Descending, NullOrdering.NullsFirst, false);

        // Assert
        Assert.That(operation.SortKeys.Count, Is.EqualTo(1));
        Assert.That(operation.SortKeys[0].ColumnName, Is.EqualTo("TestColumn"));
        Assert.That(operation.SortKeys[0].Direction, Is.EqualTo(SortDirection.Descending));
        Assert.That(operation.SortKeys[0].NullOrdering, Is.EqualTo(NullOrdering.NullsFirst));
        Assert.That(operation.IsStable, Is.False);
    }

    [Test]
    public void SortOperation_Constructor_WithNullSortKeys_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SortOperation((IEnumerable<SortKey>)null!));
    }

    [Test]
    public void SortOperation_Constructor_WithEmptySortKeys_ShouldThrowArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SortOperation(Array.Empty<SortKey>()));
    }

    [Test]
    public void TransformSchema_WithValidSchema_ShouldReturnSameSchema()
    {
        // Arrange
        var schema = new Schema(new[]
        {
            ("Numbers", typeof(int)),
            ("Names", typeof(string))
        });
        var operation = new SortOperation("Numbers");

        // Act
        var result = operation.TransformSchema(schema);

        // Assert
        Assert.That(result.ColumnNames.Count, Is.EqualTo(schema.ColumnNames.Count));
        Assert.That(result.ColumnNames, Is.EqualTo(schema.ColumnNames));
    }

    [Test]
    public void TransformSchema_WithMissingColumn_ShouldThrowSchemaValidationException()
    {
        // Arrange
        var schema = new Schema(new[]
        {
            ("Numbers", typeof(int)),
            ("Names", typeof(string))
        });
        var operation = new SortOperation("MissingColumn");

        // Act & Assert
        var ex = Assert.Throws<SchemaValidationException>(() => operation.TransformSchema(schema));
        Assert.That(ex.Message, Contains.Substring("MissingColumn"));
    }

    [Test]
    public void Execute_WithSingleColumnAscending_ShouldSortCorrectly()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 3, 1, 4, 1, 5 });
        var names = NivaraColumn<string>.Create(new[] { "c", "a", "d", "a", "e" });
        var input = new Dictionary<string, IColumn>
        {
            ["Numbers"] = numbers,
            ["Names"] = names
        };

        var operation = new SortOperation("Numbers", SortDirection.Ascending);

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedNumbers = (NivaraColumn<int>)result["Numbers"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        Assert.That(sortedNumbers.ToArray(), Is.EqualTo(new[] { 1, 1, 3, 4, 5 }));
        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "a", "a", "c", "d", "e" }));
    }

    [Test]
    public void Execute_WithSingleColumnDescending_ShouldSortCorrectly()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 3, 1, 4, 1, 5 });
        var names = NivaraColumn<string>.Create(new[] { "c", "a", "d", "a", "e" });
        var input = new Dictionary<string, IColumn>
        {
            ["Numbers"] = numbers,
            ["Names"] = names
        };

        var operation = new SortOperation("Numbers", SortDirection.Descending);

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedNumbers = (NivaraColumn<int>)result["Numbers"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        Assert.That(sortedNumbers.ToArray(), Is.EqualTo(new[] { 5, 4, 3, 1, 1 }));
        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "e", "d", "c", "a", "a" }));
    }

    [Test]
    public void Execute_WithMultipleColumns_ShouldSortByPriority()
    {
        // Arrange
        var category = NivaraColumn<string>.Create(new[] { "A", "B", "A", "B", "A" });
        var priority = NivaraColumn<int>.Create(new[] { 2, 1, 1, 2, 3 });
        var names = NivaraColumn<string>.Create(new[] { "item1", "item2", "item3", "item4", "item5" });
        var input = new Dictionary<string, IColumn>
        {
            ["Category"] = category,
            ["Priority"] = priority,
            ["Names"] = names
        };

        var sortKeys = new[]
        {
            new SortKey("Category", SortDirection.Ascending),
            new SortKey("Priority", SortDirection.Ascending)
        };
        var operation = new SortOperation(sortKeys);

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedCategory = (NivaraColumn<string>)result["Category"];
        var sortedPriority = (NivaraColumn<int>)result["Priority"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        // Expected order: A,1 -> A,2 -> A,3 -> B,1 -> B,2
        Assert.That(sortedCategory.ToArray(), Is.EqualTo(new[] { "A", "A", "A", "B", "B" }));
        Assert.That(sortedPriority.ToArray(), Is.EqualTo(new[] { 1, 2, 3, 1, 2 }));
        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "item3", "item1", "item5", "item2", "item4" }));
    }

    [Test]
    public void Execute_WithNullValues_ShouldHandleNullsCorrectly()
    {
        // Arrange
        var nullableNumbers = NivaraColumn<int>.CreateFromNullable(new int?[] { 3, null, 1, null, 2 });
        var names = NivaraColumn<string>.Create(new[] { "c", "null1", "a", "null2", "b" });
        var input = new Dictionary<string, IColumn>
        {
            ["Numbers"] = nullableNumbers,
            ["Names"] = names
        };

        var operation = new SortOperation("Numbers", SortDirection.Ascending, NullOrdering.NullsFirst);

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedNumbers = (NivaraColumn<int>)result["Numbers"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        // Expected order: null, null, 1, 2, 3 (nulls first)
        Assert.That(sortedNumbers.GetValue(0), Is.Null);
        Assert.That(sortedNumbers.GetValue(1), Is.Null);
        Assert.That(sortedNumbers.GetValue(2), Is.EqualTo(1));
        Assert.That(sortedNumbers.GetValue(3), Is.EqualTo(2));
        Assert.That(sortedNumbers.GetValue(4), Is.EqualTo(3));

        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "null1", "null2", "a", "b", "c" }));
    }

    [Test]
    public void Execute_WithNullsLast_ShouldPlaceNullsAtEnd()
    {
        // Arrange
        var nullableNumbers = NivaraColumn<int>.CreateFromNullable(new int?[] { 3, null, 1, null, 2 });
        var names = NivaraColumn<string>.Create(new[] { "c", "null1", "a", "null2", "b" });
        var input = new Dictionary<string, IColumn>
        {
            ["Numbers"] = nullableNumbers,
            ["Names"] = names
        };

        var operation = new SortOperation("Numbers", SortDirection.Ascending, NullOrdering.NullsLast);

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedNumbers = (NivaraColumn<int>)result["Numbers"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        // Expected order: 1, 2, 3, null, null (nulls last)
        Assert.That(sortedNumbers.GetValue(0), Is.EqualTo(1));
        Assert.That(sortedNumbers.GetValue(1), Is.EqualTo(2));
        Assert.That(sortedNumbers.GetValue(2), Is.EqualTo(3));
        Assert.That(sortedNumbers.GetValue(3), Is.Null);
        Assert.That(sortedNumbers.GetValue(4), Is.Null);

        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "a", "b", "c", "null1", "null2" }));
    }

    [Test]
    public void Execute_WithEmptyInput_ShouldReturnEmptyResult()
    {
        // Arrange
        var input = new Dictionary<string, IColumn>();
        var operation = new SortOperation("TestColumn");

        // Act
        var result = operation.Execute(input);

        // Assert
        Assert.That(result.Count, Is.EqualTo(0));
    }

    [Test]
    public void Execute_WithSingleRow_ShouldReturnSameData()
    {
        // Arrange
        var numbers = NivaraColumn<int>.Create(new[] { 42 });
        var names = NivaraColumn<string>.Create(new[] { "single" });
        var input = new Dictionary<string, IColumn>
        {
            ["Numbers"] = numbers,
            ["Names"] = names
        };

        var operation = new SortOperation("Numbers");

        // Act
        var result = operation.Execute(input);

        // Assert
        var sortedNumbers = (NivaraColumn<int>)result["Numbers"];
        var sortedNames = (NivaraColumn<string>)result["Names"];

        Assert.That(sortedNumbers.ToArray(), Is.EqualTo(new[] { 42 }));
        Assert.That(sortedNames.ToArray(), Is.EqualTo(new[] { "single" }));
    }

    [Test]
    public void ToString_ShouldReturnFormattedString()
    {
        // Arrange
        var sortKeys = new[]
        {
            new SortKey("Column1", SortDirection.Ascending, NullOrdering.NullsFirst),
            new SortKey("Column2", SortDirection.Descending, NullOrdering.NullsLast)
        };
        var operation = new SortOperation(sortKeys, stable: false);

        // Act
        var result = operation.ToString();

        // Assert
        Assert.That(result, Contains.Substring("Sort("));
        Assert.That(result, Contains.Substring("Column1"));
        Assert.That(result, Contains.Substring("Column2"));
    }
}