using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Tests for DataFrame concatenation operations including vertical and horizontal concatenation
/// </summary>
[TestFixture]
public class ConcatenationOperationTests
{
    [Test]
    public void ConcatenateVertical_WithCompatibleSchemas_ShouldCombineRows()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie", "Diana" })),
            ("Age", NivaraColumn<int>.Create(new[] { 35, 28 }))
        );

        // Act
        var result = frame1.ConcatenateVertical(frame2);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(4));
        Assert.That(result.ColumnCount, Is.EqualTo(2));

        var names = result.GetColumn<string>("Name");
        var ages = result.GetColumn<int>("Age");

        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(names[1], Is.EqualTo("Bob"));
        Assert.That(names[2], Is.EqualTo("Charlie"));
        Assert.That(names[3], Is.EqualTo("Diana"));

        Assert.That(ages[0], Is.EqualTo(25));
        Assert.That(ages[1], Is.EqualTo(30));
        Assert.That(ages[2], Is.EqualTo(35));
        Assert.That(ages[3], Is.EqualTo(28));
    }

    [Test]
    public void ConcatenateVertical_WithMissingColumns_FillWithNulls_ShouldAddNullColumns()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie" })),
            ("Salary", NivaraColumn<double>.Create(new[] { 50000.0 }))
        );

        // Act
        var result = frame1.ConcatenateVertical(frame2, ConcatenationMismatchHandling.FillWithNulls);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnCount, Is.EqualTo(3));
        Assert.That(result.ColumnNames, Contains.Item("Name"));
        Assert.That(result.ColumnNames, Contains.Item("Age"));
        Assert.That(result.ColumnNames, Contains.Item("Salary"));

        var names = result.GetColumn<string>("Name");
        var ages = result.GetColumn<int>("Age");
        var salaries = result.GetColumn<double>("Salary");

        // Check names
        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(names[1], Is.EqualTo("Bob"));
        Assert.That(names[2], Is.EqualTo("Charlie"));

        // Check ages (should have null for Charlie)
        Assert.That(ages[0], Is.EqualTo(25));
        Assert.That(ages[1], Is.EqualTo(30));
        Assert.That(ages.IsNull(2), Is.True);

        // Check salaries (should have nulls for Alice and Bob)
        Assert.That(salaries.IsNull(0), Is.True);
        Assert.That(salaries.IsNull(1), Is.True);
        Assert.That(salaries[2], Is.EqualTo(50000.0));
    }

    [Test]
    public void ConcatenateVertical_WithEmptyFrames_ShouldHandleGracefully()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25 }))
        );

        var emptyFrame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(Array.Empty<string>())),
            ("Age", NivaraColumn<int>.Create(Array.Empty<int>()))
        );

        // Act
        var result = frame1.ConcatenateVertical(emptyFrame);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.ColumnCount, Is.EqualTo(2));

        var names = result.GetColumn<string>("Name");
        var ages = result.GetColumn<int>("Age");

        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(ages[0], Is.EqualTo(25));
    }

    [Test]
    public void ConcatenateHorizontal_WithCompatibleRowCounts_ShouldCombineColumns()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Salary", NivaraColumn<double>.Create(new[] { 50000.0, 60000.0 })),
            ("Department", NivaraColumn<string>.CreateForReferenceType(new[] { "Engineering", "Sales" }))
        );

        // Act
        var result = frame1.ConcatenateHorizontal(frame2);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.ColumnCount, Is.EqualTo(4));
        Assert.That(result.ColumnNames, Contains.Item("Name"));
        Assert.That(result.ColumnNames, Contains.Item("Age"));
        Assert.That(result.ColumnNames, Contains.Item("Salary"));
        Assert.That(result.ColumnNames, Contains.Item("Department"));

        var names = result.GetColumn<string>("Name");
        var ages = result.GetColumn<int>("Age");
        var salaries = result.GetColumn<double>("Salary");
        var departments = result.GetColumn<string>("Department");

        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(names[1], Is.EqualTo("Bob"));
        Assert.That(ages[0], Is.EqualTo(25));
        Assert.That(ages[1], Is.EqualTo(30));
        Assert.That(salaries[0], Is.EqualTo(50000.0));
        Assert.That(salaries[1], Is.EqualTo(60000.0));
        Assert.That(departments[0], Is.EqualTo("Engineering"));
        Assert.That(departments[1], Is.EqualTo("Sales"));
    }

    [Test]
    public void ConcatenateHorizontal_WithMismatchedRowCounts_ShouldThrowException()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Salary", NivaraColumn<double>.Create(new[] { 50000.0 })) // Only one row
        );

        // Act & Assert
        var ex = Assert.Throws<QueryExecutionException>(() => frame1.ConcatenateHorizontal(frame2));
        Assert.That(ex.InnerException, Is.TypeOf<ArgumentException>());
        Assert.That(ex.Message, Contains.Substring("Row count mismatch"));
    }

    [Test]
    public void ConcatenateHorizontal_WithColumnNameConflicts_ShouldThrowException()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob" })),
            ("Age", NivaraColumn<int>.Create(new[] { 25, 30 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Charlie", "Diana" })), // Conflicting column name
            ("Salary", NivaraColumn<double>.Create(new[] { 50000.0, 60000.0 }))
        );

        // Act & Assert
        var ex = Assert.Throws<QueryExecutionException>(() => frame1.ConcatenateHorizontal(frame2));
        Assert.That(ex.InnerException, Is.TypeOf<ArgumentException>());
        Assert.That(ex.Message, Contains.Substring("Column name conflict"));
    }

    [Test]
    public void ConcatenateVertical_MultipleFrames_ShouldCombineAllRows()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Value", NivaraColumn<int>.Create(new[] { 1, 2 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Value", NivaraColumn<int>.Create(new[] { 3, 4 }))
        );

        var frame3 = NivaraFrame.Create(
            ("Value", NivaraColumn<int>.Create(new[] { 5, 6 }))
        );

        // Act
        var result = NivaraFrameExtensions.ConcatenateVertical(new[] { frame1, frame2, frame3 });

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(6));
        Assert.That(result.ColumnCount, Is.EqualTo(1));

        var values = result.GetColumn<int>("Value");
        Assert.That(values[0], Is.EqualTo(1));
        Assert.That(values[1], Is.EqualTo(2));
        Assert.That(values[2], Is.EqualTo(3));
        Assert.That(values[3], Is.EqualTo(4));
        Assert.That(values[4], Is.EqualTo(5));
        Assert.That(values[5], Is.EqualTo(6));
    }

    [Test]
    public void ConcatenateHorizontal_MultipleFrames_ShouldCombineAllColumns()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2 }))
        );

        var frame2 = NivaraFrame.Create(
            ("B", NivaraColumn<int>.Create(new[] { 3, 4 }))
        );

        var frame3 = NivaraFrame.Create(
            ("C", NivaraColumn<int>.Create(new[] { 5, 6 }))
        );

        // Act
        var result = NivaraFrameExtensions.ConcatenateHorizontal(new[] { frame1, frame2, frame3 });

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.ColumnCount, Is.EqualTo(3));

        var colA = result.GetColumn<int>("A");
        var colB = result.GetColumn<int>("B");
        var colC = result.GetColumn<int>("C");

        Assert.That(colA[0], Is.EqualTo(1));
        Assert.That(colA[1], Is.EqualTo(2));
        Assert.That(colB[0], Is.EqualTo(3));
        Assert.That(colB[1], Is.EqualTo(4));
        Assert.That(colC[0], Is.EqualTo(5));
        Assert.That(colC[1], Is.EqualTo(6));
    }

    [Test]
    public void Append_ShouldBeAliasForConcatenateVertical()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice" }))
        );

        var frame2 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Bob" }))
        );

        // Act
        var result = frame1.Append(frame2);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        var names = result.GetColumn<string>("Name");
        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(names[1], Is.EqualTo("Bob"));
    }

    [Test]
    public void Combine_ShouldBeAliasForConcatenateHorizontal()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("A", NivaraColumn<int>.Create(new[] { 1 }))
        );

        var frame2 = NivaraFrame.Create(
            ("B", NivaraColumn<int>.Create(new[] { 2 }))
        );

        // Act
        var result = frame1.Combine(frame2);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(1));
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        Assert.That(result.GetColumn<int>("A")[0], Is.EqualTo(1));
        Assert.That(result.GetColumn<int>("B")[0], Is.EqualTo(2));
    }

    [Test]
    public void ConcatenateVertical_WithNullValues_ShouldPreserveNulls()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(new string?[] { "Alice", null }!)),
            ("Age", NivaraColumn<int>.CreateFromNullable(new int?[] { 25, null }))
        );

        var frame2 = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.Create(new string?[] { null, "Bob" }!)),
            ("Age", NivaraColumn<int>.CreateFromNullable(new int?[] { null, 30 }))
        );

        // Act
        var result = frame1.ConcatenateVertical(frame2);

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(4));

        var names = result.GetColumn<string>("Name");
        var ages = result.GetColumn<int>("Age");

        Assert.That(names[0], Is.EqualTo("Alice"));
        Assert.That(names.IsNull(1), Is.True);
        Assert.That(names.IsNull(2), Is.True);
        Assert.That(names[3], Is.EqualTo("Bob"));

        Assert.That(ages[0], Is.EqualTo(25));
        Assert.That(ages.IsNull(1), Is.True);
        Assert.That(ages.IsNull(2), Is.True);
        Assert.That(ages[3], Is.EqualTo(30));
    }

    [Test]
    public void ConcatenateVertical_WithDifferentTypes_ShouldThrowException()
    {
        // Arrange
        var frame1 = NivaraFrame.Create(
            ("Value", NivaraColumn<int>.Create(new[] { 1, 2 }))
        );

        var frame2 = NivaraFrame.Create(
            ("Value", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b" }))
        );

        // Act & Assert
        var ex = Assert.Throws<QueryExecutionException>(() => frame1.ConcatenateVertical(frame2));
        Assert.That(ex.InnerException, Is.TypeOf<ArgumentException>());
        Assert.That(ex.Message, Contains.Substring("Cannot concatenate columns of different types"));
    }

    [Test]
    public void ConcatenateVertical_SingleFrame_ShouldReturnSameFrame()
    {
        // Arrange
        var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice" }))
        );

        // Act
        var result = NivaraFrameExtensions.ConcatenateVertical(new[] { frame });

        // Assert
        Assert.That(result, Is.SameAs(frame));
    }

    [Test]
    public void ConcatenateHorizontal_SingleFrame_ShouldReturnSameFrame()
    {
        // Arrange
        var frame = NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice" }))
        );

        // Act
        var result = NivaraFrameExtensions.ConcatenateHorizontal(new[] { frame });

        // Assert
        Assert.That(result, Is.SameAs(frame));
    }

    [Test]
    public void ConcatenateVertical_EmptyFramesList_ShouldThrowException()
    {
        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraFrameExtensions.ConcatenateVertical(Array.Empty<NivaraFrame>()));
        Assert.That(ex.Message, Contains.Substring("Must provide at least one frame"));
    }

    [Test]
    public void ConcatenateHorizontal_EmptyFramesList_ShouldThrowException()
    {
        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraFrameExtensions.ConcatenateHorizontal(Array.Empty<NivaraFrame>()));
        Assert.That(ex.Message, Contains.Substring("Must provide at least one frame"));
    }
}