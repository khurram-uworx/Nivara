using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class FrameProjectionTests
{
    private NivaraFrame testFrame;

    [SetUp]
    public void SetUp()
    {
        // Create a test frame with multiple columns
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d", "e" });
        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

        testFrame = NivaraFrame.Create(
            ("IntCol", intColumn),
            ("StringCol", stringColumn),
            ("DoubleCol", doubleColumn)
        );
    }

    [Test]
    public void Select_WithColumnNames_ShouldReturnSelectedColumns()
    {
        // Act
        var selected = testFrame.Select("IntCol", "StringCol");

        // Assert
        Assert.That(selected.ColumnCount, Is.EqualTo(2));
        Assert.That(selected.ColumnNames, Does.Contain("IntCol"));
        Assert.That(selected.ColumnNames, Does.Contain("StringCol"));
        Assert.That(selected.ColumnNames, Does.Not.Contain("DoubleCol"));
        Assert.That(selected.RowCount, Is.EqualTo(5));
    }

    [Test]
    public void SelectAndRename_WithMappings_ShouldRenameColumns()
    {
        // Arrange
        var mappings = new Dictionary<string, string?>
        {
            { "IntCol", "Numbers" },
            { "StringCol", "Letters" }
        };

        // Act
        var renamed = testFrame.SelectAndRename(mappings);

        // Assert
        Assert.That(renamed.ColumnCount, Is.EqualTo(2));
        Assert.That(renamed.ColumnNames, Does.Contain("Numbers"));
        Assert.That(renamed.ColumnNames, Does.Contain("Letters"));
        Assert.That(renamed.ColumnNames, Does.Not.Contain("IntCol"));
        Assert.That(renamed.ColumnNames, Does.Not.Contain("StringCol"));
        Assert.That(renamed.RowCount, Is.EqualTo(5));

        // Verify data integrity
        var numbersCol = renamed.GetColumn<int>("Numbers");
        Assert.That(numbersCol[0], Is.EqualTo(1));
        Assert.That(numbersCol[4], Is.EqualTo(5));
    }

    [Test]
    public void RenameColumn_WithValidNames_ShouldRenameColumn()
    {
        // Act
        var renamed = testFrame.RenameColumn("IntCol", "Numbers");

        // Assert
        Assert.That(renamed.ColumnCount, Is.EqualTo(3));
        Assert.That(renamed.ColumnNames, Does.Contain("Numbers"));
        Assert.That(renamed.ColumnNames, Does.Not.Contain("IntCol"));
        Assert.That(renamed.ColumnNames, Does.Contain("StringCol"));
        Assert.That(renamed.ColumnNames, Does.Contain("DoubleCol"));

        // Verify data integrity
        var numbersCol = renamed.GetColumn<int>("Numbers");
        Assert.That(numbersCol[0], Is.EqualTo(1));
        Assert.That(numbersCol[4], Is.EqualTo(5));
    }

    [Test]
    public void RenameColumns_WithMultipleMappings_ShouldRenameAllColumns()
    {
        // Arrange
        var renames = new Dictionary<string, string>
        {
            { "IntCol", "Numbers" },
            { "StringCol", "Letters" }
        };

        // Act
        var renamed = testFrame.RenameColumns(renames);

        // Assert
        Assert.That(renamed.ColumnCount, Is.EqualTo(3));
        Assert.That(renamed.ColumnNames, Does.Contain("Numbers"));
        Assert.That(renamed.ColumnNames, Does.Contain("Letters"));
        Assert.That(renamed.ColumnNames, Does.Contain("DoubleCol")); // Not renamed
        Assert.That(renamed.ColumnNames, Does.Not.Contain("IntCol"));
        Assert.That(renamed.ColumnNames, Does.Not.Contain("StringCol"));
    }

    [Test]
    public void Exclude_WithColumnNames_ShouldExcludeColumns()
    {
        // Act
        var excluded = testFrame.Exclude("DoubleCol");

        // Assert
        Assert.That(excluded.ColumnCount, Is.EqualTo(2));
        Assert.That(excluded.ColumnNames, Does.Contain("IntCol"));
        Assert.That(excluded.ColumnNames, Does.Contain("StringCol"));
        Assert.That(excluded.ColumnNames, Does.Not.Contain("DoubleCol"));
        Assert.That(excluded.RowCount, Is.EqualTo(5));
    }

    [Test]
    public void WithTransformedColumn_ShouldTransformAndReplaceColumn()
    {
        // Act
        var transformed = testFrame.WithTransformedColumn<int, int>("IntCol", x => x * 2);

        // Assert
        Assert.That(transformed.ColumnCount, Is.EqualTo(3));
        Assert.That(transformed.ColumnNames, Does.Contain("IntCol"));

        var intCol = transformed.GetColumn<int>("IntCol");
        Assert.That(intCol[0], Is.EqualTo(2));
        Assert.That(intCol[1], Is.EqualTo(4));
        Assert.That(intCol[4], Is.EqualTo(10));
    }

    [Test]
    public void WithTransformedColumn_WithNewName_ShouldAddNewColumn()
    {
        // Act
        var transformed = testFrame.WithTransformedColumn<int, int>("IntCol", x => x * 2, "DoubledInt");

        // Assert
        Assert.That(transformed.ColumnCount, Is.EqualTo(4));
        Assert.That(transformed.ColumnNames, Does.Contain("IntCol")); // Original preserved
        Assert.That(transformed.ColumnNames, Does.Contain("DoubledInt")); // New column added

        var originalCol = transformed.GetColumn<int>("IntCol");
        var doubledCol = transformed.GetColumn<int>("DoubledInt");

        Assert.That(originalCol[0], Is.EqualTo(1)); // Original unchanged
        Assert.That(doubledCol[0], Is.EqualTo(2)); // New column transformed
    }

    [Test]
    public void WithComputedColumn_WithTwoSources_ShouldCreateComputedColumn()
    {
        // Act
        var computed = testFrame.WithComputedColumn<int, double, double>(
            "IntCol", "DoubleCol",
            (i, d) => i + d,
            "Sum");

        // Assert
        Assert.That(computed.ColumnCount, Is.EqualTo(4));
        Assert.That(computed.ColumnNames, Does.Contain("Sum"));

        var sumCol = computed.GetColumn<double>("Sum");
        Assert.That(sumCol[0], Is.EqualTo(2.1).Within(0.001)); // 1 + 1.1
        Assert.That(sumCol[1], Is.EqualTo(4.2).Within(0.001)); // 2 + 2.2
    }

    [Test]
    public void Select_WithNonExistentColumn_ShouldThrowColumnNotFoundException()
    {
        // Act & Assert
        Assert.Throws<ColumnNotFoundException>(() =>
            testFrame.Select("NonExistent"));
    }

    [Test]
    public void RenameColumn_WithNonExistentColumn_ShouldThrowColumnNotFoundException()
    {
        // Act & Assert
        Assert.Throws<ColumnNotFoundException>(() =>
            testFrame.RenameColumn("NonExistent", "NewName"));
    }

    [Test]
    public void Exclude_AllColumns_ShouldThrowInvalidOperationException()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            testFrame.Exclude("IntCol", "StringCol", "DoubleCol"));
    }

    [TearDown]
    public void TearDown()
    {
        testFrame?.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}