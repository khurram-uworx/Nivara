using Nivara.Exceptions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Operations;

[TestFixture]
public class JoinOperationTests
{
    private NivaraFrame CreateLeftFrame()
    {
        return NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2, 3, 4 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string[] { "Alice", "Bob", "Charlie", "David" })),
            ("Age", NivaraColumn<int>.Create(new int[] { 25, 30, 35, 40 }))
        );
    }

    private NivaraFrame CreateRightFrame()
    {
        return NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 2, 3, 4, 5 })),
            ("Department", NivaraColumn<string>.CreateForReferenceType(new string[] { "HR", "IT", "Finance", "Marketing" })),
            ("Salary", NivaraColumn<decimal>.Create(new decimal[] { 50000m, 60000m, 70000m, 80000m }))
        );
    }

    private NivaraFrame CreateLeftFrameWithNulls()
    {
        return NivaraFrame.Create(
            ("Id", NivaraColumn.CreateFromNullable(new int?[] { 1, 2, null, 4 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string?[] { "Alice", "Bob", null, "David" }!))
        );
    }

    private NivaraFrame CreateRightFrameWithNulls()
    {
        return NivaraFrame.Create(
            ("Id", NivaraColumn.CreateFromNullable(new int?[] { 2, null, 4, 5 })),
            ("Department", NivaraColumn<string>.CreateForReferenceType(new string?[] { "HR", null, "Finance", "Marketing" }!))
        );
    }

    [Test]
    public void InnerJoin_WithMatchingKeys_ReturnsCorrectResult()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = CreateRightFrame();

        // Act
        var result = left.InnerJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(3), "Inner join should return 3 matching rows");
        Assert.That(result.ColumnCount, Is.EqualTo(5), "Result should have 5 columns (Id from left, Name, Age, Department, Salary)");

        // Verify column names
        var expectedColumns = new string[] { "Id", "Name", "Age", "Department", "Salary" };
        Assert.That(result.ColumnNames, Is.EquivalentTo(expectedColumns));

        // Verify data
        var ids = result.GetColumn<int>("Id");
        var names = result.GetColumn<string>("Name");
        var departments = result.GetColumn<string>("Department");

        Assert.That(ids.ToArray(), Is.EqualTo(new int[] { 2, 3, 4 }));
        Assert.That(names.ToArray(), Is.EqualTo(new string[] { "Bob", "Charlie", "David" }));
        Assert.That(departments.ToArray(), Is.EqualTo(new string[] { "HR", "IT", "Finance" }));
    }

    [Test]
    public void LeftJoin_WithPartialMatches_ReturnsAllLeftRows()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = CreateRightFrame();

        // Act
        var result = left.LeftJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(4), "Left join should return all 4 left rows");
        Assert.That(result.ColumnCount, Is.EqualTo(5), "Result should have 5 columns");

        // Verify data - first row (Id=1) should have nulls for right columns
        var ids = result.GetColumn<int>("Id");
        var names = result.GetColumn<string>("Name");
        var departments = result.GetColumn<string>("Department");

        Assert.That(ids.ToArray(), Is.EqualTo(new int[] { 1, 2, 3, 4 }));
        Assert.That(names.ToArray(), Is.EqualTo(new string[] { "Alice", "Bob", "Charlie", "David" }));

        // Check that first row has null department (no match for Id=1)
        Assert.That(departments.IsNull(0), Is.True, "First row should have null department");
        Assert.That(departments[1], Is.EqualTo("HR"));
        Assert.That(departments[2], Is.EqualTo("IT"));
        Assert.That(departments[3], Is.EqualTo("Finance"));
    }

    [Test]
    public void RightJoin_WithPartialMatches_ReturnsAllRightRows()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = CreateRightFrame();

        // Act
        var result = left.RightJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(4), "Right join should return all 4 right rows");
        Assert.That(result.ColumnCount, Is.EqualTo(5), "Result should have 5 columns");

        // Verify data - last row (Id=5) should have nulls for left columns
        var ids = result.GetColumn<int>("Id");
        var names = result.GetColumn<string>("Name");
        var departments = result.GetColumn<string>("Department");

        // The order might be different, so let's check by finding the row with Id=5
        var id5Index = -1;
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] == 5)
            {
                id5Index = i;
                break;
            }
        }

        Assert.That(id5Index, Is.GreaterThanOrEqualTo(0), "Should find row with Id=5");
        Assert.That(names.IsNull(id5Index), Is.True, "Row with Id=5 should have null name");
        Assert.That(departments[id5Index], Is.EqualTo("Marketing"));
    }

    [Test]
    public void FullOuterJoin_WithPartialMatches_ReturnsAllRows()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = CreateRightFrame();

        // Act
        var result = left.FullOuterJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(5), "Full outer join should return 5 rows (1 from left only, 3 matches, 1 from right only)");
        Assert.That(result.ColumnCount, Is.EqualTo(5), "Result should have 5 columns");

        // Verify we have all expected IDs
        var ids = result.GetColumn<int>("Id");
        var idArray = ids.ToArray();
        var expectedIds = new int[] { 1, 2, 3, 4, 5 };

        Assert.That(idArray.OrderBy(x => x), Is.EqualTo(expectedIds.OrderBy(x => x)));
    }

    [Test]
    public void InnerJoin_WithDifferentColumnNames_ReturnsCorrectResult()
    {
        // Arrange
        var left = NivaraFrame.Create(
            ("LeftId", NivaraColumn<int>.Create(new int[] { 1, 2, 3 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string[] { "Alice", "Bob", "Charlie" }))
        );

        var right = NivaraFrame.Create(
            ("RightId", NivaraColumn<int>.Create(new int[] { 2, 3, 4 })),
            ("Department", NivaraColumn<string>.CreateForReferenceType(new string[] { "HR", "IT", "Finance" }))
        );

        // Act
        var result = left.InnerJoin(right, "LeftId", "RightId");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2), "Should have 2 matching rows");
        Assert.That(result.ColumnCount, Is.EqualTo(3), "Should have 3 columns (LeftId, Name, Department)");

        var leftIds = result.GetColumn<int>("LeftId");
        var names = result.GetColumn<string>("Name");
        var departments = result.GetColumn<string>("Department");

        Assert.That(leftIds.ToArray(), Is.EqualTo(new int[] { 2, 3 }));
        Assert.That(names.ToArray(), Is.EqualTo(new string[] { "Bob", "Charlie" }));
        Assert.That(departments.ToArray(), Is.EqualTo(new string[] { "HR", "IT" }));
    }

    [Test]
    public void InnerJoin_WithColumnNameConflicts_UsesDisambiguationStrategy()
    {
        // Arrange
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "X", "Y" }))
        );

        // Act - using suffix disambiguation (default)
        var result = left.InnerJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.ColumnCount, Is.EqualTo(3), "Should have 3 columns (Id, Value_left, Value_right)");

        var columnNames = result.ColumnNames.ToArray();
        Assert.That(columnNames, Contains.Item("Id"));
        Assert.That(columnNames, Contains.Item("Value_left"));
        Assert.That(columnNames, Contains.Item("Value_right"));

        var leftValues = result.GetColumn<string>("Value_left");
        var rightValues = result.GetColumn<string>("Value_right");

        Assert.That(leftValues.ToArray(), Is.EqualTo(new string[] { "A", "B" }));
        Assert.That(rightValues.ToArray(), Is.EqualTo(new string[] { "X", "Y" }));
    }

    [Test]
    public void InnerJoin_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange
        var left = CreateLeftFrameWithNulls();
        var right = CreateRightFrameWithNulls();

        // Act
        var result = left.InnerJoin(right, "Id");

        // Assert
        // Only rows with matching non-null IDs should be included
        // Left: [1, 2, null, 4], Right: [2, null, 4, 5]
        // Matches: Id=2 and Id=4
        Assert.That(result.RowCount, Is.EqualTo(2), "Should have 2 matching rows (Id=2 and Id=4)");

        var ids = result.GetColumn<int>("Id");
        var names = result.GetColumn<string>("Name");
        var departments = result.GetColumn<string>("Department");

        Assert.That(ids.ToArray(), Is.EqualTo(new int[] { 2, 4 }));
        Assert.That(names.ToArray(), Is.EqualTo(new string[] { "Bob", "David" }));
        Assert.That(departments.ToArray(), Is.EqualTo(new string[] { "HR", "Finance" }));
    }

    [Test]
    public void InnerJoin_WithEmptyDataFrame_ReturnsEmpty()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(Array.Empty<int>())),
            ("Department", NivaraColumn<string>.CreateForReferenceType(Array.Empty<string>()))
        );

        // Act
        var result = left.InnerJoin(right, "Id");

        // Assert
        Assert.That(result.RowCount, Is.EqualTo(0), "Join with empty DataFrame should return empty result");
        Assert.That(result.ColumnCount, Is.EqualTo(4), "Should still have all columns from schema");
    }

    [Test]
    public void InnerJoin_WithIncompatibleJoinKeys_ThrowsException()
    {
        // Arrange
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2, 3 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string[] { "Alice", "Bob", "Charlie" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B", "C" })),
            ("Department", NivaraColumn<string>.CreateForReferenceType(new string[] { "HR", "IT", "Finance" }))
        );

        // Act & Assert
        Assert.Throws<SchemaValidationException>(() => left.InnerJoin(right, "Id"),
            "Should throw exception when join key types are incompatible");
    }

    [Test]
    public void InnerJoin_WithMissingJoinKey_ThrowsException()
    {
        // Arrange
        var left = CreateLeftFrame();
        var right = CreateRightFrame();

        // Act & Assert
        Assert.Throws<SchemaValidationException>(() => left.InnerJoin(right, "NonExistentColumn"),
            "Should throw exception when join key doesn't exist");
    }

    [Test]
    public void Join_WithPrefixDisambiguation_UsesPrefixes()
    {
        // Arrange
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "X", "Y" }))
        );

        // Act
        var result = left.InnerJoin(right, "Id", ColumnDisambiguationStrategy.Prefix, "L", "R");

        // Assert
        var columnNames = result.ColumnNames.ToArray();
        Assert.That(columnNames, Contains.Item("Id"));
        Assert.That(columnNames, Contains.Item("L_Value"));
        Assert.That(columnNames, Contains.Item("R_Value"));
    }

    [Test]
    public void Join_WithErrorDisambiguation_ThrowsOnConflict()
    {
        // Arrange
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "X", "Y" }))
        );

        // Act & Assert
        Assert.Throws<SchemaValidationException>(() =>
            left.InnerJoin(right, "Id", ColumnDisambiguationStrategy.Error),
            "Should throw exception when column names conflict and error strategy is used");
    }

    [Test]
    public void FullOuterJoin_WithHalfKeyColumn_ProducesTypedCoalescedKey()
    {
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<Half>.Create(new Half[] { (Half)1.5, (Half)2.5 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<Half>.Create(new Half[] { (Half)2.5, (Half)3.5 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "X", "Y" }))
        );

        var result = left.FullOuterJoin(right, "Id");

        var ids = result.GetColumn("Id");
        Assert.That(ids, Is.InstanceOf<NivaraColumn<Half>>(), "coalesced join key must stay typed");
        Assert.That(ids.Length, Is.EqualTo(3));
        var idValues = new[] { ids.GetValue(0), ids.GetValue(1), ids.GetValue(2) };
        Assert.That(idValues, Contains.Item((Half)1.5));
        Assert.That(idValues, Contains.Item((Half)2.5));
        Assert.That(idValues, Contains.Item((Half)3.5));
    }

    [Test]
    public void FullOuterJoin_WithNIntKeyColumn_ProducesTypedCoalescedKey()
    {
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<nint>.Create(new nint[] { 1, 2 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<nint>.Create(new nint[] { 2, 3 })),
            ("Value", NivaraColumn<string>.CreateForReferenceType(new string[] { "X", "Y" }))
        );

        var result = left.FullOuterJoin(right, "Id");

        var ids = result.GetColumn("Id");
        Assert.That(ids, Is.InstanceOf<NivaraColumn<nint>>(), "coalesced join key must stay typed");
        Assert.That(ids.Length, Is.EqualTo(3));
    }

    [Test]
    public void LeftJoin_WithUIntRightValueColumn_ProducesTypedGatheredColumn()
    {
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2, 3 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B", "C" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 3 })),
            ("Score", NivaraColumn<uint>.Create(new uint[] { 10, 30 }))
        );

        var result = left.LeftJoin(right, "Id");

        var scores = result.GetColumn("Score");
        Assert.That(scores, Is.InstanceOf<NivaraColumn<uint>>(), "gathered column must stay typed");
        Assert.That(scores.Length, Is.EqualTo(3));
        Assert.That(scores.GetValue(0), Is.EqualTo(10u));
        Assert.That(scores.IsNull(1), Is.True, "unmatched right value must be null");
        Assert.That(scores.GetValue(2), Is.EqualTo(30u));
    }

    [Test]
    public void LeftJoin_WithHalfRightValueColumn_ProducesTypedGatheredColumn()
    {
        var left = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 2, 3 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new string[] { "A", "B", "C" }))
        );

        var right = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(new int[] { 1, 3 })),
            ("Score", NivaraColumn<Half>.Create(new Half[] { (Half)1.5, (Half)3.5 }))
        );

        var result = left.LeftJoin(right, "Id");

        var scores = result.GetColumn("Score");
        Assert.That(scores, Is.InstanceOf<NivaraColumn<Half>>(), "gathered column must stay typed");
        Assert.That(scores.Length, Is.EqualTo(3));
        Assert.That(scores.GetValue(0), Is.EqualTo((Half)1.5));
        Assert.That(scores.IsNull(1), Is.True, "unmatched right value must be null");
        Assert.That(scores.GetValue(2), Is.EqualTo((Half)3.5));
    }
}
