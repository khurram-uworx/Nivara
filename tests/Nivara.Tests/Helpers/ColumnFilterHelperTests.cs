using Nivara.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Helpers;

/// <summary>
/// Tests that ColumnFilterHelper preserves the source column's element type
/// for all element types, not just the previously hardcoded list (issue #104).
/// </summary>
[TestFixture]
public class ColumnFilterHelperTests
{
    [Test]
    public void CreateFilteredColumn_GuidColumn_PreservesElementTypeAndNulls()
    {
        var target = Guid.NewGuid();
        var ids = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid() });
        var indices = new List<int> { 0, 2 };

        var filtered = ColumnFilterHelper.CreateFilteredColumn(ids, indices);

        Assert.That(filtered.ElementType, Is.EqualTo(typeof(Guid)), "Guid element type must be preserved");
        Assert.That(filtered.Length, Is.EqualTo(2));
        Assert.That(filtered.HasNulls, Is.False, "no nulls selected");
        Assert.That(filtered.GetValue(0), Is.EqualTo(target));
    }

    [Test]
    public void CreateFilteredColumn_GuidColumnWithNullIndex_PreservesNullMask()
    {
        var target = Guid.NewGuid();
        var ids = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid() });
        var indices = new List<int> { 0, 1 };

        var filtered = ColumnFilterHelper.CreateFilteredColumn(ids, indices);

        Assert.That(filtered.ElementType, Is.EqualTo(typeof(Guid)), "Guid element type must be preserved");
        Assert.That(filtered.HasNulls, Is.True, "null row must remain masked");
        Assert.That(filtered.IsNull(1), Is.True, "second row was null in source");
    }

    [Test]
    public void CreateFilteredColumn_TimeSpanColumn_PreservesElementType()
    {
        var ts = new[] { TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(3) };
        var column = NivaraColumn<TimeSpan>.Create(ts);
        var indices = new List<int> { 2, 0 };

        var filtered = ColumnFilterHelper.CreateFilteredColumn(column, indices);

        Assert.That(filtered.ElementType, Is.EqualTo(typeof(TimeSpan)), "TimeSpan element type must be preserved");
        Assert.That(filtered.GetValue(0), Is.EqualTo(TimeSpan.FromSeconds(3)));
        Assert.That(filtered.GetValue(1), Is.EqualTo(TimeSpan.FromSeconds(1)));
    }

    [Test]
    public void CreateFilteredColumn_ReferenceType_PreservesElementType()
    {
        var column = NivaraColumn<Uri>.CreateForReferenceType(new Uri[]
        {
            new("https://example.com/a"),
            null!,
            new("https://example.com/b")
        });
        var indices = new List<int> { 1, 2 };

        var filtered = ColumnFilterHelper.CreateFilteredColumn(column, indices);

        Assert.That(filtered.ElementType, Is.EqualTo(typeof(Uri)), "reference element type must be preserved");
        Assert.That(filtered.HasNulls, Is.True);
        Assert.That(filtered.IsNull(0), Is.True);
        Assert.That(filtered.GetValue(1), Is.EqualTo(new Uri("https://example.com/b")));
    }

    [Test]
    public void CreateFilteredColumn_EnumColumn_PreservesElementType()
    {
        var values = new[] { DayOfWeek.Monday, DayOfWeek.Wednesday, DayOfWeek.Friday };
        var column = NivaraColumn<DayOfWeek>.Create(values);
        var indices = new List<int> { 1 };

        var filtered = ColumnFilterHelper.CreateFilteredColumn(column, indices);

        Assert.That(filtered.ElementType, Is.EqualTo(typeof(DayOfWeek)), "enum element type must be preserved");
        Assert.That(filtered.GetValue(0), Is.EqualTo(DayOfWeek.Wednesday));
    }

    [Test]
    public void CreateEmptyColumn_Guid_PreservesElementType()
    {
        var empty = ColumnFilterHelper.CreateEmptyColumn(typeof(Guid));

        Assert.That(empty.ElementType, Is.EqualTo(typeof(Guid)), "empty Guid column must keep element type");
        Assert.That(empty.Length, Is.EqualTo(0));
    }

    [Test]
    public void CreateEmptyColumn_TimeSpan_PreservesElementType()
    {
        var empty = ColumnFilterHelper.CreateEmptyColumn(typeof(TimeSpan));

        Assert.That(empty.ElementType, Is.EqualTo(typeof(TimeSpan)), "empty TimeSpan column must keep element type");
        Assert.That(empty.Length, Is.EqualTo(0));
    }

    [Test]
    public void ReorderColumn_GuidColumn_PreservesElementType()
    {
        var g1 = Guid.NewGuid();
        var g2 = Guid.NewGuid();
        var column = NivaraColumn<Guid>.Create(new[] { g1, g2 });
        var indices = new[] { 1, 0 };

        var reordered = ColumnFilterHelper.ReorderColumn(column, indices);

        Assert.That(reordered.ElementType, Is.EqualTo(typeof(Guid)), "reorder must preserve Guid element type");
        Assert.That(reordered.GetValue(0), Is.EqualTo(g2));
        Assert.That(reordered.GetValue(1), Is.EqualTo(g1));
    }

    [Test]
    public void ConcatenateColumns_GuidColumns_PreservesElementType()
    {
        var g1 = Guid.NewGuid();
        var g2 = Guid.NewGuid();
        var left = NivaraColumn<Guid>.Create(new[] { g1 });
        var right = NivaraColumn<Guid>.Create(new[] { g2 });

        var concatenated = ColumnFilterHelper.ConcatenateColumns(new List<IColumn> { left, right });

        Assert.That(concatenated.ElementType, Is.EqualTo(typeof(Guid)), "concatenation must preserve Guid element type");
        Assert.That(concatenated.Length, Is.EqualTo(2));
        Assert.That(concatenated.GetValue(0), Is.EqualTo(g1));
        Assert.That(concatenated.GetValue(1), Is.EqualTo(g2));
    }

    [Test]
    public void ConcatenateColumns_NullableGuid_ThrowsOnTypeMismatch()
    {
        var left = NivaraColumn<Guid>.Create(new[] { Guid.NewGuid() });
        var right = NivaraColumn<int>.Create(new[] { 1 });

        var ex = Assert.Throws<ArgumentException>(() =>
            ColumnFilterHelper.ConcatenateColumns(new List<IColumn> { left, right }));

        Assert.That(ex!.Message, Does.Contain("different types"));
    }

    [Test]
    public void CreateNullColumn_Guid_PreservesElementType()
    {
        var nullColumn = ColumnFilterHelper.CreateNullColumn(typeof(Guid), 3);

        Assert.That(nullColumn.ElementType, Is.EqualTo(typeof(Guid)), "null column must keep Guid element type");
        Assert.That(nullColumn.Length, Is.EqualTo(3));
        Assert.That(nullColumn.HasNulls, Is.True);
        Assert.That(nullColumn.IsNull(0), Is.True);
    }
}
