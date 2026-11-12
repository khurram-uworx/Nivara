using Nivara;
using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class NivaraFrameEmptyMaterialisationTests
{
    [Test]
    public void FilterByMask_AllFalse_PreservesRuntimeType_NullableInt()
    {
        using var frame = NivaraFrame.Create(
            ("A", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })));
        using var empty = frame.FilterByMask(NivaraColumn<bool>.Create(new[] { false, false, false }));

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("A").GetType(), Is.EqualTo(frame.GetColumn("A").GetType()));
        var col = empty.GetColumn<int?>("A");
        Assert.That(col.Length, Is.EqualTo(0));
    }

    [Test]
    public void FilterByMask_AllFalse_PreservesRuntimeType_String()
    {
        using var frame = NivaraFrame.Create(
            ("B", NivaraColumn<string>.CreateForReferenceType(new[] { "a", null!, "c" })));
        using var empty = frame.FilterByMask(NivaraColumn<bool>.Create(new[] { false, false, false }));

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("B").GetType(), Is.EqualTo(frame.GetColumn("B").GetType()));
        var col = empty.GetColumn<string>("B");
        Assert.That(col.Length, Is.EqualTo(0));
    }

    [Test]
    public void FilterByMask_AllFalse_PreservesRuntimeType_ValueType()
    {
        using var frame = NivaraFrame.Create(
            ("C", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));
        using var empty = frame.FilterByMask(NivaraColumn<bool>.Create(new[] { false, false, false }));

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("C").GetType(), Is.EqualTo(frame.GetColumn("C").GetType()));
        var col = empty.GetColumn<int>("C");
        Assert.That(col.Length, Is.EqualTo(0));
    }

    [Test]
    public void Take_Zero_PreservesRuntimeType()
    {
        using var frame = NivaraFrame.Create(
            ("A", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })));
        using var empty = frame.Take(0);

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("A").GetType(), Is.EqualTo(frame.GetColumn("A").GetType()));
        Assert.That(empty.GetColumn<int?>("A").Length, Is.EqualTo(0));
    }

    [Test]
    public void Skip_All_PreservesRuntimeType()
    {
        using var frame = NivaraFrame.Create(
            ("A", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })));
        using var empty = frame.Skip(3);

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("A").GetType(), Is.EqualTo(frame.GetColumn("A").GetType()));
        Assert.That(empty.GetColumn<int?>("A").Length, Is.EqualTo(0));
    }

    [Test]
    public void Slice_ZeroLength_PreservesRuntimeType()
    {
        using var frame = NivaraFrame.Create(
            ("A", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })));
        using var empty = frame.Slice(0, 0);

        Assert.That(empty.RowCount, Is.EqualTo(0));
        Assert.That(empty.GetColumn("A").GetType(), Is.EqualTo(frame.GetColumn("A").GetType()));
        Assert.That(empty.GetColumn<int?>("A").Length, Is.EqualTo(0));
    }

    [Test]
    public void EmptyFrame_SchemaUnchanged()
    {
        using var frame = NivaraFrame.Create(
            ("A", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })));
        using var empty = frame.FilterByMask(NivaraColumn<bool>.Create(new[] { false, false, false }));

        Assert.That(empty.Schema, Is.EqualTo(frame.Schema));
        Assert.That(empty.ColumnCount, Is.EqualTo(frame.ColumnCount));
        Assert.That(empty.ColumnNames, Is.EqualTo(frame.ColumnNames));
    }
}
