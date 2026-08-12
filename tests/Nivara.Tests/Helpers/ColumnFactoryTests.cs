using Nivara.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Helpers;

/// <summary>
/// Tests that ColumnFactory creates typed columns for the extended CLR domain
/// instead of falling through to an object column (issue #158).
/// </summary>
[TestFixture]
public class ColumnFactoryTests
{
    static void AssertTypedColumn(IColumn column, Type expectedType, object? first, object? last)
    {
        Assert.That(column.ElementType, Is.EqualTo(expectedType), $"{expectedType.Name} element type must be preserved");
        Assert.That(column.Length, Is.EqualTo(3));
        Assert.That(column.HasNulls, Is.True, "null position must be preserved");
        Assert.That(column.IsNull(1), Is.True);
        Assert.That(column.IsNull(0), Is.False);
        Assert.That(column.GetValue(0), Is.EqualTo(first));
        Assert.That(column.GetValue(2), Is.EqualTo(last));
    }

    [Test]
    public void Create_HalfValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(Half), new object?[] { (Half)1, null, (Half)3 }), typeof(Half), (Half)1, (Half)3);

    [Test]
    public void Create_NIntValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(nint), new object?[] { (nint)1, null, (nint)3 }), typeof(nint), (nint)1, (nint)3);

    [Test]
    public void Create_NUIntValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(nuint), new object?[] { (nuint)1, null, (nuint)3 }), typeof(nuint), (nuint)1, (nuint)3);

    [Test]
    public void Create_Int128Values_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(Int128), new object?[] { (Int128)1, null, (Int128)3 }), typeof(Int128), (Int128)1, (Int128)3);

    [Test]
    public void Create_UInt128Values_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(UInt128), new object?[] { (UInt128)1, null, (UInt128)3 }), typeof(UInt128), (UInt128)1, (UInt128)3);

    [Test]
    public void Create_SByteValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(sbyte), new object?[] { (sbyte)1, null, (sbyte)3 }), typeof(sbyte), (sbyte)1, (sbyte)3);

    [Test]
    public void Create_UShortValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(ushort), new object?[] { (ushort)1, null, (ushort)3 }), typeof(ushort), (ushort)1, (ushort)3);

    [Test]
    public void Create_UIntValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(uint), new object?[] { 1u, null, 3u }), typeof(uint), 1u, 3u);

    [Test]
    public void Create_CharValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(char), new object?[] { 'a', null, 'c' }), typeof(char), 'a', 'c');

    [Test]
    public void Create_DateOnlyValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(DateOnly), new object?[] { new DateOnly(2024, 1, 1), null, new DateOnly(2024, 1, 3) }),
            typeof(DateOnly), new DateOnly(2024, 1, 1), new DateOnly(2024, 1, 3));

    [Test]
    public void Create_TimeOnlyValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(TimeOnly), new object?[] { new TimeOnly(1, 0), null, new TimeOnly(3, 0) }),
            typeof(TimeOnly), new TimeOnly(1, 0), new TimeOnly(3, 0));

    [Test]
    public void Create_DateTimeOffsetValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(DateTimeOffset), new object?[] { new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero), null, new DateTimeOffset(2024, 1, 3, 0, 0, 0, TimeSpan.Zero) }),
            typeof(DateTimeOffset), new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero), new DateTimeOffset(2024, 1, 3, 0, 0, 0, TimeSpan.Zero));

    [Test]
    public void Create_GuidValues_ProducesTypedColumn()
    {
        var first = Guid.NewGuid();
        var last = Guid.NewGuid();
        AssertTypedColumn(ColumnFactory.Create(typeof(Guid), new object?[] { first, null, last }), typeof(Guid), first, last);
    }

    [Test]
    public void Create_TimeSpanValues_ProducesTypedColumn()
        => AssertTypedColumn(ColumnFactory.Create(typeof(TimeSpan), new object?[] { TimeSpan.FromSeconds(1), null, TimeSpan.FromSeconds(3) }),
            typeof(TimeSpan), TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(3));

    [Test]
    public void Create_NullableUnderlyingType_UnwrapsToTypedColumn()
    {
        var column = ColumnFactory.Create(typeof(int?), new object?[] { 1, null, 3 });

        Assert.That(column, Is.InstanceOf<NivaraColumn<int>>(), "Nullable<int> must be unwrapped to NivaraColumn<int>");
        Assert.That(column.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(column.IsNull(1), Is.True);
        Assert.That(column.GetValue(0), Is.EqualTo(1));
    }

    [Test]
    public void Create_StringValues_ProducesReferenceTypedColumn()
    {
        var column = ColumnFactory.Create(typeof(string), new object?[] { "a", null, "c" });

        Assert.That(column, Is.InstanceOf<NivaraColumn<string>>());
        Assert.That(column.HasNulls, Is.True);
        Assert.That(column.IsNull(1), Is.True);
        Assert.That(column.GetValue(0), Is.EqualTo("a"));
    }

    [Test]
    public void Create_AllNulls_ProducesTypedColumnWithNullMask()
    {
        var column = ColumnFactory.Create(typeof(Half), new object?[] { null, null });

        Assert.That(column, Is.InstanceOf<NivaraColumn<Half>>());
        Assert.That(column.HasNulls, Is.True);
        Assert.That(column.IsNull(0), Is.True);
        Assert.That(column.IsNull(1), Is.True);
    }
}
