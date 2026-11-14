using Nivara.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Helpers;

[TestFixture]
public class CreateEmptyInvariantTests
{
    [Test]
    public void CreateEmpty_PreservesRuntimeType_ForAllSupportedColumns()
    {
        var cases = new (string Name, IColumn Column)[]
        {
            ("int", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("int?", NivaraColumn<int?>.Create(new int?[] { 1, null, 3 })),
            ("string", NivaraColumn<string>.Create(new[] { "a", null!, "c" })),
            ("double", NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 })),
            ("guid", NivaraColumn<Guid>.Create(new[] { Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid() })),
            ("timespan", NivaraColumn<TimeSpan>.Create(new[] { TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(2) })),
            ("bool", NivaraColumn<bool>.Create(new[] { true, false, true })),
        };

        foreach (var (name, column) in cases)
        {
            var empty = column.CreateEmpty();
            Assert.That(empty.GetType(), Is.EqualTo(column.GetType()), $"CreateEmpty must preserve runtime type for {name}");
            Assert.That(empty.ElementType, Is.EqualTo(column.ElementType), $"CreateEmpty must preserve ElementType for {name}");
            Assert.That(empty.Length, Is.EqualTo(0), $"CreateEmpty must be zero-length for {name}");
            Assert.That(empty.HasNulls, Is.False, $"empty column has no nulls for {name}");
        }
    }

    [Test]
    public void CreateEmpty_OnChunkedColumn_PreservesRuntimeType()
    {
        var left = NivaraColumn<int?>.Create(new int?[] { 1, null });
        var right = NivaraColumn<int?>.Create(new int?[] { null, 4 });
        var chunked = (IColumn)ColumnFilterHelper.ConcatenateColumns(new System.Collections.Generic.List<IColumn> { left, right });
        // chunked is a NivaraColumn<int?> wrapping a ChunkedColumn; runtime type is still NivaraColumn<int?>
        var empty = chunked.CreateEmpty();
        Assert.That(empty.GetType(), Is.EqualTo(chunked.GetType()));
        Assert.That(empty.Length, Is.EqualTo(0));
    }
}
