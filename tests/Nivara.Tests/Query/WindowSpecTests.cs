using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for the <see cref="WindowSpec"/> builder (issue #162) plus eager/lazy parity
/// when the same spec is passed to both window layers.
/// </summary>
/// <remarks>Added as part of issue #162 Over/WindowSpec builder delivery.</remarks>
[TestFixture]
public class WindowSpecTests
{
    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new NivaraFrame(columns);

    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    // ── Builder ──

    [Test]
    public void DefaultSpec_IsEmpty()
    {
        var spec = new WindowSpec();

        Assert.That(spec.IsEmpty, Is.True);
        Assert.That(spec.PartitionColumns, Is.Empty);
        Assert.That(spec.OrderKeys, Is.Empty);
    }

    [Test]
    public void NivaraFrameOver_ReturnsEmptySpec()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2)));

        Assert.That(frame.Over().IsEmpty, Is.True);
    }

    [Test]
    public void QueryFrameOver_ReturnsEmptySpec()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2)));

        Assert.That(frame.AsQueryFrame().Over().IsEmpty, Is.True);
    }

    [Test]
    public void PartitionBy_AddsPartitionKeys()
    {
        var spec = new WindowSpec().PartitionBy("g", "h");

        Assert.That(spec.PartitionColumns, Is.EqualTo(new[] { "g", "h" }));
        Assert.That(spec.OrderKeys, Is.Empty);
        Assert.That(spec.IsEmpty, Is.False);
    }

    [Test]
    public void OrderBy_SortKeys_AddsOrderKeys()
    {
        var spec = new WindowSpec().OrderBy(new SortKey("t", SortDirection.Descending));

        Assert.That(spec.OrderKeys, Has.Count.EqualTo(1));
        Assert.That(spec.OrderKeys[0].ColumnName, Is.EqualTo("t"));
        Assert.That(spec.OrderKeys[0].Direction, Is.EqualTo(SortDirection.Descending));
        Assert.That(spec.OrderKeys[0].NullOrdering, Is.EqualTo(NullOrdering.NullsLast));
    }

    [Test]
    public void OrderBy_StringColumns_AddsAscendingKeys()
    {
        var spec = new WindowSpec().OrderBy("t");

        Assert.That(spec.OrderKeys[0].ColumnName, Is.EqualTo("t"));
        Assert.That(spec.OrderKeys[0].Direction, Is.EqualTo(SortDirection.Ascending));
        Assert.That(spec.OrderKeys[0].NullOrdering, Is.EqualTo(NullOrdering.NullsLast));
    }

    [Test]
    public void OrderBy_ColumnDirection_AddsKey()
    {
        var spec = new WindowSpec().OrderBy("t", SortDirection.Descending, NullOrdering.NullsFirst);

        Assert.That(spec.OrderKeys[0].ColumnName, Is.EqualTo("t"));
        Assert.That(spec.OrderKeys[0].Direction, Is.EqualTo(SortDirection.Descending));
        Assert.That(spec.OrderKeys[0].NullOrdering, Is.EqualTo(NullOrdering.NullsFirst));
    }

    [Test]
    public void FluentMethods_AreImmutable()
    {
        var spec = new WindowSpec().PartitionBy("g");
        var extended = spec.OrderBy("t");

        Assert.That(spec.OrderKeys, Is.Empty);
        Assert.That(spec.PartitionColumns, Is.EqualTo(new[] { "g" }));
        Assert.That(extended.PartitionColumns, Is.EqualTo(new[] { "g" }));
        Assert.That(extended.OrderKeys, Has.Count.EqualTo(1));
    }

    [Test]
    public void ChainOrder_IsIrrelevant()
    {
        var a = new WindowSpec().PartitionBy("g").OrderBy("t");
        var b = new WindowSpec().OrderBy("t").PartitionBy("g");

        Assert.That(a.PartitionColumns, Is.EqualTo(b.PartitionColumns));
        Assert.That(a.OrderKeys.Select(k => k.ColumnName), Is.EqualTo(b.OrderKeys.Select(k => k.ColumnName)));
    }

    // ── Builder validation ──

    [Test]
    public void PartitionBy_NullArgument_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new WindowSpec().PartitionBy(null!));
    }

    [Test]
    public void PartitionBy_WhitespaceEntry_Throws()
    {
        Assert.Throws<ArgumentException>(() => new WindowSpec().PartitionBy("  "));
    }

    [Test]
    public void OrderBy_NullKeys_Throws()
    {
        Assert.Throws<ArgumentException>(() => new WindowSpec().OrderBy(new SortKey?[] { null! }!));
    }

    [Test]
    public void OrderBy_WhitespaceColumn_Throws()
    {
        Assert.Throws<ArgumentException>(() => new WindowSpec().OrderBy("  "));
        Assert.Throws<ArgumentException>(() => new WindowSpec().OrderBy("", SortDirection.Ascending));
    }

    // ── Eager / lazy parity ──

    [Test]
    public void EagerAndLazy_WithSameSpec_ProduceSameResults()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");

        var eager = frame.RollingSum("v", "rs", 2, spec);
        using var lazy = frame.AsQueryFrame().RollingSum("v", "rs", 2, spec).Collect();

        Assert.That(eager.GetColumn<int>("rs").ToArray(), Is.EqualTo(lazy.GetColumn<int>("rs").ToArray()));
    }

    [Test]
    public void EagerAndLazy_RankFamily_WithSameSpec_ProduceSameResults()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "B" })),
            ("t", IntColumn(2, 1, 2, 1)),
            ("v", IntColumn(10, 20, 30, 40)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");

        var eager = frame.Rank("rnk", spec);
        using var lazy = frame.AsQueryFrame().Rank("rnk", spec).Collect();

        Assert.That(eager.GetColumn<long>("rnk").ToArray(), Is.EqualTo(lazy.GetColumn<long>("rnk").ToArray()));
    }
}
