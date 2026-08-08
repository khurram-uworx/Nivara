using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for the eager NivaraFrame rank-family window extensions
/// (row_number / rank / dense_rank / percent_rank).
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery.</remarks>
[TestFixture]
public class RankFunctionsFrameTests
{
    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new NivaraFrame(columns);

    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    // ── Rank family ──

    [Test]
    public void Rank_AddsResultColumn_KeepsSourceColumns()
    {
        var frame = FrameWith(("v", IntColumn(10, 20, 20, 30)));

        var result = frame.Rank("rnk", new[] { new SortKey("v") });

        Assert.That(result.HasColumn("v"), Is.True);
        Assert.That(result.HasColumn("rnk"), Is.True);
        var rnk = result.GetColumn<long>("rnk");
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk[1], Is.EqualTo(2));
        Assert.That(rnk[2], Is.EqualTo(2));
        Assert.That(rnk[3], Is.EqualTo(4));
    }

    [Test]
    public void DenseRank_NoGapsOnTies()
    {
        var frame = FrameWith(("v", IntColumn(10, 20, 20, 30, 20)));

        var result = frame.DenseRank("dense", new[] { new SortKey("v") });

        var dense = result.GetColumn<long>("dense");
        Assert.That(dense[0], Is.EqualTo(1));
        Assert.That(dense[1], Is.EqualTo(2));
        Assert.That(dense[2], Is.EqualTo(2));
        Assert.That(dense[3], Is.EqualTo(3));
        Assert.That(dense[4], Is.EqualTo(2));
    }

    [Test]
    public void PercentRank_AddsDoubleColumn()
    {
        var frame = FrameWith(("v", IntColumn(10, 20, 20, 30)));

        var result = frame.PercentRank("pct", new[] { new SortKey("v") });

        Assert.That(result.GetColumn("pct").ElementType, Is.EqualTo(typeof(double)));
        var pct = result.GetColumn<double>("pct");
        Assert.That(pct[0], Is.EqualTo(0.0));
        Assert.That(pct[1], Is.EqualTo(1.0 / 3.0).Within(1e-9));
        Assert.That(pct[2], Is.EqualTo(1.0 / 3.0).Within(1e-9));
        Assert.That(pct[3], Is.EqualTo(1.0));
    }

    [Test]
    public void RowNumber_NoOrderBy_Sequential()
    {
        var frame = FrameWith(("v", IntColumn(5, 1, 3)));

        var result = frame.RowNumber("rn");

        var rn = result.GetColumn<long>("rn");
        Assert.That(rn[0], Is.EqualTo(1));
        Assert.That(rn[1], Is.EqualTo(2));
        Assert.That(rn[2], Is.EqualTo(3));
    }

    [Test]
    public void RowNumber_WithPartition_ResetsPerGroup()
    {
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "A" })),
            ("v", IntColumn(10, 20, 30, 20)));

        var result = frame.RowNumber("rn", new[] { "g" });

        var rn = result.GetColumn<long>("rn");
        Assert.That(rn[0], Is.EqualTo(1));
        Assert.That(rn[1], Is.EqualTo(2));
        Assert.That(rn[2], Is.EqualTo(1));
        Assert.That(rn[3], Is.EqualTo(3));
    }

    [Test]
    public void Rank_Descending_OrdersHighToLow()
    {
        var frame = FrameWith(("v", IntColumn(30, 20, 20, 10)));

        var result = frame.Rank("rnk", new[] { new SortKey("v", SortDirection.Descending) });

        var rnk = result.GetColumn<long>("rnk");
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk[1], Is.EqualTo(2));
        Assert.That(rnk[2], Is.EqualTo(2));
        Assert.That(rnk[3], Is.EqualTo(4));
    }

    [Test]
    public void Rank_NullOrderKey_NullOutput()
    {
        var source = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false });
        var frame = FrameWith(("v", source));

        var result = frame.Rank("rnk", new[] { new SortKey("v") });

        var rnk = result.GetColumn<long>("rnk");
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk.IsNull(1), Is.True);
        Assert.That(rnk[2], Is.EqualTo(2));
    }

    [Test]
    public void Rank_ChainsWithCumulativeOps()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var result = frame.CumulativeCount("v", "count").Rank("rnk", new[] { new SortKey("v") });

        Assert.That(result.HasColumn("count"), Is.True);
        Assert.That(result.HasColumn("rnk"), Is.True);
        Assert.That(result.GetColumn<long>("count")[2], Is.EqualTo(3));
        Assert.That(result.GetColumn<long>("rnk")[0], Is.EqualTo(1));
    }

    // ── Validation ──

    [Test]
    public void Rank_NoOrderKeys_Throws()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.Rank("rnk", Array.Empty<SortKey>()));
    }

    [Test]
    public void Rank_MissingPartitionColumn_Throws()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.Rank("rnk", new[] { new SortKey("v") }, "missing"));
    }

    [Test]
    public void Rank_MissingOrderColumn_Throws()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.Rank("rnk", new[] { new SortKey("missing") }));
    }

    [Test]
    public void Rank_ResultColumnCollision_Throws()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.Rank("v", new[] { new SortKey("v") }));
    }
}
