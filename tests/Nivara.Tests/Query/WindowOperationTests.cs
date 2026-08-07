using Nivara.Exceptions;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for window operations in the lazy query pipeline
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
[TestFixture]
public class WindowOperationTests
{
    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new NivaraFrame(columns);

    // ── Rolling ──

    [Test]
    public void RollingSum_AddsResultColumn_KeepsSourceColumns()
    {
        using var frame = FrameWith(("price", IntColumn(1, 2, 3, 4, 5)));
        using var result = frame.AsQueryFrame().RollingSum("price", "ma", 3).Collect();

        Assert.That(result.HasColumn("price"), Is.True);
        Assert.That(result.HasColumn("ma"), Is.True);
        var ma = result.GetColumn<int>("ma");
        Assert.That(ma.IsNull(0), Is.True);
        Assert.That(ma.IsNull(1), Is.True);
        Assert.That(ma[2], Is.EqualTo(6));
        Assert.That(ma[3], Is.EqualTo(9));
        Assert.That(ma[4], Is.EqualTo(12));
    }

    [Test]
    public void RollingMean_AddsDoubleColumn()
    {
        using var frame = FrameWith(("price", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().RollingMean("price", "mean3", 3).Collect();

        var mean = result.GetColumn<double>("mean3");
        Assert.That(mean[2], Is.EqualTo(2.0).Within(1e-9));
    }

    [Test]
    public void RollingMinMax_AddResultColumns()
    {
        using var frame = FrameWith(("v", IntColumn(3, 1, 4, 1, 5)));
        using var result = frame.AsQueryFrame()
            .RollingMin("v", "min3", 3)
            .RollingMax("v", "max3", 3)
            .Collect();

        var min = result.GetColumn<int>("min3");
        var max = result.GetColumn<int>("max3");
        Assert.That(min[2], Is.EqualTo(1));
        Assert.That(max[2], Is.EqualTo(4));
        Assert.That(max[4], Is.EqualTo(5));
    }

    [Test]
    public void Rolling_WithNullHandler_ReplacesNulls()
    {
        var source = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false });
        using var frame = FrameWith(("v", source));
        using var result = frame.AsQueryFrame().RollingSum("v", "sum", 3, nullHandler: () => 0).Collect();

        var sum = result.GetColumn<int>("sum");
        Assert.That(sum.HasNulls, Is.False);
        Assert.That(sum[2], Is.EqualTo(4));
    }

    [Test]
    public void Rolling_NonNumericColumn_Throws()
    {
        using var frame = FrameWith(("name", NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c" })));

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("name", "x", 3).Collect());
    }

    // ── Cumulative ──

    [Test]
    public void CumulativeSum_AddsColumn_PreservesNulls()
    {
        var source = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2, 0, 3 }, new[] { false, true, false, true, false });
        using var frame = FrameWith(("v", source));
        using var result = frame.AsQueryFrame().CumulativeSum("v", "cum").Collect();

        var cum = result.GetColumn<int>("cum");
        Assert.That(cum[0], Is.EqualTo(1));
        Assert.That(cum.IsNull(1), Is.True);
        Assert.That(cum[2], Is.EqualTo(3));
        Assert.That(cum[4], Is.EqualTo(6));
    }

    [Test]
    public void CumulativeCount_AddsLongColumn()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().CumulativeCount("v", "n").Collect();

        var n = result.GetColumn<long>("n");
        Assert.That(n[0], Is.EqualTo(1));
        Assert.That(n[1], Is.EqualTo(2));
        Assert.That(n[2], Is.EqualTo(3));
    }

    // ── Shift / Lead ──

    [Test]
    public void Shift_AddsLagColumn_NullBoundaries()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().Shift("v", "lag", 1).Collect();

        var lag = result.GetColumn<int>("lag");
        Assert.That(lag.IsNull(0), Is.True);
        Assert.That(lag[1], Is.EqualTo(1));
        Assert.That(lag[2], Is.EqualTo(2));
    }

    [Test]
    public void Shift_WithFillValue_FillsBoundaries()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().Shift("v", "lag", 1, fillValue: 0).Collect();

        Assert.That(result.GetColumn<int>("lag").HasNulls, Is.False);
        Assert.That(result.GetColumn<int>("lag")[0], Is.EqualTo(0));
    }

    [Test]
    public void Lead_AddsLeadColumn_NullBoundaries()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().Lead("v", "lead", 1).Collect();

        var lead = result.GetColumn<int>("lead");
        Assert.That(lead[0], Is.EqualTo(2));
        Assert.That(lead[1], Is.EqualTo(3));
        Assert.That(lead.IsNull(2), Is.True);
    }

    // ── Schema ──

    [Test]
    public void Window_ResultColumnReflectedInSchema()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var queryFrame = frame.AsQueryFrame().RollingMean("v", "mean3", 3);

        var schema = queryFrame.Schema;
        Assert.That(schema.HasColumn("mean3"), Is.True);
        Assert.That(schema.GetColumnType("mean3"), Is.EqualTo(typeof(double)));
        Assert.That(schema.ColumnNames.Count, Is.EqualTo(2));
    }

    // ── Validation ──

    [Test]
    public void Window_MissingSourceColumn_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("missing", "x", 3).Collect());
    }

    [Test]
    public void Window_ResultColumnCollision_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("v", "v", 3).Collect());
    }

    [Test]
    public void Window_Pipeline_SchemaReflectsMultipleColumns()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var queryFrame = frame.AsQueryFrame()
            .RollingSum("v", "sum", 2)
            .CumulativeCount("v", "count")
            .Shift("v", "lag", 1);

        var schema = queryFrame.Schema;
        Assert.That(schema.HasColumn("sum"), Is.True);
        Assert.That(schema.HasColumn("count"), Is.True);
        Assert.That(schema.HasColumn("lag"), Is.True);
        Assert.That(schema.GetColumnType("count"), Is.EqualTo(typeof(long)));
    }
}
