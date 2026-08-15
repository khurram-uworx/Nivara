using Nivara.Exceptions;
using Nivara.Operations;
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

    // ── WindowSpec ──

    [Test]
    public void RollingSum_WithSpec_PartitionsAndOrders()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g").OrderBy("t");

        using var result = frame.AsQueryFrame().RollingSum("v", "rs", 2, spec).Collect();

        var rs = result.GetColumn<int>("rs");
        Assert.That(rs.IsNull(0), Is.True);
        Assert.That(rs.IsNull(1), Is.True);
        Assert.That(rs[2], Is.EqualTo(40));
        Assert.That(rs[3], Is.EqualTo(60));
        Assert.That(rs[4], Is.EqualTo(80));
        Assert.That(rs[5], Is.EqualTo(100));
    }

    [Test]
    public void CumulativeSum_WithSpec_PartitionsAndOrders()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g").OrderBy("t");

        using var result = frame.AsQueryFrame().CumulativeSum("v", "cum", spec).Collect();

        var cum = result.GetColumn<int>("cum");
        Assert.That(cum[0], Is.EqualTo(10));
        Assert.That(cum[1], Is.EqualTo(20));
        Assert.That(cum[2], Is.EqualTo(40));
        Assert.That(cum[3], Is.EqualTo(60));
        Assert.That(cum[4], Is.EqualTo(90));
        Assert.That(cum[5], Is.EqualTo(120));
    }

    [Test]
    public void CumulativeCount_WithSpec_Partitions()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g");

        using var result = frame.AsQueryFrame().CumulativeCount("v", "n", spec).Collect();

        var n = result.GetColumn<long>("n");
        Assert.That(n[0], Is.EqualTo(1));
        Assert.That(n[1], Is.EqualTo(1));
        Assert.That(n[2], Is.EqualTo(2));
        Assert.That(n[3], Is.EqualTo(2));
        Assert.That(n[4], Is.EqualTo(3));
        Assert.That(n[5], Is.EqualTo(3));
    }

    [Test]
    public void Shift_WithSpec_PartitionsAndOrders()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g").OrderBy("t");

        using var result = frame.AsQueryFrame().Shift("v", "lag", 1, spec).Collect();

        var lag = result.GetColumn<int>("lag");
        Assert.That(lag.IsNull(0), Is.True);
        Assert.That(lag.IsNull(1), Is.True);
        Assert.That(lag[2], Is.EqualTo(10));
        Assert.That(lag[3], Is.EqualTo(20));
        Assert.That(lag[4], Is.EqualTo(30));
        Assert.That(lag[5], Is.EqualTo(40));
    }

    [Test]
    public void RollingSum_WithSpec_NullOrderKeyRow_Participates()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "A" })),
            ("t", NivaraColumn<int>.CreateFromSpans(new[] { 3, 0, 1 }, new[] { false, true, false })),
            ("v", IntColumn(30, 20, 10)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g").OrderBy("t");

        using var result = frame.AsQueryFrame().RollingSum("v", "rs", 2, spec).Collect();

        var rs = result.GetColumn<int>("rs");
        Assert.That(rs[0], Is.EqualTo(40));
        Assert.That(rs[1], Is.EqualTo(50));
        Assert.That(rs.IsNull(2), Is.True);
    }

    [Test]
    public void Window_WithSpec_MissingPartitionColumn_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var spec = new WindowSpec().PartitionBy("missing").OrderBy("v");

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("v", "x", 2, spec).Collect());
    }

    [Test]
    public void Window_WithSpec_MissingOrderColumn_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var spec = new WindowSpec().OrderBy("missing");

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("v", "x", 2, spec).Collect());
    }

    [Test]
    public void Window_WithSpec_NonComparableOrderColumn_Throws()
    {
        using var frame = FrameWith(
            ("v", IntColumn(1, 2, 3)),
            ("o", NivaraColumn<object>.Create(new object[] { new object(), new object(), new object() })));
        var spec = frame.AsQueryFrame().Over().OrderBy("o");

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().RollingSum("v", "x", 2, spec).Collect());
    }

    // ── All-null columns and window-size boundaries (#252) ──

    [Test]
    public void Rolling_AllNullColumn_ThroughPipeline_AllMasked()
    {
        var source = NivaraColumn.CreateFromNullable(new int?[] { null, null, null, null });
        using var frame = FrameWith(("v", source));
        using var result = frame.AsQueryFrame()
            .RollingSum("v", "sum", 2)
            .RollingMin("v", "min", 2)
            .RollingMax("v", "max", 2)
            .RollingMean("v", "mean", 2)
            .Collect();

        var sum = result.GetColumn<int>("sum");
        var min = result.GetColumn<int>("min");
        var max = result.GetColumn<int>("max");
        var mean = result.GetColumn<double>("mean");

        foreach (var col in new IColumn[] { sum, min, max, mean })
            for (int i = 0; i < 4; i++)
                Assert.That(col.IsNull(i), Is.True);
    }

    [Test]
    public void Rolling_AllNullColumn_ThroughPipeline_WithNullHandler_Fills()
    {
        var source = NivaraColumn.CreateFromNullable(new int?[] { null, null, null, null });
        using var frame = FrameWith(("v", source));
        using var result = frame.AsQueryFrame()
            .RollingSum("v", "sum", 2, nullHandler: () => 0)
            .RollingMax("v", "max", 2, nullHandler: () => 5)
            .Collect();

        var sum = result.GetColumn<int>("sum");
        var max = result.GetColumn<int>("max");
        Assert.That(sum.HasNulls, Is.False);
        Assert.That(max.HasNulls, Is.False);
        Assert.That(sum[3], Is.EqualTo(0));
        Assert.That(max[3], Is.EqualTo(5));
    }

    [Test]
    public void Rolling_WindowLargerThanData_ThroughPipeline_AllMasked()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().RollingSum("v", "sum", 5).Collect();

        var sum = result.GetColumn<int>("sum");
        for (int i = 0; i < 3; i++)
            Assert.That(sum.IsNull(i), Is.True);
    }

    [Test]
    public void Rolling_WindowEqualsDataLength_ThroughPipeline_FirstOutputAtLastRow()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().RollingSum("v", "sum", 3).Collect();

        var sum = result.GetColumn<int>("sum");
        Assert.That(sum.IsNull(0), Is.True);
        Assert.That(sum.IsNull(1), Is.True);
        Assert.That(sum[2], Is.EqualTo(6));
    }

    [Test]
    public void Shift_PeriodEqualToLength_ThroughPipeline_AllMasked()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame()
            .Shift("v", "lag", 3)
            .Lead("v", "lead", 3)
            .Collect();

        var lag = result.GetColumn<int>("lag");
        var lead = result.GetColumn<int>("lead");
        for (int i = 0; i < 3; i++)
        {
            Assert.That(lag.IsNull(i), Is.True);
            Assert.That(lead.IsNull(i), Is.True);
        }
    }

    [Test]
    public void Shift_AllNullColumn_ThroughPipeline_AllMasked()
    {
        var source = NivaraColumn.CreateFromNullable(new int?[] { null, null, null });
        using var frame = FrameWith(("v", source));
        using var result = frame.AsQueryFrame().Shift("v", "lag", 1).Collect();

        var lag = result.GetColumn<int>("lag");
        for (int i = 0; i < 3; i++)
            Assert.That(lag.IsNull(i), Is.True);
    }
}
