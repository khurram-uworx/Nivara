using Nivara.Exceptions;
using Nivara.Operations;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for the eager NivaraFrame window-function extensions
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
[TestFixture]
public class WindowFunctionsFrameTests
{
    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new NivaraFrame(columns);

    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    // ── Rolling ──

    [Test]
    public void RollingSum_AddsResultColumn_KeepsSourceColumns()
    {
        var frame = FrameWith(("price", IntColumn(1, 2, 3, 4, 5)));

        var result = frame.RollingSum("price", "ma", 3);

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
        var frame = FrameWith(("price", IntColumn(1, 2, 3)));

        var result = frame.RollingMean("price", "mean3", 3);

        var mean = result.GetColumn<double>("mean3");
        Assert.That(mean[2], Is.EqualTo(2.0).Within(1e-9));
    }

    [Test]
    public void RollingMinMax_AddResultColumns()
    {
        var frame = FrameWith(("v", IntColumn(3, 1, 4, 1, 5)));

        var result = frame.RollingMin("v", "min3", 3).RollingMax("v", "max3", 3);

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
        var frame = FrameWith(("v", source));

        var result = frame.RollingSum("v", "sum", 3, nullHandler: () => 0);

        var sum = result.GetColumn<int>("sum");
        Assert.That(sum.HasNulls, Is.False);
        Assert.That(sum[2], Is.EqualTo(4));
    }

    [Test]
    public void Rolling_NonNumericColumn_ThrowsNotSupported()
    {
        var frame = FrameWith(("name", NivaraColumn<string>.Create(new[] { "a", "b", "c" })));

        Assert.Throws<NotSupportedException>(() => frame.RollingSum("name", "x", 3));
    }

    // ── Cumulative ──

    [Test]
    public void CumulativeSum_AddsColumn_PreservesNulls()
    {
        var source = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2, 0, 3 }, new[] { false, true, false, true, false });
        var frame = FrameWith(("v", source));

        var result = frame.CumulativeSum("v", "cum");

        var cum = result.GetColumn<int>("cum");
        Assert.That(cum[0], Is.EqualTo(1));
        Assert.That(cum.IsNull(1), Is.True);
        Assert.That(cum[2], Is.EqualTo(3));
        Assert.That(cum[4], Is.EqualTo(6));
    }

    [Test]
    public void CumulativeCount_AddsLongColumn()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var result = frame.CumulativeCount("v", "n");

        var n = result.GetColumn<long>("n");
        Assert.That(n[0], Is.EqualTo(1));
        Assert.That(n[1], Is.EqualTo(2));
        Assert.That(n[2], Is.EqualTo(3));
    }

    // ── Shift / Lead ──

    [Test]
    public void Shift_AddsLagColumn_NullBoundaries()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var result = frame.Shift("v", "lag", 1);

        var lag = result.GetColumn<int>("lag");
        Assert.That(lag.IsNull(0), Is.True);
        Assert.That(lag[1], Is.EqualTo(1));
        Assert.That(lag[2], Is.EqualTo(2));
    }

    [Test]
    public void Shift_WithFillValue_FillsBoundaries()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var result = frame.Shift("v", "lag", 1, fillValue: 0);

        Assert.That(result.GetColumn<int>("lag").HasNulls, Is.False);
        Assert.That(result.GetColumn<int>("lag")[0], Is.EqualTo(0));
    }

    [Test]
    public void Lead_AddsLeadColumn_NullBoundaries()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var result = frame.Lead("v", "lead", 1);

        var lead = result.GetColumn<int>("lead");
        Assert.That(lead[0], Is.EqualTo(2));
        Assert.That(lead[1], Is.EqualTo(3));
        Assert.That(lead.IsNull(2), Is.True);
    }

    [Test]
    public void Shift_StringColumn_Works()
    {
        var frame = FrameWith(("s", NivaraColumn<string>.Create(new[] { "a", "b", "c" })));

        var result = frame.Shift("s", "lag", 1);

        Assert.That(result.GetColumn<string>("lag")[1], Is.EqualTo("a"));
    }

    // ── Extended numeric domain (issue #158) ──

    [Test]
    public void RollingSum_HalfColumn_ProducesTypedColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3 })));

        var result = frame.RollingSum("v", "sum", 3);

        var sum = result.GetColumn<Half>("sum");
        Assert.That(sum[2], Is.EqualTo((Half)6));
    }

    [Test]
    public void RollingSum_BFloat16Column_ProducesTypedColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)1, (BFloat16)2, (BFloat16)3 })));

        var result = frame.RollingSum("v", "sum", 3);

        var sum = result.GetColumn<BFloat16>("sum");
        Assert.That(sum[2], Is.EqualTo((BFloat16)6));
    }

    [Test]
    public void RollingMean_HalfColumn_ProducesDoubleColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3 })));

        var result = frame.RollingMean("v", "mean", 3);

        var mean = result.GetColumn<double>("mean");
        Assert.That(mean[2], Is.EqualTo(2.0).Within(1e-9));
    }

    [Test]
    public void CumulativeSum_NIntColumn_ProducesTypedColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<nint>.Create(new nint[] { 1, 2, 3 })));

        var result = frame.CumulativeSum("v", "cum");

        var cum = result.GetColumn<nint>("cum");
        Assert.That(cum[2], Is.EqualTo((nint)6));
    }

    [Test]
    public void CumulativeProduct_Int128Column_ProducesTypedColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<Int128>.Create(new Int128[] { 1, 2, 3 })));

        var result = frame.CumulativeProduct("v", "prod");

        var prod = result.GetColumn<Int128>("prod");
        Assert.That(prod[2], Is.EqualTo((Int128)6));
    }

    [Test]
    public void CumulativeCount_HalfColumn_ProducesLongColumn()
    {
        var frame = FrameWith(("v", NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3 })));

        var result = frame.CumulativeCount("v", "n");

        var n = result.GetColumn<long>("n");
        Assert.That(n[2], Is.EqualTo(3L));
    }

    [Test]
    public void Shift_HalfColumn_WithTypedFillValue_FillsBoundary()
    {
        var frame = FrameWith(("v", NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3 })));

        var result = frame.Shift("v", "lag", 1, fillValue: (Half)0);

        var lag = result.GetColumn<Half>("lag");
        Assert.That(lag.HasNulls, Is.False);
        Assert.That(lag[0], Is.EqualTo((Half)0));
        Assert.That(lag[1], Is.EqualTo((Half)1));
    }

    [Test]
    public void Shift_Int128Column_WithStringFillValue_UsesTryParse()
    {
        var frame = FrameWith(("v", NivaraColumn<Int128>.Create(new Int128[] { 1, 2, 3 })));

        var result = frame.Shift("v", "lag", 1, fillValue: "7");

        var lag = result.GetColumn<Int128>("lag");
        Assert.That(lag.HasNulls, Is.False);
        Assert.That(lag[0], Is.EqualTo((Int128)7));
    }

    // ── Validation ──

    [Test]
    public void Window_MissingSourceColumn_ThrowsColumnNotFound()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ColumnNotFoundException>(() => frame.RollingSum("missing", "x", 3));
    }

    [Test]
    public void Window_ResultColumnCollision_ThrowsArgumentException()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.RollingSum("v", "v", 3));
    }

    // ── WindowSpec overloads ──

    [Test]
    public void RollingSum_WithSpec_PartitionsAndOrders()
    {
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");
        var result = frame.RollingSum("v", "rs", 2, spec);

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
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");
        var result = frame.CumulativeSum("v", "cum", spec);

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
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));

        var spec = frame.Over().PartitionBy("g");
        var result = frame.CumulativeCount("v", "n", spec);

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
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "B", "A", "B", "A", "B" })),
            ("t", IntColumn(1, 1, 2, 2, 3, 3)),
            ("v", IntColumn(10, 20, 30, 40, 50, 60)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");
        var result = frame.Shift("v", "lag", 1, spec);

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
        var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "A" })),
            ("t", NivaraColumn<int>.CreateFromSpans(new[] { 3, 0, 1 }, new[] { false, true, false })),
            ("v", IntColumn(30, 20, 10)));

        var spec = frame.Over().PartitionBy("g").OrderBy("t");
        var result = frame.RollingSum("v", "rs", 2, spec);

        var rs = result.GetColumn<int>("rs");
        Assert.That(rs[0], Is.EqualTo(40));
        Assert.That(rs[1], Is.EqualTo(50));
        Assert.That(rs.IsNull(2), Is.True);
    }

    [Test]
    public void RollingSum_WithEmptySpec_MatchesMethodArgs()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3, 4)));

        var viaSpec = frame.RollingSum("v", "rs", 3, new WindowSpec());
        var viaArgs = frame.RollingSum("v", "rsArgs", 3);

        Assert.That(viaSpec.GetColumn<int>("rs").ToArray(), Is.EqualTo(viaArgs.GetColumn<int>("rsArgs").ToArray()));
    }

    [Test]
    public void Window_WithSpec_MissingPartitionColumn_ThrowsArgumentException()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var spec = frame.Over().PartitionBy("missing");
        Assert.Throws<ArgumentException>(() => frame.RollingSum("v", "x", 2, spec));
    }

    [Test]
    public void Window_WithSpec_MissingOrderColumn_ThrowsArgumentException()
    {
        var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        var spec = frame.Over().OrderBy("missing");
        Assert.Throws<ArgumentException>(() => frame.RollingSum("v", "x", 2, spec));
    }

    [Test]
    public void Window_WithSpec_NonComparableOrderColumn_ThrowsArgumentException()
    {
        var frame = FrameWith(
            ("v", IntColumn(1, 2, 3)),
            ("o", NivaraColumn<object>.Create(new object[] { new object(), new object(), new object() })));

        var spec = frame.Over().OrderBy("o");
        Assert.Throws<ArgumentException>(() => frame.RollingSum("v", "x", 2, spec));
    }
}
