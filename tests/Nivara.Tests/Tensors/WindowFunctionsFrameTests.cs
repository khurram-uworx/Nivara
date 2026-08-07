using Nivara.Exceptions;
using Nivara.Tensors;
using NUnit.Framework;

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
}
