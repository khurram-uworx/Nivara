using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

/// <summary>
/// Allocation regression guard for the #966d3f9 read-in-place path in
/// <c>RegressionMetrics.buildSummary</c>: a null-free <c>double</c> pair must go through
/// <c>TryGetSpan</c> and rent only the residual buffer, while any other pairing converts
/// once into pooled/managed buffers.
/// </summary>
/// <remarks>
/// Length 512 is deliberate and informative: it sits below
/// <c>RegressionMetrics.PoolThreshold</c> (1024), so every rent becomes a plain
/// <c>new double[length]</c> allocation that <c>GC.GetAllocatedBytesForCurrentThread</c> can
/// see. At or above the threshold the same rents come from <c>ArrayPool&lt;double&gt;.Shared</c>
/// and are invisible once the pool is warm, so no allocation assertion could distinguish
/// the two paths there. The converted path therefore pays for two extra 512-element
/// buffers on top of the residual buffer that both paths rent.
/// </remarks>
[TestFixture]
[Category("Performance")]
public class MetricsAllocationTests
{
    const int Length = 512;

    static long MeasureOnce(Action operation)
    {
        for (int i = 0; i < 3; i++)
            operation();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long pre = GC.GetAllocatedBytesForCurrentThread();
        operation();
        return GC.GetAllocatedBytesForCurrentThread() - pre;
    }

    static long MeasureBestOf(Action operation, int samples = 5)
    {
        long best = long.MaxValue;
        for (int i = 0; i < samples; i++)
            best = Math.Min(best, MeasureOnce(operation));
        return best;
    }

    [Test]
    public void RegressionSummaryFrame_NullFreeDoublePath_AllocatesAtMostOneBufferLessThanConvertedFloatPath()
    {
        var doublePredicted = NivaraColumn<double>.Create(BuildValues(Length, i => (double)i));
        var doubleActual = NivaraColumn<double>.Create(BuildValues(Length, i => (double)(i + 1)));
        var floatPredicted = NivaraColumn<float>.Create(BuildValues(Length, i => (float)i));
        var floatActual = NivaraColumn<float>.Create(BuildValues(Length, i => (float)(i + 1)));

        long doubleAlloc = MeasureBestOf(() => _ = NivaraMetrics.RegressionSummaryFrame(doublePredicted, doubleActual));
        long floatAlloc = MeasureBestOf(() => _ = NivaraMetrics.RegressionSummaryFrame(floatPredicted, floatActual));

        TestContext.Out.WriteLine(
            $"double in-place: {doubleAlloc} bytes; float converted: {floatAlloc} bytes");

        // The converted path rents two 512-element buffers the in-place path avoids.
        Assert.That(
            doubleAlloc + (Length * 8),
            Is.LessThanOrEqualTo(floatAlloc),
            "a null-free double pair must read through TryGetSpan and avoid at least one converted buffer");
    }

    static T[] BuildValues<T>(int length, Func<int, T> factory)
    {
        var values = new T[length];
        for (int i = 0; i < length; i++)
            values[i] = factory(i);
        return values;
    }
}
