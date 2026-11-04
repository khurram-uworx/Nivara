using NivaraClassifier;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// The statistics behind <c>NivaraClassifier --benchmark</c>. These assert the summarisation,
/// the round-length rule, and the noise band on plain numbers; nothing here measures wall clock,
/// so no timing guard and no <c>Performance</c> category is involved.
/// </summary>
[TestFixture]
public class BenchStatsTests
{
    [Test]
    public void Summarize_OddCount_MedianIsMiddleSample()
    {
        var summary = BenchStats.Summarize([3.0, 1.0, 2.0]);

        Assert.Multiple(() =>
        {
            Assert.That(summary.MedianMs, Is.EqualTo(2.0));
            Assert.That(summary.MinMs, Is.EqualTo(1.0));
            Assert.That(summary.MaxMs, Is.EqualTo(3.0));
        });
    }

    [Test]
    public void Summarize_EvenCount_MedianIsAverageOfTwoMiddleSamples()
    {
        var summary = BenchStats.Summarize([4.0, 1.0, 3.0, 2.0]);

        Assert.That(summary.MedianMs, Is.EqualTo(2.5));
    }

    [Test]
    public void Summarize_SingleSample_MedianMinMaxAgree()
    {
        var summary = BenchStats.Summarize([7.5]);

        Assert.Multiple(() =>
        {
            Assert.That(summary.MedianMs, Is.EqualTo(7.5));
            Assert.That(summary.MinMs, Is.EqualTo(7.5));
            Assert.That(summary.MaxMs, Is.EqualTo(7.5));
            Assert.That(summary.Measured, Is.True);
        });
    }

    [Test]
    public void Summarize_EmptySamples_Throws()
    {
        Assert.That(
            () => BenchStats.Summarize(ReadOnlySpan<double>.Empty),
            Throws.ArgumentException.With.Message.Contains("At least one sample"));
    }

    [Test]
    public void NotMeasured_Sentinel_ReportsUnmeasured()
    {
        Assert.That(TimingSummary.NotMeasured.Measured, Is.False);
    }

    [Test]
    public void Iterations_SubMillisecondCost_ScalesToBudget()
    {
        Assert.That(BenchStats.Iterations(perIterationMs: 0.05, budgetMs: 20), Is.EqualTo(400));
    }

    [Test]
    public void Iterations_CostBelowResolution_AsksForMaximum()
    {
        Assert.That(BenchStats.Iterations(perIterationMs: 0, budgetMs: 20, max: 2000), Is.EqualTo(2000));
    }

    [Test]
    public void Iterations_CostExceedingBudget_ClampsToMinimum()
    {
        Assert.That(BenchStats.Iterations(perIterationMs: 25.0, budgetMs: 20), Is.EqualTo(1));
    }

    [Test]
    public void Iterations_NonPositiveBudget_Throws()
    {
        Assert.That(
            () => BenchStats.Iterations(perIterationMs: 1.0, budgetMs: 0),
            Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void Classify_RatioOutsideBand_ReportsDirection()
    {
        Assert.Multiple(() =>
        {
            Assert.That(BenchStats.Classify(0.80), Is.EqualTo(BenchStats.Verdict.Faster));
            Assert.That(BenchStats.Classify(1.30), Is.EqualTo(BenchStats.Verdict.Slower));
        });
    }

    [Test]
    public void Classify_RatioInsideBand_ReportsNoise()
    {
        Assert.Multiple(() =>
        {
            Assert.That(BenchStats.Classify(0.96), Is.EqualTo(BenchStats.Verdict.WithinNoise));
            Assert.That(BenchStats.Classify(1.04), Is.EqualTo(BenchStats.Verdict.WithinNoise));
            Assert.That(BenchStats.Classify(1.0), Is.EqualTo(BenchStats.Verdict.WithinNoise));
        });
    }

    [Test]
    public void Classify_WiderBand_AbsorbsDifferenceThatDefaultBandWouldFlag()
    {
        Assert.Multiple(() =>
        {
            Assert.That(BenchStats.Classify(0.90), Is.EqualTo(BenchStats.Verdict.Faster));
            Assert.That(BenchStats.Classify(0.90, noiseBand: 0.15), Is.EqualTo(BenchStats.Verdict.WithinNoise));
        });
    }

    [Test]
    public void Ratio_CandidateOverReference_IsOneAtParity()
    {
        var reference = new TimingSummary(10.0, 9.0, 11.0);
        var candidate = new TimingSummary(20.0, 18.0, 22.0);

        Assert.That(BenchStats.Ratio(candidate, reference), Is.EqualTo(2.0));
    }

    [Test]
    public void Ratio_UnmeasuredSide_IsNaN()
    {
        Assert.That(BenchStats.Ratio(TimingSummary.NotMeasured, new TimingSummary(10.0, 9.0, 11.0)), Is.NaN);
    }
}
