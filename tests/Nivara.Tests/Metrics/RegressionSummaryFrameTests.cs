using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class RegressionSummaryFrameTests
{
    [Test]
    public void RegressionSummaryFrame_PerfectPrediction_ReturnsZeroErrorAndUnitR2()
    {
        var frame = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 }),
            NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 }));

        Assert.That(frame.ColumnNames, Is.EqualTo(new[] { "Count", "Mae", "Mse", "Rmse", "R2" }));
        Assert.That(frame.RowCount, Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Count")[0], Is.EqualTo(3));
        Assert.That(frame.GetColumn<double>("Mae")[0], Is.EqualTo(0.0));
        Assert.That(frame.GetColumn<double>("Mse")[0], Is.EqualTo(0.0));
        Assert.That(frame.GetColumn<double>("Rmse")[0], Is.EqualTo(0.0));
        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(1.0));
    }

    [Test]
    public void RegressionSummaryFrame_HandComputedCase_MatchesExpectedValues()
    {
        var frame = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 }),
            NivaraColumn<double>.Create(new[] { 2.0, 4.0, 6.0 }));

        Assert.That(frame.GetColumn<double>("Mae")[0], Is.EqualTo(2.0));
        Assert.That(frame.GetColumn<double>("Mse")[0], Is.EqualTo(14.0 / 3.0).Within(1e-15));
        Assert.That(frame.GetColumn<double>("Rmse")[0], Is.EqualTo(Math.Sqrt(14.0 / 3.0)).Within(1e-15));
        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(-0.75));
    }

    [Test]
    public void RegressionSummaryFrame_ConstantTargetWithPerfectPrediction_ReturnsUnitR2()
    {
        var frame = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 5.0, 5.0, 5.0 }),
            NivaraColumn<double>.Create(new[] { 5.0, 5.0, 5.0 }));

        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(1.0));
    }

    [Test]
    public void RegressionSummaryFrame_ConstantTargetWithError_ReturnsZeroR2()
    {
        var frame = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 5.0, 6.0, 5.0 }),
            NivaraColumn<double>.Create(new[] { 5.0, 5.0, 5.0 }));

        Assert.That(frame.GetColumn<double>("Mae")[0], Is.EqualTo(1.0 / 3.0).Within(1e-15));
        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(0.0));
    }

    [Test]
    public void RegressionSummaryFrame_SinglePair_UsesDegenerateTargetVariance()
    {
        var perfect = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 3.0 }),
            NivaraColumn<double>.Create(new[] { 3.0 }));

        var imperfect = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<double>.Create(new[] { 3.0 }),
            NivaraColumn<double>.Create(new[] { 4.0 }));

        Assert.That(perfect.GetColumn<long>("Count")[0], Is.EqualTo(1));
        Assert.That(perfect.GetColumn<double>("R2")[0], Is.EqualTo(1.0));
        Assert.That(imperfect.GetColumn<long>("Count")[0], Is.EqualTo(1));
        Assert.That(imperfect.GetColumn<double>("R2")[0], Is.EqualTo(0.0));
    }

    [Test]
    public void RegressionSummaryFrame_MixedNumericElementTypes_ConvertsToDouble()
    {
        var frame = NivaraMetrics.RegressionSummaryFrame(
            NivaraColumn<float>.Create(new[] { 2.0f, 4.0f, 6.0f }),
            NivaraColumn<int>.Create(new[] { 1, 2, 3 }));

        Assert.That(frame.GetColumn<double>("Mae")[0], Is.EqualTo(2.0));
        Assert.That(frame.GetColumn<double>("Mse")[0], Is.EqualTo(14.0 / 3.0).Within(1e-15));
        Assert.That(frame.GetColumn<double>("Rmse")[0], Is.EqualTo(Math.Sqrt(14.0 / 3.0)).Within(1e-15));
        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(-6.0));
    }

    [Test]
    public void RegressionSummaryFrame_NoValidPairs_Throws()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable<double>(new double?[] { null, 1.0 });
        var actual = NivaraColumnFactory.CreateFromNullable<double>(new double?[] { 1.0, null });

        var exception = Assert.Throws<ArgumentException>(
            () => NivaraMetrics.RegressionSummaryFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("at least one non-null pair"));
    }

    [Test]
    public void RegressionSummaryFrame_NonNumericElementType_Throws()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "b" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "b" });

        var exception = Assert.Throws<ArgumentException>(
            () => NivaraMetrics.RegressionSummaryFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("numeric"));
    }

    [Test]
    public void RegressionSummaryFrame_LengthMismatch_Throws()
    {
        var predicted = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
        var actual = NivaraColumn<double>.Create(new[] { 1.0, 2.0 });

        var exception = Assert.Throws<ArgumentException>(
            () => NivaraMetrics.RegressionSummaryFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("same length"));
    }
}
