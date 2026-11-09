using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class MetricsNullHandlingTests
{
    static readonly double?[] predictedValues = [1.0, 2.0, null, 3.0, 1.0, 4.0];
    static readonly double?[] actualValues = [1.0, null, 3.0, 3.0, 2.0, 4.0];

    [Test]
    public void ConfusionMatrixFrame_NullCells_AreExcludedFromEveryTotal()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(predictedValues);
        var actual = NivaraColumnFactory.CreateFromNullable(actualValues);

        var frame = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        long cellTotal = 0;
        long supportTotal = 0;
        for (int row = 0; row < frame.RowCount; row++)
        {
            supportTotal += frame.GetColumn<long>("Support")[row];
            foreach (var name in frame.ColumnNames)
            {
                if (name.StartsWith("Actual:", StringComparison.Ordinal))
                    cellTotal += frame.GetColumn<long>(name)[row];
            }
        }

        Assert.That(frame.RowCount, Is.EqualTo(4));
        Assert.That(cellTotal, Is.EqualTo(4));
        Assert.That(supportTotal, Is.EqualTo(4));
        Assert.That(cellTotal + 2, Is.EqualTo(predicted.Length));
    }

    [Test]
    public void ConfusionMatrixFrame_LabelOnlyInExcludedRows_IsNotAClass()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, null, 2.0 });
        var actual = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, 3.0, 2.0 });

        var frame = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        Assert.That(frame.RowCount, Is.EqualTo(2));
        Assert.That(frame.ColumnNames, Does.Not.Contain("Actual:3"));
    }

    [Test]
    public void PerClassMetricsFrame_NullCells_KeepCountsConsistentWithValidPairs()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(predictedValues);
        var actual = NivaraColumnFactory.CreateFromNullable(actualValues);

        var frame = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        long truePositives = 0;
        long falsePositives = 0;
        long falseNegatives = 0;
        for (int row = 0; row < frame.RowCount; row++)
        {
            truePositives += frame.GetColumn<long>("Tp")[row];
            falsePositives += frame.GetColumn<long>("Fp")[row];
            falseNegatives += frame.GetColumn<long>("Fn")[row];
        }

        Assert.That(truePositives, Is.EqualTo(3));
        Assert.That(truePositives + falsePositives, Is.EqualTo(4));
        Assert.That(truePositives + falseNegatives, Is.EqualTo(4));
    }

    [Test]
    public void PerClassMetricsFrame_AllPairsNull_Throws()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(new double?[] { null, null });
        var actual = NivaraColumnFactory.CreateFromNullable(new double?[] { null, null });

        var exception = Assert.Throws<ArgumentException>(
            () => NivaraMetrics.PerClassMetricsFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("at least one non-null pair"));
    }

    [Test]
    public void RegressionSummaryFrame_NullCells_ReduceCountToValidPairs()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, null, 3.0, 4.0 });
        var actual = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, 2.0, null, 4.0 });

        var frame = NivaraMetrics.RegressionSummaryFrame(predicted, actual);

        Assert.That(frame.GetColumn<long>("Count")[0], Is.EqualTo(2));
        Assert.That(frame.GetColumn<double>("Mae")[0], Is.EqualTo(0.0));
        Assert.That(frame.GetColumn<double>("R2")[0], Is.EqualTo(1.0));
    }

    [Test]
    public void ConfusionMatrixFrame_NullOnlyPredicted_Throws()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(new double?[] { null, null });
        var actual = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, 2.0 });

        var exception = Assert.Throws<ArgumentException>(
            () => NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("at least one non-null pair"));
    }
}
