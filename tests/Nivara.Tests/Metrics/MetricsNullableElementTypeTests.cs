using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class MetricsNullableElementTypeTests
{
    [Test]
    public void ConfusionMatrixFrame_NullableElementColumns_RejectedWithCreateFromNullableGuidance()
    {
        var predicted = NivaraColumn<int?>.Create(new int?[] { 1, 2, 3 });
        var actual = NivaraColumn<int?>.Create(new int?[] { 1, 2, 3 });

        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(ex!.Message, Does.Contain("CreateFromNullable"));
    }

    [Test]
    public void RegressionSummaryFrame_NullableElementColumns_RejectedWithCreateFromNullableGuidance()
    {
        var predicted = NivaraColumn<double?>.Create(new double?[] { 1.0, 2.0, 3.0 });
        var actual = NivaraColumn<double?>.Create(new double?[] { 1.0, 2.0, 3.0 });

        var ex = Assert.Throws<ArgumentException>(() =>
            NivaraMetrics.RegressionSummaryFrame(predicted, actual));

        Assert.That(ex!.Message, Does.Contain("CreateFromNullable"));
    }
}
