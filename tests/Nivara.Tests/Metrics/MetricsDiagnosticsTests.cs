using Nivara.Diagnostics;
using Nivara.Metrics;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class MetricsDiagnosticsTests
{
    [SetUp]
    public void EnableDiagnostics()
    {
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();
    }

    [TearDown]
    public void DisableDiagnostics()
    {
        DiagnosticsTracker.ClearRecordedOperations();
        DiagnosticsTracker.IsEnabled = false;
    }

    [Test]
    public void ConfusionMatrixFrame_RecordsScalarKernelAndUsedPairs()
    {
        var predicted = NivaraColumn<int>.Create(Enumerable.Range(0, 200).Select(i => i % 3).ToArray());
        var actual = NivaraColumn<int>.Create(Enumerable.Range(0, 200).Select(i => (i + 1) % 3).ToArray());

        NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        var operation = single();
        Assert.That(operation.OperationType, Is.EqualTo("Metrics.ConfusionMatrix"));
        Assert.That(operation.KernelUsed, Is.EqualTo(KernelType.Scalar));
        Assert.That(operation.InputLength, Is.EqualTo(200));
        Assert.That(operation.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(operation.HadNulls, Is.False);
        Assert.That(operation.Notes, Is.EqualTo("200 of 200 pairs used"));
    }

    [Test]
    public void PerClassMetricsFrame_RecordsScalarKernelAndUsedPairs()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "b", "a" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "a", "b" });

        NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        var operation = single();
        Assert.That(operation.OperationType, Is.EqualTo("Metrics.PerClassMetrics"));
        Assert.That(operation.KernelUsed, Is.EqualTo(KernelType.Scalar));
        Assert.That(operation.Notes, Is.EqualTo("3 of 3 pairs used"));
    }

    [Test]
    public void RegressionSummaryFrame_RecordsKernelSelectorVerdictForConvertedLength()
    {
        int length = 2048;
        var predicted = NivaraColumn<double>.Create(Enumerable.Repeat(1.0, length).ToArray());
        var actual = NivaraColumn<double>.Create(Enumerable.Repeat(2.0, length).ToArray());

        NivaraMetrics.RegressionSummaryFrame(predicted, actual);

        var expected = Vector.IsHardwareAccelerated && length >= Vector<byte>.Count * 4
            ? KernelType.Vectorized
            : KernelType.Scalar;

        var operation = single();
        Assert.That(operation.OperationType, Is.EqualTo("Metrics.RegressionSummary"));
        Assert.That(operation.KernelUsed, Is.EqualTo(expected));
        Assert.That(operation.InputLength, Is.EqualTo(length));
        Assert.That(operation.Notes, Is.EqualTo("2048 of 2048 pairs used"));
    }

    [Test]
    public void RegressionSummaryFrame_NullExclusion_IsReportedInNotesAndHadNulls()
    {
        var predicted = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, 2.0, null, 4.0, 5.0 });
        var actual = NivaraColumnFactory.CreateFromNullable(new double?[] { 1.0, null, 3.0, 4.0, 5.0 });

        NivaraMetrics.RegressionSummaryFrame(predicted, actual);

        var operation = single();
        Assert.That(operation.HadNulls, Is.True);
        Assert.That(operation.Notes, Is.EqualTo("3 of 5 pairs used"));
    }

    static OperationDiagnostics single()
    {
        var operations = DiagnosticsTracker.GetRecordedOperations();
        Assert.That(operations, Has.Length.EqualTo(1));
        return operations[0];
    }
}
