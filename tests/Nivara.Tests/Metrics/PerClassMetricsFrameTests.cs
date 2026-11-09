using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class PerClassMetricsFrameTests
{
    [Test]
    public void PerClassMetricsFrame_ThreeClasses_ComputesCountsAndDerivedMetrics()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "a", "b", "b", "c" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "b", "a", "c", "c" });

        var frame = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        Assert.That(frame.ColumnNames, Is.EqualTo(new[]
        {
            "Class", "Tp", "Fp", "Fn", "Precision", "Recall", "F1",
        }));

        var classes = frame.GetColumn<string>("Class");
        var tp = frame.GetColumn<long>("Tp");
        var fp = frame.GetColumn<long>("Fp");
        var fn = frame.GetColumn<long>("Fn");
        var precision = frame.GetColumn<double>("Precision");
        var recall = frame.GetColumn<double>("Recall");
        var f1 = frame.GetColumn<double>("F1");

        Assert.That(classes[0], Is.EqualTo("a"));
        Assert.That(classes[1], Is.EqualTo("b"));
        Assert.That(classes[2], Is.EqualTo("c"));

        Assert.That(tp[0], Is.EqualTo(1));
        Assert.That(tp[1], Is.EqualTo(0));
        Assert.That(tp[2], Is.EqualTo(1));

        Assert.That(fp[0], Is.EqualTo(1));
        Assert.That(fp[1], Is.EqualTo(2));
        Assert.That(fp[2], Is.EqualTo(0));

        Assert.That(fn[0], Is.EqualTo(1));
        Assert.That(fn[1], Is.EqualTo(1));
        Assert.That(fn[2], Is.EqualTo(1));

        Assert.That(precision[0], Is.EqualTo(0.5));
        Assert.That(precision[1], Is.EqualTo(0.0));
        Assert.That(precision[2], Is.EqualTo(1.0));

        Assert.That(recall[0], Is.EqualTo(0.5));
        Assert.That(recall[1], Is.EqualTo(0.0));
        Assert.That(recall[2], Is.EqualTo(0.5));

        Assert.That(f1[0], Is.EqualTo(0.5));
        Assert.That(f1.IsNull(1), Is.True);
        Assert.That(f1[2], Is.EqualTo(2.0 / 3.0).Within(1e-15));
    }

    [Test]
    public void PerClassMetricsFrame_ClassNeverPredicted_HasNullPrecisionAndF1ButZeroRecall()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "a", "a" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "a", "b" });

        var frame = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        var classes = frame.GetColumn<string>("Class");
        var precision = frame.GetColumn<double>("Precision");
        var recall = frame.GetColumn<double>("Recall");
        var f1 = frame.GetColumn<double>("F1");

        Assert.That(classes[1], Is.EqualTo("b"));
        Assert.That(frame.GetColumn<long>("Tp")[1], Is.EqualTo(0));
        Assert.That(frame.GetColumn<long>("Fp")[1], Is.EqualTo(0));
        Assert.That(frame.GetColumn<long>("Fn")[1], Is.EqualTo(1));
        Assert.That(precision.IsNull(1), Is.True);
        Assert.That(recall[1], Is.EqualTo(0.0));
        Assert.That(f1.IsNull(1), Is.True);
    }

    [Test]
    public void PerClassMetricsFrame_ClassNeverActual_HasNullRecallAndF1ButZeroPrecision()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "b", "a" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "a", "a" });

        var frame = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        var classes = frame.GetColumn<string>("Class");
        var precision = frame.GetColumn<double>("Precision");
        var recall = frame.GetColumn<double>("Recall");
        var f1 = frame.GetColumn<double>("F1");

        Assert.That(classes[1], Is.EqualTo("b"));
        Assert.That(frame.GetColumn<long>("Tp")[1], Is.EqualTo(0));
        Assert.That(frame.GetColumn<long>("Fp")[1], Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Fn")[1], Is.EqualTo(0));
        Assert.That(precision[1], Is.EqualTo(0.0));
        Assert.That(recall.IsNull(1), Is.True);
        Assert.That(f1.IsNull(1), Is.True);
    }

    [Test]
    public void PerClassMetricsFrame_CountsAgreeWithConfusionMatrix()
    {
        var predicted = NivaraColumn<int>.Create(new[] { 0, 1, 2, 2, 1, 0, 1 });
        var actual = NivaraColumn<int>.Create(new[] { 0, 1, 2, 1, 1, 2, 0 });

        var matrix = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);
        var perClass = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

        var support = matrix.GetColumn<long>("Support");
        var correct = matrix.GetColumn<long>("Correct");
        var tp = perClass.GetColumn<long>("Tp");
        var fp = perClass.GetColumn<long>("Fp");
        var fn = perClass.GetColumn<long>("Fn");

        long expectedPredictedTotal = 0;
        long expectedActualTotal = 0;
        for (int row = 0; row < matrix.RowCount; row++)
        {
            expectedPredictedTotal += support[row];
            expectedActualTotal += tp[row] + fn[row];
            Assert.That(tp[row], Is.EqualTo(correct[row]));
            Assert.That(tp[row] + fp[row], Is.EqualTo(support[row]));
        }

        Assert.That(expectedPredictedTotal, Is.EqualTo(predicted.Length));
        Assert.That(expectedActualTotal, Is.EqualTo(predicted.Length));
    }
}
