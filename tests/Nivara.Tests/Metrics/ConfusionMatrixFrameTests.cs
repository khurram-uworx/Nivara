using Nivara.Metrics;
using NUnit.Framework;

namespace Nivara.Tests.Metrics;

[TestFixture]
public class ConfusionMatrixFrameTests
{
    [Test]
    public void ConfusionMatrixFrame_TwoClasses_BuildsPredictedRowsAndActualColumns()
    {
        var predicted = NivaraColumn<int>.Create(new[] { 0, 0, 1, 1, 1 });
        var actual = NivaraColumn<int>.Create(new[] { 0, 1, 0, 1, 1 });

        var frame = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        Assert.That(frame.ColumnNames, Is.EqualTo(new[] { "Predicted", "Actual:0", "Actual:1", "Correct", "Support" }));
        Assert.That(frame.GetColumn<int>("Predicted")[0], Is.EqualTo(0));
        Assert.That(frame.GetColumn<int>("Predicted")[1], Is.EqualTo(1));

        var actualZero = frame.GetColumn<long>("Actual:0");
        var actualOne = frame.GetColumn<long>("Actual:1");
        var correct = frame.GetColumn<long>("Correct");
        var support = frame.GetColumn<long>("Support");

        Assert.That(actualZero[0], Is.EqualTo(1));
        Assert.That(actualZero[1], Is.EqualTo(1));
        Assert.That(actualOne[0], Is.EqualTo(1));
        Assert.That(actualOne[1], Is.EqualTo(2));
        Assert.That(correct[0], Is.EqualTo(1));
        Assert.That(correct[1], Is.EqualTo(2));
        Assert.That(support[0], Is.EqualTo(2));
        Assert.That(support[1], Is.EqualTo(3));
    }

    [Test]
    public void ConfusionMatrixFrame_ImbalancedStrings_OrdersClassesOrdinally()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "b", "b", "b", "b", "b", "b", "b", "b", "b", "b" });
        var actual = NivaraColumn<string>.Create(new[] { "b", "b", "b", "b", "b", "b", "b", "b", "b", "a" });

        var frame = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);

        Assert.That(frame.ColumnNames, Is.EqualTo(new[] { "Predicted", "Actual:a", "Actual:b", "Correct", "Support" }));
        Assert.That(frame.GetColumn<string>("Predicted")[0], Is.EqualTo("a"));
        Assert.That(frame.GetColumn<string>("Predicted")[1], Is.EqualTo("b"));
        Assert.That(frame.GetColumn<long>("Support")[0], Is.EqualTo(0));
        Assert.That(frame.GetColumn<long>("Support")[1], Is.EqualTo(10));
        Assert.That(frame.GetColumn<long>("Correct")[0], Is.EqualTo(0));
        Assert.That(frame.GetColumn<long>("Correct")[1], Is.EqualTo(9));
        Assert.That(frame.GetColumn<long>("Actual:a")[1], Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Actual:b")[1], Is.EqualTo(9));
    }

    [Test]
    public void ConfusionMatrixFrame_LabelsNamedLikeReservedColumns_DoNotCollide()
    {
        var values = new[] { "Correct", "Predicted", "Support", "Correct" };

        var frame = NivaraMetrics.ConfusionMatrixFrame(
            NivaraColumn<string>.Create(values),
            NivaraColumn<string>.Create(values));

        Assert.That(frame.ColumnNames, Is.EqualTo(new[]
        {
            "Predicted",
            "Actual:Correct",
            "Actual:Predicted",
            "Actual:Support",
            "Correct",
            "Support",
        }));
        Assert.That(frame.GetColumn<long>("Correct")[0], Is.EqualTo(2));
        Assert.That(frame.GetColumn<long>("Correct")[1], Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Correct")[2], Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Support")[0], Is.EqualTo(2));
        Assert.That(frame.GetColumn<long>("Support")[1], Is.EqualTo(1));
        Assert.That(frame.GetColumn<long>("Support")[2], Is.EqualTo(1));
    }

    [Test]
    public void ConfusionMatrixFrame_LabelsDifferingOnlyByCase_Throws()
    {
        var predicted = NivaraColumn<string>.Create(new[] { "a", "A" });
        var actual = NivaraColumn<string>.Create(new[] { "a", "A" });

        var exception = Assert.Throws<ArgumentException>(() => NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("ignoring case"));
    }

    [Test]
    public void ConfusionMatrixFrame_DifferentElementTypes_Throws()
    {
        var predicted = NivaraColumn<int>.Create(new[] { 1, 2 });
        var actual = NivaraColumn<long>.Create(new[] { 1L, 2L });

        var exception = Assert.Throws<ArgumentException>(() => NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("same element type"));
    }

    [Test]
    public void ConfusionMatrixFrame_NonComparableLabelType_Throws()
    {
        var predicted = NivaraColumn<object>.Create(new[] { new object(), new object() });
        var actual = NivaraColumn<object>.Create(new[] { new object(), new object() });

        var exception = Assert.Throws<ArgumentException>(() => NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("comparable"));
    }

    [Test]
    public void ConfusionMatrixFrame_LengthMismatch_Throws()
    {
        var predicted = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
        var actual = NivaraColumn<int>.Create(new[] { 1, 2 });

        var exception = Assert.Throws<ArgumentException>(() => NivaraMetrics.ConfusionMatrixFrame(predicted, actual));

        Assert.That(exception!.Message, Does.Contain("same length"));
    }

    [Test]
    public void ConfusionMatrixFrame_NullColumn_Throws()
    {
        var actual = NivaraColumn<int>.Create(new[] { 1, 2 });

        Assert.Throws<ArgumentNullException>(() => NivaraMetrics.ConfusionMatrixFrame(null!, actual));
        Assert.Throws<ArgumentNullException>(() => NivaraMetrics.ConfusionMatrixFrame(actual, null!));
    }
}
