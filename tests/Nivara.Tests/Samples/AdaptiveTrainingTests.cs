using Nivara.AI;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveTrainingTests
{
    const int HiddenSize = 64;

    static AdaptiveTrainingExample Example(string query, int target, int seed)
    {
        var features = new float[4 * HiddenSize];
        for (var row = 0; row < 4; row++)
            for (var column = 0; column < HiddenSize; column++)
                features[row * HiddenSize + column] = MathF.Sin((row + 1) * (column + seed + 1) * 0.13f);

        return new AdaptiveTrainingExample(query, $"chunk-{query}", features, 4, HiddenSize, [1, 2], target);
    }

    static float[] Values(Nivara.AutoDiff.Nn.Parameter<float> parameter)
    {
        parameter.Tensor.Data.TryGetSpan(out var span);
        return span.ToArray();
    }

    [Test]
    public void TrainAndCompare_UpdatesChoicePathAndLeavesActionPathUnchanged()
    {
        var head = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 1);
        var training = new[]
        {
            Example("train-a", 0, 1),
            Example("train-b", 1, 2),
            Example("train-c", 0, 3),
            Example("train-d", 1, 4)
        };
        var holdout = new[] { Example("test-a", 0, 5), Example("test-b", 1, 6) };
        var scorerBefore = Values(head.scorerOut.Weight!);
        var actionBefore = Values(head.actOut.Weight!);

        var report = AdaptiveTraining.TrainAndCompare(head, training, holdout, epochs: 2);

        Assert.Multiple(() =>
        {
            Assert.That(report.Baseline.Count, Is.EqualTo(holdout.Length));
            Assert.That(report.Trained.Count, Is.EqualTo(holdout.Length));
            Assert.That(report.TrainingExamples, Is.EqualTo(training.Length));
            Assert.That(Values(head.scorerOut.Weight!), Is.Not.EqualTo(scorerBefore));
            Assert.That(Values(head.actOut.Weight!), Is.EqualTo(actionBefore));
        });
    }

    [Test]
    public void TrainAndCompare_RejectsQueryOverlapBetweenTrainAndHoldout()
    {
        var head = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 0);
        Assert.Throws<ArgumentException>(() => AdaptiveTraining.TrainAndCompare(
            head,
            [Example("same-query", 0, 1)],
            [Example("same-query", 1, 2)],
            epochs: 1));
    }
}
