using Nivara.AI;
using Nivara.AutoDiff.Serialization;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveFeedbackCollectorTests
{
    const int HiddenSize = 64;

    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "nivara-adaptive-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, recursive: true);
    }

    static AdaptiveTrainingExample Example(int index)
    {
        var features = new float[4 * HiddenSize];
        for (var i = 0; i < features.Length; i++)
            features[i] = MathF.Cos((i + index + 1) * 0.07f);

        return new AdaptiveTrainingExample(
            $"query-{index}", $"chunk-{index}", features, 4, HiddenSize, [1, 2], index % 2);
    }

    static Dictionary<string, int[]> ParameterBits(LayaDecisionHead<float, float> head)
        => head.StateDict().ToDictionary(
            entry => entry.Key,
            entry =>
            {
                entry.Value.Data.TryGetSpan(out var span);
                return span.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
            },
            StringComparer.Ordinal);

    static float[] ChoiceLogits(LayaDecisionHead<float, float> head)
    {
        var values = Enumerable.Range(0, 4 * HiddenSize)
            .Select(index => MathF.Sin((index + 1) * 0.11f))
            .ToArray();
        var hidden = Nivara.AutoDiff.ReverseGradTensor<float>.FromMatrix(values, 4, HiddenSize);
        var output = head.Forward(hidden, LayaQuestionType.Choice, [1, 2], 4);
        return output.Logits;
    }

    static LayaDecision DecodeChoice(LayaDecisionHead<float, float> head)
    {
        var question = new LayaQuestion
        {
            Id = "adaptive-roundtrip",
            Type = LayaQuestionType.Choice,
            Instructions = "Select the relevant option.",
            Options = [new("relevant", null), new("not_relevant", null)]
        };
        var calibration = new LayaCalibration([1.0, 1.0, 1.0]);
        return calibration.Decode(question, ChoiceLogits(head), actProbability: 0.5);
    }

    [Test]
    public void Add_UpdatesOnlyAtThresholdAndReloadsBitExactly()
    {
        var head = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 0);
        var path = Path.Combine(tempDir, AdaptiveHeadPersistence.FileName);
        var collector = new AdaptiveFeedbackCollector(head, path, batchThreshold: 10, epochs: 1);
        var baseline = ParameterBits(head);
        var logitsBefore = ChoiceLogits(head);

        for (var i = 0; i < 9; i++)
            Assert.That(collector.Add(Example(i)), Is.False);

        Assert.Multiple(() =>
        {
            Assert.That(collector.BufferCount, Is.EqualTo(9));
            Assert.That(File.Exists(path), Is.False);
            Assert.That(collector.SuccessfulUpdates, Is.Zero);
        });

        Assert.That(collector.Add(Example(9)), Is.True);
        var decision = DecodeChoice(head);

        var restored = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 0);
        Assert.That(AdaptiveHeadPersistence.LoadIfPresent(restored, path), Is.True);
        var restoredDecision = DecodeChoice(restored);
        Assert.Multiple(() =>
        {
            Assert.That(collector.BufferCount, Is.Zero);
            Assert.That(collector.SuccessfulUpdates, Is.EqualTo(1));
            Assert.That(ParameterBits(head), Is.EqualTo(ParameterBits(restored)));
            Assert.That(ChoiceLogits(head), Is.EqualTo(ChoiceLogits(restored)));
            Assert.That(ParameterBits(head), Is.Not.EqualTo(baseline));
            Assert.That(restoredDecision.QuestionId, Is.EqualTo(decision.QuestionId));
            Assert.That(restoredDecision.Type, Is.EqualTo(decision.Type));
            Assert.That(restoredDecision.Choice, Is.EqualTo(decision.Choice));
            Assert.That(restoredDecision.Probabilities, Is.EqualTo(decision.Probabilities));
            Assert.That(restoredDecision.Confidence, Is.EqualTo(decision.Confidence));
            Assert.That(restoredDecision.AnswerConfidence, Is.EqualTo(decision.AnswerConfidence));
            Assert.That(restoredDecision.ActionProbability, Is.EqualTo(decision.ActionProbability));
            Assert.That(restoredDecision.Temperature, Is.EqualTo(decision.Temperature));
        });
    }

    [Test]
    public void Add_PersistenceFailureRestoresWeightsAndRetainsBatch()
    {
        var head = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 0);
        var baseline = ParameterBits(head);
        var collector = new AdaptiveFeedbackCollector(head, tempDir, batchThreshold: 2, epochs: 1);

        Assert.That(collector.Add(Example(0)), Is.False);
        var exception = Assert.Catch(() => collector.Add(Example(1)));

        Assert.Multiple(() =>
        {
            Assert.That(exception, Is.InstanceOf<IOException>().Or.InstanceOf<UnauthorizedAccessException>());
            Assert.That(collector.BufferCount, Is.EqualTo(2));
            Assert.That(collector.SuccessfulUpdates, Is.Zero);
            Assert.That(ParameterBits(head), Is.EqualTo(baseline));
        });
    }

    [Test]
    public void LoadIfPresent_RejectsIncompleteHeadState()
    {
        var head = new LayaDecisionHead<float>(HiddenSize, headLayerCount: 0);
        var state = head.StateDict();
        state.Remove(state.Keys.First());
        var path = Path.Combine(tempDir, AdaptiveHeadPersistence.FileName);
        File.WriteAllText(path, ModelSerializer.StateDictToJson(state));

        Assert.Throws<InvalidOperationException>(() => AdaptiveHeadPersistence.LoadIfPresent(head, path));
    }
}
