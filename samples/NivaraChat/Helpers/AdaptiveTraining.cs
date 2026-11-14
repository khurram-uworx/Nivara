using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;

namespace NivaraChat.Helpers;

public sealed record AdaptiveChoiceMetrics(
    int Count,
    int Correct,
    int PredictedRelevant,
    int TrueRelevant,
    double Accuracy,
    double RelevantPrecision,
    double RelevantRecall,
    double RelevantF1);

public sealed record AdaptiveTrainingReport(
    AdaptiveChoiceMetrics Baseline,
    AdaptiveChoiceMetrics Trained,
    int TrainingExamples,
    int HoldoutExamples);

public static class AdaptiveTraining
{
    public const int DefaultEpochs = 5;
    public const float DefaultLearningRate = 0.0005f;

    public static void Update(
        LayaDecisionHead<float, float> head,
        IReadOnlyList<AdaptiveTrainingExample> examples,
        int epochs = DefaultEpochs,
        float learningRate = DefaultLearningRate)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate));
        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("AdaptiveTraining.Update owns its GradientUtils.Grad() scope.");
        if (examples.Count == 0)
            throw new ArgumentException("At least one example is required for an adaptive update.", nameof(examples));

        foreach (var example in examples)
            ValidateExample(head, example);

        using var optimizer = new Adam<float>(learningRate);
        optimizer.AddParameterGroup(ChoiceParameters(head));
        var loss = new CrossEntropyLoss<float>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            optimizer.ZeroGrad();
            using (GradientUtils.Grad())
            {
                ReverseGradTensor<float>? batchLoss = null;
                foreach (var example in examples)
                {
                    var hidden = ReverseGradTensor<float>.FromMatrix(
                        (float[])example.Features.Clone(), example.SequenceLength, example.HiddenSize);
                    var logits = head.ForwardChoiceLogits(
                        hidden, LayaQuestionType.Choice, example.MarkerPositions, example.SequenceLength);
                    var exampleLoss = loss.Forward(logits, [example.Target]);
                    batchLoss = batchLoss is null
                        ? exampleLoss
                        : ReverseGradOperations.Add(batchLoss, exampleLoss);
                }

                ReverseGradOperations.DivideScalar(batchLoss!, examples.Count).Backward();
            }

            optimizer.Step();
        }
    }

    public static AdaptiveTrainingReport TrainAndCompare(
        LayaDecisionHead<float, float> head,
        IReadOnlyList<AdaptiveTrainingExample> trainingExamples,
        IReadOnlyList<AdaptiveTrainingExample> holdoutExamples,
        int epochs = DefaultEpochs,
        float learningRate = DefaultLearningRate)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(trainingExamples);
        ArgumentNullException.ThrowIfNull(holdoutExamples);
        EnsureQueryDisjoint(trainingExamples, holdoutExamples);

        var baseline = Evaluate(head, holdoutExamples);
        Update(head, trainingExamples, epochs, learningRate);
        var trained = Evaluate(head, holdoutExamples);
        return new AdaptiveTrainingReport(baseline, trained, trainingExamples.Count, holdoutExamples.Count);
    }

    public static AdaptiveChoiceMetrics Evaluate(
        LayaDecisionHead<float, float> head,
        IReadOnlyList<AdaptiveTrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(examples);
        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("Adaptive choice evaluation must run outside GradientUtils.Grad().");
        if (examples.Count == 0)
            throw new ArgumentException("At least one example is required for adaptive evaluation.", nameof(examples));

        var correct = 0;
        var predictedRelevant = 0;
        var trueRelevant = 0;
        var truePositive = 0;
        foreach (var example in examples)
        {
            ValidateExample(head, example);
            var hidden = ReverseGradTensor<float>.FromMatrix(
                (float[])example.Features.Clone(), example.SequenceLength, example.HiddenSize);
            var logits = head.ForwardChoiceLogits(
                hidden, LayaQuestionType.Choice, example.MarkerPositions, example.SequenceLength);
            logits.Data.TryGetSpan(out var values);
            var predicted = values[0] >= values[1] ? 0 : 1;
            if (predicted == example.Target)
                correct++;
            if (predicted == 0)
                predictedRelevant++;
            if (example.Target == 0)
            {
                trueRelevant++;
                if (predicted == 0)
                    truePositive++;
            }
        }

        var precision = predictedRelevant == 0 ? 0d : (double)truePositive / predictedRelevant;
        var recall = trueRelevant == 0 ? 0d : (double)truePositive / trueRelevant;
        var f1 = precision + recall == 0d ? 0d : 2d * precision * recall / (precision + recall);
        return new AdaptiveChoiceMetrics(
            examples.Count,
            correct,
            predictedRelevant,
            trueRelevant,
            (double)correct / examples.Count,
            precision,
            recall,
            f1);
    }

    static IReadOnlyList<Parameter<float>> ChoiceParameters(LayaDecisionHead<float, float> head)
    {
        var parameters = new List<Parameter<float>>();
        parameters.AddRange(head.typeEmb.GetParameters().Values);
        foreach (var layer in head.HeadLayers)
            parameters.AddRange(layer.GetParameters().Values);
        parameters.AddRange(head.scorerNorm.GetParameters().Values);
        parameters.AddRange(head.scorerHidden.GetParameters().Values);
        parameters.AddRange(head.scorerOut.GetParameters().Values);
        return parameters;
    }

    internal static void ValidateExample(LayaDecisionHead<float, float> head, AdaptiveTrainingExample example)
    {
        ArgumentNullException.ThrowIfNull(example);
        ArgumentNullException.ThrowIfNull(example.Features);
        ArgumentNullException.ThrowIfNull(example.MarkerPositions);
        ArgumentException.ThrowIfNullOrWhiteSpace(example.Query);
        ArgumentException.ThrowIfNullOrWhiteSpace(example.ChunkId);
        if (example.Target is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(example), example.Target, "Binary choice targets must be 0 or 1.");
        if (example.SequenceLength <= 0 || example.HiddenSize != head.HiddenSize
            || example.Features.Length != (long)example.SequenceLength * example.HiddenSize)
            throw new ArgumentException("Example features must match [sequenceLength, head.HiddenSize].", nameof(example));
        if (example.MarkerPositions.Length != 2
            || example.MarkerPositions[0] == example.MarkerPositions[1]
            || example.MarkerPositions.Any(marker => marker < 0 || marker >= example.SequenceLength))
            throw new ArgumentException("Binary relevance examples require two valid option markers.", nameof(example));
        if (example.Features.Any(value => !float.IsFinite(value)))
            throw new ArgumentException("Example feature values must be finite.", nameof(example));
    }

    static void EnsureQueryDisjoint(
        IReadOnlyList<AdaptiveTrainingExample> trainingExamples,
        IReadOnlyList<AdaptiveTrainingExample> holdoutExamples)
    {
        var queries = trainingExamples.Select(example => example.Query).ToHashSet(StringComparer.Ordinal);
        if (holdoutExamples.Any(example => queries.Contains(example.Query)))
            throw new ArgumentException("Training and holdout examples must use disjoint queries.", nameof(holdoutExamples));
    }
}
