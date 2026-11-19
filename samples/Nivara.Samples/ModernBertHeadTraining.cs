using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AI;

/// <summary>
/// Head-only training helper for <see cref="ModernBertClassificationHead{TModel,TWeight}"/>:
/// precomputes frozen <c>[CLS]</c> features from a ModernBERT encoder, then trains only
/// the head with Adam + cross-entropy. The encoder is never differentiated.
/// </summary>
public static class ModernBertHeadTraining
{
    public sealed record Result(int Examples, int Epochs, float FinalLoss, float Accuracy);

    public static Result Train(
        ModernBertEncoder<float, float> encoder,
        ModernBertClassificationHead<float, float> head,
        (int[] Ids, int ValidLength, int Label)[] examples,
        int epochs = 5,
        float learningRate = 0.001f)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(examples);
        if (examples.Length == 0) throw new ArgumentException("At least one example is required.", nameof(examples));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (learningRate <= 0f) throw new ArgumentOutOfRangeException(nameof(learningRate));

        int hiddenSize = head.HiddenSize;
        int numClasses = head.NumClasses;

        // Precompute frozen [CLS] features (row 0 of the encoder's last hidden state).
        var features = new float[examples.Length][];
        var labels = new int[examples.Length];
        for (int i = 0; i < examples.Length; i++)
        {
            var hidden = encoder.Forward(examples[i].Ids, examples[i].ValidLength);
            hidden.Data.TryGetSpan(out var span);
            var cls = new float[hiddenSize];
            for (int d = 0; d < hiddenSize; d++) cls[d] = span[d];
            features[i] = cls;
            labels[i] = examples[i].Label;
        }

        var parameters = new List<Parameter<float>>();
        parameters.AddRange(head.Hidden.GetParameters().Values);
        parameters.AddRange(head.Classifier.GetParameters().Values);

        using var optimizer = new Adam<float>(learningRate);
        optimizer.AddParameterGroup(parameters);
        var loss = new CrossEntropyLoss<float>();

        float finalLoss = 0f;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            optimizer.ZeroGrad();
            using (GradientUtils.Grad())
            {
                ReverseGradTensor<float>? batchLoss = null;
                for (int i = 0; i < examples.Length; i++)
                {
                    var clsTensor = ReverseGradTensor<float>.FromMatrix(
                        (float[])features[i].Clone(), 1, hiddenSize, requiresGrad: false);
                    var logits = head.Forward(clsTensor);
                    var exLoss = loss.Forward(logits, new[] { labels[i] });
                    batchLoss = batchLoss is null ? exLoss : ReverseGradOperations.Add(batchLoss, exLoss);
                }
                finalLoss = float.CreateChecked(ReverseGradOperations.DivideScalar(batchLoss!, examples.Length).Data.TryGetSpan(out var ls) ? ls[0] : 0f);
                ReverseGradOperations.DivideScalar(batchLoss!, examples.Length).Backward();
            }
            optimizer.Step();
        }

        // Accuracy on the training set (inference, no grad).
        head.Eval();
        int correct = 0;
        for (int i = 0; i < examples.Length; i++)
        {
            var clsTensor = ReverseGradTensor<float>.FromMatrix(features[i], 1, hiddenSize, requiresGrad: false);
            var logits = head.Forward(clsTensor);
            logits.Data.TryGetSpan(out var span);
            int pred = 0; float best = span[0];
            for (int c = 1; c < numClasses; c++)
                if (span[c] > best) { best = span[c]; pred = c; }
            if (pred == labels[i]) correct++;
        }
        float accuracy = (float)correct / examples.Length;
        return new Result(examples.Length, epochs, finalLoss, accuracy);
    }
}
