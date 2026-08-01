using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.Samples;
using System.Numerics;

namespace NivaraFineTuning;

public sealed class DistilBertForSequenceClassification<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    internal readonly BertEncoder<T> encoder;
    internal readonly Linear<T> preClassifier;
    internal readonly Linear<T> classifier;
    readonly int hiddenDim;

    public int NumClasses { get; }

    public DistilBertForSequenceClassification(BertConfig config, int numClasses)
    {
        NumClasses = numClasses;
        hiddenDim = config.HiddenSize;
        encoder = new BertEncoder<T>(config);
        preClassifier = new Linear<T>(hiddenDim, hiddenDim, bias: true);
        classifier = new Linear<T>(hiddenDim, numClasses, bias: true);
        RegisterModules(encoder, preClassifier, classifier);
    }

    public ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> inputIds,
        ReverseGradTensor<T> attentionMask,
        int batchSize,
        int seqLen)
    {
        var encoded = encoder.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
        var clsTokens = ExtractClsTokens(encoded, batchSize, seqLen);
        var h = preClassifier.Forward(clsTokens);
        h = ReverseGradOperations.Gelu(h);
        var logits = classifier.Forward(h);
        return logits;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => throw new NotImplementedException("Use Forward(inputIds, attentionMask, batchSize, seqLen).");

    static ReverseGradTensor<T> ExtractClsTokens(ReverseGradTensor<T> encoded, int batchSize, int seqLen)
    {
        var indices = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
            indices[b] = b * seqLen;
        return ReverseGradOperations.Gather(encoded, indices, axis: 0);
    }

    public void LoadWeights<TWeight>(
        Dictionary<string, (TWeight[] Data, int[] Shape)> tensors)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        DistilBertLoader.LoadEncoderWeights<T, TWeight>(encoder, tensors, "distilbert");

        if (tensors.ContainsKey("pre_classifier.weight") || tensors.ContainsKey("pre_classifier.bias"))
            StateDictLoader.LoadLinear(preClassifier, tensors, "pre_classifier");

        if (tensors.ContainsKey("classifier.weight") || tensors.ContainsKey("classifier.bias"))
            StateDictLoader.LoadLinear(classifier, tensors, "classifier");

        Eval();
    }
}
