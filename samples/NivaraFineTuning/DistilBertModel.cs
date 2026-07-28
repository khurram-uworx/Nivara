using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using System.Numerics;

namespace NivaraFineTuning;

public sealed class DistilBertForSequenceClassification<T> : Module<T> where T : struct, INumber<T>
{
    readonly MiniLMDistilled<T> backbone;
    readonly Linear<T> classifier;
    readonly int hiddenDim;
    readonly int numClasses;

    public int NumClasses => numClasses;

    public DistilBertForSequenceClassification(BertConfig config, int numClasses)
    {
        this.numClasses = numClasses;
        hiddenDim = config.HiddenSize;
        backbone = new MiniLMDistilled<T>(config);
        classifier = new Linear<T>(hiddenDim, numClasses, bias: true);
        RegisterModules(backbone, classifier);
    }

    public ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> inputIds,
        ReverseGradTensor<T> attentionMask,
        int batchSize,
        int seqLen)
    {
        var encoded = backbone.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
        var pooled = ExtractClsTokens(encoded, batchSize, seqLen);
        var logits = classifier.Forward(pooled);
        return logits;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => throw new NotImplementedException("Use Forward(inputIds, attentionMask, batchSize, seqLen).");

    ReverseGradTensor<T> ExtractClsTokens(ReverseGradTensor<T> encoded, int batchSize, int seqLen)
    {
        var srcSpan = encoded.Data.AsReadOnlySpan();
        var clsData = new T[batchSize * hiddenDim];
        for (int b = 0; b < batchSize; b++)
            srcSpan.Slice(b * seqLen * hiddenDim, hiddenDim).CopyTo(clsData.AsSpan(b * hiddenDim, hiddenDim));
        var clsCol = NivaraColumn<T>.Create(clsData);
        var clsTensor = new ReverseGradTensor<T>(clsCol, requiresGrad: encoded.RequiresGrad);
        clsTensor.Reshape(batchSize, hiddenDim);
        return clsTensor;
    }
}
