using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace NivaraClassifier;

public sealed class TextClassifierModel<T> : Module<T> where T : struct, INumber<T>
{
    readonly Embedding<T> embedding;
    readonly Linear<T> fc1;
    readonly Linear<T> fc2;
    readonly int embeddingDim;
    readonly int maxSeqLen;

    public TextClassifierModel(int vocabSize, int embeddingDim, int hiddenDim, int numClasses, int maxSeqLen)
    {
        this.embeddingDim = embeddingDim;
        this.maxSeqLen = maxSeqLen;

        embedding = new Embedding<T>(vocabSize, embeddingDim);
        fc1 = new Linear<T>(embeddingDim, hiddenDim, useBias: true);
        fc2 = new Linear<T>(hiddenDim, numClasses, useBias: true);

        RegisterSubModule("embedding", embedding);
        RegisterSubModule("fc1", fc1);
        RegisterSubModule("fc2", fc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        var embedded = embedding.Forward(input);
        int batchTokens = embedded.Length;
        int batchSize = batchTokens / embeddingDim;
        int seqLen = batchTokens / (batchSize * embeddingDim);

        var pooled = ReverseGradOperations.MeanPool(embedded, seqLen, embeddingDim);
        var h = fc1.Forward(pooled);
        var hRelu = ReverseGradOperations.Relu(h);
        var logits = fc2.Forward(hRelu);
        return logits;
    }

    public int[] ForwardWithLogits(NivaraColumn<T> batchInput, int batchSize)
    {
        using var gradScope = GradientUtils.Grad();
        var input = ReverseGradTensor<T>.FromMatrix(
            batchInput, batchSize, batchInput.Length / batchSize, requiresGrad: false);

        var logits = Forward(input);
        var result = new int[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            result[i] = int.CreateChecked(logits.Data[i]);
        return result;
    }
}
