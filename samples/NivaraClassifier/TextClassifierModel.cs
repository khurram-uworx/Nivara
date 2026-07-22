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
        int batchSize = input.Length / maxSeqLen;

        var embedded = embedding.Forward(input);
        var pooled = ReverseGradOperations.MeanPool(embedded, maxSeqLen, embeddingDim);
        var h = fc1.Forward(pooled);
        var hRelu = ReverseGradOperations.Relu(h);
        var logits = fc2.Forward(hRelu);
        return logits;
    }

    public int[] Predict(NivaraColumn<T> tokenIds, int batchSize)
    {
        var input = ReverseGradTensor<T>.FromMatrix(
            tokenIds, batchSize, tokenIds.Length / batchSize, requiresGrad: false);
        var logits = Forward(input);
        var result = new int[batchSize];
        int numClasses = logits.Length / batchSize;
        for (int b = 0; b < batchSize; b++)
        {
            int bestClass = 0;
            T bestVal = logits.Data[b * numClasses];
            for (int c = 1; c < numClasses; c++)
            {
                T val = logits.Data[b * numClasses + c];
                if (val > bestVal) { bestVal = val; bestClass = c; }
            }
            result[b] = bestClass;
        }
        return result;
    }
}
