using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class TokenClassifierModel<T> : Module<T> where T : struct, INumber<T>
{
    readonly Embedding<T> embedding;
    readonly Linear<T> fc1;
    readonly Linear<T> fc2;
    readonly int embeddingDim;
    readonly int maxSeqLen;
    readonly int numClasses;

    public int VocabSize => embedding.NumEmbeddings;
    public int EmbeddingDim => embeddingDim;
    public int MaxSeqLen => maxSeqLen;
    public int NumClasses => numClasses;

    public TokenClassifierModel(int vocabSize, int embeddingDim, int hiddenDim, int numClasses, int maxSeqLen)
    {
        this.embeddingDim = embeddingDim;
        this.maxSeqLen = maxSeqLen;
        this.numClasses = numClasses;

        embedding = new Embedding<T>(vocabSize, embeddingDim);
        fc1 = new Linear<T>(embeddingDim, hiddenDim, bias: true);
        fc2 = new Linear<T>(hiddenDim, numClasses, bias: true);

        RegisterModules(embedding, fc1, fc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var embedded = embedding.Forward(input);

        int totalTokens = 1;
        for (int i = 0; i < embedded.Shape.Length - 1; i++)
            totalTokens *= embedded.Shape[i];
        embedded.Reshape(totalTokens, embeddingDim);

        var h = fc1.Forward(embedded);
        var hRelu = ReverseGradOperations.Relu(h);
        var logits = fc2.Forward(hRelu);
        return logits;
    }

    public int[] Predict(int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        if (tokenIds.Length % maxSeqLen != 0)
            throw new ArgumentException(
                $"tokenIds length ({tokenIds.Length}) must be divisible by MaxSeqLen ({maxSeqLen}).",
                nameof(tokenIds));

        int batchSize = tokenIds.Length / maxSeqLen;
        var data = new T[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            data[i] = T.CreateChecked(tokenIds[i]);
        var input = ReverseGradTensor<T>.FromMatrix(data, batchSize, maxSeqLen, requiresGrad: false);
        var logits = Forward(input);
        var result = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            result[i] = ArgMax(logits, i, numClasses);
        return result;
    }
}
