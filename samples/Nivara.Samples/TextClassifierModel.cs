using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class TextClassifierModel<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Embedding<T> embedding;
    readonly Linear<T> fc1;
    readonly Linear<T> fc2;
    readonly int embeddingDim;
    readonly int maxSeqLen;

    public int VocabSize => embedding.NumEmbeddings;
    public int EmbeddingDim => embeddingDim;
    public int MaxSeqLen => maxSeqLen;

    public TextClassifierModel(int vocabSize, int embeddingDim, int hiddenDim, int numClasses, int maxSeqLen)
    {
        this.embeddingDim = embeddingDim;
        this.maxSeqLen = maxSeqLen;

        embedding = new Embedding<T>(vocabSize, embeddingDim);
        fc1 = new Linear<T>(embeddingDim, hiddenDim, bias: true);
        fc2 = new Linear<T>(hiddenDim, numClasses, bias: true);

        RegisterModules(embedding, fc1, fc2);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var embedded = embedding.Forward(input);
        var pooled = ReverseGradOperations.MeanPool(embedded, maxSeqLen, embeddingDim);
        var h = fc1.Forward(pooled);
        var hRelu = ReverseGradOperations.Relu(h);
        var logits = fc2.Forward(hRelu);
        return logits;
    }

    public override string ExportToOnnx(IOnnxExportContext<T> context, string inputName) =>
        context.Linear(
            context.Relu(
                context.Linear(
                    context.ReduceMean(
                        context.Embedding(inputName, embedding),
                        new[] { 1 },
                        keepDims: false),
                    fc1)),
            fc2);

    public void SetWeights(T[] embeddingWeights, T[] fc1WeightsCpuLayout, T[] fc1Biases, T[] fc2WeightsCpuLayout, T[] fc2Biases)
    {
        var ps = GetParameters().Values.ToArray();
        ps[0].Tensor = ReverseGradTensor<T>.FromMatrix(embeddingWeights, VocabSize, EmbeddingDim, requiresGrad: false);
        ps[1].Tensor = ReverseGradTensor<T>.FromMatrix(fc1WeightsCpuLayout, fc1WeightsCpuLayout.Length / EmbeddingDim, EmbeddingDim, requiresGrad: false);
        ps[2].Tensor = ReverseGradTensor<T>.FromMatrix(fc1Biases, 1, fc1Biases.Length, requiresGrad: false);
        ps[3].Tensor = ReverseGradTensor<T>.FromMatrix(fc2WeightsCpuLayout, 2, fc2WeightsCpuLayout.Length / 2, requiresGrad: false);
        ps[4].Tensor = ReverseGradTensor<T>.FromMatrix(fc2Biases, 1, fc2Biases.Length, requiresGrad: false);
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
        int numClasses = logits.Length / batchSize;
        var result = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
            result[b] = ArgMax(logits, b, numClasses);
        return result;
    }
}
