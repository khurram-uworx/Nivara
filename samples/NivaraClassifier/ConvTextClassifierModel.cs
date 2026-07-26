using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace NivaraClassifier;

public sealed class ConvTextClassifierModel<T> : Module<T> where T : struct, INumber<T>
{
    readonly Embedding<T> embedding;
    readonly Conv1d<T> conv1;
    readonly Conv1d<T> conv2;
    readonly Conv1d<T> conv3;
    readonly Linear<T> fc1;
    readonly Linear<T> fc2;
    readonly Dropout<T> drop;
    readonly int embedDim;
    readonly int maxSeqLen;
    readonly int convOutputSize;

    public int VocabSize => embedding.NumEmbeddings;
    public int EmbeddingDim => embedDim;
    public int MaxSeqLen => maxSeqLen;

    public ConvTextClassifierModel(int vocabSize, int embedDim, int hiddenDim, int numClasses, int maxSeqLen, float dropout = 0.3f)
    {
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;

        embedding = new Embedding<T>(vocabSize, embedDim);
        conv1 = new Conv1d<T>(embedDim, 64, kernelSize: 3, padding: 1);
        conv2 = new Conv1d<T>(embedDim, 64, kernelSize: 5, padding: 2);
        conv3 = new Conv1d<T>(embedDim, 64, kernelSize: 7, padding: 3);
        convOutputSize = 64 * 3 * maxSeqLen;
        fc1 = new Linear<T>(convOutputSize, hiddenDim, bias: true);
        fc2 = new Linear<T>(hiddenDim, numClasses, bias: true);
        drop = new Dropout<T>(dropout);

        RegisterModules(embedding, conv1, conv2, conv3, fc1, fc2, drop);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var emb = embedding.Forward(input);
        var ncl = ReverseGradOperations.TransposeAxes(emb, 1, 2);

        var c1 = ReverseGradOperations.Relu(conv1.Forward(ncl));
        var c2 = ReverseGradOperations.Relu(conv2.Forward(ncl));
        var c3 = ReverseGradOperations.Relu(conv3.Forward(ncl));

        int B = c1.Shape[0];
        int c1Flat = c1.Shape[1] * c1.Shape[2];
        int c2Flat = c2.Shape[1] * c2.Shape[2];
        int c3Flat = c3.Shape[1] * c3.Shape[2];

        c1.Reshape(B, c1Flat);
        c2.Reshape(B, c2Flat);
        c3.Reshape(B, c3Flat);

        var cat = ReverseGradOperations.Concat([c1, c2, c3], axis: 1);

        var h = drop.Forward(ReverseGradOperations.Relu(fc1.Forward(cat)));
        return fc2.Forward(h);
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
