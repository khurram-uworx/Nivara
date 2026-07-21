using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Sparse embedding bag for fixed-width batches of active feature indices.
/// Input shape is [batchSize, maxActiveFeatures]; output shape is [batchSize, embeddingDim].
/// Entries equal to <see cref="PaddingIndex"/> are ignored.
/// </summary>
public sealed class SparseEmbedding<T> : Module<T> where T : struct, INumber<T>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    readonly int paddingIndex;
    readonly Parameter<T> weight;

    public int NumEmbeddings => numEmbeddings;
    public int EmbeddingDim => embeddingDim;
    public int PaddingIndex => paddingIndex;
    public ReverseGradTensor<T> Weight => weight.Tensor;
    public Parameter<T> WeightParam => weight;

    public SparseEmbedding(int numEmbeddings, int embeddingDim, int paddingIndex = -1)
    {
        if (numEmbeddings <= 0)
            throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIndex = paddingIndex;

        var data = new T[numEmbeddings * embeddingDim];
        var tensor = ReverseGradTensor<T>.FromMatrix(data, numEmbeddings, embeddingDim, requiresGrad: true);
        weight = new Parameter<T>("Weight", tensor);
        RegisterParameters(weight);

        var init = new NormalInitializer<T>(T.Zero, T.CreateChecked(0.02));
        init.Initialize(weight);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2)
            throw new ArgumentException("SparseEmbedding input must have shape [batchSize, maxActiveFeatures].", nameof(input));

        return ReverseGradOperations.SparseEmbeddingBag(weight.Tensor, input, paddingIndex);
    }
}
