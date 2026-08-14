using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Sparse embedding bag for fixed-width batches of active feature indices.
/// Input shape is [batchSize, maxActiveFeatures]; output shape is [batchSize, embeddingDim].
/// Entries equal to <see cref="PaddingIndex"/> are ignored.
/// </summary>
public sealed class SparseEmbedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    readonly int paddingIndex;
    readonly Parameter<T> weight;

    /// <summary>Gets the size of the vocabulary (number of embedding rows).</summary>
    public int NumEmbeddings => numEmbeddings;
    /// <summary>Gets the dimension of each embedding vector.</summary>
    public int EmbeddingDim => embeddingDim;
    /// <summary>Gets the padding index whose entries are ignored during the sum.</summary>
    public int PaddingIndex => paddingIndex;
    /// <summary>Gets the weight parameter (shape <c>[numEmbeddings, embeddingDim]</c>).</summary>
    public Parameter<T>? Weight => weight;

    /// <summary>
    /// Creates a sparse embedding bag.
    /// </summary>
    /// <param name="numEmbeddings">Size of the vocabulary (must be positive)</param>
    /// <param name="embeddingDim">Dimension of each embedding vector (must be positive)</param>
    /// <param name="paddingIndex">Index that marks padding; its row is ignored</param>
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

    /// <summary>
    /// Sums the embeddings of the active feature indices in each row of a
    /// <c>[batchSize, maxActiveFeatures]</c> input, producing <c>[batchSize, embeddingDim]</c>.
    /// </summary>
    /// <param name="input">Tensor of feature indices (rank 2)</param>
    /// <returns>The summed embedding vectors</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2)
            throw new ArgumentException("SparseEmbedding input must have shape [batchSize, maxActiveFeatures].", nameof(input));

        return ReverseGradOperations.SparseEmbeddingBag(weight.Tensor, input, paddingIndex);
    }
}
