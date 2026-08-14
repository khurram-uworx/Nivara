using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Learnable lookup table mapping token IDs to dense vectors. Supports both single-token and
/// batched lookups via <see cref="ReverseGradOperations.Gather"/>. Weights are initialized
/// with a normal distribution (std 0.02).
/// </summary>
public sealed class Embedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    readonly Parameter<T> weight;

    /// <summary>Gets the size of the vocabulary (number of embedding rows).</summary>
    public int NumEmbeddings => numEmbeddings;
    /// <summary>Gets the dimension of each embedding vector.</summary>
    public int EmbeddingDim => embeddingDim;
    /// <summary>Gets the weight parameter (shape <c>[numEmbeddings, embeddingDim]</c>).</summary>
    public Parameter<T>? Weight => weight;

    /// <summary>
    /// Creates an embedding table.
    /// </summary>
    /// <param name="numEmbeddings">Size of the vocabulary (must be positive)</param>
    /// <param name="embeddingDim">Dimension of each embedding vector (must be positive)</param>
    public Embedding(int numEmbeddings, int embeddingDim)
    {
        if (numEmbeddings <= 0) throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        var data = new T[numEmbeddings * embeddingDim];
        var tensor = ReverseGradTensor<T>.FromMatrix(data, numEmbeddings, embeddingDim, requiresGrad: true);
        weight = new Parameter<T>("Weight", tensor);
        RegisterParameters(weight);

        var init = new NormalInitializer<T>(T.Zero, T.CreateChecked(0.02));
        init.Initialize(weight);
    }

    /// <summary>
    /// Looks up the embedding vector for each token ID in the input, producing a tensor of
    /// shape <c>inputShape + [embeddingDim]</c> (or <c>[numTokens, embeddingDim]</c> for flat input).
    /// </summary>
    /// <param name="input">Tensor of token IDs, each in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedded tensor</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        if (input.Length == 0)
            throw new ArgumentException("Embedding.Forward input tensor is empty.", nameof(input));

        int[] originalShape = input.Shape;
        int totalTokens = input.Length;

        var tokenIds = new int[totalTokens];
        for (int i = 0; i < totalTokens; i++)
            tokenIds[i] = int.CreateChecked(input.Data[i]);

        for (int i = 0; i < totalTokens; i++)
        {
            if (tokenIds[i] < 0 || tokenIds[i] >= numEmbeddings)
                throw new ArgumentOutOfRangeException(
                    nameof(input),
                    $"Token ID at position {i} is {tokenIds[i]}, " +
                    $"must be in range [0, {numEmbeddings}).");
        }

        var result = ReverseGradOperations.Gather(weight.Tensor, tokenIds);

        if (originalShape.Length > 1)
            result.Reshape(originalShape.Append(embeddingDim).ToArray());

        return result;
    }

    /// <summary>
    /// Looks up the embedding vector for a single token ID.
    /// </summary>
    /// <param name="tokenId">The token ID in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedding vector of length <c>embeddingDim</c></returns>
    public ReverseGradTensor<T> Forward(int tokenId)
    {
        if (tokenId < 0 || tokenId >= numEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        return ReverseGradOperations.Gather(weight.Tensor, [tokenId]);
    }
}
