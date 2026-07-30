using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Embedding<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    readonly Parameter<T> weight;

    public int NumEmbeddings => numEmbeddings;
    public int EmbeddingDim => embeddingDim;
    public ReverseGradTensor<T> Weight => weight.Tensor;
    public Parameter<T> WeightParam => weight;

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

    public ReverseGradTensor<T> Forward(int tokenId)
    {
        if (tokenId < 0 || tokenId >= numEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        return ReverseGradOperations.Gather(weight.Tensor, [tokenId]);
    }
}
