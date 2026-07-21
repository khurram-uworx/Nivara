using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Embedding<T> : Module<T> where T : struct, INumber<T>
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

    static ReverseGradTensor<T> OneHot(int index, int vocabSize)
    {
        var data = new T[vocabSize];
        data[index] = T.One;
        var col = NivaraColumn<T>.Create(data);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(1, vocabSize);
        return tensor;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        if (input.Length == 0)
            throw new ArgumentException("Embedding.Forward input tensor is empty.", nameof(input));

        if (input.Length == 1)
        {
            var tokenId = int.CreateChecked(input.Data[0]);
            return Forward(tokenId);
        }

        return ForwardBatched(input);
    }

    ReverseGradTensor<T> ForwardBatched(ReverseGradTensor<T> input)
    {
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

        int vocabSize = numEmbeddings;
        int embedDim = embeddingDim;

        var oneHotData = new T[totalTokens * vocabSize];
        for (int i = 0; i < totalTokens; i++)
            oneHotData[i * vocabSize + tokenIds[i]] = T.One;

        var oneHotCol = NivaraColumn<T>.Create(oneHotData);
        var oneHotTensor = new ReverseGradTensor<T>(oneHotCol, requiresGrad: false);
        oneHotTensor.Reshape(totalTokens, vocabSize);

        var result = ReverseGradOperations.MatMul(oneHotTensor, weight.Tensor);

        if (originalShape.Length > 1)
            result.Reshape(originalShape.Append(embedDim).ToArray());

        return result;
    }

    public ReverseGradTensor<T> Forward(int tokenId)
    {
        if (tokenId < 0 || tokenId >= numEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        var oneHot = OneHot(tokenId, numEmbeddings);
        var emb = ReverseGradOperations.MatMul(oneHot, weight.Tensor);
        return emb;
    }
}
