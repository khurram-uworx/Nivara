using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Embedding<T> : IDisposable where T : struct, INumber<T>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    readonly Parameter<T> weight;
    bool disposed;

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
        weights.Add(weight);

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

    public ReverseGradTensor<T> Forward(int tokenId)
    {
        if (tokenId < 0 || tokenId >= numEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(tokenId));

        var oneHot = OneHot(tokenId, numEmbeddings);
        var emb = ReverseGradOperations.MatMul(oneHot, weight.Tensor);
        return emb;
    }

    readonly List<Parameter<T>> weights = [];

    public IReadOnlyList<Parameter<T>> Parameters => weights.AsReadOnly();

    public void Dispose()
    {
        if (!disposed)
        {
            foreach (var p in weights)
                p.Dispose();
            disposed = true;
        }
    }
}
