using Microsoft.Extensions.AI;

namespace NivaraChess;

public sealed class ChessEmbeddingGenerator : IEmbeddingGenerator<ChessBoard, Embedding<float>>
{
    readonly ChessEvalModelBase model;
    readonly EmbeddingGeneratorMetadata metadata;

    public ChessEmbeddingGenerator(ChessEvalModelBase model)
    {
        ArgumentNullException.ThrowIfNull(model);

        this.model = model;
        this.model.Eval();
        metadata = new EmbeddingGeneratorMetadata(
            providerName: "NivaraChess",
            defaultModelId: $"chess-eval-phase{model.Phase}",
            defaultModelDimensions: model.EmbeddingDim);
    }

    public int EmbeddingDimension => model.EmbeddingDim;

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<ChessBoard> boards,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(boards);

        var results = new List<Embedding<float>>();
        foreach (var board in boards)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var vector = model.ComputeEmbedding(board);
            results.Add(new Embedding<float>(vector));
        }

        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(results));
    }

    public object? GetService(Type serviceType, object? serviceKey)
    {
        if (serviceKey is not null)
            return null;

        if (serviceType == typeof(EmbeddingGeneratorMetadata))
            return metadata;

        if (serviceType.IsInstanceOfType(this))
            return this;

        return null;
    }

    public void Dispose()
    {
        model.Dispose();
    }
}
