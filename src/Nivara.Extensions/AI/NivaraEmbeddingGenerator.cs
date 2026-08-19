using Microsoft.Extensions.AI;

namespace Nivara.AI;

public sealed class NivaraEmbeddingGenerator<TInput> : IEmbeddingGenerator<TInput, Embedding<float>>
{
    readonly Func<TInput, float[]> embeddingFactory;
    readonly EmbeddingGeneratorMetadata metadata;

    public int EmbeddingDimension { get; }

    public NivaraEmbeddingGenerator(
        Func<TInput, float[]> embeddingFactory,
        int embeddingDimension,
        string providerName = "Nivara",
        string? defaultModelId = null)
    {
        ArgumentNullException.ThrowIfNull(embeddingFactory);
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));

        this.embeddingFactory = embeddingFactory;
        EmbeddingDimension = embeddingDimension;
        metadata = new EmbeddingGeneratorMetadata(
            providerName: providerName,
            defaultModelId: defaultModelId,
            defaultModelDimensions: embeddingDimension);
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<TInput> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);

        var results = new List<Embedding<float>>();
        foreach (var value in values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var vector = embeddingFactory(value);
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

    public void Dispose() { }
}
