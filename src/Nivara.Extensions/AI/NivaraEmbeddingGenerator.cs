using Microsoft.Extensions.AI;

namespace Nivara.AI;

public sealed class NivaraEmbeddingGenerator<TInput> : IEmbeddingGenerator<TInput, Embedding<float>>
{
    readonly Func<TInput, float[]> embeddingFactory;
    readonly Func<IEnumerable<TInput>, IEnumerable<float[]>>? batchEmbeddingFactory;
    readonly EmbeddingGeneratorMetadata metadata;

    public int EmbeddingDimension { get; }

    public NivaraEmbeddingGenerator(
        Func<TInput, float[]> embeddingFactory,
        int embeddingDimension,
        string providerName = "Nivara",
        string? defaultModelId = null)
        : this(embeddingFactory, null, embeddingDimension, providerName, defaultModelId)
    {
    }

    public NivaraEmbeddingGenerator(
        Func<TInput, float[]> embeddingFactory,
        Func<IEnumerable<TInput>, IEnumerable<float[]>>? batchEmbeddingFactory,
        int embeddingDimension,
        string providerName = "Nivara",
        string? defaultModelId = null)
    {
        ArgumentNullException.ThrowIfNull(embeddingFactory);
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));

        this.embeddingFactory = embeddingFactory;
        this.batchEmbeddingFactory = batchEmbeddingFactory;
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

        var input = values.ToList();
        var results = new List<Embedding<float>>(input.Count);

        if (batchEmbeddingFactory is null || input.Count <= 1)
        {
            foreach (var value in input)
            {
                cancellationToken.ThrowIfCancellationRequested();
                results.Add(new Embedding<float>(embeddingFactory(value)));
            }
        }
        else
        {
            cancellationToken.ThrowIfCancellationRequested();
            var vectors = batchEmbeddingFactory(input).ToList();
            if (vectors.Count != input.Count)
                throw new InvalidOperationException(
                    $"The batch embedding factory returned {vectors.Count} vectors for {input.Count} inputs.");

            foreach (var vector in vectors)
                results.Add(new Embedding<float>(vector));
            cancellationToken.ThrowIfCancellationRequested();
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
