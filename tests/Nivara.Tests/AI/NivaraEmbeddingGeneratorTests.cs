using Nivara.AI;
using NUnit.Framework;

namespace Nivara.Tests.AI;

/// <summary>
/// Covers the optional batch path on <see cref="NivaraEmbeddingGenerator{TInput}"/>: the batch
/// factory serves multi-input requests, while single-input requests stay on the per-item path.
/// </summary>
[TestFixture]
public class NivaraEmbeddingGeneratorTests
{
    [Test]
    public async Task GenerateAsync_WithBatchFactory_UsesItForMultipleInputs()
    {
        var batchCalls = 0;
        var generator = new NivaraEmbeddingGenerator<string>(
            _ => throw new InvalidOperationException("single-item path should not run"),
            texts =>
            {
                batchCalls++;
                return texts.Select(text => new[] { (float)text.Length });
            },
            embeddingDimension: 1);

        var generated = await generator.GenerateAsync(["a", "bb", "ccc"]);

        Assert.That(batchCalls, Is.EqualTo(1));
        Assert.That(generated.Select(e => e.Vector.ToArray()), Is.EqualTo(new[]
        {
            new[] { 1f },
            new[] { 2f },
            new[] { 3f }
        }));
    }

    [Test]
    public async Task GenerateAsync_WithBatchFactory_UsesPerItemPathForSingleInput()
    {
        var generator = new NivaraEmbeddingGenerator<string>(
            text => [(float)text.Length],
            _ => throw new InvalidOperationException("batch path should not run for one input"),
            embeddingDimension: 1);

        var generated = await generator.GenerateAsync(["abc"]);

        Assert.That(generated, Has.Count.EqualTo(1));
        Assert.That(generated[0].Vector.ToArray(), Is.EqualTo(new[] { 3f }));
    }

    [Test]
    public async Task GenerateAsync_WithBatchFactoryReturningWrongCount_Throws()
    {
        var generator = new NivaraEmbeddingGenerator<string>(
            _ => [0f],
            _ => new[] { new[] { 0f } },
            embeddingDimension: 1);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await generator.GenerateAsync(["a", "b"]));
    }
}
