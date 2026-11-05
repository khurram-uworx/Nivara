using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the null/empty and failure semantics of <see cref="MiniLMEmbedder.Handle.EmbedBatch"/>
/// that the RAG ingestion loop relies on, without loading the MiniLM model files.
/// </summary>
[TestFixture]
public class MiniLMEmbedderBatchTests
{
    [Test]
    public void EmbedBatch_WithoutBatchDelegate_UsesPerItemFallback()
    {
        using var handle = new MiniLMEmbedder.Handle(
            text =>
            {
                if (text == "boom") throw new InvalidOperationException("bad text");
                return [text.Length];
            },
            dimension: 1,
            lifetime: null);

        var result = handle.EmbedBatch(["a", "", "  ", null!, "boom", "cccc"]);

        Assert.That(result, Has.Length.EqualTo(6));
        Assert.That(result[0], Is.EqualTo(new[] { 1f }));
        Assert.That(result[1], Is.Null);
        Assert.That(result[2], Is.Null);
        Assert.That(result[3], Is.Null);
        Assert.That(result[4], Is.Null, "A per-item failure must yield a null embedding.");
        Assert.That(result[5], Is.EqualTo(new[] { 4f }));
    }

    [Test]
    public void EmbedBatch_StaticHelper_DelegatesToHandle()
    {
        var expected = new[] { new[] { 0.5f, 0.25f } };
        using var handle = new MiniLMEmbedder.Handle(
            _ => throw new InvalidOperationException("single-item path should not run"),
            _ => expected,
            dimension: 2,
            lifetime: null);

        var result = MiniLMEmbedder.EmbedBatch(handle, ["ignored"]);

        Assert.That(result, Is.SameAs(expected));
    }

    [Test]
    public void EmbedBatch_StaticHelper_WhenHandleNull_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => MiniLMEmbedder.EmbedBatch(null!, ["text"]));
    }
}
