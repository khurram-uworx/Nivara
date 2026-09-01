using Microsoft.Extensions.AI;
using Nivara.Samples;
using System.Numerics.Tensors;

namespace NivaraChat.Modes;

/// <summary>
/// --embed: embedding search — index documents in-memory and retrieve the most relevant ones
/// for a query via IEmbeddingGenerator + TensorPrimitives cosine similarity.
/// </summary>
public static class EmbeddingMode
{
    public static void Run()
    {
        Console.WriteLine("=== NivaraChat — Embedding Search ===\n");

        var minilmDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "minilm");
        if (!Directory.Exists(minilmDir))
        {
            Console.WriteLine($"MiniLM model not found at: {minilmDir}");
            Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
            return;
        }

        Console.WriteLine("Loading MiniLM embedding model...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
        var metadata = generator.GetService(typeof(EmbeddingGeneratorMetadata), null) as EmbeddingGeneratorMetadata;
        sw.Stop();
        Console.WriteLine($"  Provider:     {metadata?.ProviderName ?? "Nivara-MiniLM"}");
        Console.WriteLine($"  Model:        {metadata?.DefaultModelId ?? "all-minilm-l6-v2"}");
        Console.WriteLine($"  Dimensions:   {generator.EmbeddingDimension}");
        Console.WriteLine($"  Loaded in:    {sw.ElapsedMilliseconds} ms\n");

        var documents = new[]
        {
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "The stock market closed at record highs today",
            "Neural networks are inspired by biological brains",
            "The weather forecast predicts rain tomorrow",
            "Deep learning has revolutionized computer vision",
            "Interest rates are expected to rise next quarter",
            "Natural language processing enables text understanding"
        };

        Console.WriteLine($"Indexing {documents.Length} knowledge documents via IEmbeddingGenerator...");
        sw.Restart();
        var docEmbeddings = generator.GenerateAsync(documents).GetAwaiter().GetResult();
        sw.Stop();
        for (int i = 0; i < documents.Length; i++)
            Console.WriteLine($"  [{i}] \"{documents[i]}\"");
        Console.WriteLine($"\nIndexed in {sw.ElapsedMilliseconds} ms — ready for chat\n");

        Console.WriteLine("Type a message and press Enter (or 'quit' to exit):\n");

        while (true)
        {
            Console.Write("> ");
            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query) || query == "quit") break;

            sw.Restart();
            var queryEmbedding = generator.GenerateVectorAsync(query).GetAwaiter().GetResult();
            sw.Stop();
            var queryVector = queryEmbedding.ToArray();

            var scores = new float[documents.Length];
            for (int i = 0; i < documents.Length; i++)
            {
                var docVector = docEmbeddings[i].Vector;
                scores[i] = TensorPrimitives.CosineSimilarity(queryVector, docVector.Span);
            }

            var ranked = scores
                .Select((score, idx) => (Score: score, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(4)
                .ToList();

            Console.WriteLine($"  Retrieved {ranked.Count} relevant documents ({sw.ElapsedMilliseconds} ms)\n");
            Console.WriteLine("  Context for LLM:");
            for (int rank = 0; rank < ranked.Count; rank++)
                Console.WriteLine($"    #{rank + 1}  {ranked[rank].Score:F4}  \"{documents[ranked[rank].Index]}\"");
            Console.WriteLine("\n  (In a full pipeline, these would be injected into the LLM prompt\n   via TextSearchProvider — see README.md \"RAG agent\" section)\n");
        }
    }
}