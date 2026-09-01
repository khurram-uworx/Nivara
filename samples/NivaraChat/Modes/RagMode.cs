using Microsoft.Extensions.AI;
using Nivara.Samples;
using NivaraChat.Helpers;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --rag: RAG pipeline — chunk the docs into vector embeddings, retrieve the top-K chunks per
/// query, and let the Ollama LLM answer from that context.
/// </summary>
public static class RagMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — RAG Pipeline ===\n");

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --rag requires --ollama for LLM generation.");
            return;
        }

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
        sw.Stop();
        Console.WriteLine($"  Dimensions: {generator.EmbeddingDimension}");
        Console.WriteLine($"  Loaded in:  {sw.ElapsedMilliseconds} ms\n");

        var vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(
            new() { EmbeddingGenerator = generator });
        var collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
        await collection.EnsureCollectionExistsAsync();

        var repoRoot = ModeHelpers.GetRepoRoot();
        var resolvedDocsDir = ctx.DocsDir ?? Path.Combine(repoRoot, "docs");
        var mdFiles = Directory.Exists(resolvedDocsDir)
            ? Directory.GetFiles(resolvedDocsDir, "*.md")
            : [];

        var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
        if (File.Exists(readmePath))
            mdFiles = [.. mdFiles, readmePath];

        if (mdFiles.Length == 0)
        {
            Console.WriteLine($"No markdown files found in {resolvedDocsDir}");
            return;
        }

        Console.WriteLine($"Indexing {mdFiles.Length} markdown files...");
        sw.Restart();
        await DocumentChunker.IndexMarkdownFiles(collection, mdFiles);
        sw.Stop();
        Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        Console.WriteLine($"Top-K: {ctx.TopK} chunks per query. Type a question (or 'quit' to exit):\n");

        async Task RunQuery(string query)
        {
            sw.Restart();
            var searchResults = new List<(DocumentChunk Record, double? Score)>();
            await foreach (var result in collection.SearchAsync(query, ctx.TopK))
            {
                searchResults.Add((result.Record, result.Score));
            }
            var retrievalMs = sw.ElapsedMilliseconds;

            Console.WriteLine($"\n  Retrieved {searchResults.Count} chunks ({retrievalMs} ms):");
            for (int i = 0; i < searchResults.Count; i++)
                Console.WriteLine($"    #{i + 1}  {searchResults[i].Score:F4}  [{searchResults[i].Record.Source}]  \"{searchResults[i].Record.Text[..Math.Min(100, searchResults[i].Record.Text.Length)]}...\"");

            var context = string.Join("\n\n", searchResults.Select(r => r.Record.Text));
            var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";

            sw.Restart();
            var response = await chatClient.GetResponseAsync(prompt, new ChatOptions { ModelId = ctx.ModelName });
            var llmMs = sw.ElapsedMilliseconds;

            Console.WriteLine($"\n  LLM response ({llmMs} ms):");
            Console.WriteLine($"  {response.Text}");
            Console.WriteLine();
        }

        if (ctx.SingleShotText != null)
        {
            await RunQuery(ctx.SingleShotText);
        }
        else
        {
            while (true)
            {
                Console.Write("> ");
                var query = Console.ReadLine()?.Trim();
                if (string.IsNullOrEmpty(query) || query == "quit") break;
                await RunQuery(query);
            }
        }
    }
}