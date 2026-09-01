using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Nivara.Samples;
using NivaraChat;
using OllamaSharp;

namespace NivaraChat.Modes;

/// <summary>
/// --rag-agent: RAG pipeline with a TextSearchProvider that auto-injects retrieved context
/// before each LLM call inside the agent workflow.
/// </summary>
public static class RagAgentMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — RAG Agent (TextSearchProvider) ===\n");

        if (!ctx.UseOllama)
        {
            Console.WriteLine("Error: --rag-agent requires --ollama for LLM generation.");
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

        var textSearch = new TextSearchProvider(
            async (query, ct) =>
            {
                var results = new List<TextSearchProvider.TextSearchResult>();
                await foreach (var result in collection.SearchAsync(query, ctx.TopK, cancellationToken: ct))
                {
                    results.Add(new TextSearchProvider.TextSearchResult
                    {
                        Text = result.Record.Text,
                        SourceName = result.Record.Source
                    });
                }
                return results;
            },
            new TextSearchProviderOptions
            {
                SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
                RecentMessageMemoryLimit = 2
            });

        var baseAgent = new ChatClientAgent(chatClient,
            name: "NivaraRagAgent",
            instructions: "You are a helpful assistant that answers questions about the Nivara project using the provided context. Always base your answers on the retrieved context.");

        var agent = new AIAgentBuilder(baseAgent)
            .UseAIContextProviders(textSearch)
            .Build();

        Console.WriteLine($"Top-K: {ctx.TopK} chunks. TextSearchProvider auto-injects context before each LLM call.");
        Console.WriteLine("Type a question (or 'quit' to exit):\n");

        async Task RunQuery(string query)
        {
            var run = await InProcessExecution.RunAsync(
                new WorkflowBuilder(agent).WithOutputFrom(agent).Build(),
                query);

            Console.WriteLine("\n--- RAG Agent Response ---");
            foreach (var evt in run.NewEvents)
            {
                switch (evt)
                {
                    case AgentResponseUpdateEvent updateEvt:
                        if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text))
                            Console.Write(text);
                        break;
                    case AgentResponseEvent agentEvt:
                        if (agentEvt.Data is string data && !string.IsNullOrEmpty(data))
                            Console.WriteLine(data);
                        break;
                }
            }
            Console.WriteLine("\n");
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