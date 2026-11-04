using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Nivara;
using Nivara.AI;
using NivaraChat.Helpers;
using Nivara.Samples;
using OllamaSharp;
using System.Numerics.Tensors;

namespace NivaraChat.Modes;

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
            return;
        }

        Console.WriteLine("Loading MiniLM embedding model...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var embedder = MiniLMEmbedder.Create(minilmDir, ctx.UseSilk);
        sw.Stop();
        Console.WriteLine($"  Backend:    {(ctx.UseSilk ? "Silk (OpenCL)" : "CPU")}");
        Console.WriteLine($"  Dimensions: {embedder.Dimension}");
        Console.WriteLine($"  Loaded in:  {sw.ElapsedMilliseconds} ms\n");

        var repoRoot = ModeHelpers.GetRepoRoot();
        var resolvedDocsDir = ctx.DocsDir ?? Path.Combine(repoRoot, "docs");
        var mdFiles = Directory.Exists(resolvedDocsDir)
            ? Directory.GetFiles(resolvedDocsDir, "*.md", SearchOption.AllDirectories)
            : [];
        var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
        if (File.Exists(readmePath))
            mdFiles = [.. mdFiles, readmePath];

        if (mdFiles.Length == 0)
        {
            Console.WriteLine($"No markdown files found in {resolvedDocsDir}");
            return;
        }

        var allChunks = new List<RagChunk>();
        foreach (var f in mdFiles)
        {
            var content = await File.ReadAllTextAsync(f);
            allChunks.AddRange(RagChunker.ChunkMarkdown(content, f, repoRoot));
        }

        Console.WriteLine($"Chunked {allChunks.Count} chunks from {mdFiles.Length} files (store={ctx.Store})...");
        sw.Restart();

        bool useFrame = !string.Equals(ctx.Store, "inmemory", StringComparison.OrdinalIgnoreCase);

        float[]?[] embeddings = new float[allChunks.Count][];
        NivaraFrame? frame = null;
        CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore = null;
        VectorStoreCollection<string, DocumentChunk>? collection = null;

        var indexPath = RagIndexCache.ResolvePath(ctx, resolvedDocsDir);
        bool useCache = !ctx.NoCache && useFrame;
        if (useCache && !ctx.Rebuild && RagIndexCache.IsValid(indexPath, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk))
        {
            Console.WriteLine($"Loading cached index: {indexPath}");
            var loaded = RagIndexCache.Load(indexPath, out int dim);
            allChunks = RagIndexCache.ToChunks(loaded, dim);
            embeddings = RagIndexCache.ToEmbeddings(loaded, dim);
            var cols = new List<(string Name, IColumn)>();
            cols.Add(("ChunkId", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.ChunkId).ToArray())));
            cols.Add(("Source", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Source).ToArray())));
            cols.Add(("Title", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Title).ToArray())));
            cols.Add(("SectionPath", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.SectionPath).ToArray())));
            cols.Add(("AclGroup", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.AclGroup).ToArray())));
            cols.Add(("Content", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Content).ToArray())));
            var embCol = NivaraColumn<float[]>.CreateForReferenceType(embeddings.Select(e => e).ToArray()!);
            cols.Add(("Embedding", (IColumn)embCol));
            frame = NivaraFrame.Create(cols.ToArray());
            sw.Stop();
            Console.WriteLine($"  Loaded {allChunks.Count} chunks in {sw.ElapsedMilliseconds} ms\n");
        }
        else
        {
            for (int i = 0; i < allChunks.Count; i++)
            {
                var c = allChunks[i];
                if (string.IsNullOrWhiteSpace(c.Content)) { embeddings[i] = null; continue; }
                try { embeddings[i] = embedder.Embed(c.Content); } catch { embeddings[i] = null; }
                if ((i + 1) % 50 == 0 || i == allChunks.Count - 1)
                    Console.Write($"\r  Embedded {i + 1}/{allChunks.Count} chunks...   ");
            }
            Console.WriteLine();

            if (useFrame)
            {
                var cols = new List<(string Name, IColumn)>();
                cols.Add(("ChunkId", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.ChunkId).ToArray())));
                cols.Add(("Source", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Source).ToArray())));
                cols.Add(("Title", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Title).ToArray())));
                cols.Add(("SectionPath", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.SectionPath).ToArray())));
                cols.Add(("AclGroup", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.AclGroup).ToArray())));
                cols.Add(("Content", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Content).ToArray())));
                cols.Add(("Embedding", (IColumn)NivaraColumn<float[]>.CreateForReferenceType(embeddings.Select(e => e).ToArray()!)));
                frame = NivaraFrame.Create(cols.ToArray());
            }
            else
            {
                var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
                vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(new() { EmbeddingGenerator = generator });
                collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
                await collection.EnsureCollectionExistsAsync();
                foreach (var chunk in allChunks)
                {
                    await collection.UpsertAsync(new DocumentChunk { Key = chunk.ChunkId, Text = chunk.Content, Source = chunk.Source });
                }
            }

            if (useCache && useFrame && frame != null)
            {
                try
                {
                    RagIndexCache.Save(indexPath, frame, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk, embedder.Dimension);
                    Console.WriteLine($"  Cached index to {indexPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Cache save failed: {ex.Message}");
                }
            }

            sw.Stop();
            Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");
        }

        // ACL filter applied in search delegate (row op)

        Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
        var chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
        Console.WriteLine("Ollama connected.\n");

        // TextSearchProvider search delegate
        async Task<IEnumerable<TextSearchProvider.TextSearchResult>> SearchAsync(string query, CancellationToken ct)
        {
            var results = new List<TextSearchProvider.TextSearchResult>();
            if (useFrame && frame != null)
            {
                NivaraFrame workFrame = frame;
                List<RagChunk> workChunks = allChunks;
                if (!string.IsNullOrEmpty(ctx.AclGroup))
                {
                    var mask = NivaraColumn<bool>.Create(allChunks.Select(c => c.AclGroup == ctx.AclGroup).ToArray());
                    workFrame = frame.FilterByMask(mask);
                    workChunks = allChunks.Where(c => c.AclGroup == ctx.AclGroup).ToList();
                    if (workChunks.Count == 0) return results;
                }

                float[]? qVec = null;
                try { qVec = embedder.Embed(query); } catch { qVec = null; }
                NivaraSeries<float> scores;
                if (qVec == null)
                    scores = workFrame.RowCosineSimilarity("Embedding", (float[]?)null, labels: workFrame.GetColumn<string>("ChunkId"));
                else
                    scores = workFrame.RowCosineSimilarity("Embedding", qVec, labels: workFrame.GetColumn<string>("ChunkId"));
                var top = scores.TopKDescending(ctx.TopK);
                foreach (var (label, _) in top)
                {
                    var chunk = workChunks.FirstOrDefault(c => c.ChunkId == label);
                    if (chunk == null) continue;
                    results.Add(new TextSearchProvider.TextSearchResult { Text = chunk.Content, SourceName = chunk.Source });
                }
            }
            else if (collection != null)
            {
                await foreach (var r in collection.SearchAsync(query, ctx.TopK, cancellationToken: ct))
                    results.Add(new TextSearchProvider.TextSearchResult { Text = r.Record.Text, SourceName = r.Record.Source });
            }
            return results;
        }

        var textSearch = new TextSearchProvider(SearchAsync, new TextSearchProviderOptions
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
            RecentMessageMemoryLimit = 2
        });

        var baseAgent = new ChatClientAgent(chatClient, name: "NivaraRagAgent",
            instructions: "You are a helpful assistant that answers questions about the Nivara project using the provided context. Always base your answers on the retrieved context.");

        var agent = new AIAgentBuilder(baseAgent).UseAIContextProviders(textSearch).Build();

        Console.WriteLine($"Top-K: {ctx.TopK} chunks. TextSearchProvider auto-injects context before each LLM call.");
        Console.WriteLine("Type a question (or 'quit' to exit):\n");

        async Task RunQuery(string query)
        {
            var run = await InProcessExecution.RunAsync(new WorkflowBuilder(agent).WithOutputFrom(agent).Build(), query);
            Console.WriteLine("\n--- RAG Agent Response ---");
            foreach (var evt in run.NewEvents)
            {
                switch (evt)
                {
                    case AgentResponseUpdateEvent updateEvt:
                        if (updateEvt.Update?.Text is string text && !string.IsNullOrEmpty(text)) Console.Write(text);
                        break;
                    case AgentResponseEvent agentEvt:
                        if (agentEvt.Data is string data && !string.IsNullOrEmpty(data)) Console.WriteLine(data);
                        break;
                }
            }
            Console.WriteLine("\n");
        }

        if (ctx.SingleShotText != null)
            await RunQuery(ctx.SingleShotText);
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
