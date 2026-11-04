using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Nivara;
using Nivara.AI;
using NivaraChat.Helpers;
using Nivara.Samples;
using OllamaSharp;
using System.Numerics.Tensors;

namespace NivaraChat.Modes;

/// <summary>
/// --rag: RAG pipeline. Chunks docs, embeds via MiniLM (CPU or Silk), stores in a
/// NivaraFrame (convention from #867) or legacy InMemoryVectorStore, retrieves Top-K
/// via RowCosineSimilarity, and optionally grounds an Ollama answer.
/// </summary>
public static class RagMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — RAG Pipeline ===\n");

        var minilmDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "minilm");
        if (!Directory.Exists(minilmDir))
        {
            Console.WriteLine($"MiniLM model not found at: {minilmDir}");
            Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
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

        // Chunk
        var allChunks = new List<RagChunk>();
        foreach (var f in mdFiles)
        {
            var content = await File.ReadAllTextAsync(f);
            allChunks.AddRange(RagChunker.ChunkMarkdown(content, f, repoRoot));
        }

        // ACL filter (row op before scoring)
        if (!string.IsNullOrEmpty(ctx.AclGroup))
        {
            var before = allChunks.Count;
            allChunks = allChunks.Where(c => c.AclGroup == ctx.AclGroup).ToList();
            Console.WriteLine($"ACL group '{ctx.AclGroup}': {allChunks.Count}/{before} chunks visible.");
            if (allChunks.Count == 0)
            {
                Console.WriteLine("No chunks visible under this ACL group.");
                return;
            }
        }

        Console.WriteLine($"Indexing {allChunks.Count} chunks from {mdFiles.Length} files (store={ctx.Store})...");
        sw.Restart();

        // Embed
        float[]?[] embeddings = new float[allChunks.Count][];
        for (int i = 0; i < allChunks.Count; i++)
        {
            var c = allChunks[i];
            if (string.IsNullOrWhiteSpace(c.Content))
            {
                embeddings[i] = null;
                continue;
            }
            try { embeddings[i] = embedder.Embed(c.Content); }
            catch { embeddings[i] = null; }
            if ((i + 1) % 50 == 0 || i == allChunks.Count - 1)
                Console.Write($"\r  Embedded {i + 1}/{allChunks.Count} chunks...   ");
        }
        Console.WriteLine();

        bool useFrame = !string.Equals(ctx.Store, "inmemory", StringComparison.OrdinalIgnoreCase);

        // Legacy in-memory path
        CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore = null!;
        VectorStoreCollection<string, DocumentChunk>? collection = null!;
        if (!useFrame)
        {
            var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
            vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(new() { EmbeddingGenerator = generator });
            collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
            await collection.EnsureCollectionExistsAsync();
            foreach (var (chunk, emb) in allChunks.Zip(embeddings))
            {
                var dc = new DocumentChunk { Key = chunk.ChunkId, Text = chunk.Content, Source = chunk.Source };
                await collection.UpsertAsync(dc);
            }
        }

        // Frame path
        NivaraFrame frame = null!;
        if (useFrame)
        {
            var cols = new List<(string Name, IColumn)>();
            cols.Add(("ChunkId", (IColumn)NivaraColumn<object>.CreateForReferenceType(allChunks.Select(c => (object)c.ChunkId).ToArray())));
            cols.Add(("Source", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Source).ToArray())));
            cols.Add(("Title", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Title).ToArray())));
            cols.Add(("SectionPath", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.SectionPath).ToArray())));
            cols.Add(("AclGroup", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.AclGroup).ToArray())));
            cols.Add(("Content", (IColumn)NivaraColumn<string>.Create(allChunks.Select(c => c.Content).ToArray())));
            var embCol = NivaraColumn<float[]>.CreateForReferenceType(embeddings.Select(e => e).ToArray()!);
            cols.Add(("Embedding", (IColumn)embCol));
            frame = NivaraFrame.Create(cols.ToArray());
        }

        sw.Stop();
        Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");

        // Ollama (optional)
        OllamaApiClient? chatClient = null!;
        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            chatClient = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            Console.WriteLine("Ollama connected.\n");
        }
        else
        {
            Console.WriteLine("No --ollama: will list retrieved chunks only.\n");
        }

        Console.WriteLine($"Top-K: {ctx.TopK} chunks per query. Type a question (or 'quit' to exit):\n");

        async Task RunQuery(string query)
        {
            sw.Restart();
            var results = new List<(string ChunkId, string Source, string Title, string Content, double Score)>();

            if (useFrame)
            {
                float[]? qVec = null;
                try { qVec = embedder.Embed(query); } catch { qVec = null; }

                NivaraSeries<float> scores;
                if (qVec == null)
                {
                    // null query -> all scores null
                    scores = frame.RowCosineSimilarity("Embedding", (float[]?)null,
                        labels: frame.GetColumn<object>("ChunkId"));
                }
                else
                {
                    scores = frame.RowCosineSimilarity("Embedding", qVec,
                        labels: frame.GetColumn<object>("ChunkId"));
                }

                var top = scores.TopKDescending(ctx.TopK);
                foreach (var (label, score) in top)
                {
                    var cid = label ?? "";
                    var chunk = allChunks.FirstOrDefault(c => c.ChunkId == cid);
                    if (chunk == null) continue;
                    results.Add((chunk.ChunkId, chunk.Source, chunk.Title, chunk.Content, score));
                }
            }
            else if (!useFrame)
            {
                var searchResults = new List<(DocumentChunk Record, double? Score)>();
                await foreach (var r in collection!.SearchAsync(query, ctx.TopK))
                    searchResults.Add((r.Record, r.Score));
                foreach (var (rec, sc) in searchResults)
                {
                    var chunk = allChunks.FirstOrDefault(c => c.ChunkId == rec.Key);
                    results.Add((rec.Key, rec.Source, chunk?.Title ?? rec.Source, rec.Text, sc ?? 0));
                }
            }

            var retrievalMs = sw.ElapsedMilliseconds;

            Console.WriteLine($"\n  Retrieved {results.Count} chunks ({retrievalMs} ms):");
            for (int i = 0; i < results.Count; i++)
            {
                var r = results[i];
                Console.WriteLine($"    #{i + 1}  {r.Score:F4}  [{r.Source}]  \"{r.Content[..Math.Min(100, r.Content.Length)]}...\"");
                Console.WriteLine($"          ChunkId={r.ChunkId}  Title={r.Title}");
            }

            if (ctx.ShowRoundTrip && useFrame)
            {
                Console.WriteLine("\n  Round-trip:");
                var matrix = frame.ToEmbeddingMatrix("Embedding");
                var rebuilt = matrix.ToEmbeddingFrame("Embedding");
                Console.WriteLine($"    frame rows={frame.RowCount}, matrix cols={matrix.ColumnCount}, rebuilt rows={rebuilt.RowCount}");
                try
                {
                    var gen = frame.ToGeneratedEmbeddings("Embedding", nullReplacement: (row, dim) => new float[dim]);
                    Console.WriteLine($"    ToGeneratedEmbeddings count={gen.Count}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"    ToGeneratedEmbeddings: {ex.Message}");
                }
            }

            if (chatClient != null && results.Count > 0)
            {
                var context = string.Join("\n\n", results.Select(r => r.Content));
                var prompt = $"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:";

                sw.Restart();
                var response = await chatClient.GetResponseAsync(prompt, new ChatOptions { ModelId = ctx.ModelName });
                var llmMs = sw.ElapsedMilliseconds;

                Console.WriteLine($"\n  LLM response ({llmMs} ms):");
                Console.WriteLine($"  {response.Text}");
                Console.WriteLine();
            }
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