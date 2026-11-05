using Microsoft.Extensions.VectorData;
using Nivara;
using Nivara.AI;
using Nivara.Samples;
using NivaraChat.Modes;
using System.Diagnostics;

namespace NivaraChat.Helpers;

/// <summary>
/// A retrieved chunk with the score and metadata the retrieval report prints. Shared by
/// <c>--rag</c> and <c>--rag-agent</c> so both modes report the same thing.
/// </summary>
public sealed record RetrievedChunk(
    string ChunkId,
    string Source,
    string Title,
    string Content,
    double Score);

/// <summary>
/// Pure ranking over the convention frame. Kept separate from <see cref="RagIndex"/> so it can be
/// tested without MiniLM model files.
/// </summary>
public static class RagSearch
{
    /// <summary>
    /// Scores the frame's <c>Embedding</c> column against <paramref name="query"/>, applies the
    /// ACL filter first, and returns the top <paramref name="topK"/> rows in descending score order.
    /// </summary>
    /// <remarks>
    /// The frame rows must line up with <paramref name="chunks"/>. Top-K labels are positional, so a
    /// duplicate <see cref="RagChunk.ChunkId"/> cannot make the reported chunk differ from the
    /// scored row.
    /// </remarks>
    public static IReadOnlyList<RetrievedChunk> Rank(
        NivaraFrame frame,
        IReadOnlyList<RagChunk> chunks,
        float[]? query,
        string? aclGroup,
        int topK)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(chunks);
        if (frame.RowCount != chunks.Count)
            throw new ArgumentException(
                $"Frame row count ({frame.RowCount}) must match the chunk count ({chunks.Count}).",
                nameof(chunks));
        if (topK <= 0)
            return [];

        var workFrame = frame;
        IReadOnlyList<RagChunk> visible = chunks;
        if (!string.IsNullOrEmpty(aclGroup))
        {
            var mask = NivaraColumn<bool>.Create(chunks.Select(c => c.AclGroup == aclGroup).ToArray());
            workFrame = frame.FilterByMask(mask);
            visible = chunks.Where(c => c.AclGroup == aclGroup).ToArray();
        }

        if (visible.Count == 0)
            return [];

        var scores = workFrame.RowCosineSimilarity("Embedding", query);
        var top = scores.TopKDescending(topK);
        var results = new List<RetrievedChunk>(top.Length);
        foreach (var (label, score) in top)
        {
            if (!int.TryParse(label, out var row) || row < 0 || row >= visible.Count)
                continue;
            var chunk = visible[row];
            results.Add(new RetrievedChunk(chunk.ChunkId, chunk.Source, chunk.Title, chunk.Content, score));
        }
        return results;
    }
}

/// <summary>
/// Owns the chunk/embed/cache pipeline shared by <c>--rag</c> and <c>--rag-agent</c>, plus the
/// retrieval report and round-trip diagnostics. One implementation so the two modes cannot drift
/// again.
/// </summary>
public sealed class RagIndex : IDisposable
{
    readonly MiniLMEmbedder.Handle embedder;
    readonly List<RagChunk> chunks;
    readonly NivaraFrame? frame;
    readonly VectorStoreCollection<string, DocumentChunk>? collection;
    readonly CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore;
    readonly string? aclGroup;
    readonly int topK;

    RagIndex(
        MiniLMEmbedder.Handle embedder,
        List<RagChunk> chunks,
        NivaraFrame? frame,
        VectorStoreCollection<string, DocumentChunk>? collection,
        CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore,
        string? aclGroup,
        int topK)
    {
        this.embedder = embedder;
        this.chunks = chunks;
        this.frame = frame;
        this.collection = collection;
        this.vectorStore = vectorStore;
        this.aclGroup = aclGroup;
        this.topK = topK;
    }

    /// <summary>Whether retrieval runs over the convention frame (the default) rather than the legacy store.</summary>
    public bool UseFrame => frame != null;

    /// <summary>Embedding dimension of the loaded model.</summary>
    public int Dimension => embedder.Dimension;

    /// <summary>Whether any chunk passes the configured ACL filter. Always true when no group is set.</summary>
    public bool HasVisibleChunks => string.IsNullOrEmpty(aclGroup) || chunks.Any(c => c.AclGroup == aclGroup);

    /// <summary>
    /// Loads MiniLM, chunks the corpus, and either loads the parquet cache or embeds and saves it.
    /// Returns <see langword="null"/> after printing the reason when the model or corpus is missing.
    /// </summary>
    public static async Task<RagIndex?> BuildAsync(ModeContext ctx, CancellationToken ct = default)
    {
        var minilmDir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "minilm");
        if (!Directory.Exists(minilmDir))
        {
            Console.WriteLine($"MiniLM model not found at: {minilmDir}");
            Console.WriteLine("Download model files (model.safetensors, config.json, vocab.txt) to that directory.");
            return null;
        }

        Console.WriteLine("Loading MiniLM embedding model...");
        var sw = Stopwatch.StartNew();
        var embedder = MiniLMEmbedder.Create(minilmDir, ctx.UseSilk);
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
            embedder.Dispose();
            return null;
        }

        var allChunks = new List<RagChunk>();
        foreach (var f in mdFiles)
        {
            var content = await File.ReadAllTextAsync(f, ct);
            allChunks.AddRange(RagChunker.ChunkMarkdown(content, f, repoRoot, ctx.ChunkSize));
        }

        Console.WriteLine($"Chunked {allChunks.Count} chunks from {mdFiles.Length} files (store={ctx.Store})...");
        sw.Restart();

        bool useFrame = !string.Equals(ctx.Store, "inmemory", StringComparison.OrdinalIgnoreCase);
        var indexPath = RagIndexCache.ResolvePath(ctx, resolvedDocsDir);
        bool useCache = !ctx.NoCache && useFrame;
        float[]?[] embeddings = new float[allChunks.Count][];
        NivaraFrame? frame = null;
        CommunityToolkit.VectorData.InMemory.InMemoryVectorStore? vectorStore = null;
        VectorStoreCollection<string, DocumentChunk>? collection = null;

        try
        {
            if (useCache && !ctx.Rebuild && RagIndexCache.IsValid(indexPath, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk, ctx.ChunkSize))
            {
                Console.WriteLine($"Loading cached index: {indexPath}");
                sw.Restart();
                var loaded = RagIndexCache.Load(indexPath, out int dim);
                allChunks = RagIndexCache.ToChunks(loaded, dim);
                embeddings = RagIndexCache.ToEmbeddings(loaded, dim);
                frame = BuildFrame(allChunks, embeddings);
                sw.Stop();
                Console.WriteLine($"  Loaded {allChunks.Count} chunks in {sw.ElapsedMilliseconds} ms\n");
            }
            else
            {
                Console.WriteLine($"Indexing {allChunks.Count} chunks from {mdFiles.Length} files (store={ctx.Store})...");
                sw.Restart();

                int batchSize = Math.Max(1, ctx.BatchSize);
                for (int start = 0; start < allChunks.Count; start += batchSize)
                {
                    int count = Math.Min(batchSize, allChunks.Count - start);
                    var texts = new string[count];
                    for (int j = 0; j < count; j++)
                        texts[j] = allChunks[start + j].Content;

                    float[][]? batch = null;
                    try { batch = embedder.EmbedBatch(texts); }
                    catch { batch = null; }

                    for (int j = 0; j < count; j++)
                    {
                        int i = start + j;
                        if (batch is { } values && j < values.Length && values[j] is { } vector)
                        {
                            embeddings[i] = vector;
                            continue;
                        }

                        var content = allChunks[i].Content;
                        if (string.IsNullOrWhiteSpace(content))
                        {
                            embeddings[i] = null;
                            continue;
                        }
                        try { embeddings[i] = embedder.Embed(content); }
                        catch { embeddings[i] = null; }
                    }

                    if ((start + count) % 50 == 0 || start + count == allChunks.Count)
                        Console.Write($"\r  Embedded {start + count}/{allChunks.Count} chunks...   ");
                }
                Console.WriteLine();

                if (useFrame)
                {
                    frame = BuildFrame(allChunks, embeddings);
                }
                else
                {
                    var generator = MiniLMEmbeddingGenerator.Create(minilmDir);
                    vectorStore = new CommunityToolkit.VectorData.InMemory.InMemoryVectorStore(new() { EmbeddingGenerator = generator });
                    collection = vectorStore.GetCollection<string, DocumentChunk>("nivaradocs");
                    await collection.EnsureCollectionExistsAsync(ct);
                    foreach (var chunk in allChunks)
                        await collection.UpsertAsync(new DocumentChunk { Key = chunk.ChunkId, Text = chunk.Content, Source = chunk.Source }, ct);
                }

                if (useCache && frame != null)
                {
                    try
                    {
                        if (RagIndexCache.TrySave(indexPath, frame, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk, embedder.Dimension, ctx.ChunkSize))
                            Console.WriteLine($"  Cached index to {indexPath}");
                        else
                            Console.WriteLine("  Cache save skipped: another process is writing the index.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Cache save failed: {ex.Message}");
                    }
                }

                sw.Stop();
                Console.WriteLine($"  Indexed in {sw.ElapsedMilliseconds} ms\n");
            }
        }
        catch
        {
            embedder.Dispose();
            throw;
        }

        return new RagIndex(embedder, allChunks, frame, collection, vectorStore, ctx.AclGroup, ctx.TopK);
    }

    /// <summary>Retrieves the top-K chunks for a query and reports how long retrieval took.</summary>
    public async Task<(IReadOnlyList<RetrievedChunk> Results, long ElapsedMs)> RetrieveAsync(
        string query, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        IReadOnlyList<RetrievedChunk> results;

        if (frame != null)
        {
            float[]? queryVector = null;
            try { queryVector = embedder.Embed(query); }
            catch { queryVector = null; }
            results = RagSearch.Rank(frame, chunks, queryVector, aclGroup, topK);
        }
        else
        {
            var found = new List<RetrievedChunk>();
            await foreach (var r in collection!.SearchAsync(query, topK, cancellationToken: ct))
            {
                var chunk = chunks.FirstOrDefault(c => c.ChunkId == r.Record.Key);
                found.Add(new RetrievedChunk(
                    r.Record.Key,
                    r.Record.Source,
                    chunk?.Title ?? r.Record.Source,
                    r.Record.Text,
                    r.Score ?? 0));
            }
            results = found;
        }

        return (results, sw.ElapsedMilliseconds);
    }

    /// <summary>Prints the retrieval report shared by both modes.</summary>
    public static void PrintRetrieved(IReadOnlyList<RetrievedChunk> results, long elapsedMs)
    {
        Console.WriteLine($"\n  Retrieved {results.Count} chunks ({elapsedMs} ms):");
        for (int i = 0; i < results.Count; i++)
        {
            var r = results[i];
            Console.WriteLine($"    #{i + 1}  {r.Score:F4}  [{r.Source}]  \"{r.Content[..Math.Min(100, r.Content.Length)]}...\"");
            Console.WriteLine($"          ChunkId={r.ChunkId}  Title={r.Title}");
        }
    }

    /// <summary>Prints the frame ↔ matrix ↔ GeneratedEmbeddings round-trip diagnostics.</summary>
    public void PrintRoundTrip()
    {
        if (frame == null)
            return;

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

    public void Dispose()
    {
        embedder.Dispose();
        vectorStore?.Dispose();
    }

    static NivaraFrame BuildFrame(List<RagChunk> chunks, float[]?[] embeddings)
    {
        var cols = new List<(string Name, IColumn)>
        {
            ("ChunkId", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.ChunkId).ToArray())),
            ("Source", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.Source).ToArray())),
            ("Title", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.Title).ToArray())),
            ("SectionPath", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.SectionPath).ToArray())),
            ("AclGroup", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.AclGroup).ToArray())),
            ("Content", (IColumn)NivaraColumn<string>.Create(chunks.Select(c => c.Content).ToArray())),
            ("Embedding", (IColumn)NivaraColumn<float[]>.CreateForReferenceType(embeddings.Select(e => e).ToArray()!)),
        };
        return NivaraFrame.Create(cols.ToArray());
    }
}
