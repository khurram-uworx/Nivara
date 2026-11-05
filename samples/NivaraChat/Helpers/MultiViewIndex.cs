using Nivara;
using Nivara.Samples;
using NivaraChat.Modes;
using System.Diagnostics;

namespace NivaraChat.Helpers;

/// <summary>
/// Owns the #914 build pipeline: chunk with <see cref="MultiViewChunker"/>, embed the four views
/// (deduplicated, because titles and sections repeat heavily), land them in the convention frame,
/// serve them from <see cref="MultiViewIndexCache"/> when valid, and score queries with
/// <see cref="MultiViewScoring"/>.
/// </summary>
public sealed class MultiViewIndex : IDisposable
{
    static readonly (ViewSpace Space, string Column)[] s_views =
    [
        (ViewSpace.Content, "VectorContent"),
        (ViewSpace.Title, "VectorTitle"),
        (ViewSpace.Section, "VectorSection"),
        (ViewSpace.Code, "VectorCode")
    ];

    readonly MiniLMEmbedder.Handle embedder;
    readonly List<MultiViewChunk> chunks;
    readonly NivaraFrame metadata;
    readonly MultiViewMatrices matrices;
    readonly int[] viewCounts;

    MultiViewIndex(
        MiniLMEmbedder.Handle embedder,
        List<MultiViewChunk> chunks,
        NivaraFrame metadata,
        MultiViewMatrices matrices)
    {
        this.embedder = embedder;
        this.chunks = chunks;
        this.metadata = metadata;
        this.matrices = matrices;
        viewCounts = s_views.Select(view => CountPresent(metadata, view.Column)).ToArray();
    }

    /// <summary>Embedding dimension of the loaded model.</summary>
    public int Dimension => embedder.Dimension;

    /// <summary>Chunks visible under the configured ACL filter.</summary>
    public IReadOnlyList<MultiViewChunk> Chunks => chunks;

    /// <summary>Number of chunks with a vector for <paramref name="space"/>.</summary>
    public int ViewCount(ViewSpace space) => viewCounts[(int)space];

    /// <summary>Number of distinct source files among the visible chunks.</summary>
    public int FileCount => chunks.Select(chunk => chunk.SourcePath).Distinct(StringComparer.Ordinal).Count();

    /// <summary>Whether any chunk passes the configured ACL filter.</summary>
    public bool HasVisibleChunks => chunks.Count > 0;

    /// <summary>
    /// Loads MiniLM, chunks the corpus, then loads or builds the multi-view cache. Returns
    /// <see langword="null"/> after printing the reason when the model or corpus is missing.
    /// </summary>
    public static async Task<MultiViewIndex?> BuildAsync(ModeContext ctx, CancellationToken ct = default)
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

        try
        {
            var indexPath = MultiViewIndexCache.ResolvePath(ctx, resolvedDocsDir);
            var useCache = !ctx.NoCache;
            bool cacheHit = useCache && !ctx.Rebuild &&
                MultiViewIndexCache.IsValid(indexPath, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk, ctx.ChunkSize, embedder.Dimension);

            List<MultiViewChunk> chunks;
            float[]?[] content;
            float[]?[] title;
            float[]?[] section;
            float[]?[] code;

            if (cacheHit)
            {
                Console.WriteLine($"Loading cached multi-view index: {indexPath}");
                sw.Restart();
                var loaded = MultiViewIndexCache.Load(indexPath, out var dimension);
                chunks = [.. loaded.Chunks];
                content = [.. loaded.Content];
                title = [.. loaded.Title];
                section = [.. loaded.Section];
                code = [.. loaded.Code];
                sw.Stop();
                Console.WriteLine($"  Loaded {chunks.Count} chunks ({dimension}-d) in {sw.ElapsedMilliseconds} ms\n");
            }
            else
            {
                chunks = [];
                foreach (var file in mdFiles)
                {
                    var text = await File.ReadAllTextAsync(file, ct);
                    chunks.AddRange(MultiViewChunker.ChunkMarkdown(text, file, repoRoot, ctx.ChunkSize));
                }

                Console.WriteLine($"Chunked {chunks.Count} chunks from {mdFiles.Length} files (chunker {MultiViewIndexCache.ChunkerVersion}).");
                if (chunks.Count == 0)
                {
                    Console.WriteLine($"No multi-view chunks could be produced from {resolvedDocsDir}");
                    embedder.Dispose();
                    return null;
                }

                var texts = new List<string>(chunks.Count * 3);
                var lookup = new Dictionary<string, int>(StringComparer.Ordinal);
                var contentIndexes = new int[chunks.Count];
                var titleIndexes = new int[chunks.Count];
                var sectionIndexes = new int[chunks.Count];
                var codeIndexes = new int[chunks.Count];
                for (var row = 0; row < chunks.Count; row++)
                {
                    contentIndexes[row] = IndexOf(texts, lookup, chunks[row].Content);
                    titleIndexes[row] = IndexOf(texts, lookup, chunks[row].Title);
                    sectionIndexes[row] = IndexOf(texts, lookup, chunks[row].Section);
                    codeIndexes[row] = IndexOf(texts, lookup, chunks[row].Code);
                }

                Console.WriteLine($"  {texts.Count} unique view texts after deduplication (from {chunks.Count * 4} slots).");
                sw.Restart();
                var vectors = EmbedAll(embedder, texts, Math.Max(1, ctx.BatchSize));
                content = Expand(vectors, contentIndexes);
                title = Expand(vectors, titleIndexes);
                section = Expand(vectors, sectionIndexes);
                code = Expand(vectors, codeIndexes);
                sw.Stop();
                Console.WriteLine($"  Embedded in {sw.ElapsedMilliseconds} ms\n");
            }

            var metadata = MultiViewScoring.BuildFrame(chunks, content, title, section, code);
            var matrices = MultiViewMatrices.Create(metadata, embedder.Dimension);

            if (!cacheHit && useCache)
            {
                try
                {
                    if (MultiViewIndexCache.TrySave(
                        indexPath, metadata, matrices, resolvedDocsDir, mdFiles.ToList(), ctx.UseSilk, embedder.Dimension, ctx.ChunkSize))
                        Console.WriteLine($"Cached multi-view index to {indexPath}");
                    else
                        Console.WriteLine("Cache save skipped: another process is writing the multi-view index.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Cache save failed: {ex.Message}");
                }
            }

            if (!string.IsNullOrEmpty(ctx.AclGroup))
            {
                var mask = NivaraColumn<bool>.Create(chunks.Select(c => c.AclGroup == ctx.AclGroup).ToArray());
                metadata = metadata.FilterByMask(mask);
                chunks = [.. chunks.Where(c => c.AclGroup == ctx.AclGroup)];
                matrices = MultiViewMatrices.Create(metadata, embedder.Dimension);
            }

            return new MultiViewIndex(embedder, chunks, metadata, matrices);
        }
        catch
        {
            embedder.Dispose();
            throw;
        }
    }

    /// <summary>Scores the visible chunks for one query and reports how long retrieval took.</summary>
    public (IReadOnlyList<MultiViewHit> Hits, long ElapsedMs) Retrieve(string query, FusionWeights weights, int topK)
    {
        var sw = Stopwatch.StartNew();
        float[]? vector = null;
        try { vector = embedder.Embed(query); }
        catch { vector = null; }

        var hits = MultiViewScoring.Rank(metadata, matrices, chunks, vector, weights, topK);
        sw.Stop();
        return (hits, sw.ElapsedMilliseconds);
    }

    public void Dispose() => embedder.Dispose();

    static int IndexOf(List<string> texts, Dictionary<string, int> lookup, string? text)
    {
        if (text is null)
            return -1;
        if (lookup.TryGetValue(text, out var index))
            return index;

        index = texts.Count;
        texts.Add(text);
        lookup[text] = index;
        return index;
    }

    static float[]?[] Expand(float[]?[] unique, int[] indexes)
    {
        var vectors = new float[]?[indexes.Length];
        for (var row = 0; row < indexes.Length; row++)
        {
            var index = indexes[row];
            vectors[row] = index < 0 ? null : unique[index];
        }

        return vectors;
    }

    static float[]?[] EmbedAll(MiniLMEmbedder.Handle embedder, List<string> texts, int batchSize)
    {
        var vectors = new float[]?[texts.Count];
        for (var start = 0; start < texts.Count; start += batchSize)
        {
            var count = Math.Min(batchSize, texts.Count - start);
            var batch = new string[count];
            for (var j = 0; j < count; j++)
                batch[j] = texts[start + j];

            float[][]? results = null;
            try { results = embedder.EmbedBatch(batch); }
            catch { results = null; }

            for (var j = 0; j < count; j++)
            {
                if (results is not null && j < results.Length && results[j] is { } vector)
                {
                    vectors[start + j] = vector;
                    continue;
                }

                try { vectors[start + j] = embedder.Embed(batch[j]); }
                catch { vectors[start + j] = null; }
            }

            if (start + count == texts.Count || (start + count) % 200 == 0)
                Console.Write($"\r  Embedded {start + count}/{texts.Count} unique texts...   ");
        }

        if (texts.Count > 0)
            Console.WriteLine();
        return vectors;
    }

    static int CountPresent(NivaraFrame frame, string column)
    {
        var vectors = frame.GetColumn<float[]>(column);
        var present = 0;
        for (var row = 0; row < vectors.Length; row++)
        {
            if (!vectors.IsNull(row))
                present++;
        }

        return present;
    }
}
