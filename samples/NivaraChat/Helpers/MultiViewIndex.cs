using Nivara;
using Nivara.AI;
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

    public CorpusIdentity CorpusIdentity { get; }

    MultiViewIndex(
        MiniLMEmbedder.Handle embedder,
        List<MultiViewChunk> chunks,
        NivaraFrame metadata,
        MultiViewMatrices matrices,
        CorpusIdentity corpusIdentity)
    {
        this.embedder = embedder;
        this.chunks = chunks;
        this.metadata = metadata;
        this.matrices = matrices;
        viewCounts = s_views.Select(view => CountPresent(metadata, view.Column)).ToArray();
        CorpusIdentity = corpusIdentity;
    }

    public CorpusIdentity Identity => CorpusIdentity;

    /// <summary>Embedding dimension of the loaded model.</summary>
    public int Dimension => embedder.Dimension;

    /// <summary>Whether retrieval runs over the convention frame/matrices (always true for multi-view).</summary>
    public bool UseFrame => true;

    /// <summary>Chunks visible under the configured ACL filter.</summary>
    public IReadOnlyList<MultiViewChunk> Chunks => chunks;

    /// <summary>Number of chunks with a vector for <paramref name="space"/>.</summary>
    public int ViewCount(ViewSpace space) => viewCounts[(int)space];

    /// <summary>Number of distinct source files among the visible chunks.</summary>
    public int FileCount => chunks.Select(chunk => chunk.SourcePath).Distinct(StringComparer.Ordinal).Count();

    /// <summary>Whether any chunk passes the configured ACL filter.</summary>
    public bool HasVisibleChunks => chunks.Count > 0;

    /// <summary>The four flat <c>[rows, dim]</c> per-view matrices the index scores against.</summary>
    public MultiViewMatrices Matrices => matrices;

    /// <summary>
    /// The convention frame holding chunk metadata, the four <c>float[]</c> vector columns and the
    /// Content/Title coherence. Scoring and training both need the raw rows, not just the top-K.
    /// </summary>
    public NivaraFrame Metadata => metadata;

    /// <summary>
    /// Embeds <paramref name="query"/>, throwing instead of returning null when the embedder fails.
    /// </summary>
    /// <remarks>
    /// <see cref="Retrieve"/> swallows the failure deliberately: an interactive search with no query
    /// vector is just an empty result. Training and evaluation cannot do that, because scoring nothing
    /// still produces a plausible-looking metric, so they use this method and fail loudly.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// The embedder threw, returned null, or returned a vector of the wrong dimension.
    /// </exception>
    public float[] EmbedQuery(string query)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);

        float[]? vector;
        try
        {
            vector = embedder.Embed(query);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to embed the query '{query}'.", ex);
        }

        if (vector is null || vector.Length != embedder.Dimension)
        {
            throw new InvalidOperationException(
                $"Embedding the query '{query}' returned {(vector is null ? "null" : $"{vector.Length} dimensions")}; " +
                $"expected {embedder.Dimension}.");
        }

        return vector;
    }

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
        var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
        var mdFiles = MultiViewIndex.DiscoverCorpusFiles(resolvedDocsDir, readmePath).ToArray();

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

            var contentHash = RagIndexCache.ComputeContentHash(mdFiles, repoRoot);
            var docsDirRelative = resolvedDocsDir.StartsWith(repoRoot, StringComparison.OrdinalIgnoreCase)
                ? Path.GetRelativePath(repoRoot, resolvedDocsDir)
                : resolvedDocsDir;
            var identity = new CorpusIdentity(
                contentHash,
                mdFiles.Length,
                chunks.Count,
                MultiViewIndexCache.ChunkerVersion,
                ctx.ChunkSize,
                docsDirRelative,
                ctx.UseSilk ? "silk" : "cpu",
                embedder.Dimension);

            Console.WriteLine($"  Corpus:      {identity.FileCount} files, {identity.ChunkCount} chunks, chunker {identity.ChunkerVersion}, chunk size {identity.ChunkSize}");
            Console.WriteLine($"  Content hash: {identity.ContentHash}  backend: {identity.Backend}  dimension: {identity.Dimension}");

            return new MultiViewIndex(embedder, chunks, metadata, matrices, identity);
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

    /// <summary>Prints the Content-view frame ↔ matrix ↔ GeneratedEmbeddings round-trip diagnostics.</summary>
    public void PrintRoundTrip()
    {
        Console.WriteLine("\n  Round-trip (Content view):");

        var matrix = metadata.ToEmbeddingMatrix("VectorContent", "ContentE", Dimension);
        var rebuilt = matrix.ToEmbeddingFrame("VectorContent", "ContentE");
        var original = metadata.GetColumn<float[]>("VectorContent");
        var restored = rebuilt.GetColumn<float[]>("VectorContent");

        var mismatches = 0;
        for (var row = 0; row < metadata.RowCount; row++)
        {
            var left = original.IsNull(row) ? null : original[row];
            var right = restored.IsNull(row) ? null : restored[row];
            if (!VectorsEqual(left, right))
                mismatches++;
        }

        Console.WriteLine($"    frame rows={metadata.RowCount}, matrix cols={matrix.ColumnCount}, rebuilt rows={rebuilt.RowCount}, mismatches={mismatches}");

        try
        {
            var hasNulls = false;
            for (var row = 0; row < original.Length; row++)
            {
                if (original.IsNull(row))
                {
                    hasNulls = true;
                    break;
                }
            }

            NullEmbeddingReplacement? policy = hasNulls ? ((row, dim) => new float[dim]) : null;
            var generated = metadata.ToGeneratedEmbeddings("VectorContent", policy, Dimension);
            var roundTrippedFrame = generated.ToNivaraFrame("VectorContent");
            var roundTripped = roundTrippedFrame.GetColumn<float[]>("VectorContent");

            var genMismatches = 0;
            for (var row = 0; row < metadata.RowCount; row++)
            {
                var left = original.IsNull(row) ? null : original[row];
                var right = roundTripped.IsNull(row) ? null : roundTripped[row];
                if (!VectorsEqual(left, right))
                    genMismatches++;
            }

            Console.WriteLine($"    ToGeneratedEmbeddings count={generated.Count}, round-trip mismatches={genMismatches}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"    ToGeneratedEmbeddings: {ex.Message}");
        }
    }

    static bool VectorsEqual(float[]? left, float[]? right)
    {
        if (left is null || right is null)
            return left is null && right is null;
        if (left.Length != right.Length)
            return false;
        for (var i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i])
                return false;
        }
        return true;
    }

    public static IReadOnlyList<string> DiscoverCorpusFiles(string docsDir, string readmePath)
    {
        var files = new List<string>();
        if (Directory.Exists(docsDir))
        {
            files.AddRange(Directory
                .GetFiles(docsDir, "*.md", SearchOption.AllDirectories)
                .Where(path => !DocCorpus.IsPlanDocument(docsDir, path)));
        }

        if (File.Exists(readmePath))
            files.Add(readmePath);

        files.Sort(StringComparer.Ordinal);
        return files;
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

/// <summary>Stable identity of a built corpus + chunker + backend.</summary>
public sealed record CorpusIdentity(
    string ContentHash,
    int FileCount,
    int ChunkCount,
    string ChunkerVersion,
    int ChunkSize,
    string DocsDir,
    string Backend,
    int Dimension);
