using Microsoft.Extensions.AI;
using Nivara;
using Nivara.AI;
using Nivara.IO;
using NivaraChat.Modes;
using System.Security.Cryptography;
using System.Text;

namespace NivaraChat.Helpers;

/// <summary>
/// The multi-view equivalent of <see cref="RagIndexCache"/> (#914): one Parquet holding the chunk
/// metadata plus four prefixed dimension blocks, with its own schema version and a chunker version
/// in both the file hash and the manifest. It is deliberately a separate type: the #912 cache counts
/// columns literally named <c>E&lt;int&gt;</c> and shares that schema with single-view retrieval.
/// </summary>
public static class MultiViewIndexCache
{
    /// <summary>Current cache-format version. Bump on any change to the persisted columns.</summary>
    public const int SchemaVersion = 1;

    /// <summary>
    /// Identifies the chunker's output. Bump whenever <see cref="MultiViewChunker"/> would produce
    /// different chunks for unchanged markdown, because cache validity hashes document content but
    /// not chunker code.
    /// </summary>
    public const string ChunkerVersion = "multiview-1";

    static readonly string[] s_metadataColumns =
    [
        "ChunkId", "Source", "SourcePath", "Title", "Section", "AclGroup", "Content", "Code"
    ];

    static readonly (string Prefix, string VectorColumn)[] s_views =
    [
        ("ContentE", "VectorContent"),
        ("TitleE", "VectorTitle"),
        ("SectionE", "VectorSection"),
        ("CodeE", "VectorCode")
    ];

    public sealed record Manifest(
        string Backend,
        int Dimension,
        int ChunkCount,
        DateTime DocsMaxMtime,
        string DocsDir,
        string ContentHash = "",
        int SchemaVersion = 0,
        int ChunkSize = 500,
        string ChunkerVersion = "");

    /// <summary>Chunks and their four view vectors as loaded from the cache.</summary>
    public sealed record IndexData(
        IReadOnlyList<MultiViewChunk> Chunks,
        IReadOnlyList<float[]?> Content,
        IReadOnlyList<float[]?> Title,
        IReadOnlyList<float[]?> Section,
        IReadOnlyList<float[]?> Code);

    public static string ComputeHash(string docsDir, bool useSilk, int chunkSize = 500)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(
            docsDir + "|" + (useSilk ? "silk" : "cpu") + "|" + chunkSize + "|" + ChunkerVersion));
        return Convert.ToHexString(bytes)[..12].ToLowerInvariant();
    }

    public static string DefaultPath(string docsDir, bool useSilk, int chunkSize = 500)
    {
        var hash = ComputeHash(docsDir, useSilk, chunkSize);
        var dir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "nivarachat", "rag_multiview_cache");
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, $"rag_multiview_{hash}.parquet");
    }

    public static string ResolvePath(ModeContext ctx, string docsDir)
    {
        if (!string.IsNullOrEmpty(ctx.Index)) return ctx.Index!;
        return DefaultPath(docsDir, ctx.UseSilk, ctx.ChunkSize);
    }

    public static bool IsValid(
        string indexPath,
        string docsDir,
        List<string> mdFiles,
        bool useSilk,
        int chunkSize = 500,
        int? expectedDimension = null)
    {
        if (!File.Exists(indexPath)) return false;
        var manifestPath = indexPath + ".manifest.json";
        if (!File.Exists(manifestPath)) return false;

        var lockPath = indexPath + ".lock";
        if (!IndexCacheFile.TryAcquireLock(lockPath, out var lockStream))
            return false;

        try
        {
            var m = IndexCacheFile.ReadJson<Manifest>(manifestPath);
            if (m == null) return false;
            if (m.Backend != (useSilk ? "silk" : "cpu")) return false;
            if (!string.Equals(m.DocsDir, docsDir, StringComparison.Ordinal)) return false;
            if (m.SchemaVersion != SchemaVersion) return false;
            if (!string.Equals(m.ChunkerVersion, ChunkerVersion, StringComparison.Ordinal)) return false;
            if (expectedDimension is { } expected && m.Dimension != expected) return false;
            var manifestChunkSize = m.ChunkSize == 0 ? 500 : m.ChunkSize;
            if (manifestChunkSize != chunkSize) return false;
            var maxMtime = mdFiles.Count > 0 ? mdFiles.Max(f => File.GetLastWriteTimeUtc(f)) : DateTime.MinValue;
            if (m.DocsMaxMtime != maxMtime) return false;
            var contentHash = RagIndexCache.ComputeContentHash(mdFiles, Directory.GetParent(docsDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))?.FullName);
            if (!string.Equals(m.ContentHash, contentHash, StringComparison.Ordinal)) return false;

            // The manifest must agree with the persisted footer in both directions: the row count,
            // and one dimension block per view. Reading only the footer keeps this cheap.
            if (!IndexCacheFile.TryReadFooter(indexPath, out var rowCount, out var fieldNames))
                return false;
            if (rowCount != m.ChunkCount) return false;

            foreach (var (prefix, _) in s_views)
            {
                var dimensions = fieldNames.Count(name =>
                    name.Length > prefix.Length &&
                    name.StartsWith(prefix, StringComparison.Ordinal) &&
                    int.TryParse(name.AsSpan(prefix.Length), out _));
                if (dimensions != m.Dimension) return false;
            }

            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            lockStream?.Dispose();
        }
    }

    /// <summary>
    /// Writes the metadata columns and the four dimension blocks as one Parquet. Returns
    /// <see langword="false"/> when another process holds the cache lock.
    /// </summary>
    public static bool TrySave(
        string indexPath,
        NivaraFrame metadataFrame,
        MultiViewMatrices matrices,
        string docsDir,
        List<string> mdFiles,
        bool useSilk,
        int dimension,
        int chunkSize = 500)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(indexPath);
        ArgumentNullException.ThrowIfNull(metadataFrame);
        ArgumentNullException.ThrowIfNull(matrices);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);

        var columns = new List<(string Name, IColumn Column)>();
        foreach (var name in s_metadataColumns)
            columns.Add((name, (IColumn)metadataFrame.GetColumn<string>(name)));

        var matrixFrame = matrices.ToFrame();
        foreach (var name in matrixFrame.ColumnNames)
            columns.Add((name, (IColumn)matrixFrame.GetColumn<float>(name)));

        var saveFrame = NivaraFrame.Create(columns.ToArray());
        var maxMtime = mdFiles.Count > 0 ? mdFiles.Max(f => File.GetLastWriteTimeUtc(f)) : DateTime.MinValue;
        var contentHash = RagIndexCache.ComputeContentHash(mdFiles, Directory.GetParent(docsDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))?.FullName);
        var manifest = new Manifest(
            useSilk ? "silk" : "cpu",
            dimension,
            metadataFrame.RowCount,
            maxMtime,
            docsDir,
            contentHash,
            SchemaVersion,
            chunkSize,
            ChunkerVersion);

        var directory = Path.GetDirectoryName(indexPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var manifestPath = indexPath + ".manifest.json";
        return IndexCacheFile.TrySaveAtomically(indexPath, manifestPath, (tempIndexPath, tempManifestPath) =>
        {
            saveFrame.ToParquet(tempIndexPath);
            IndexCacheFile.WriteJson(tempManifestPath, manifest);
        });
    }

    /// <summary>
    /// Loads the metadata and all four views. The dimension is counted from the
    /// <c>ContentE*</c> footer fields, so an all-null view still loads.
    /// </summary>
    public static IndexData Load(string indexPath, out int dimension)
    {
        var lockPath = indexPath + ".lock";
        if (!IndexCacheFile.TryAcquireLock(lockPath, out var lockStream))
            throw new IOException($"Cache is locked by another process: {lockPath}");

        using (lockStream)
        {
            var loaded = NivaraFrameIOExtensions.LoadParquet(indexPath);
            if (!IndexCacheFile.TryReadFooter(indexPath, out _, out var fieldNames))
                throw new InvalidOperationException($"Could not read the Parquet footer of {indexPath}.");

            dimension = fieldNames.Count(name =>
                name.Length > "ContentE".Length &&
                name.StartsWith("ContentE", StringComparison.Ordinal) &&
                int.TryParse(name.AsSpan("ContentE".Length), out _));
            if (dimension == 0)
                throw new InvalidOperationException($"Cache {indexPath} has no ContentE* dimension columns.");

            var chunks = ToChunks(loaded);
            return new IndexData(
                chunks,
                Vectors(loaded, "VectorContent", "ContentE", dimension),
                Vectors(loaded, "VectorTitle", "TitleE", dimension),
                Vectors(loaded, "VectorSection", "SectionE", dimension),
                Vectors(loaded, "VectorCode", "CodeE", dimension));
        }
    }

    static List<MultiViewChunk> ToChunks(NivaraFrame loaded)
    {
        var chunkIds = loaded.GetColumn<string>("ChunkId");
        var sources = loaded.GetColumn<string>("Source");
        var sourcePaths = loaded.GetColumn<string>("SourcePath");
        var titles = loaded.GetColumn<string>("Title");
        var sections = loaded.GetColumn<string>("Section");
        var aclGroups = loaded.GetColumn<string>("AclGroup");
        var contents = loaded.GetColumn<string>("Content");
        var codes = loaded.GetColumn<string>("Code");

        var chunks = new List<MultiViewChunk>(loaded.RowCount);
        for (var row = 0; row < loaded.RowCount; row++)
        {
            chunks.Add(new MultiViewChunk(
                chunkIds[row],
                sources[row],
                sourcePaths[row],
                titles[row],
                sections[row],
                aclGroups[row],
                contents[row],
                codes.IsNull(row) ? null : codes[row]));
        }

        return chunks;
    }

    static float[]?[] Vectors(NivaraFrame loaded, string vectorColumn, string prefix, int dimension)
    {
        var vectorFrame = loaded.ToEmbeddingFrame(vectorColumn, prefix, dimension);
        var column = vectorFrame.GetColumn<float[]>(vectorColumn);
        var vectors = new float[]?[column.Length];
        for (var row = 0; row < column.Length; row++)
            vectors[row] = column.IsNull(row) ? null : column[row];

        return vectors;
    }

    /// <summary>
    /// Converts the cached Content view dimension columns back into a <c>float[]?</c> array,
    /// mirroring <see cref="RagIndexCache.ToEmbeddings"/>.
    /// </summary>
    public static float[]?[] ToEmbeddings(NivaraFrame loaded, int dimension)
        => Vectors(loaded, "VectorContent", "ContentE", dimension);

    /// <summary>
    /// Converts the cached dimension columns for one view back into a <c>float[]?</c> array.
    /// </summary>
    public static float[]?[] ToEmbeddings(NivaraFrame loaded, string vectorColumn, string prefix, int dimension)
        => Vectors(loaded, vectorColumn, prefix, dimension);

    /// <summary>
    /// Converts the cached Content view into <see cref="GeneratedEmbeddings{TEmbedding}"/>,
    /// mirroring the <c>--rag --roundtrip</c> demonstration.
    /// </summary>
    public static GeneratedEmbeddings<Embedding<float>> ToGeneratedEmbeddings(
        NivaraFrame loaded,
        int dimension,
        NullEmbeddingReplacement? nullReplacement = null)
    {
        var frame = loaded.ToEmbeddingFrame("VectorContent", "ContentE", dimension);
        return frame.ToGeneratedEmbeddings("VectorContent", nullReplacement, dimension);
    }

    /// <summary>
    /// Converts one cached view into <see cref="GeneratedEmbeddings{TEmbedding}"/>.
    /// </summary>
    public static GeneratedEmbeddings<Embedding<float>> ToGeneratedEmbeddings(
        NivaraFrame loaded,
        string vectorColumn,
        string prefix,
        int dimension,
        NullEmbeddingReplacement? nullReplacement = null)
    {
        var frame = loaded.ToEmbeddingFrame(vectorColumn, prefix, dimension);
        return frame.ToGeneratedEmbeddings(vectorColumn, nullReplacement, dimension);
    }
}
