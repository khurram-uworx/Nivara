using Nivara;
using Nivara.AI;
using Nivara.IO;
using NivaraChat.Modes;
using System.Security.Cryptography;
using System.Text;

namespace NivaraChat.Helpers;

public static class RagIndexCache
{
    /// <summary>Current manifest schema version. Bump on any cache-format change.</summary>
    public const int SchemaVersion = 1;
    public const int CurrentSchemaVersion = SchemaVersion;
    public const int ManifestSchemaVersion = SchemaVersion;

    public sealed record Manifest(
        string Backend,
        int Dimension,
        int ChunkCount,
        DateTime DocsMaxMtime,
        string DocsDir,
        string ContentHash = "",
        int SchemaVersion = 0,
        int ChunkSize = 500);
    public static string ComputeHash(string docsDir, bool useSilk, int chunkSize = 500)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(docsDir + "|" + (useSilk ? "silk" : "cpu") + "|" + chunkSize));
        return Convert.ToHexString(bytes)[..12].ToLowerInvariant();
    }

    public static string ComputeContentHash(IEnumerable<string> mdFiles)
    {
        using var sha = SHA256.Create();
        foreach (var f in mdFiles ?? [])
        {
            var bytes = File.ReadAllBytes(f);
            if (bytes.Length > 0)
                sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
        }
        sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
        return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
    }

    public static string DefaultPath(string docsDir, bool useSilk, int chunkSize = 500)
    {
        var hash = ComputeHash(docsDir, useSilk, chunkSize);
        var dir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "nivarachat", "rag_cache");
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, $"rag_index_{hash}.parquet");
    }

    public static string ResolvePath(ModeContext ctx, string docsDir)
    {
        if (!string.IsNullOrEmpty(ctx.Index)) return ctx.Index!;
        return DefaultPath(docsDir, ctx.UseSilk, ctx.ChunkSize);
    }

    public static bool IsValid(string indexPath, string docsDir, List<string> mdFiles, bool useSilk, int chunkSize = 500)
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
            var manifestChunkSize = m.ChunkSize == 0 ? 500 : m.ChunkSize;
            if (manifestChunkSize != chunkSize) return false;
            var maxMtime = mdFiles.Count > 0 ? mdFiles.Max(f => File.GetLastWriteTimeUtc(f)) : DateTime.MinValue;
            if (m.DocsMaxMtime != maxMtime) return false;
            var contentHash = ComputeContentHash(mdFiles);
            if (!string.Equals(m.ContentHash, contentHash, StringComparison.Ordinal)) return false;
            // chunk-count + dimension check: the manifest must agree with the persisted
            // Parquet footer (row count and number of E* dimension columns). Reading only
            // the footer metadata keeps this cheap (no full load).
            if (!IndexCacheFile.TryReadFooter(indexPath, out var rowCount, out var fieldNames))
                return false;
            if (rowCount != m.ChunkCount) return false;
            var dimensionColumns = fieldNames.Count(name =>
                name.Length > 1 && name[0] == 'E' && int.TryParse(name.AsSpan(1), out _));
            return dimensionColumns == m.Dimension;
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

    public static void Save(string indexPath, NivaraFrame vectorFrame, string docsDir, List<string> mdFiles, bool useSilk, int dimension, int chunkSize = 500)
        => TrySave(indexPath, vectorFrame, docsDir, mdFiles, useSilk, dimension, chunkSize);
    public static bool TrySave(string indexPath, NivaraFrame vectorFrame, string docsDir, List<string> mdFiles, bool useSilk, int dimension, int chunkSize = 500)
    {
        var matrix = vectorFrame.ToEmbeddingMatrix("Embedding");
        var cols = new List<(string Name, IColumn)>();
        if (vectorFrame.HasColumn("ChunkId")) cols.Add(("ChunkId", (IColumn)vectorFrame.GetColumn<string>("ChunkId")));
        if (vectorFrame.HasColumn("Source")) cols.Add(("Source", (IColumn)vectorFrame.GetColumn<string>("Source")));
        if (vectorFrame.HasColumn("Title")) cols.Add(("Title", (IColumn)vectorFrame.GetColumn<string>("Title")));
        if (vectorFrame.HasColumn("SectionPath")) cols.Add(("SectionPath", (IColumn)vectorFrame.GetColumn<string>("SectionPath")));
        if (vectorFrame.HasColumn("AclGroup")) cols.Add(("AclGroup", (IColumn)vectorFrame.GetColumn<string>("AclGroup")));
        if (vectorFrame.HasColumn("Content")) cols.Add(("Content", (IColumn)vectorFrame.GetColumn<string>("Content")));
        for (int d = 0; d < dimension; d++)
        {
            var name = $"E{d}";
            if (matrix.HasColumn(name))
                cols.Add((name, (IColumn)matrix.GetColumn<float>(name)));
        }
        var saveFrame = NivaraFrame.Create(cols.ToArray());
        var maxMtime = mdFiles.Count > 0 ? mdFiles.Max(f => File.GetLastWriteTimeUtc(f)) : DateTime.MinValue;
        var contentHash = ComputeContentHash(mdFiles);
        var manifest = new Manifest(useSilk ? "silk" : "cpu", dimension, vectorFrame.RowCount, maxMtime, docsDir, contentHash, SchemaVersion, chunkSize);
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

    public static NivaraFrame Load(string indexPath, out int dimension)
    {
        var lockPath = indexPath + ".lock";
        if (!IndexCacheFile.TryAcquireLock(lockPath, out var lockStream))
            throw new IOException($"Cache is locked by another process: {lockPath}");

        using (lockStream)
        {
            var frame = NivaraFrameIOExtensions.LoadParquet(indexPath);
            // dimension = number of E* columns
            dimension = frame.ColumnNames.Where(c => c.StartsWith("E") && int.TryParse(c.Substring(1), out _)).Count();
            return frame;
        }
    }

    public static List<RagChunk> ToChunks(NivaraFrame loaded, int dimension)
    {
        var chunks = new List<RagChunk>();
        int rows = loaded.RowCount;
        for (int i = 0; i < rows; i++)
        {
            chunks.Add(new RagChunk(
                loaded.GetColumn<string>("ChunkId")[i],
                loaded.GetColumn<string>("Source")[i],
                loaded.GetColumn<string>("Title")[i],
                loaded.GetColumn<string>("SectionPath")[i],
                loaded.GetColumn<string>("AclGroup")[i],
                loaded.GetColumn<string>("Content")[i]));
        }
        return chunks;
    }

    public static float[]?[] ToEmbeddings(NivaraFrame loaded, int dimension)
    {
        float[]?[] result = new float[loaded.RowCount][];
        // reconstruct via ToEmbeddingFrame on a dimension-only frame
        var dimCols = new List<(string Name, IColumn)>();
        for (int d = 0; d < dimension; d++)
            dimCols.Add(($"E{d}", (IColumn)loaded.GetColumn<float>($"E{d}")));
        var dimFrame = NivaraFrame.Create(dimCols.ToArray());
        var vecFrame = dimFrame.ToEmbeddingFrame("Embedding");
        var embCol = vecFrame.GetColumn<float[]>("Embedding");
        for (int i = 0; i < loaded.RowCount; i++)
            result[i] = embCol[i];
        return result;
    }
}
