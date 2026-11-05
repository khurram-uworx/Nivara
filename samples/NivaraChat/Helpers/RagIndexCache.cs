using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Nivara;
using Nivara.AI;
using Nivara.IO;
using NivaraChat.Modes;

namespace NivaraChat.Helpers;

public static class RagIndexCache
{
    const int LockTimeoutMilliseconds = 2000;

    public sealed record Manifest(string Backend, int Dimension, int ChunkCount, DateTime DocsMaxMtime, string DocsDir);

    public static string ComputeHash(string docsDir, bool useSilk)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(docsDir + "|" + (useSilk ? "silk" : "cpu")));
        return Convert.ToHexString(bytes)[..12].ToLowerInvariant();
    }

    public static string DefaultPath(string docsDir, bool useSilk)
    {
        var hash = ComputeHash(docsDir, useSilk);
        var dir = Path.Combine(ModeHelpers.GetRepoRoot(), "samples", "data", "nivarachat", "rag_cache");
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, $"rag_index_{hash}.parquet");
    }

    public static string ResolvePath(ModeContext ctx, string docsDir)
    {
        if (!string.IsNullOrEmpty(ctx.Index)) return ctx.Index!;
        return DefaultPath(docsDir, ctx.UseSilk);
    }

    public static bool IsValid(string indexPath, string docsDir, List<string> mdFiles, bool useSilk)
    {
        if (!File.Exists(indexPath)) return false;
        var manifestPath = indexPath + ".manifest.json";
        if (!File.Exists(manifestPath)) return false;

        var lockPath = indexPath + ".lock";
        if (!TryAcquireLock(lockPath, out var lockStream))
            return false;
        try
        {
            var m = JsonSerializer.Deserialize<Manifest>(File.ReadAllText(manifestPath));
            if (m == null) return false;
            if (m.Backend != (useSilk ? "silk" : "cpu")) return false;
            if (!string.Equals(m.DocsDir, docsDir, StringComparison.Ordinal)) return false;
            var maxMtime = mdFiles.Count > 0 ? mdFiles.Max(f => File.GetLastWriteTimeUtc(f)) : DateTime.MinValue;
            if (m.DocsMaxMtime != maxMtime) return false;
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

    public static void Save(string indexPath, NivaraFrame vectorFrame, string docsDir, List<string> mdFiles, bool useSilk, int dimension)
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
        var manifest = new Manifest(useSilk ? "silk" : "cpu", dimension, vectorFrame.RowCount, maxMtime, docsDir);

        var directory = Path.GetDirectoryName(indexPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var lockPath = indexPath + ".lock";
        if (!TryAcquireLock(lockPath, out var lockStream))
            return;

        using (lockStream)
        {
            var manifestPath = indexPath + ".manifest.json";
            var tempIndexPath = TempPath(indexPath);
            var tempManifestPath = TempPath(manifestPath);
            try
            {
                saveFrame.ToParquet(tempIndexPath);
                File.WriteAllText(tempManifestPath, JsonSerializer.Serialize(manifest));
                File.Move(tempIndexPath, indexPath, overwrite: true);
                File.Move(tempManifestPath, manifestPath, overwrite: true);
            }
            finally
            {
                TryDelete(tempIndexPath);
                TryDelete(tempManifestPath);
            }
        }
    }

    public static NivaraFrame Load(string indexPath, out int dimension)
    {
        var lockPath = indexPath + ".lock";
        if (!TryAcquireLock(lockPath, out var lockStream))
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

    static bool TryAcquireLock(string lockPath, out FileStream? lockStream)
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(LockTimeoutMilliseconds);
        while (true)
        {
            try
            {
                lockStream = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
                return true;
            }
            catch (IOException)
            {
                if (DateTime.UtcNow >= deadline)
                {
                    lockStream = null;
                    return false;
                }
            }
            catch (UnauthorizedAccessException)
            {
                lockStream = null;
                return false;
            }

            Thread.Sleep(25);
        }
    }

    static string TempPath(string path) => path + "." + Guid.NewGuid().ToString("N") + ".tmp";

    static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
        }
    }
}
