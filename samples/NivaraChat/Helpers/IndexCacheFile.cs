using System.Text.Json;

namespace NivaraChat.Helpers;

/// <summary>
/// File-level operations shared by the RAG index caches: the cross-process lock, temp-file
/// naming, atomic manifest/index replacement, and a cheap Parquet footer read. Extracted from
/// <see cref="RagIndexCache"/> so a second cache type does not duplicate the concurrency
/// discipline that #920 established. No cache schema or validity rule lives here.
/// </summary>
internal static class IndexCacheFile
{
    internal const int LockTimeoutMilliseconds = 1000;

    static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    /// <summary>Opens an exclusive lock file, retrying briefly so a concurrent writer can finish.</summary>
    internal static bool TryAcquireLock(string lockPath, out FileStream? lockStream)
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

    internal static string TempPath(string path) => path + "." + Guid.NewGuid().ToString("N") + ".tmp";

    internal static void TryDelete(string path)
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

    internal static T? ReadJson<T>(string path) => JsonSerializer.Deserialize<T>(File.ReadAllText(path), JsonOptions);

    internal static void WriteJson<T>(string path, T value) => File.WriteAllText(path, JsonSerializer.Serialize(value));

    /// <summary>
    /// Writes both temporary files under the cache lock, then moves them into place. Returns
    /// <see langword="false"/> when the lock is held by another process.
    /// </summary>
    internal static bool TrySaveAtomically(string indexPath, string manifestPath, Action<string, string> writeTemps)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(indexPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        ArgumentNullException.ThrowIfNull(writeTemps);

        var lockPath = indexPath + ".lock";
        if (!TryAcquireLock(lockPath, out var lockStream))
            return false;

        var tempIndexPath = TempPath(indexPath);
        var tempManifestPath = TempPath(manifestPath);
        using (lockStream)
        {
            try
            {
                writeTemps(tempIndexPath, tempManifestPath);
                File.Move(tempIndexPath, indexPath, overwrite: true);
                File.Move(tempManifestPath, manifestPath, overwrite: true);
                return true;
            }
            finally
            {
                TryDelete(tempIndexPath);
                TryDelete(tempManifestPath);
            }
        }
    }

    /// <summary>
    /// Reads row count and field names from the Parquet footer. Cheap enough for a validity
    /// check because it never loads row data.
    /// </summary>
    internal static bool TryReadFooter(string indexPath, out long rowCount, out IReadOnlyList<string> fieldNames)
    {
        rowCount = 0;
        fieldNames = [];
        try
        {
            var reader = Parquet.ParquetReader.CreateAsync(indexPath, null, CancellationToken.None).GetAwaiter().GetResult();
            try
            {
                rowCount = reader.Metadata!.NumRows;
                fieldNames = [.. reader.Schema!.DataFields.Select(field => field.Name)];
                return true;
            }
            finally
            {
                reader.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }
        catch
        {
            return false;
        }
    }
}
