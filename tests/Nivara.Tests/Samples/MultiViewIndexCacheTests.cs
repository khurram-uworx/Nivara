using Nivara;
using NivaraChat.Helpers;
using NivaraChat.Modes;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the #914 multi-view cache: exact round-trip of the four views and their metadata, a
/// separate schema and file hash from the single-view cache, per-field manifest invalidation, and
/// the same lock discipline as #920.
/// </summary>
[TestFixture]
public class MultiViewIndexCacheTests
{
    string tempDir = null!;
    string docsDir = null!;
    string indexPath = null!;
    List<string> mdFiles = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "nivarachat-multiview-cache-" + Guid.NewGuid().ToString("N"));
        docsDir = Path.Combine(tempDir, "docs");
        Directory.CreateDirectory(docsDir);
        mdFiles = [Path.Combine(docsDir, "a.md"), Path.Combine(docsDir, "b.md")];
        File.WriteAllText(mdFiles[0], "# A\n\nAlpha.");
        File.WriteAllText(mdFiles[1], "# B\n\nBeta.");
        indexPath = Path.Combine(tempDir, "rag_multiview_test.parquet");
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, recursive: true);
    }

    static float[] Vector(int dimension, int seed)
    {
        var vector = new float[dimension];
        for (var d = 0; d < dimension; d++)
            vector[d] = (seed + 1) * 0.25f + d * 0.5f;
        return vector;
    }

    static (NivaraFrame Metadata, MultiViewMatrices Matrices, List<MultiViewChunk> Chunks) Sample(
        int dimension = 3,
        bool allNullCode = false)
    {
        var chunks = new List<MultiViewChunk>
        {
            new("docs/a.md#0", "a.md", "docs/a.md", "A", "A > One", "a", "content 0", "var x = 1;"),
            new("docs/a.md#1", "a.md", "docs/a.md", "A", "A > Two", "a", "content 1", null),
            new("docs/b.md#0", "b.md", "docs/b.md", "B", "B", "docs", "content 2", null)
        };

        float[]?[] content = [Vector(dimension, 0), Vector(dimension, 1), Vector(dimension, 2)];
        float[]?[] title = [Vector(dimension, 3), null, Vector(dimension, 4)];
        float[]?[] section = [Vector(dimension, 5), Vector(dimension, 6), Vector(dimension, 7)];
        float[]?[] code = allNullCode
            ? [null, null, null]
            : [Vector(dimension, 8), null, null];

        var metadata = MultiViewScoring.BuildFrame(chunks, content, title, section, code);
        return (metadata, MultiViewMatrices.Create(metadata, dimension), chunks);
    }

    static bool IsValid(string indexPath, string docsDir, List<string> mdFiles, bool useSilk = false) =>
        MultiViewIndexCache.IsValid(indexPath, docsDir, mdFiles, useSilk);

    [Test]
    public void TrySave_ThenLoad_RoundTripsMetadataAndEveryView()
    {
        var (metadata, matrices, chunks) = Sample();

        var saved = MultiViewIndexCache.TrySave(
            indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);

        var loaded = MultiViewIndexCache.Load(indexPath, out var dimension);

        Assert.Multiple(() =>
        {
            Assert.That(saved, Is.True);
            Assert.That(dimension, Is.EqualTo(3));
            Assert.That(loaded.Chunks, Is.EqualTo(chunks));
            Assert.That(loaded.Content, Has.Count.EqualTo(3));
            Assert.That(loaded.Content[0], Is.EqualTo(Vector(3, 0)));
            Assert.That(loaded.Content[2], Is.EqualTo(Vector(3, 2)));
            Assert.That(loaded.Title[1], Is.Null);
            Assert.That(loaded.Title[2], Is.EqualTo(Vector(3, 4)));
            Assert.That(loaded.Section[1], Is.EqualTo(Vector(3, 6)));
            Assert.That(loaded.Code[0], Is.EqualTo(Vector(3, 8)));
            Assert.That(loaded.Code[1], Is.Null);
        });
    }

    [Test]
    public void IsValid_FreshCache_IsTrue()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.True);
    }

    [Test]
    public void IsValid_MissingIndexOrManifest_IsFalse()
    {
        Assert.Multiple(() =>
        {
            Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False);
        });

        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);
        File.Delete(indexPath + ".manifest.json");

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False);
    }

    [Test]
    public void IsValid_AnyManifestFieldMismatch_IsFalse()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);

        var manifestPath = indexPath + ".manifest.json";
        var originalJson = File.ReadAllText(manifestPath);
        var original = JsonSerializer.Deserialize<MultiViewIndexCache.Manifest>(originalJson)!;

        void AssertInvalid(string field, MultiViewIndexCache.Manifest mutated)
        {
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(mutated));
            try
            {
                Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False, $"manifest field '{field}' must invalidate the cache");
            }
            finally
            {
                File.WriteAllText(manifestPath, originalJson);
            }
        }

        AssertInvalid(nameof(original.Backend), original with { Backend = "silk" });
        AssertInvalid(nameof(original.DocsDir), original with { DocsDir = "/elsewhere" });
        AssertInvalid(nameof(original.SchemaVersion), original with { SchemaVersion = original.SchemaVersion + 1 });
        AssertInvalid(nameof(original.ChunkerVersion), original with { ChunkerVersion = "other-chunker" });
        AssertInvalid(nameof(original.ChunkSize), original with { ChunkSize = original.ChunkSize + 1 });
        AssertInvalid(nameof(original.ChunkCount), original with { ChunkCount = original.ChunkCount + 1 });
        AssertInvalid(nameof(original.Dimension), original with { Dimension = original.Dimension + 1 });
        AssertInvalid(nameof(original.ContentHash), original with { ContentHash = "deadbeef" });
        AssertInvalid(nameof(original.DocsMaxMtime), original with { DocsMaxMtime = original.DocsMaxMtime.AddMinutes(1) });

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.True, "restoring the manifest must restore validity");
    }

    [Test]
    public void IsValid_ChangedDocumentContent_IsFalse()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);

        File.AppendAllText(mdFiles[0], "\n\nChanged.");

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False);
    }

    [Test]
    public void IsValid_TouchedDocumentTimestamp_IsFalse()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);
        var mtime = File.GetLastWriteTimeUtc(mdFiles[1]);

        File.SetLastWriteTimeUtc(mdFiles[1], mtime.AddMinutes(1));

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False);
    }

    [Test]
    public void IsValid_SilkCacheIsNotValidForCpu()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: true, dimension: 3);

        Assert.Multiple(() =>
        {
            Assert.That(IsValid(indexPath, docsDir, mdFiles, useSilk: true), Is.True);
            Assert.That(IsValid(indexPath, docsDir, mdFiles, useSilk: false), Is.False);
        });
    }

    [Test]
    public void IsValid_WhenLockIsHeld_ReturnsFalseInsteadOfBlocking()
    {
        var (metadata, matrices, _) = Sample();
        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);

        using var held = new FileStream(indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        Assert.That(IsValid(indexPath, docsDir, mdFiles), Is.False);
    }

    [Test]
    public void TrySave_WhenLockIsHeld_ReturnsFalse()
    {
        var (metadata, matrices, _) = Sample();

        using var held = new FileStream(indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        Assert.That(
            MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3),
            Is.False);
        Assert.That(File.Exists(indexPath), Is.False);
    }

    [Test]
    public void TrySave_ThenLoad_RoundTripsA384DimensionCache()
    {
        var (metadata, matrices, chunks) = Sample(dimension: 384);

        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 384);
        var loaded = MultiViewIndexCache.Load(indexPath, out var dimension);

        Assert.Multiple(() =>
        {
            Assert.That(dimension, Is.EqualTo(384));
            Assert.That(loaded.Chunks, Is.EqualTo(chunks));
            Assert.That(loaded.Content[0], Has.Length.EqualTo(384));
            Assert.That(loaded.Content[1], Is.EqualTo(Vector(384, 1)));
            Assert.That(loaded.Code[0], Is.EqualTo(Vector(384, 8)));
        });
    }

    [Test]
    public void Load_AllNullCodeView_StillLoadsAtTheManifestDimension()
    {
        var (metadata, matrices, chunks) = Sample(dimension: 3, allNullCode: true);

        MultiViewIndexCache.TrySave(indexPath, metadata, matrices, docsDir, mdFiles, useSilk: false, dimension: 3);
        var loaded = MultiViewIndexCache.Load(indexPath, out var dimension);

        Assert.Multiple(() =>
        {
            Assert.That(dimension, Is.EqualTo(3));
            Assert.That(loaded.Chunks, Is.EqualTo(chunks));
            Assert.That(loaded.Code, Has.All.Null);
            Assert.That(loaded.Content[2], Is.EqualTo(Vector(3, 2)));
        });
    }

    [Test]
    public void ResolvePath_IndexOverride_Wins()
    {
        var ctx = new ModeContext("models", "http://localhost:11434", "model", false, null, 0.5f, docsDir, 5)
        {
            Index = indexPath
        };

        Assert.That(MultiViewIndexCache.ResolvePath(ctx, docsDir), Is.EqualTo(indexPath));
    }

    [Test]
    public void DefaultPath_IsSeparateFromTheSingleViewCache()
    {
        var multiView = MultiViewIndexCache.DefaultPath(docsDir, useSilk: false);
        var singleView = RagIndexCache.DefaultPath(docsDir, useSilk: false);

        Assert.Multiple(() =>
        {
            Assert.That(multiView, Does.Contain("rag_multiview_cache"));
            Assert.That(Path.GetFileName(multiView), Does.StartWith("rag_multiview_"));
            Assert.That(multiView, Is.Not.EqualTo(singleView));
        });
    }

    [Test]
    public void ComputeHash_IsStableAndSensitiveToBackendAndChunkSize()
    {
        var hash = MultiViewIndexCache.ComputeHash(docsDir, useSilk: false, chunkSize: 500);

        Assert.Multiple(() =>
        {
            Assert.That(hash, Has.Length.EqualTo(12));
            Assert.That(hash, Is.EqualTo(MultiViewIndexCache.ComputeHash(docsDir, useSilk: false, chunkSize: 500)));
            Assert.That(hash, Is.Not.EqualTo(MultiViewIndexCache.ComputeHash(docsDir, useSilk: true, chunkSize: 500)));
            Assert.That(hash, Is.Not.EqualTo(MultiViewIndexCache.ComputeHash(docsDir, useSilk: false, chunkSize: 600)));
            Assert.That(hash, Is.Not.EqualTo(RagIndexCache.ComputeHash(docsDir, useSilk: false, chunkSize: 500)));
        });
    }
}
