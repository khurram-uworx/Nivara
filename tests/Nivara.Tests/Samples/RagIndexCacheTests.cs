using Nivara;
using Nivara.AI;
using NivaraChat.Helpers;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the #920 concurrency guard: simultaneous saves must leave a loadable cache, and a held
/// lock must make writers skip and readers report the cache as unavailable instead of hanging.
/// </summary>
[TestFixture]
public class RagIndexCacheTests
{
    string tempDir = null!;
    string indexPath = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "nivarachat-rag-cache-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        indexPath = Path.Combine(tempDir, "rag_index_test.parquet");
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, recursive: true);
    }

    static NivaraFrame BuildFrame(params float[][] embeddings)
    {
        var count = embeddings.Length;
        var ids = Enumerable.Range(0, count).Select(i => $"c{i}").ToArray();
        var sources = Enumerable.Range(0, count).Select(i => $"s{i}").ToArray();
        var titles = Enumerable.Range(0, count).Select(i => $"t{i}").ToArray();
        var contents = Enumerable.Range(0, count).Select(i => $"content {i}").ToArray();

        return NivaraFrame.Create(
            ("ChunkId", (IColumn)NivaraColumn<string>.Create(ids)),
            ("Source", (IColumn)NivaraColumn<string>.Create(sources)),
            ("Title", (IColumn)NivaraColumn<string>.Create(titles)),
            ("SectionPath", (IColumn)NivaraColumn<string>.Create(titles)),
            ("AclGroup", (IColumn)NivaraColumn<string>.Create(ids.Select(_ => "docs").ToArray())),
            ("Content", (IColumn)NivaraColumn<string>.Create(contents)),
            ("Embedding", (IColumn)NivaraColumn<float[]>.Create(embeddings)));
    }

    [Test]
    public void Save_ConcurrentWriters_LeavesValidLoadableCache()
    {
        var frame = BuildFrame(new[] { 1f, 0f }, new[] { 0f, 1f });
        var mdFiles = new List<string>();

        var tasks = Enumerable.Range(0, 4)
            .Select(_ => Task.Run(() => RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2)))
            .ToArray();

        Assert.That(Task.WaitAll(tasks, TimeSpan.FromSeconds(15)), Is.True, "Concurrent saves deadlocked.");

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.True);

        var loaded = RagIndexCache.Load(indexPath, out var dimension);
        Assert.That(dimension, Is.EqualTo(2));
        Assert.That(loaded.RowCount, Is.EqualTo(2));
        Assert.That(Directory.GetFiles(tempDir, "*.tmp"), Is.Empty);
    }

    [Test]
    public void TrySave_WhenLockHeld_ReturnsFalseWithoutHanging()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        using var heldLock = new FileStream(
            indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var start = DateTime.UtcNow;
        var saved = RagIndexCache.TrySave(indexPath, frame, tempDir, [], useSilk: false, dimension: 2);
        var elapsed = DateTime.UtcNow - start;

        Assert.That(saved, Is.False);
        Assert.That(elapsed, Is.LessThan(TimeSpan.FromSeconds(5)));
        Assert.That(File.Exists(indexPath), Is.False);
    }

    [Test]
    public void IsValid_WhenLockHeld_ReturnsFalseWithoutHanging()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        using var heldLock = new FileStream(
            indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var start = DateTime.UtcNow;
        var valid = RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false);
        var elapsed = DateTime.UtcNow - start;

        Assert.That(valid, Is.False);
        Assert.That(elapsed, Is.LessThan(TimeSpan.FromSeconds(5)));
    }

    [Test]
    public void Load_WhenLockHeld_ThrowsWithoutHanging()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        RagIndexCache.Save(indexPath, frame, tempDir, [], useSilk: false, dimension: 2);

        using var heldLock = new FileStream(
            indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var start = DateTime.UtcNow;
        Assert.Throws<IOException>(() => RagIndexCache.Load(indexPath, out _));
        var elapsed = DateTime.UtcNow - start;

        Assert.That(elapsed, Is.LessThan(TimeSpan.FromSeconds(5)));
    }

    void RoundTrip(bool useSilk)
    {
        var frame = BuildFrame(new[] { 1f, 0f }, new[] { 0f, 1f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk, dimension: 2);

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk), Is.True);

        var loaded = RagIndexCache.Load(indexPath, out var dimension);
        Assert.That(dimension, Is.EqualTo(2));
        Assert.That(loaded.RowCount, Is.EqualTo(2));

        var chunks = RagIndexCache.ToChunks(loaded, dimension);
        var embeddings = RagIndexCache.ToEmbeddings(loaded, dimension);

        Assert.That(chunks[0].ChunkId, Is.EqualTo("c0"));
        Assert.That(chunks[1].ChunkId, Is.EqualTo("c1"));
        Assert.That(chunks[0].Source, Is.EqualTo("s0"));
        Assert.That(chunks[1].Source, Is.EqualTo("s1"));
        Assert.That(chunks[0].Title, Is.EqualTo("t0"));
        Assert.That(chunks[1].Title, Is.EqualTo("t1"));
        Assert.That(chunks[0].SectionPath, Is.EqualTo("t0"));
        Assert.That(chunks[1].SectionPath, Is.EqualTo("t1"));
        Assert.That(chunks[0].AclGroup, Is.EqualTo("docs"));
        Assert.That(chunks[1].AclGroup, Is.EqualTo("docs"));
        Assert.That(chunks[0].Content, Is.EqualTo("content 0"));
        Assert.That(chunks[1].Content, Is.EqualTo("content 1"));

        Assert.That(embeddings[0], Is.EqualTo(new[] { 1f, 0f }).AsCollection);
        Assert.That(embeddings[1], Is.EqualTo(new[] { 0f, 1f }).AsCollection);
    }

    [Test]
    public void Save_Load_RoundTrips_Exact_Cpu() => RoundTrip(false);

    [Test]
    public void Save_Load_RoundTrips_Exact_Silk() => RoundTrip(true);

    [Test]
    public void Save_Load_NullEmbeddings_SurviveAndExcludedFromTopK()
    {
        var frame = BuildFrame(new float[][] { new[] { 1f, 0f }, null! });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        var loaded = RagIndexCache.Load(indexPath, out var dimension);
        var embeddings = RagIndexCache.ToEmbeddings(loaded, dimension);

        Assert.That(embeddings[0], Is.EqualTo(new[] { 1f, 0f }).AsCollection);
        Assert.That(embeddings[1], Is.Null);

        // null row stays excluded from Top-K
        var vecFrame = loaded.ToEmbeddingFrame("Embedding");
        var scores = vecFrame.RowCosineSimilarity("Embedding", new[] { 1f, 0f });
        var top = scores.TopKDescending(5);
        Assert.That(top.Length, Is.EqualTo(1));
    }

    [Test]
    public void IsValid_MtimeBump_ReturnsFalse()
    {
        var file = Path.Combine(tempDir, "doc.md");
        File.WriteAllText(file, "hello");
        var mdFiles = new List<string> { file };

        var frame = BuildFrame(new[] { 1f, 0f });
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.True);

        File.SetLastWriteTimeUtc(file, File.GetLastWriteTimeUtc(file).AddMinutes(1));
        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.False);
    }

    [Test]
    public void IsValid_BackendMismatch_ReturnsFalse()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: true), Is.False);
    }

    [Test]
    public void IsValid_BackendMismatch_SilkSaved_CpuQueried_ReturnsFalse()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: true, dimension: 2);

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.False);
    }

    [Test]
    public void IsValid_BadManifest_ReturnsFalse()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        File.WriteAllText(indexPath + ".manifest.json", "{ not json");
        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.False);
    }

    [Test]
    public void IsValid_ChunkCountMismatch_ReturnsFalse()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        var manifestPath = indexPath + ".manifest.json";
        var manifest = JsonSerializer.Deserialize<RagIndexCache.Manifest>(File.ReadAllText(manifestPath))!;
        var tampered = manifest with { ChunkCount = manifest.ChunkCount + 1 };
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(tampered));

        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.False);
    }

    [Test]
    public void IsValid_MissingFiles_ReturnsFalse()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        var mdFiles = new List<string>();
        RagIndexCache.Save(indexPath, frame, tempDir, mdFiles, useSilk: false, dimension: 2);

        File.Delete(indexPath);
        Assert.That(RagIndexCache.IsValid(indexPath, tempDir, mdFiles, useSilk: false), Is.False);
    }
}
