using Nivara;
using NivaraChat.Helpers;
using NUnit.Framework;

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
            ("Embedding", (IColumn)NivaraColumn<float[]>.CreateForReferenceType(embeddings)));
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
    public void Save_WhenLockHeld_SkipsWithoutThrowing()
    {
        var frame = BuildFrame(new[] { 1f, 0f });
        using var heldLock = new FileStream(
            indexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);

        var start = DateTime.UtcNow;
        Assert.DoesNotThrow(() => RagIndexCache.Save(indexPath, frame, tempDir, [], useSilk: false, dimension: 2));
        var elapsed = DateTime.UtcNow - start;

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
}
