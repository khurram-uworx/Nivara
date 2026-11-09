using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public class RetrievalReproducibilityTests
{
    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "nivarachat-repro-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, recursive: true);
    }

    static string WriteFile(string dir, string relative, string content)
    {
        var path = Path.Combine(dir, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        return path;
    }

    [Test]
    public void ComputeContentHash_FileOrderDoesNotChangeTheHash()
    {
        var a = WriteFile(tempDir, "a.md", "hello");
        var b = WriteFile(tempDir, "b.md", "world");
        var h1 = RagIndexCache.ComputeContentHash(new[] { a, b }, tempDir);
        var h2 = RagIndexCache.ComputeContentHash(new[] { b, a }, tempDir);
        Assert.That(h2, Is.EqualTo(h1));
    }

    [Test]
    public void ComputeContentHash_RenamedFileChangesTheHash()
    {
        var a = WriteFile(tempDir, "a.md", "hello");
        var h1 = RagIndexCache.ComputeContentHash(new[] { a }, tempDir);
        var renamed = WriteFile(tempDir, "renamed.md", "hello");
        var h2 = RagIndexCache.ComputeContentHash(new[] { renamed }, tempDir);
        Assert.That(h2, Is.Not.EqualTo(h1));
    }

    [Test]
    public void ComputeContentHash_IsStableAcrossEquivalentCheckoutRoots()
    {
        var root1 = Path.Combine(tempDir, "r1");
        var root2 = Path.Combine(tempDir, "r2");
        Directory.CreateDirectory(root1);
        Directory.CreateDirectory(root2);
        WriteFile(root1, "docs" + "/a.md", "hello");
        WriteFile(root2, "docs" + "/a.md", "hello");
        var f1 = Directory.GetFiles(root1, "*" + ".md", SearchOption.AllDirectories).ToList();
        var f2 = Directory.GetFiles(root2, "*" + ".md", SearchOption.AllDirectories).ToList();
        var h1 = RagIndexCache.ComputeContentHash(f1, root1);
        var h2 = RagIndexCache.ComputeContentHash(f2, root2);
        Assert.That(h2, Is.EqualTo(h1));
    }

    [Test]
    public void DiscoverCorpusFiles_IsOrdinalSortedAndIncludesReadme()
    {
        var docs = Path.Combine(tempDir, "docs");
        Directory.CreateDirectory(docs);
        File.WriteAllText(Path.Combine(docs, "b.md"), "b");
        File.WriteAllText(Path.Combine(docs, "a.md"), "a");
        var readme = Path.Combine(tempDir, "README.md");
        File.WriteAllText(readme, "readme");
        var files = MultiViewIndex.DiscoverCorpusFiles(docs, readme);
        Assert.That(files.Count, Is.EqualTo(3));
        // sorted
        for (int i = 1; i < files.Count; i++)
            Assert.That(string.Compare(files[i - 1], files[i], StringComparison.Ordinal), Is.LessThan(0));
    }

    [Test]
    public void ChunkMarkdown_RepeatedRuns_ProduceIdenticalIdsAndOrder()
    {
        var content = "# Title\n\nSome content here.\n";
        var ids1 = MultiViewChunker.ChunkMarkdown(content, Path.Combine(tempDir, "doc.md"), tempDir, 500).Select(c => c.ChunkId).ToList();
        var ids2 = MultiViewChunker.ChunkMarkdown(content, Path.Combine(tempDir, "doc.md"), tempDir, 500).Select(c => c.ChunkId).ToList();
        Assert.That(ids2, Is.EqualTo(ids1));
    }

    [Test]
    public void RetrievalBaseline_WriteTwice_IsByteIdentical()
    {
        var corpus = new CorpusIdentity("abc", 1, 1, "v", 500, "docs", "cpu", 2);
        var labels = new ResolvedLabels(LabelKind.Judge, "desc", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 1, 1);
        var split = QuerySplit.Create(new[] { "q1" }, 42);
        var baseline = RetrievalBaseline.Create(corpus, labels, "src", 42, 10, 4, split, 0,
            new Dictionary<string, RankerMetrics> { ["fixed(default)"] = new RankerMetrics(0.1, 0.2, 0.3, 0.4) },
            new Dictionary<string, RankerMetrics> { ["fixed(default)"] = new RankerMetrics(0.1, 0.2, 0.3, 0.4) });
        var p1 = Path.Combine(tempDir, "b1.json");
        var p2 = Path.Combine(tempDir, "b2.json");
        RetrievalBaseline.Write(p1, baseline);
        RetrievalBaseline.Write(p2, baseline);
        Assert.That(File.ReadAllBytes(p2), Is.EqualTo(File.ReadAllBytes(p1)));
    }

    [Test]
    public void RetrievalBaseline_RoundTrips()
    {
        var corpus = new CorpusIdentity("abc", 1, 1, "v", 500, "docs", "cpu", 2);
        var labels = new ResolvedLabels(LabelKind.Judge, "desc", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 1, 1);
        var split = QuerySplit.Create(new[] { "q1" }, 42);
        var baseline = RetrievalBaseline.Create(corpus, labels, "src", 42, 10, 4, split, 0,
            new Dictionary<string, RankerMetrics> { ["fixed(default)"] = new RankerMetrics(0.1, 0.2, 0.3, 0.4) },
            new Dictionary<string, RankerMetrics> { ["fixed(default)"] = new RankerMetrics(0.1, 0.2, 0.3, 0.4) });
        var p = Path.Combine(tempDir, "b.json");
        RetrievalBaseline.Write(p, baseline);
        var loaded = RetrievalBaseline.Read(p);
        Assert.That(loaded.SchemaVersion, Is.EqualTo(baseline.SchemaVersion));
        Assert.That(loaded.Corpus.ContentHash, Is.EqualTo("abc"));
    }
}
