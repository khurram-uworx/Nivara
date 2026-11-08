using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the shared markdown corpus enumeration: plan documents under <c>docs/plan/</c> and
/// <c>docs/plans/</c> must never be chunked, indexed or turned into auto queries, while the rest of
/// <c>docs/</c> and the sample README stay in the corpus.
/// </summary>
[TestFixture]
public class DocCorpusTests
{
    string root = null!;
    string docsDir = null!;

    [SetUp]
    public void SetUp()
    {
        root = Path.Combine(Path.GetTempPath(), "nivaradocs-" + Guid.NewGuid().ToString("N"));
        docsDir = Path.Combine(root, "docs");
        Directory.CreateDirectory(docsDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(root))
            Directory.Delete(root, recursive: true);
    }

    void Write(string relativePath)
    {
        var absolute = Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(absolute)!);
        File.WriteAllText(absolute, "# document");
    }

    [Test]
    public void EnumerateMarkdownFiles_ExcludesPlanDirectories_AndIncludesSampleReadme()
    {
        Write("docs/GUIDE.md");
        Write("docs/blog/post.md");
        Write("docs/plan/ROADMAP.md");
        Write("docs/plans/EVAL.md");
        Write("samples/NivaraChat/README.md");

        var files = DocCorpus.EnumerateMarkdownFiles(docsDir, root);

        Assert.Multiple(() =>
        {
            Assert.That(files.Select(Path.GetFileName), Is.EquivalentTo(new[] { "GUIDE.md", "post.md", "README.md" }));
            Assert.That(files.Where(path => DocCorpus.IsPlanDocument(docsDir, path)), Is.Empty);
        });
    }

    [Test]
    public void EnumerateMarkdownFiles_EmptyDocsDirectory_StillReturnsReadmeWhenPresent()
    {
        Write("samples/NivaraChat/README.md");

        var files = DocCorpus.EnumerateMarkdownFiles(docsDir, root);

        Assert.That(files.Select(Path.GetFileName), Is.EqualTo(new[] { "README.md" }));
    }

    [Test]
    public void IsPlanDocument_DirectChildOfPlanDirectory_ReturnsTrue()
    {
        Assert.Multiple(() =>
        {
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "plan", "ROADMAP.md")), Is.True);
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "plans", "EVAL.md")), Is.True);
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "Plan", "ROADMAP.md")), Is.True);
        });
    }

    [Test]
    public void IsPlanDocument_OtherLocations_ReturnsFalse()
    {
        Assert.Multiple(() =>
        {
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "GUIDE.md")), Is.False);
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "blog", "post.md")), Is.False);
            Assert.That(DocCorpus.IsPlanDocument(docsDir, Path.Combine(docsDir, "planning", "ROADMAP.md")), Is.False);
        });
    }
}
