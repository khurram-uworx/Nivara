using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the fence-aware chunker behind #914: heading breadcrumbs, fences that must not be read as
/// headings, the Content/Code split that makes the two views differ, prose splitting, ACL groups and
/// the stable document title.
/// </summary>
[TestFixture]
public class MultiViewChunkerTests
{
    const string RepoRoot = "repo";

    static string Source(string relative) =>
        Path.Combine(RepoRoot, relative.Replace('/', Path.DirectorySeparatorChar));

    static List<MultiViewChunk> Chunk(string markdown, int maxChunkSize = 500, string relative = "corpus/multi.md") =>
        MultiViewChunker.ChunkMarkdown(markdown, Source(relative), RepoRoot, maxChunkSize);

    [Test]
    public void ChunkMarkdown_SectionsAndHeadings_BuildTitleAndBreadcrumb()
    {
        const string markdown = """
            # Document Title

            Intro text.

            ## Section A

            Alpha prose.

            ### Deep

            Deep prose.

            ## Section B

            Beta prose.
            """;

        var chunks = Chunk(markdown);

        Assert.That(chunks, Has.Count.EqualTo(4));
        Assert.Multiple(() =>
        {
            Assert.That(chunks.Select(c => c.Title), Is.All.EqualTo("Document Title"));
            Assert.That(chunks[0].Section, Is.EqualTo("Document Title"));
            Assert.That(chunks[0].Content, Is.EqualTo("Intro text."));
            Assert.That(chunks[1].Section, Is.EqualTo("Document Title > Section A"));
            Assert.That(chunks[2].Section, Is.EqualTo("Document Title > Section A > Deep"));
            Assert.That(chunks[3].Section, Is.EqualTo("Document Title > Section B"));
        });
    }

    [Test]
    public void ChunkMarkdown_HashInsideFence_IsNotTreatedAsHeading()
    {
        const string markdown = """
            # Title

            ## Code

            ```csharp
            // # not a heading
            var x = 1;
            ```

            ## After

            Trailing prose.
            """;

        var chunks = Chunk(markdown);

        Assert.That(chunks, Has.Count.EqualTo(2));
        Assert.Multiple(() =>
        {
            Assert.That(chunks[0].Section, Is.EqualTo("Title > Code"));
            Assert.That(chunks[0].Content, Does.Contain("# not a heading"));
            Assert.That(chunks[1].Section, Is.EqualTo("Title > After"));
            Assert.That(chunks[1].Content, Is.EqualTo("Trailing prose."));
        });
    }

    [Test]
    public void ChunkMarkdown_FenceInsideChunk_ContentKeepsMarkersAndCodeKeepsBodyOnly()
    {
        const string markdown = """
            # Title

            ## Code

            Use this:

            ```csharp
            var x = 1;
            ```

            After.
            """;

        var chunk = Chunk(markdown).Single();

        Assert.Multiple(() =>
        {
            Assert.That(chunk.Content, Does.Contain("Use this:"));
            Assert.That(chunk.Content, Does.Contain("```csharp"));
            Assert.That(chunk.Content, Does.Contain("After."));
            Assert.That(chunk.Code, Is.EqualTo("var x = 1;"));
            Assert.That(chunk.Code, Does.Not.Contain("```"));
        });
    }

    [Test]
    public void ChunkMarkdown_ChunkWithoutFence_HasNullCodeView()
    {
        const string markdown = """
            # Title

            ## Prose

            Only prose here.

            More prose.
            """;

        var chunks = Chunk(markdown);

        Assert.That(chunks, Is.Not.Empty);
        Assert.That(chunks.Select(c => c.Code), Is.All.Null);
    }

    [Test]
    public void ChunkMarkdown_LongProse_SplitsAtChunkSize()
    {
        var markdown = "# Title\n\n" + new string('a', 1200);

        var chunks = Chunk(markdown, maxChunkSize: 500);

        Assert.Multiple(() =>
        {
            Assert.That(chunks, Has.Count.EqualTo(3));
            Assert.That(chunks.Select(c => c.Content.Length), Is.All.LessThanOrEqualTo(500));
            Assert.That(chunks.Select(c => c.Code), Is.All.Null);
        });
    }

    [Test]
    public void ChunkMarkdown_FenceLargerThanChunkSize_IsNotSplit()
    {
        var body = new string('b', 600);
        var markdown = $"# Title\n\n```csharp\n{body}\n```";

        var chunks = Chunk(markdown, maxChunkSize: 500);

        Assert.Multiple(() =>
        {
            Assert.That(chunks, Has.Count.EqualTo(1));
            Assert.That(chunks[0].Code, Is.EqualTo(body));
            Assert.That(chunks[0].Content.Length, Is.GreaterThan(500));
        });
    }

    [Test]
    public void ChunkMarkdown_NoH1_UsesFileNameAsTitle()
    {
        const string markdown = """
            ## Only Section

            Body.
            """;

        var chunks = Chunk(markdown, relative: "corpus/guide.md");

        Assert.That(chunks, Has.Count.EqualTo(1));
        Assert.Multiple(() =>
        {
            Assert.That(chunks[0].Title, Is.EqualTo("guide"));
            Assert.That(chunks[0].Section, Is.EqualTo("guide > Only Section"));
        });
    }

    [Test]
    public void ChunkMarkdown_ContentBeforeH1_StillCarriesDocumentTitle()
    {
        const string markdown = """
            Preamble.

            # Real Title

            Body.
            """;

        var chunks = Chunk(markdown);

        Assert.That(chunks, Has.Count.EqualTo(2));
        Assert.That(chunks.Select(c => c.Title), Is.All.EqualTo("Real Title"));
    }

    [Test]
    public void ChunkMarkdown_CrlfInput_MatchesLfInput()
    {
        const string lf = "# Title\n\nBody.\n\n## Next\n\nMore.";
        var crlf = lf.Replace("\n", "\r\n");

        var lfChunks = Chunk(lf);
        var crlfChunks = Chunk(crlf);

        Assert.Multiple(() =>
        {
            Assert.That(crlfChunks, Has.Count.EqualTo(lfChunks.Count));
            Assert.That(crlfChunks.Select(c => c.Content), Is.EqualTo(lfChunks.Select(c => c.Content)));
            Assert.That(crlfChunks.Select(c => c.Section), Is.EqualTo(lfChunks.Select(c => c.Section)));
        });
    }

    [Test]
    public void ChunkMarkdown_H4BoundsChunkButDoesNotExtendBreadcrumb()
    {
        const string markdown = """
            # Title

            ## Section

            First.

            #### Detail

            Second.
            """;

        var chunks = Chunk(markdown);

        Assert.That(chunks, Has.Count.EqualTo(2));
        Assert.That(chunks.Select(c => c.Section), Is.All.EqualTo("Title > Section"));
    }

    [Test]
    public void ChunkMarkdown_EmptyOrWhitespace_ReturnsNoChunks()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Chunk(""), Is.Empty);
            Assert.That(Chunk("   \n\n\t\n"), Is.Empty);
        });
    }

    [Test]
    public void ChunkMarkdown_AclGroup_IsParentFolderName()
    {
        var nested = Chunk("# T\n\nBody.", relative: "corpus/adr/001-decision.md");
        var root = Chunk("# T\n\nBody.", relative: "README.md");

        Assert.Multiple(() =>
        {
            Assert.That(nested[0].AclGroup, Is.EqualTo("adr"));
            Assert.That(root[0].AclGroup, Is.EqualTo("public"));
        });
    }

    [Test]
    public void ChunkMarkdown_DuplicateFileNamesAcrossFolders_ProduceDistinctChunkIds()
    {
        var first = Chunk("# A\n\nBody.", relative: "corpus/a/README.md");
        var second = Chunk("# B\n\nBody.", relative: "corpus/b/README.md");

        Assert.Multiple(() =>
        {
            Assert.That(first[0].ChunkId, Is.EqualTo("corpus/a/README.md#0"));
            Assert.That(second[0].ChunkId, Is.EqualTo("corpus/b/README.md#0"));
            Assert.That(first[0].ChunkId, Is.Not.EqualTo(second[0].ChunkId));
        });
    }

    [Test]
    public void ChunkMarkdown_HashtagWithoutSpace_StaysProse()
    {
        var chunks = Chunk("# Title\n\n#notaheading");

        Assert.That(chunks, Has.Count.EqualTo(1));
        Assert.That(chunks[0].Content, Is.EqualTo("#notaheading"));
    }

    [Test]
    public void ChunkMarkdown_UnterminatedFence_KeepsRemainingContent()
    {
        const string markdown = "# Title\n\n```csharp\nvar x = 1;";

        var chunk = Chunk(markdown).Single();

        Assert.Multiple(() =>
        {
            Assert.That(chunk.Content, Does.Contain("var x = 1;"));
            Assert.That(chunk.Code, Is.EqualTo("var x = 1;"));
        });
    }
}
