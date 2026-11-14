using Nivara;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the shared ranking helper extracted for #918: ordering, ACL filtering, null query,
/// null embedding rows, top-K clamping, and the row-alignment guarantee that protects against
/// duplicate <c>ChunkId</c> values (e.g. two README.md files in the corpus).
/// </summary>
[TestFixture]
public class RagSearchTests
{
    static NivaraFrame BuildFrame(params (string ChunkId, string Source, string Title, string AclGroup, float[]? Embedding)[] rows)
    {
        var chunkIds = rows.Select(r => r.ChunkId).ToArray();
        var sources = rows.Select(r => r.Source).ToArray();
        var titles = rows.Select(r => r.Title).ToArray();
        var sections = rows.Select(r => r.Title).ToArray();
        var acls = rows.Select(r => r.AclGroup).ToArray();
        var contents = rows.Select(r => r.ChunkId + " content").ToArray();
        var embeddingArray = new float[rows.Length][];
        for (int i = 0; i < rows.Length; i++)
            embeddingArray[i] = rows[i].Embedding!;

        var embeddings = NivaraColumn<float[]>.Create(embeddingArray);

        return NivaraFrame.Create(
            ("ChunkId", (IColumn)NivaraColumn<string>.Create(chunkIds)),
            ("Source", (IColumn)NivaraColumn<string>.Create(sources)),
            ("Title", (IColumn)NivaraColumn<string>.Create(titles)),
            ("SectionPath", (IColumn)NivaraColumn<string>.Create(sections)),
            ("AclGroup", (IColumn)NivaraColumn<string>.Create(acls)),
            ("Content", (IColumn)NivaraColumn<string>.Create(contents)),
            ("Embedding", (IColumn)embeddings));
    }

    static List<RagChunk> BuildChunks(params (string ChunkId, string Source, string Title, string AclGroup, float[]? Embedding)[] rows) =>
        rows.Select(r => new RagChunk(r.ChunkId, r.Source, r.Title, r.Title, r.AclGroup, r.ChunkId + " content")).ToList();

    static (NivaraFrame Frame, List<RagChunk> Chunks) Scenario(params (string ChunkId, string Source, string Title, string AclGroup, float[]? Embedding)[] rows) =>
        (BuildFrame(rows), BuildChunks(rows));

    [Test]
    public void Rank_NoAcl_ReturnsTopKDescending()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", new[] { 1f, 0f }),
            ("c1", "s1", "t1", "docs", new[] { 0f, 1f }),
            ("c2", "s2", "t2", "docs", new[] { 0.5f, 0.5f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 2);

        Assert.That(results.Count, Is.EqualTo(2));
        Assert.That(results[0].ChunkId, Is.EqualTo("c0"));
        Assert.That(results[1].ChunkId, Is.EqualTo("c2"));
        Assert.That(results[0].Score, Is.GreaterThan(results[1].Score));
    }

    [Test]
    public void Rank_DuplicateChunkIds_ReturnsScoredRowContent()
    {
        // Two rows share the same ChunkId; the second is the better match. Row alignment
        // (not ChunkId lookup) must surface the second row's content.
        var (frame, chunks) = Scenario(
            ("dup", "src-a", "title-a", "docs", new[] { 0f, 1f }),
            ("dup", "src-b", "title-b", "docs", new[] { 1f, 0f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 1);

        Assert.That(results.Count, Is.EqualTo(1));
        Assert.That(results[0].Source, Is.EqualTo("src-b"));
        Assert.That(results[0].Title, Is.EqualTo("title-b"));
        Assert.That(results[0].Content, Is.EqualTo("dup content"));
    }

    [Test]
    public void Rank_AclGroup_FiltersBeforeScoring()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "adr", new[] { 1f, 0f }),
            ("c1", "s1", "t1", "docs", new[] { 1f, 0f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, "adr", 5);

        Assert.That(results.Count, Is.EqualTo(1));
        Assert.That(results[0].ChunkId, Is.EqualTo("c0"));
    }

    [Test]
    public void Rank_AclGroupNoMatches_ReturnsEmpty()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", new[] { 1f, 0f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, "adr", 5);

        Assert.That(results, Is.Empty);
    }

    [Test]
    public void Rank_NullQuery_ReturnsEmpty()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", new[] { 1f, 0f }));

        var results = RagSearch.Rank(frame, chunks, null, null, 5);

        Assert.That(results, Is.Empty);
    }

    [Test]
    public void Rank_NullEmbeddingRow_Excluded()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", null),
            ("c1", "s1", "t1", "docs", new[] { 1f, 0f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 5);

        Assert.That(results.Count, Is.EqualTo(1));
        Assert.That(results[0].ChunkId, Is.EqualTo("c1"));
    }

    [Test]
    public void Rank_TopKGreaterThanVisible_ReturnsAllVisible()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", new[] { 1f, 0f }),
            ("c1", "s1", "t1", "docs", new[] { 0f, 1f }));

        var results = RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 100);

        Assert.That(results.Count, Is.EqualTo(2));
    }

    [Test]
    public void Rank_NonPositiveTopK_ReturnsEmpty()
    {
        var (frame, chunks) = Scenario(
            ("c0", "s0", "t0", "docs", new[] { 1f, 0f }));

        Assert.That(RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 0), Is.Empty);
    }

    [Test]
    public void Rank_FrameRowCountMismatch_Throws()
    {
        var (frame, _) = Scenario(("c0", "s0", "t0", "docs", new[] { 1f, 0f }));
        var chunks = new List<RagChunk>();

        Assert.Throws<ArgumentException>(() => RagSearch.Rank(frame, chunks, new[] { 1f, 0f }, null, 1));
    }
}
