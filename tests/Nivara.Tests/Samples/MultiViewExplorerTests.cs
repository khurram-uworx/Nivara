using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the #914 agreement report: absolute-divergence buckets, the insufficient-views bucket for
/// hits that cannot be compared, and the one-line explanation the console and markdown reports use.
/// </summary>
[TestFixture]
public class MultiViewExplorerTests
{
    static readonly MultiViewChunk Chunk =
        new("corpus/a.md#0", "a.md", "corpus/a.md", "Doc", "Doc > S", "corpus", "content", "var x = 1;");

    static MultiViewHit Hit(
        ViewScores scores,
        float? fused = 0.5f,
        ViewSpace? best = ViewSpace.Content,
        float? contentTitle = null,
        float? contentSection = null,
        float? contentCode = null,
        float? maxDivergence = null,
        float? coherence = null,
        string? band = "Near") =>
        new(
            0, Chunk, scores, fused, best, contentTitle, contentSection, contentCode,
            maxDivergence, coherence, band);

    [Test]
    public void BucketFor_UsesInclusiveAgreeAndMildBounds()
    {
        var agree = Hit(new ViewScores(0.9f, 0.85f, null, null), maxDivergence: MultiViewExplorer.AgreeBound);
        var mild = Hit(new ViewScores(0.9f, 0.85f, null, null), maxDivergence: MultiViewExplorer.MildBound);
        var strong = Hit(new ViewScores(0.9f, 0.85f, null, null), maxDivergence: MultiViewExplorer.MildBound + 0.001f);

        Assert.Multiple(() =>
        {
            Assert.That(MultiViewExplorer.BucketFor(agree), Is.EqualTo(MultiViewExplorer.AgreeBucket));
            Assert.That(MultiViewExplorer.BucketFor(mild), Is.EqualTo(MultiViewExplorer.MildBucket));
            Assert.That(MultiViewExplorer.BucketFor(strong), Is.EqualTo(MultiViewExplorer.StrongBucket));
        });
    }

    [Test]
    public void BucketFor_FewerThanTwoScoredViews_IsInsufficientViews()
    {
        var oneView = Hit(new ViewScores(0.9f, null, null, null), maxDivergence: null, best: ViewSpace.Content);
        var noViews = Hit(new ViewScores(null, null, null, null), fused: null, maxDivergence: null, best: null);

        Assert.Multiple(() =>
        {
            Assert.That(MultiViewExplorer.BucketFor(oneView), Is.EqualTo(MultiViewExplorer.InsufficientViewsBucket));
            Assert.That(MultiViewExplorer.BucketFor(noViews), Is.EqualTo(MultiViewExplorer.InsufficientViewsBucket));
        });
    }

    [Test]
    public void Bucketize_CountsEveryHitAndKeepsThreeExamplesPerBucket()
    {
        var agreeHits = Enumerable.Range(0, 5)
            .Select(index => Hit(new ViewScores(0.9f, 0.89f, null, null), maxDivergence: 0.01f) with { Row = index })
            .ToArray();
        var mild = Hit(new ViewScores(0.9f, 0.8f, null, null), maxDivergence: 0.1f) with { Row = 5 };
        var strong = Hit(new ViewScores(0.9f, 0.4f, null, null), maxDivergence: 0.5f) with { Row = 6 };
        var insufficient = Hit(new ViewScores(0.9f, null, null, null), best: ViewSpace.Content) with { Row = 7 };

        var buckets = MultiViewExplorer.Bucketize([.. agreeHits, mild, strong, insufficient]);

        Assert.Multiple(() =>
        {
            Assert.That(buckets, Has.Count.EqualTo(4));
            Assert.That(buckets[0].Name, Is.EqualTo(MultiViewExplorer.AgreeBucket));
            Assert.That(buckets[0].Count, Is.EqualTo(5));
            Assert.That(buckets[0].Examples, Has.Count.EqualTo(3));
            Assert.That(buckets[1].Count, Is.EqualTo(1));
            Assert.That(buckets[2].Count, Is.EqualTo(1));
            Assert.That(buckets[3].Count, Is.EqualTo(1));
            Assert.That(buckets[3].Examples.Single().Row, Is.EqualTo(7));
        });
    }

    [Test]
    public void Bucketize_EmptyInput_ReturnsAllFourBucketsEmpty()
    {
        var buckets = MultiViewExplorer.Bucketize([]);

        Assert.That(buckets.Select(bucket => bucket.Name), Is.EqualTo(new[]
        {
            MultiViewExplorer.AgreeBucket,
            MultiViewExplorer.MildBucket,
            MultiViewExplorer.StrongBucket,
            MultiViewExplorer.InsufficientViewsBucket
        }));
        Assert.That(buckets.Sum(bucket => bucket.Count), Is.EqualTo(0));
    }

    [Test]
    public void Describe_RendersCountsBoundsAndExamples()
    {
        var buckets = MultiViewExplorer.Bucketize(
        [
            Hit(new ViewScores(0.9f, 0.85f, null, null), maxDivergence: 0.05f)
        ]);

        var text = MultiViewExplorer.Describe(buckets);

        Assert.Multiple(() =>
        {
            Assert.That(text, Does.Contain("Agree (max divergence <= 0.05): 1"));
            Assert.That(text, Does.Contain("Mild (max divergence <= 0.15): 0"));
            Assert.That(text, Does.Contain("Strong: 0"));
            Assert.That(text, Does.Contain("InsufficientViews: 0"));
            Assert.That(text, Does.Contain("wins at 0.900"));
        });
    }

    [Test]
    public void Explain_NamesWinnerAndWidestDisagreement()
    {
        var hit = Hit(
            new ViewScores(1f / MathF.Sqrt(2f), 0f, 1f, null),
            fused: 0.615f,
            best: ViewSpace.Section,
            contentTitle: 1f / MathF.Sqrt(2f),
            contentSection: 1f - 1f / MathF.Sqrt(2f),
            contentCode: null,
            maxDivergence: 1f / MathF.Sqrt(2f),
            coherence: 1f / MathF.Sqrt(2f),
            band: "Distinct");

        var text = MultiViewExplorer.Explain(hit);

        Assert.That(text, Is.EqualTo(
            "Section wins at 1.000 (fused 0.615, Distinct); widest disagreement: Content vs Title 0.707 (views strongly disagree)."));
    }

    [Test]
    public void Explain_EqualDivergence_PrefersDeclarationOrder()
    {
        var hit = Hit(
            new ViewScores(0.9f, 0.8f, 0.8f, null),
            best: ViewSpace.Content,
            contentTitle: 0.1f,
            contentSection: 0.1f,
            contentCode: null,
            maxDivergence: 0.1f,
            band: "Near");

        Assert.That(MultiViewExplorer.Explain(hit), Does.Contain("Content vs Title 0.100"));
    }

    [Test]
    public void Explain_OneScoredView_SaysThereIsNoDisagreement()
    {
        var hit = Hit(new ViewScores(null, 1f, null, null), fused: 0.5f, best: ViewSpace.Title);

        Assert.That(MultiViewExplorer.Explain(hit), Is.EqualTo(
            "Title wins at 1.000 (fused 0.500, Near); only one view scored, so there is no disagreement to report."));
    }

    [Test]
    public void Explain_NoScoredView_SaysSo()
    {
        var hit = Hit(new ViewScores(null, null, null, null), fused: null, best: null);

        Assert.That(MultiViewExplorer.Explain(hit), Is.EqualTo("corpus/a.md#0: no view scored."));
    }
}
