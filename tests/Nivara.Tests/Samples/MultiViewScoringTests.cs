using Nivara;
using Nivara.AI;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the #914 late-fusion core: four independently scored views, per-row weight
/// renormalization for missing views, the winning-space tie order, divergence and coherence, and
/// the rank bands measured from the query's top hit.
/// </summary>
[TestFixture]
public class MultiViewScoringTests
{
    static readonly float[] Query = [1f, 0f];

    static readonly float[] Content01 = [0f, 1f];
    static readonly float[] Content11 = [1f, 1f];
    static readonly float[] Content10 = [1f, 0f];

    static MultiViewChunk Chunk(int index, string? code = null) => new(
        $"docs/a.md#{index}",
        "a.md",
        "docs/a.md",
        "Doc",
        $"Doc > S{index}",
        "docs",
        $"content {index}",
        code);

    static List<MultiViewChunk> Chunks(int count) =>
        [.. Enumerable.Range(0, count).Select(index => Chunk(index))];

    static NivaraFrame BuildFrame(
        IReadOnlyList<MultiViewChunk> chunks,
        float[]?[] content,
        float[]?[] title,
        float[]?[] section,
        float[]?[] code) =>
        MultiViewScoring.BuildFrame(chunks, content, title, section, code);

    /// <summary>Rows 0..4 as described in the fixture comment; row 4 has no view at all.</summary>
    static (NivaraFrame Frame, IReadOnlyList<MultiViewChunk> Chunks) Sample()
    {
        var chunks = Chunks(5);
        var content = new float[]?[] { Content10, Content01, Content11, Content10, null };
        var title = new float[]?[] { Content10, null, Content01, Content10, null };
        var section = new float[]?[] { Content10, Content10, Content10, null, null };
        var code = new float[]?[] { Content10, Content01, null, Content01, null };
        return (BuildFrame(chunks, content, title, section, code), chunks);
    }

    static MultiViewHit HitFor(IReadOnlyList<MultiViewHit> hits, int row) =>
        hits.Single(hit => hit.Row == row);

    [Test]
    public void BuildFrame_ProducesOneFrameWithMetadataFourVectorColumnsAndCoherence()
    {
        var (frame, _) = Sample();

        Assert.Multiple(() =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(5));
            Assert.That(frame.Schema.GetColumnType("VectorContent"), Is.EqualTo(typeof(float[])));
            Assert.That(frame.Schema.GetColumnType("VectorTitle"), Is.EqualTo(typeof(float[])));
            Assert.That(frame.Schema.GetColumnType("VectorSection"), Is.EqualTo(typeof(float[])));
            Assert.That(frame.Schema.GetColumnType("VectorCode"), Is.EqualTo(typeof(float[])));
            Assert.That(frame.GetColumn<float[]>("VectorCode").IsNull(2), Is.True);
            Assert.That(frame.GetColumn<float[]>("VectorCode").IsNull(0), Is.False);
        });
    }

    [Test]
    public void BuildFrame_DifferentViewCounts_Throws()
    {
        var chunks = Chunks(2);
        var vectors = new float[]?[] { Content10, Content01 };

        Assert.Throws<ArgumentException>(() =>
            BuildFrame(chunks, vectors, vectors, vectors, [Content10]));
    }

    [Test]
    public void Rank_PerViewScores_MatchHandComputedCosines()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 4);

        Assert.Multiple(() =>
        {
            var row0 = HitFor(hits, 0);
            Assert.That(row0.Scores.Content, Is.EqualTo(1f));
            Assert.That(row0.Scores.Title, Is.EqualTo(1f));
            Assert.That(row0.Scores.Section, Is.EqualTo(1f));
            Assert.That(row0.Scores.Code, Is.EqualTo(1f));
            Assert.That(row0.Fused, Is.EqualTo(1f));

            var row2 = HitFor(hits, 2);
            Assert.That(row2.Scores.Content, Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));
            Assert.That(row2.Scores.Title, Is.EqualTo(0f));
            Assert.That(row2.Scores.Code, Is.Null);
        });
    }

    [Test]
    public void Rank_MissingView_RenormalizesWeightsOverPresentViews()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 4);

        Assert.Multiple(() =>
        {
            // Row 1 has Content 0, Section 1 and Code 0; Title is missing, so the remaining
            // weights (0.5 + 0.2 + 0.1 = 0.8) are renormalized: 0.2 / 0.8.
            Assert.That(HitFor(hits, 1).Fused, Is.EqualTo(0.25f).Within(1e-6f));
            // Row 2 has no Code view: (0.5 * 1/sqrt(2) + 0.2 * 0 + 0.2 * 1) / 0.9.
            Assert.That(HitFor(hits, 2).Fused, Is.EqualTo((0.5f * (1f / MathF.Sqrt(2f)) + 0.2f) / 0.9f).Within(1e-6f));
            // Row 3 has no Section view: (0.5 * 1 + 0.2 * 1 + 0.1 * 0) / 0.8.
            Assert.That(HitFor(hits, 3).Fused, Is.EqualTo(0.875f).Within(1e-6f));
        });
    }

    [Test]
    public void Rank_RowWithNoView_HasNullFusedAndNeverEntersTopK()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 10);

        Assert.Multiple(() =>
        {
            Assert.That(hits.Select(hit => hit.Row), Is.EqualTo(new[] { 0, 3, 2, 1 }));
            Assert.That(hits.Any(hit => hit.Row == 4), Is.False);
            Assert.That(hits.All(hit => hit.Fused is not null), Is.True);
        });
    }

    [Test]
    public void Rank_BestSpace_IsHighestScoringViewAndBreaksTiesByDeclarationOrder()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 4);

        Assert.Multiple(() =>
        {
            // Row 0: every view is 1.0, so Content wins by declaration order.
            Assert.That(HitFor(hits, 0).BestSpace, Is.EqualTo(ViewSpace.Content));
            // Row 1: only Section scored 1.0.
            Assert.That(HitFor(hits, 1).BestSpace, Is.EqualTo(ViewSpace.Section));
            // Row 2: Section 1.0 beats Content 0.707 and Title 0.
            Assert.That(HitFor(hits, 2).BestSpace, Is.EqualTo(ViewSpace.Section));
            // Row 3: Content and Title both 1.0, so Content wins.
            Assert.That(HitFor(hits, 3).BestSpace, Is.EqualTo(ViewSpace.Content));
        });
    }

    [Test]
    public void Rank_Divergence_IsAbsoluteDifferenceAndNullWhenEitherViewIsMissing()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 4);

        Assert.Multiple(() =>
        {
            var row1 = HitFor(hits, 1);
            Assert.That(row1.DivergenceContentTitle, Is.Null);
            Assert.That(row1.DivergenceContentSection, Is.EqualTo(1f));
            Assert.That(row1.DivergenceContentCode, Is.EqualTo(0f));
            Assert.That(row1.MaxDivergence, Is.EqualTo(1f));

            var row2 = HitFor(hits, 2);
            Assert.That(row2.DivergenceContentTitle, Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));
            Assert.That(row2.DivergenceContentSection, Is.EqualTo(1f - 1f / MathF.Sqrt(2f)).Within(1e-6f));
            Assert.That(row2.DivergenceContentCode, Is.Null);
            Assert.That(row2.MaxDivergence, Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));

            var row3 = HitFor(hits, 3);
            Assert.That(row3.DivergenceContentTitle, Is.EqualTo(0f));
            Assert.That(row3.DivergenceContentSection, Is.Null);
            Assert.That(row3.DivergenceContentCode, Is.EqualTo(1f));
        });
    }

    [Test]
    public void Rank_Coherence_IsCosineBetweenContentAndTitle()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 4);

        Assert.Multiple(() =>
        {
            Assert.That(HitFor(hits, 0).CoherenceContentTitle, Is.EqualTo(1f).Within(1e-6f));
            Assert.That(HitFor(hits, 1).CoherenceContentTitle, Is.Null);
            Assert.That(HitFor(hits, 2).CoherenceContentTitle, Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));
        });
    }

    [Test]
    public void Rank_NonFiniteScore_IsTreatedAsMissingView()
    {
        var chunks = Chunks(2);
        var zero = new float[] { 0f, 0f };
        var frame = BuildFrame(
            chunks,
            [zero, Content10],
            [Content10, null],
            [null, null],
            [null, null]);
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 2);

        Assert.Multiple(() =>
        {
            var row0 = HitFor(hits, 0);
            Assert.That(row0.Scores.Content, Is.Null);
            Assert.That(row0.Fused, Is.EqualTo(1f));
            Assert.That(row0.Fused is { } fused && float.IsFinite(fused), Is.True);
            Assert.That(row0.BestSpace, Is.EqualTo(ViewSpace.Title));
        });
    }

    [Test]
    public void Rank_OrdersByFusedScoreDescending()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, topK: 2);

        Assert.Multiple(() =>
        {
            Assert.That(hits, Has.Count.EqualTo(2));
            Assert.That(hits[0].Row, Is.EqualTo(0));
            Assert.That(hits[1].Row, Is.EqualTo(3));
            Assert.That(hits[0].Fused!.Value, Is.GreaterThan(hits[1].Fused!.Value));
        });
    }

    [Test]
    public void Rank_BandEdges_AreInclusiveAtNearAndDistinctBounds()
    {
        Assert.Multiple(() =>
        {
            Assert.That(MultiViewScoring.BandFor(0f), Is.EqualTo("Near"));
            Assert.That(MultiViewScoring.BandFor(MultiViewScoring.NearBandBound), Is.EqualTo("Near"));
            Assert.That(MultiViewScoring.BandFor(MultiViewScoring.NearBandBound + 1e-5f), Is.EqualTo("Distinct"));
            Assert.That(MultiViewScoring.BandFor(MultiViewScoring.DistinctBandBound), Is.EqualTo("Distinct"));
            Assert.That(MultiViewScoring.BandFor(MultiViewScoring.DistinctBandBound + 1e-5f), Is.EqualTo("Weak"));
        });
    }

    [Test]
    public void Rank_Bands_AreMeasuredFromTheTopFusedScore()
    {
        var near = Content10;
        var distinct = new float[] { 0.95f, MathF.Sqrt(1f - 0.95f * 0.95f) };
        var weak = Content01;
        var chunks = Chunks(3);
        var frame = BuildFrame(
            chunks,
            [near, distinct, weak],
            [null, null, null],
            [null, null, null],
            [null, null, null]);
        var matrices = MultiViewMatrices.Create(frame);

        var hits = MultiViewScoring.Rank(
            frame, matrices, chunks, Query, new FusionWeights(1f, 0f, 0f, 0f), topK: 3);

        Assert.Multiple(() =>
        {
            Assert.That(hits[0].RankBand, Is.EqualTo("Near"));
            Assert.That(hits[1].RankBand, Is.EqualTo("Distinct"));
            Assert.That(hits[2].RankBand, Is.EqualTo("Weak"));
        });
    }

    [Test]
    public void Rank_NullQueryOrZeroTopK_ReturnsEmpty()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        Assert.Multiple(() =>
        {
            Assert.That(MultiViewScoring.Rank(frame, matrices, chunks, null, FusionWeights.Default, 5), Is.Empty);
            Assert.That(MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, 0), Is.Empty);
        });
    }

    [Test]
    public void Rank_QueryDimensionMismatch_Throws()
    {
        var (frame, chunks) = Sample();
        var matrices = MultiViewMatrices.Create(frame);

        Assert.Throws<ArgumentException>(() =>
            MultiViewScoring.Rank(frame, matrices, chunks, [1f, 0f, 0f], FusionWeights.Default, 5));
    }

    [Test]
    public void Frame_FilterByMask_KeepsFourVectorColumnsAligned()
    {
        var (frame, _) = Sample();
        var mask = NivaraColumn<bool>.Create([true, false, true, false, false]);

        var filtered = frame.FilterByMask(mask);

        Assert.Multiple(() =>
        {
            Assert.That(filtered.RowCount, Is.EqualTo(2));
            Assert.That(filtered.GetColumn<string>("ChunkId")[0], Is.EqualTo("docs/a.md#0"));
            Assert.That(filtered.GetColumn<float[]>("VectorContent")[1], Is.EqualTo(Content11));
            Assert.That(filtered.GetColumn<float[]>("VectorTitle")[1], Is.EqualTo(Content01));
            Assert.That(filtered.GetColumn<float[]>("VectorSection")[1], Is.EqualTo(Content10));
            Assert.That(filtered.GetColumn<float[]>("VectorCode")[1], Is.Null);
            Assert.That(filtered.GetColumn<float>("CoherenceContentTitle")[1],
                Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));
        });
    }

    [Test]
    public void Frame_Slice_KeepsFourVectorColumnsAligned()
    {
        var (frame, _) = Sample();

        var sliced = frame.Slice(1, 2);

        Assert.Multiple(() =>
        {
            Assert.That(sliced.RowCount, Is.EqualTo(2));
            Assert.That(sliced.GetColumn<string>("ChunkId")[0], Is.EqualTo("docs/a.md#1"));
            Assert.That(sliced.GetColumn<float[]>("VectorContent")[0], Is.EqualTo(Content01));
            Assert.That(sliced.GetColumn<float[]>("VectorContent")[1], Is.EqualTo(Content11));
            Assert.That(sliced.GetColumn<float[]>("VectorTitle")[0], Is.Null);
            Assert.That(sliced.GetColumn<float[]>("VectorSection")[1], Is.EqualTo(Content10));
            Assert.That(sliced.GetColumn<float[]>("VectorCode")[0], Is.EqualTo(Content01));
            Assert.That(sliced.GetColumn<float>("CoherenceContentTitle").IsNull(0), Is.True);
        });
    }

    [Test]
    public void Matrices_ToFrame_RoundTripsEveryViewThroughTheEmbeddingConvention()
    {
        var (frame, _) = Sample();

        var merged = MultiViewMatrices.Create(frame).ToFrame();

        Assert.Multiple(() =>
        {
            Assert.That(merged.ColumnCount, Is.EqualTo(8));
            var content = merged.ToEmbeddingFrame("VectorContent", "ContentE", dimension: 2);
            var title = merged.ToEmbeddingFrame("VectorTitle", "TitleE", dimension: 2);
            var code = merged.ToEmbeddingFrame("VectorCode", "CodeE", dimension: 2);

            Assert.That(content.GetColumn<float[]>("VectorContent")[2], Is.EqualTo(Content11));
            Assert.That(title.GetColumn<float[]>("VectorTitle").IsNull(1), Is.True);
            Assert.That(code.GetColumn<float[]>("VectorCode").IsNull(2), Is.True);
            Assert.That(code.GetColumn<float[]>("VectorCode")[3], Is.EqualTo(Content01));
        });
    }

    [Test]
    public void Rank_WithDiagonalTitleVector_ScoresCosineNotDotProduct()
    {
        var chunks = Chunks(1);
        var frame = BuildFrame(
            chunks,
            [Content10],
            [Content11],
            [null],
            [null]);
        var matrices = MultiViewMatrices.Create(frame);

        var hit = MultiViewScoring.Rank(frame, matrices, chunks, Query, FusionWeights.Default, 1).Single();

        Assert.That(hit.Scores.Title, Is.EqualTo(1f / MathF.Sqrt(2f)).Within(1e-6f));
        Assert.That(hit.Scores.Content, Is.EqualTo(1f));
    }

    [Test]
    public void FusionWeights_Default_MatchesTheDocumentedSplit()
    {
        var weights = FusionWeights.Default;

        Assert.Multiple(() =>
        {
            Assert.That(weights.Content, Is.EqualTo(0.5f));
            Assert.That(weights.Title, Is.EqualTo(0.2f));
            Assert.That(weights.Section, Is.EqualTo(0.2f));
            Assert.That(weights.Code, Is.EqualTo(0.1f));
            Assert.That(weights[ViewSpace.Code], Is.EqualTo(0.1f));
        });
    }

    [Test]
    public void FusionWeights_TryParse_OverridesOnlyNamedViews()
    {
        Assert.Multiple(() =>
        {
            Assert.That(FusionWeights.TryParse(" title=0.5 , code=0 ", out var weights, out var error), Is.True);
            Assert.That(error, Is.Null);
            Assert.That(weights.Title, Is.EqualTo(0.5f));
            Assert.That(weights.Code, Is.EqualTo(0f));
            Assert.That(weights.Content, Is.EqualTo(FusionWeights.Default.Content));
            Assert.That(weights.Section, Is.EqualTo(FusionWeights.Default.Section));
        });
    }

    [Test]
    public void FusionWeights_TryParse_NullOrBlank_KeepsDefaults()
    {
        Assert.Multiple(() =>
        {
            Assert.That(FusionWeights.TryParse(null, out var fromNull, out _), Is.True);
            Assert.That(fromNull, Is.EqualTo(FusionWeights.Default));
            Assert.That(FusionWeights.TryParse("  ", out var fromBlank, out _), Is.True);
            Assert.That(fromBlank, Is.EqualTo(FusionWeights.Default));
        });
    }

    [Test]
    public void FusionWeights_TryParse_RejectsUnknownNegativeAndAllZero()
    {
        Assert.Multiple(() =>
        {
            Assert.That(FusionWeights.TryParse("body=1", out _, out var unknown), Is.False);
            Assert.That(unknown, Does.Contain("body"));
            Assert.That(FusionWeights.TryParse("content=-1", out _, out var negative), Is.False);
            Assert.That(negative, Does.Contain("negative"));
            Assert.That(FusionWeights.TryParse("content=0,title=0,section=0,code=0", out _, out var zero), Is.False);
            Assert.That(zero, Does.Contain("positive"));
            Assert.That(FusionWeights.TryParse("content=", out _, out var empty), Is.False);
            Assert.That(empty, Is.Not.Null);
        });
    }
}
