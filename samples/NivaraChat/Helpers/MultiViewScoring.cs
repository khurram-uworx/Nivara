using Nivara;
using Nivara.AI;
using System.Globalization;
using System.Numerics.Tensors;

namespace NivaraChat.Helpers;

/// <summary>
/// The four representations #914 scores independently. Declaration order is the tie order for
/// <see cref="MultiViewHit.BestSpace"/>, so Content wins a tie against every other view.
/// </summary>
public enum ViewSpace
{
    Content,
    Title,
    Section,
    Code
}

/// <summary>
/// Late-fusion weights. They are renormalized per row over the views that actually have a score, so
/// a chunk without code still gets a full-strength fused score from the remaining views.
/// </summary>
public sealed record FusionWeights(float Content, float Title, float Section, float Code)
{
    public static FusionWeights Default { get; } = new(0.5f, 0.2f, 0.2f, 0.1f);

    /// <summary>
    /// Parses <c>content=0.4,title=0.3,section=0.2,code=0.1</c>. Omitted views keep their default
    /// weight; unknown names, negatives and an all-zero total are rejected.
    /// </summary>
    public static bool TryParse(string? text, out FusionWeights weights, out string? error)
    {
        weights = Default;
        error = null;
        if (string.IsNullOrWhiteSpace(text))
            return true;

        var content = weights.Content;
        var title = weights.Title;
        var section = weights.Section;
        var code = weights.Code;
        foreach (var part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var separator = part.IndexOf('=');
            if (separator <= 0 || separator == part.Length - 1)
            {
                error = $"Expected 'view=weight', got '{part}'.";
                return false;
            }

            var name = part[..separator].Trim().ToLowerInvariant();
            if (!float.TryParse(part[(separator + 1)..].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            {
                error = $"'{part}' is not a number.";
                return false;
            }

            if (value < 0)
            {
                error = $"Weight for '{name}' is negative.";
                return false;
            }

            switch (name)
            {
                case "content": content = value; break;
                case "title": title = value; break;
                case "section": section = value; break;
                case "code": code = value; break;
                default:
                    error = $"Unknown view '{name}'. Expected content, title, section or code.";
                    return false;
            }
        }

        if (content + title + section + code <= 0)
        {
            error = "At least one weight must be positive.";
            return false;
        }

        weights = new FusionWeights(content, title, section, code);
        return true;
    }

    public float this[ViewSpace space] => space switch
    {
        ViewSpace.Content => Content,
        ViewSpace.Title => Title,
        ViewSpace.Section => Section,
        _ => Code
    };
}

/// <summary>One cosine score per view. A null score means the chunk has no vector for that view.</summary>
public sealed record ViewScores(float? Content, float? Title, float? Section, float? Code)
{
    public float? this[ViewSpace space] => space switch
    {
        ViewSpace.Content => Content,
        ViewSpace.Title => Title,
        ViewSpace.Section => Section,
        _ => Code
    };

    public int PresentCount => (Content is null ? 0 : 1) + (Title is null ? 0 : 1) + (Section is null ? 0 : 1) + (Code is null ? 0 : 1);
}

/// <summary>A scored chunk: the fused score, the view that carried it, and how far the views diverge.</summary>
public sealed record MultiViewHit(
    int Row,
    MultiViewChunk Chunk,
    ViewScores Scores,
    float? Fused,
    ViewSpace? BestSpace,
    float? DivergenceContentTitle,
    float? DivergenceContentSection,
    float? DivergenceContentCode,
    float? MaxDivergence,
    float? CoherenceContentTitle,
    string? RankBand);

/// <summary>
/// The four flat <c>[rows, dim]</c> matrices, one per view, built once when the index is built so a
/// query only pays for scoring. <see cref="MultiViewScoring.Rank"/> scores every matrix with the
/// frame's <see cref="NivaraFrame.RowCosineSimilarity{T}(NivaraSeries{T}, IColumn?)"/> kernel.
/// </summary>
public sealed record MultiViewMatrices(NivaraFrame Content, NivaraFrame Title, NivaraFrame Section, NivaraFrame Code)
{
    public static MultiViewMatrices Create(NivaraFrame vectorFrame, int? dimension = null)
    {
        ArgumentNullException.ThrowIfNull(vectorFrame);
        var resolved = dimension ?? ResolveDimension(vectorFrame);
        return new MultiViewMatrices(
            vectorFrame.ToEmbeddingMatrix("VectorContent", "ContentE", resolved),
            vectorFrame.ToEmbeddingMatrix("VectorTitle", "TitleE", resolved),
            vectorFrame.ToEmbeddingMatrix("VectorSection", "SectionE", resolved),
            vectorFrame.ToEmbeddingMatrix("VectorCode", "CodeE", resolved));
    }

    /// <summary>
    /// Infers the embedding dimension from the first view that has a vector. A corpus with no code
    /// fences leaves <c>VectorCode</c> entirely null, so the dimension cannot come from one column.
    /// </summary>
    static int ResolveDimension(NivaraFrame vectorFrame)
    {
        foreach (var name in new[] { "VectorContent", "VectorTitle", "VectorSection", "VectorCode" })
        {
            var column = vectorFrame.GetColumn<float[]>(name);
            for (var row = 0; row < column.Length; row++)
            {
                if (!column.IsNull(row) && column[row] is { Length: > 0 } vector)
                    return vector.Length;
            }
        }

        throw new InvalidOperationException(
            "No view holds a non-null embedding, so the dimension cannot be inferred. Pass 'dimension' explicitly.");
    }

    /// <summary>Merges the four matrices into one frame with a distinct prefix per view.</summary>
    public NivaraFrame ToFrame()
    {
        var frame = Content;
        foreach (var matrix in new[] { Title, Section, Code })
        {
            foreach (var name in matrix.ColumnNames)
                frame = frame.WithColumn(name, matrix.GetColumn<float>(name));
        }

        return frame;
    }
}

/// <summary>
/// Scores the four #914 views independently, fuses them late, names the winning view and reports
/// where the views disagree. Everything is sample-local and works on the embedding convention in
/// <see cref="NivaraEmbeddingExtensions"/>.
/// </summary>
public static class MultiViewScoring
{
    static readonly ViewSpace[] s_spaces = Enum.GetValues<ViewSpace>();

    /// <summary>Gap from the top fused score up to which two hits are called near-equal.</summary>
    public const float NearBandBound = 0.02f;

    /// <summary>Gap from the top fused score up to which a hit is still distinct.</summary>
    public const float DistinctBandBound = 0.08f;

    public static string BandFor(float gap) => gap switch
    {
        <= NearBandBound => "Near",
        <= DistinctBandBound => "Distinct",
        _ => "Weak"
    };

    /// <summary>
    /// Lands the chunks, their four vectors and the Content/Title coherence as one frame. All four
    /// vector columns are reference-type <c>float[]</c> columns, so they must arrive in one
    /// <see cref="NivaraFrame.Create((string, IColumn)[])"/> call: <c>WithColumn</c> rejects
    /// <c>float[]</c>.
    /// </summary>
    public static NivaraFrame BuildFrame(
        IReadOnlyList<MultiViewChunk> chunks,
        IReadOnlyList<float[]?> contentVectors,
        IReadOnlyList<float[]?> titleVectors,
        IReadOnlyList<float[]?> sectionVectors,
        IReadOnlyList<float[]?> codeVectors)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(contentVectors);
        ArgumentNullException.ThrowIfNull(titleVectors);
        ArgumentNullException.ThrowIfNull(sectionVectors);
        ArgumentNullException.ThrowIfNull(codeVectors);

        var count = chunks.Count;
        if (contentVectors.Count != count || titleVectors.Count != count ||
            sectionVectors.Count != count || codeVectors.Count != count)
            throw new ArgumentException("Every view must supply one vector per chunk.", nameof(chunks));

        var source = new string[count];
        var sourcePath = new string[count];
        var title = new string[count];
        var section = new string[count];
        var aclGroup = new string[count];
        var content = new string[count];
        var code = new string[count];
        var chunkId = new string[count];
        var coherence = new float[count];
        var coherenceNull = new bool[count];

        for (var row = 0; row < count; row++)
        {
            var chunk = chunks[row];
            chunkId[row] = chunk.ChunkId;
            source[row] = chunk.Source;
            sourcePath[row] = chunk.SourcePath;
            title[row] = chunk.Title;
            section[row] = chunk.Section;
            aclGroup[row] = chunk.AclGroup;
            content[row] = chunk.Content;
            code[row] = chunk.Code!;

            var value = Cosine(contentVectors[row], titleVectors[row]);
            if (value is null)
                coherenceNull[row] = true;
            else
                coherence[row] = value.Value;
        }

        return NivaraFrame.Create(
            ("ChunkId", (IColumn)NivaraColumn<string>.CreateForReferenceType(chunkId)),
            ("Source", (IColumn)NivaraColumn<string>.CreateForReferenceType(source)),
            ("SourcePath", (IColumn)NivaraColumn<string>.CreateForReferenceType(sourcePath)),
            ("Title", (IColumn)NivaraColumn<string>.CreateForReferenceType(title)),
            ("Section", (IColumn)NivaraColumn<string>.CreateForReferenceType(section)),
            ("AclGroup", (IColumn)NivaraColumn<string>.CreateForReferenceType(aclGroup)),
            ("Content", (IColumn)NivaraColumn<string>.CreateForReferenceType(content)),
            ("Code", (IColumn)NivaraColumn<string>.CreateForReferenceType(code)),
            ("VectorContent", (IColumn)VectorColumn(contentVectors)),
            ("VectorTitle", (IColumn)VectorColumn(titleVectors)),
            ("VectorSection", (IColumn)VectorColumn(sectionVectors)),
            ("VectorCode", (IColumn)VectorColumn(codeVectors)),
            ("CoherenceContentTitle", (IColumn)NivaraColumn<float>.CreateFromSpans(coherence, coherenceNull)));
    }

    /// <summary>
    /// Scores each view, fuses the scores with <paramref name="weights"/> renormalized per row over
    /// the views that have one, and returns the top <paramref name="topK"/> rows by fused score.
    /// </summary>
    /// <remarks>
    /// A null, non-finite or null-vector score counts as a missing view: it is excluded from fusion,
    /// from <see cref="MultiViewHit.BestSpace"/> and from divergence. Top-K labels are positional, so
    /// a duplicate chunk id cannot make the reported chunk differ from the scored row.
    /// </remarks>
    public static IReadOnlyList<MultiViewHit> Rank(
        NivaraFrame vectorFrame,
        MultiViewMatrices matrices,
        IReadOnlyList<MultiViewChunk> chunks,
        float[]? query,
        FusionWeights weights,
        int topK)
    {
        ArgumentNullException.ThrowIfNull(vectorFrame);
        ArgumentNullException.ThrowIfNull(matrices);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(weights);

        var count = chunks.Count;
        if (vectorFrame.RowCount != count)
            throw new ArgumentException(
                $"Vector frame row count ({vectorFrame.RowCount}) must match the chunk count ({count}).",
                nameof(chunks));
        foreach (var matrix in new[] { matrices.Content, matrices.Title, matrices.Section, matrices.Code })
        {
            if (matrix.RowCount != count)
                throw new ArgumentException(
                    $"Matrix row count ({matrix.RowCount}) must match the chunk count ({count}).",
                    nameof(matrices));
        }

        if (topK <= 0 || count == 0)
            return [];

        var scoreRows = ScoreAll(matrices, count, query);
        if (scoreRows.Length == 0)
            return [];

        var coherence = vectorFrame.GetColumn<float>("CoherenceContentTitle");
        var fusedValues = new float[count];
        var fusedNull = new bool[count];
        var hits = new MultiViewHit?[count];

        for (var row = 0; row < count; row++)
        {
            var viewScores = scoreRows[row];
            var fused = Fuse(viewScores, weights);
            if (fused is null)
            {
                fusedNull[row] = true;
                hits[row] = new MultiViewHit(
                    row, chunks[row], viewScores, null, null, null, null, null, null,
                    coherence.IsNull(row) ? null : coherence[row], null);
                continue;
            }

            fusedValues[row] = fused.Value;
            var titleDivergence = Divergence(viewScores.Content, viewScores.Title);
            var sectionDivergence = Divergence(viewScores.Content, viewScores.Section);
            var codeDivergence = Divergence(viewScores.Content, viewScores.Code);
            hits[row] = new MultiViewHit(
                row,
                chunks[row],
                viewScores,
                fused,
                BestSpace(viewScores),
                titleDivergence,
                sectionDivergence,
                codeDivergence,
                Max(titleDivergence, sectionDivergence, codeDivergence),
                coherence.IsNull(row) ? null : coherence[row],
                null);
        }

        var fusedSeries = new NivaraSeries<float>(NivaraColumn<float>.CreateFromSpans(fusedValues, fusedNull));
        var top = fusedSeries.TopKDescending(topK);
        var ranked = new List<MultiViewHit>(top.Length);
        float? topScore = null;
        foreach (var (label, score) in top)
        {
            if (!int.TryParse(label, out var row) || row < 0 || row >= count || hits[row] is not { } hit)
                continue;

            topScore ??= score;
            var gap = topScore.Value - score;
            ranked.Add(hit with { RankBand = BandFor(gap) });
        }

        return ranked;
    }

    /// <summary>
    /// Scores every row under each of the four views, without ranking or fusing. A null entry means
    /// the row has no vector for that view (or the score was not finite). Returns an empty array when
    /// the query is missing, so <see cref="Rank"/> and the training pipeline share one empty-input
    /// rule instead of each inventing its own.
    /// </summary>
    /// <param name="matrices">The four <c>[rows, dim]</c> view matrices</param>
    /// <param name="rowCount">Number of rows the caller expects; must match every matrix</param>
    /// <param name="query">The query embedding, or null when embedding failed</param>
    /// <exception cref="ArgumentException">
    /// A matrix row or column count does not match the caller's expectation.
    /// </exception>
    public static ViewScores[] ScoreAll(MultiViewMatrices matrices, int rowCount, float[]? query)
    {
        ArgumentNullException.ThrowIfNull(matrices);
        ArgumentOutOfRangeException.ThrowIfNegative(rowCount);

        foreach (var matrix in new[] { matrices.Content, matrices.Title, matrices.Section, matrices.Code })
        {
            if (matrix.RowCount != rowCount)
                throw new ArgumentException(
                    $"Matrix row count ({matrix.RowCount}) must match the expected row count ({rowCount}).",
                    nameof(rowCount));
        }

        if (rowCount == 0 || query is null || query.Length == 0)
            return [];

        if (matrices.Content.ColumnCount != query.Length)
            throw new ArgumentException(
                $"Query length ({query.Length}) must match the embedding dimension ({matrices.Content.ColumnCount}).",
                nameof(query));

        var querySeries = NivaraSeries<float>.Create(query);
        var scores = new[]
        {
            ReadScores(matrices.Content.RowCosineSimilarity(querySeries), rowCount),
            ReadScores(matrices.Title.RowCosineSimilarity(querySeries), rowCount),
            ReadScores(matrices.Section.RowCosineSimilarity(querySeries), rowCount),
            ReadScores(matrices.Code.RowCosineSimilarity(querySeries), rowCount)
        };

        var rows = new ViewScores[rowCount];
        for (var row = 0; row < rowCount; row++)
            rows[row] = new ViewScores(scores[0][row], scores[1][row], scores[2][row], scores[3][row]);

        return rows;
    }

    static NivaraColumn<float[]> VectorColumn(IReadOnlyList<float[]?> vectors)
    {
        var values = new float[vectors.Count][];
        for (var i = 0; i < vectors.Count; i++)
            values[i] = vectors[i]!;
        return NivaraColumn<float[]>.CreateForReferenceType(values);
    }

    static float?[] ReadScores(NivaraSeries<float> series, int count)
    {
        var values = new float?[count];
        for (var row = 0; row < count; row++)
        {
            if (series.Values.IsNull(row))
                continue;

            var score = series.Values[row];
            values[row] = float.IsFinite(score) ? score : null;
        }

        return values;
    }

    /// <summary>
    /// Fuses one row's view scores with <paramref name="weights"/>, renormalizing per row over the
    /// views that actually have a score. A view whose weight is zero is skipped rather than allowed
    /// to shrink the denominator, and a row with no scored view returns null.
    /// </summary>
    /// <remarks>
    /// This is the single fusion arithmetic for the sample: <see cref="Rank"/> calls it, and the
    /// #915 trainer asserts its AutoDiff representation agrees with it at initialisation.
    /// </remarks>
    public static float? Fuse(ViewScores scores, FusionWeights weights)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(weights);

        var total = 0f;
        var available = 0f;
        foreach (var space in s_spaces)
        {
            if (scores[space] is not { } score)
                continue;

            var weight = weights[space];
            if (weight <= 0)
                continue;

            total += weight * score;
            available += weight;
        }

        return available > 0 ? total / available : null;
    }

    static ViewSpace? BestSpace(ViewScores scores)
    {
        ViewSpace? best = null;
        var bestScore = float.NegativeInfinity;
        foreach (var space in s_spaces)
        {
            if (scores[space] is not { } score || score <= bestScore)
                continue;

            best = space;
            bestScore = score;
        }

        return best;
    }

    static float? Divergence(float? left, float? right) =>
        left is { } l && right is { } r ? MathF.Abs(l - r) : null;

    static float? Max(params float?[] values)
    {
        float? max = null;
        foreach (var value in values)
        {
            if (value is { } present && (max is null || present > max.Value))
                max = present;
        }

        return max;
    }

    /// <summary>Cosine of two view vectors, or null when either is missing or degenerate.</summary>
    static float? Cosine(float[]? left, float[]? right)
    {
        if (left is null || right is null || left.Length == 0 || left.Length != right.Length)
            return null;

        var score = TensorPrimitives.CosineSimilarity(left, right);
        return float.IsFinite(score) ? score : null;
    }
}
