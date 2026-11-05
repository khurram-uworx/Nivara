using System.Text;

namespace NivaraChat.Helpers;

/// <summary>One bucket of the disagreement report and the examples to show for it.</summary>
public sealed record AgreementBucket(string Name, float? UpperBound, int Count, IReadOnlyList<MultiViewHit> Examples);

/// <summary>
/// Turns ranked <see cref="MultiViewHit"/>s into the #914 agreement report: how many hits have views
/// that agree, and how widely they disagree when they do not. Bounds are absolute cosine differences,
/// so they are comparable across queries.
/// </summary>
public static class MultiViewExplorer
{
    public const float AgreeBound = 0.05f;
    public const float MildBound = 0.15f;

    public const string AgreeBucket = "Agree";
    public const string MildBucket = "Mild";
    public const string StrongBucket = "Strong";
    public const string InsufficientViewsBucket = "InsufficientViews";

    const int ExampleLimit = 3;

    /// <summary>
    /// Buckets every supplied hit. A hit with fewer than two scored views has nothing to compare, so
    /// it lands in <see cref="InsufficientViewsBucket"/> rather than being counted as agreement.
    /// </summary>
    public static IReadOnlyList<AgreementBucket> Bucketize(IReadOnlyList<MultiViewHit> hits)
    {
        ArgumentNullException.ThrowIfNull(hits);

        var agree = new List<MultiViewHit>();
        var mild = new List<MultiViewHit>();
        var strong = new List<MultiViewHit>();
        var insufficient = new List<MultiViewHit>();

        foreach (var hit in hits)
        {
            switch (BucketFor(hit))
            {
                case AgreeBucket: agree.Add(hit); break;
                case MildBucket: mild.Add(hit); break;
                case StrongBucket: strong.Add(hit); break;
                default: insufficient.Add(hit); break;
            }
        }

        return
        [
            new AgreementBucket(AgreeBucket, AgreeBound, agree.Count, [.. agree.Take(ExampleLimit)]),
            new AgreementBucket(MildBucket, MildBound, mild.Count, [.. mild.Take(ExampleLimit)]),
            new AgreementBucket(StrongBucket, null, strong.Count, [.. strong.Take(ExampleLimit)]),
            new AgreementBucket(InsufficientViewsBucket, null, insufficient.Count, [.. insufficient.Take(ExampleLimit)])
        ];
    }

    /// <summary>Bucket name for a single hit.</summary>
    public static string BucketFor(MultiViewHit hit)
    {
        ArgumentNullException.ThrowIfNull(hit);
        if (hit.Scores.PresentCount < 2)
            return InsufficientViewsBucket;

        var divergence = hit.MaxDivergence ?? 0f;
        if (divergence <= AgreeBound)
            return AgreeBucket;
        return divergence <= MildBound ? MildBucket : StrongBucket;
    }

    /// <summary>
    /// One line naming the winning view and the widest disagreement, e.g.
    /// <c>Section wins at 1.000 (fused 0.615, Distinct); widest disagreement: Content vs Section 0.293</c>.
    /// </summary>
    public static string Explain(MultiViewHit hit)
    {
        ArgumentNullException.ThrowIfNull(hit);

        if (hit.Scores.PresentCount == 0)
            return $"{hit.Chunk.ChunkId}: no view scored.";

        var best = hit.BestSpace is { } space ? space.ToString() : "no view";
        var bestScore = hit.BestSpace is { } bestSpace ? hit.Scores[bestSpace] ?? 0f : 0f;
        var fused = hit.Fused is { } value ? value.ToString("0.000") : "none";
        var band = hit.RankBand ?? "Unranked";

        if (hit.Scores.PresentCount == 1)
            return $"{best} wins at {bestScore:0.000} (fused {fused}, {band}); only one view scored, so there is no disagreement to report.";

        var (pair, divergence) = WidestDivergence(hit);
        var verdict = divergence <= AgreeBound
            ? "views agree"
            : divergence <= MildBound ? "views mildly disagree" : "views strongly disagree";

        return $"{best} wins at {bestScore:0.000} (fused {fused}, {band}); widest disagreement: {pair} {divergence:0.000} ({verdict}).";
    }

    /// <summary>Renders the bucket table shared by the console report and the markdown report.</summary>
    public static string Describe(IReadOnlyList<AgreementBucket> buckets)
    {
        ArgumentNullException.ThrowIfNull(buckets);

        var builder = new StringBuilder();
        foreach (var bucket in buckets)
        {
            var bound = bucket.UpperBound is { } upper ? $" (max divergence <= {upper:0.00})" : string.Empty;
            builder.AppendLine($"{bucket.Name}{bound}: {bucket.Count}");
            foreach (var example in bucket.Examples)
                builder.AppendLine($"  {Explain(example)}");
        }

        return builder.ToString().TrimEnd();
    }

    static (string Pair, float Divergence) WidestDivergence(MultiViewHit hit)
    {
        var pair = "Content vs Title";
        var divergence = 0f;
        var found = false;

        void Consider(string name, float? value)
        {
            if (value is not { } present || (found && present <= divergence))
                return;

            pair = $"Content vs {name}";
            divergence = present;
            found = true;
        }

        Consider("Title", hit.DivergenceContentTitle);
        Consider("Section", hit.DivergenceContentSection);
        Consider("Code", hit.DivergenceContentCode);
        return (pair, divergence);
    }
}
