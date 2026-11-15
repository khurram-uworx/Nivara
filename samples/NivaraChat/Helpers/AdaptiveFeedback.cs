using Nivara.AI;
using Nivara.AutoDiff.Utilities;

namespace NivaraChat.Helpers;

/// <summary>Per-split caps for adaptive example extraction.</summary>
public sealed record AdaptiveBuildLimits(int MaxQueries, int MaxExamples);

/// <summary>Result of an adaptive example extraction pass.</summary>
public sealed record AdaptiveFeedbackBuild(
    IReadOnlyList<AdaptiveTrainingExample> Examples,
    int QueriesConsidered,
    int CandidatesConsidered,
    int SkippedUnlabelled,
    int SkippedByCap,
    bool Truncated);

/// <summary>
/// Builds adaptive training examples from the retrieval path (top-K per query), so the training
/// distribution matches what the deployed decision layer actually sees. Cap-aware and deterministic.
/// </summary>
public static class AdaptiveFeedback
{
    public static AdaptiveFeedbackBuild Build(
        ResolvedLabels labels,
        IReadOnlyList<MultiViewChunk> chunks,
        IReadOnlyList<string> queries,
        Func<string, IReadOnlyList<int>> retrieve,
        Func<string, MultiViewChunk, AdaptiveFrozenFeatures> extract,
        AdaptiveBuildLimits limits)
    {
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(queries);
        ArgumentNullException.ThrowIfNull(retrieve);
        ArgumentNullException.ThrowIfNull(extract);
        ArgumentNullException.ThrowIfNull(limits);
        if (limits.MaxQueries <= 0)
            throw new ArgumentOutOfRangeException(nameof(limits), limits.MaxQueries, "MaxQueries must be positive.");
        if (limits.MaxExamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(limits), limits.MaxExamples, "MaxExamples must be positive.");

        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("Feature extraction must run outside GradientUtils.Grad().");

        var examples = new List<AdaptiveTrainingExample>();
        var queriesConsidered = 0;
        var candidatesConsidered = 0;
        var skippedUnlabelled = 0;
        var skippedByCap = 0;
        var truncated = false;

        foreach (var query in queries.OrderBy(q => q, StringComparer.Ordinal))
        {
            if (queriesConsidered >= limits.MaxQueries)
            {
                truncated = true;
                break;
            }

            if (!labels.Grades.TryGetValue(query, out var grades))
                continue;

            queriesConsidered++;
            var seenRows = new HashSet<int>();
            foreach (var row in retrieve(query).OrderBy(r => r))
            {
                if (row < 0 || row >= chunks.Count)
                    continue;
                if (!seenRows.Add(row))
                    continue;

                candidatesConsidered++;
                if (!grades.TryGetValue(row, out var grade))
                {
                    skippedUnlabelled++;
                    continue;
                }

                if (examples.Count >= limits.MaxExamples)
                {
                    skippedByCap++;
                    truncated = true;
                    continue;
                }

                var features = extract(query, chunks[row]);
                AdaptiveTrainingData.ValidateFeatures(features);
                examples.Add(AdaptiveTrainingData.CreateExample(query, chunks[row], grade, features));
            }
        }

        return new AdaptiveFeedbackBuild(
            examples, queriesConsidered, candidatesConsidered,
            skippedUnlabelled, skippedByCap, truncated);
    }

    public static bool AllowsAutoLabels(LabelKind kind) =>
        kind is LabelKind.Judge or LabelKind.Auto;
}
