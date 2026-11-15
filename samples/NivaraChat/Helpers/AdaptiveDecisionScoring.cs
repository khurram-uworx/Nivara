namespace NivaraChat.Helpers;

/// <summary>
/// One retrieved candidate as seen by the adaptive decision layer, with its fused score and the
/// head's verdict on it.
/// </summary>
public sealed record AdaptiveCandidate(
    int Row,
    string ChunkId,
    double FusedScore,
    bool Accepted,
    double AnswerConfidence);

/// <summary>
/// Pure reranking/filtering over the fixed top-K, so the tie-break and the membership invariant
/// can be tested without MiniLM or Laya checkpoints.
/// </summary>
public static class AdaptiveDecisionScoring
{
    /// <summary>
    /// Accepted chunks first (by confidence, then fused score, then row), then the rejected chunks
    /// (by fused score, then row). Membership is preserved, so P@K and R@K are invariant by
    /// construction.
    /// </summary>
    public static IReadOnlyList<int> Rerank(IReadOnlyList<AdaptiveCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        var accepted = candidates
            .Where(candidate => candidate.Accepted)
            .OrderByDescending(candidate => candidate.AnswerConfidence)
            .ThenByDescending(candidate => candidate.FusedScore)
            .ThenBy(candidate => candidate.Row)
            .Select(candidate => candidate.Row);
        var rejected = candidates
            .Where(candidate => !candidate.Accepted)
            .OrderByDescending(candidate => candidate.FusedScore)
            .ThenBy(candidate => candidate.Row)
            .Select(candidate => candidate.Row);
        return accepted.Concat(rejected).ToList();
    }

    /// <summary>
    /// Accepted chunks only, in the same order as <see cref="Rerank"/>. A shorter list than the
    /// fixed top-K, which is what lets precision and recall move.
    /// </summary>
    public static IReadOnlyList<int> Filter(IReadOnlyList<AdaptiveCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        return candidates
            .Where(candidate => candidate.Accepted)
            .OrderByDescending(candidate => candidate.AnswerConfidence)
            .ThenByDescending(candidate => candidate.FusedScore)
            .ThenBy(candidate => candidate.Row)
            .Select(candidate => candidate.Row)
            .ToList();
    }
}
