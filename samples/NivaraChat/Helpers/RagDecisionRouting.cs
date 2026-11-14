namespace NivaraChat.Helpers;

/// <summary>
/// Chooses the single chunk the adaptive RAG flow should present to the answer stage.
/// </summary>
public static class RagDecisionRouting
{
    public static LayaChunkDecision SelectBestDecision(
        string query,
        IReadOnlyList<RetrievedChunk> chunks,
        float confidenceThreshold)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(chunks);
        if (chunks.Count == 0)
            throw new ArgumentException("At least one retrieved chunk is required.", nameof(chunks));

        return SelectBestDecisionFrom(LayaRetrievalDecisionHead.Evaluate(query, chunks, confidenceThreshold));
    }

    public static LayaChunkDecision SelectBestDecisionFrom(IReadOnlyList<LayaChunkDecision> decisions)
    {
        ArgumentNullException.ThrowIfNull(decisions);
        if (decisions.Count == 0)
            throw new ArgumentException("At least one Laya decision is required.", nameof(decisions));

        var accepted = decisions
            .Where(d => d.Accepted)
            .OrderByDescending(d => d.AnswerConfidence)
            .ThenByDescending(d => d.RetrievalScore)
            .FirstOrDefault();

        if (accepted is not null)
            return accepted;

        return decisions
            .OrderByDescending(d => d.RetrievalScore)
            .ThenByDescending(d => d.AnswerConfidence)
            .First();
    }
}
