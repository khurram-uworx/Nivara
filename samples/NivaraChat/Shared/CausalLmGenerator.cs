namespace NivaraChat.Shared;

/// <summary>
/// Shared autoregressive generation loop for causal LMs. Works over a <see cref="CausalLmOps"/>
/// bundle so improvements (sampling, overflow) propagate once.
/// </summary>
internal static class CausalLmGenerator
{
    public static IEnumerable<int> Generate(
        CausalLmOps ops,
        IReadOnlyList<int> promptIds,
        int maxNewTokens,
        float temperature,
        float topP,
        Random rng,
        bool useKvCache,
        CancellationToken cancellationToken,
        bool windowPrompt = true,
        bool windowNonCached = true,
        int overflowKeep = 0,
        bool breakOnContextLimit = false)
    {
        var ids = new List<int>(promptIds);
        int maxNew = maxNewTokens > 0 ? maxNewTokens : 1;

        if (useKvCache)
        {
            ops.ResetCache();

            int seqLen = windowPrompt
                ? Math.Min(ids.Count, ops.MaxSequenceLength)
                : ids.Count;
            if (seqLen < 1) seqLen = 1;

            var window = ids.GetRange(ids.Count - seqLen, seqLen);
            var logits = ops.Prefill(window.ToArray());
            int position = window.Count;

            int generated = 0;
            while (generated < maxNew)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int next = TokenSampling.Sample(logits.AsSpan(), ops.VocabSize, temperature, topP, rng);
                if (ops.IsStopToken(next))
                    yield break;

                ids.Add(next);
                generated++;
                yield return next;

                if (overflowKeep > 0 && position >= ops.MaxSequenceLength)
                {
                    int keep = overflowKeep;
                    var w = ids.GetRange(ids.Count - 1 - keep, keep);
                    ops.ResetCache();
                    ops.Prefill(w.ToArray());
                    position = keep;
                }

                logits = ops.Cached(next, position++);
            }
        }
        else
        {
            int position = ids.Count;
            int generated = 0;
            while (generated < maxNew)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int seqLen = windowNonCached
                    ? Math.Min(ids.Count, ops.MaxSequenceLength)
                    : ids.Count;
                if (seqLen < 1) seqLen = 1;

                var window = ids.GetRange(ids.Count - seqLen, seqLen);
                var logits = ops.LastLogits(window.ToArray());

                int next = TokenSampling.Sample(logits.AsSpan(), ops.VocabSize, temperature, topP, rng);
                if (ops.IsStopToken(next))
                    yield break;

                ids.Add(next);
                generated++;
                yield return next;

                if (breakOnContextLimit && generated >= ops.MaxSequenceLength - position)
                    yield break;
            }
        }
    }
}
