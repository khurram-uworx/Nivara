namespace Nivara.Samples;

/// <summary>
/// The host-side half of the Laya decision head: the softmax, the two-against-1 marker padding
/// mask, and the four features concatenated onto the pooled row.
/// </summary>
/// <remarks>
/// These are host arithmetic in the reference too — <c>DecisionModel.forward</c> computes
/// <c>softmax</c> and the feature vector in Python and hands the result to
/// <c>act_head</c> — so both backends share this one implementation rather than each keeping a
/// copy. That is what lets the GPU gate (docs/LAYA.md) treat the four features as identical by
/// construction on both sides, leaving the comparison to measure only the device math.
/// </remarks>
public static class LayaHeadScoring
{
    /// <summary>
    /// A 1-D key-padding mask of length <paramref name="seqLen"/>: 1 for the real tokens, 0 for the
    /// padding. Suppressing a key is expressed as a value below 0.5, which is what the reference's
    /// <c>~attention_mask</c> means once read as an additive mask.
    /// </summary>
    /// <remarks>
    /// The mask must be one entry per <em>position</em>, not per valid token. A shorter mask leaves
    /// the trailing positions unconstrained, which is not a weaker mask but the wrong one: nothing
    /// after it gets suppressed and the padding is attended to normally.
    /// </remarks>
    public static float[] PrefixMask(int seqLen, int validLength)
    {
        var mask = new float[seqLen];
        for (int i = 0; i < seqLen; i++)
            mask[i] = i < validLength ? 1f : 0f;
        return mask;
    }

    /// <summary>Softmax at temperature 1, in double. The reference's <c>p</c>.</summary>
    public static double[] Softmax(ReadOnlySpan<float> logits)
    {
        var p = new double[logits.Length];
        if (logits.Length == 0)
            return p;

        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            max = Math.Max(max, logits[i]);

        double sum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            p[i] = Math.Exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < logits.Length; i++)
            p[i] /= sum;
        return p;
    }

    /// <summary>
    /// The act head's input row, minus the pooled row: top-1 probability, top-1 minus top-2,
    /// entropy normalized by <c>log(max(2, k))</c>, and <c>max(2, k) / 255</c>.
    /// </summary>
    public static double[] BuildFeatures(ReadOnlySpan<double> probabilities)
    {
        int k = probabilities.Length;
        double clampedK = Math.Max(2, k);
        double entropy = 0.0;
        for (int i = 0; i < k; i++)
        {
            double p = Math.Max(probabilities[i], 1e-9);
            entropy -= p * Math.Log(p);
        }
        entropy /= Math.Log(clampedK);

        // p.topk(2) raises on a one-element distribution, so the reference special-cases it by
        // padding the missing second entry with 0.0. Softmax over one logit is 1.0 whatever the logit
        // is, so top1 - top2 == 1.0, the same "fully decided" signal an unambiguous question gives.
        double top1 = probabilities[0];
        double top2 = k >= 2 ? SecondHighest(probabilities) : 0.0;

        var features = new double[LayaHeadOutput.FeatureCount];
        features[LayaHeadOutput.FeatureTop1] = top1;
        features[LayaHeadOutput.FeatureGap] = top1 - top2;
        features[LayaHeadOutput.FeatureEntropy] = entropy;
        features[LayaHeadOutput.FeatureOptionCount] = clampedK / 255.0;
        return features;
    }

    static double SecondHighest(ReadOnlySpan<double> values)
    {
        double top = double.NegativeInfinity, second = double.NegativeInfinity;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] > top)
            {
                second = top;
                top = values[i];
            }
            else if (values[i] > second)
            {
                second = values[i];
            }
        }
        return second;
    }
}
