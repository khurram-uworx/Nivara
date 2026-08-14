using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Samples token indices from logits using temperature-scaled softmax with optional top-k
/// filtering. Intended for autoregressive text generation.
/// </summary>
public sealed class Sampler<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Random rng;

    /// <summary>
    /// Creates a sampler with an optional seed for reproducible results.
    /// </summary>
    /// <param name="seed">Optional RNG seed; when null, a shared non-deterministic RNG is used</param>
    public Sampler(int? seed = null)
    {
        rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    }

    /// <summary>
    /// Samples one token index from a vector of logits.
    /// </summary>
    /// <param name="logits">The logits tensor (one value per candidate)</param>
    /// <param name="temperature">Softmax temperature; higher values flatten the distribution</param>
    /// <param name="topK">When positive, restrict sampling to the top-k highest logits</param>
    /// <returns>The sampled token index</returns>
    public int Sample(ReverseGradTensor<T> logits, double temperature = 1.0, int topK = 0)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (logits.Length == 0) throw new ArgumentException("Logits tensor is empty.", nameof(logits));

        int vocabSize = logits.Length;
        var probs = new float[vocabSize];

        // Copy logits to float array and apply temperature
        float maxVal = float.MinValue;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = float.CreateChecked(logits[i]) / (float)temperature;
            if (probs[i] > maxVal) maxVal = probs[i];
        }

        // Apply top-k filtering
        if (topK > 0 && topK < vocabSize)
        {
            // Find the k-th largest value
            var sorted = probs.OrderByDescending(x => x).ToArray();
            float threshold = sorted[topK - 1];
            for (int i = 0; i < vocabSize; i++)
            {
                if (probs[i] < threshold)
                    probs[i] = float.NegativeInfinity;
            }
        }

        // Softmax (numerically stable)
        float sumExp = 0f;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = MathF.Exp(probs[i] - maxVal);
            sumExp += probs[i];
        }
        for (int i = 0; i < vocabSize; i++)
            probs[i] /= sumExp;

        // Categorical sampling
        double d = rng.NextDouble();
        double cum = 0f;
        int next = vocabSize - 1;
        for (int i = 0; i < vocabSize; i++)
        {
            cum += probs[i];
            if (d < cum) { next = i; break; }
        }

        return next;
    }
}
