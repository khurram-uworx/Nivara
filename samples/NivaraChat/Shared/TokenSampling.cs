using System.Buffers;
using System.Numerics;

namespace NivaraChat.Shared;

public static class TokenSampling
{
    public static int Sample<T>(ReadOnlySpan<T> logits, int vocab, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => SampleCore(logits, vocab, temperature, topP, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, int vocab, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => SampleCore(logits, vocab, temperature, topP, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => SampleCore(logits, logits.Length, temperature, topP, rng);

    static int SampleCore<T>(ReadOnlySpan<T> logits, int vocab, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (vocab <= 0 || logits.IsEmpty) return 0;
        if (logits.Length < vocab) vocab = logits.Length;

        if (temperature <= 0f)
            return ArgMax<T>(logits, vocab);

        var exp = ArrayPool<double>.Shared.Rent(vocab);
        try
        {
            double max = double.CreateChecked(logits[0]);
            for (int i = 1; i < vocab; i++)
            {
                double v = double.CreateChecked(logits[i]);
                if (v > max) max = v;
            }

            double sum = 0;
            for (int i = 0; i < vocab; i++)
            {
                double v = double.CreateChecked(logits[i]);
                exp[i] = Math.Exp((v - max) / temperature);
                sum += exp[i];
            }

            if (sum <= 0 || !double.IsFinite(sum))
                return ArgMax<T>(logits, vocab);

            if (topP < 1f)
            {
                // Avoid full-vocab sort: use a max-heap to extract descending until cum >= topP.
                var pq = new PriorityQueue<int, double>(vocab);
                for (int i = 0; i < vocab; i++)
                    pq.Enqueue(i, -exp[i]);

                var nucleus = new List<int>(Math.Min(vocab, 64));
                double cum = 0;
                int cut = vocab;
                while (pq.Count > 0)
                {
                    int idx = pq.Dequeue();
                    nucleus.Add(idx);
                    cum += exp[idx];
                    if (cum >= topP) { cut = nucleus.Count; break; }
                }
                if (nucleus.Count == 0)
                    return ArgMax<T>(logits, vocab);

                double kept = 0;
                for (int i = 0; i < cut; i++) kept += exp[nucleus[i]];

                double r = rng.NextDouble() * kept;
                double acc = 0;
                for (int i = 0; i < cut; i++)
                {
                    acc += exp[nucleus[i]];
                    if (r <= acc) return nucleus[i];
                }
                return nucleus[0];
            }

            double rFull = rng.NextDouble() * sum;
            double accFull = 0;
            for (int i = 0; i < vocab; i++)
            {
                accFull += exp[i];
                if (rFull <= accFull) return i;
            }
            return vocab - 1;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(exp);
        }
    }

    static int ArgMax<U>(ReadOnlySpan<U> logits, int vocab) where U : struct, IFloatingPointIeee754<U>
    {
        int best = 0;
        double bestValue = double.CreateChecked(logits[0]);
        for (int i = 1; i < vocab; i++)
        {
            double v = double.CreateChecked(logits[i]);
            if (v > bestValue)
            {
                bestValue = v;
                best = i;
            }
        }
        return best;
    }
}

public static class SamplingHelper
{
    public static int Sample<T>(ReadOnlySpan<T> logits, int vocab, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, vocab, temperature, topP, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, int vocab, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, temperature, topP, vocab, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, temperature, topP, rng);
}

public static class TokenSampler
{
    public static int Sample<T>(ReadOnlySpan<T> logits, int vocab, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, vocab, temperature, topP, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, int vocab, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, temperature, topP, vocab, rng);

    public static int Sample<T>(ReadOnlySpan<T> logits, float temperature, float topP, Random rng)
        where T : struct, IFloatingPointIeee754<T>
        => TokenSampling.Sample(logits, temperature, topP, rng);
}