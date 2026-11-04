namespace NivaraClassifier;

/// <summary>
/// Median/min/max of one measurement round, in milliseconds. <see cref="Measured"/> is false only
/// for the sentinel used when a cell could not be timed at all.
/// </summary>
internal readonly record struct TimingSummary(double MedianMs, double MinMs, double MaxMs)
{
    internal static readonly TimingSummary NotMeasured = new(double.NaN, double.NaN, double.NaN);

    internal bool Measured => !double.IsNaN(MedianMs);
}

/// <summary>
/// The statistics behind the <c>--benchmark</c> table. Pure arithmetic, kept apart from the
/// measurement loop so the summary and the round-length rule can be asserted without timing.
/// </summary>
internal static class BenchStats
{
    /// <summary>
    /// Band within which a ratio is reported as noise rather than as a speed difference. Five
    /// percent is well above what the interleaved rounds showed on the classifier legs, and a
    /// narrower band would manufacture findings out of a single scheduling hiccup.
    /// </summary>
    internal const double NoiseBand = 0.05;

    /// <summary>
    /// Median is the primary figure, with min and max alongside it. A round is an inner loop of N
    /// iterations divided by N, so the samples here are already per-operation milliseconds.
    /// </summary>
    internal static TimingSummary Summarize(ReadOnlySpan<double> samplesMs)
    {
        if (samplesMs.IsEmpty)
            throw new ArgumentException("At least one sample is required.", nameof(samplesMs));

        double[] sorted = samplesMs.ToArray();
        Array.Sort(sorted);
        double median = sorted.Length % 2 == 1
            ? sorted[sorted.Length / 2]
            : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2.0;
        return new TimingSummary(median, sorted[0], sorted[^1]);
    }

    /// <summary>
    /// Round length: enough iterations that a round of <paramref name="budgetMs"/> dominates the
    /// timer's resolution, however fast the operation is. A cost at or below the resolution cannot
    /// be divided out, so it asks for the maximum instead.
    /// </summary>
    internal static int Iterations(double perIterationMs, double budgetMs, int min = 1, int max = 2000)
    {
        if (budgetMs <= 0) throw new ArgumentOutOfRangeException(nameof(budgetMs));
        if (min <= 0) throw new ArgumentOutOfRangeException(nameof(min));
        if (max < min) throw new ArgumentOutOfRangeException(nameof(max));

        if (perIterationMs <= 0 || double.IsNaN(perIterationMs))
            return max;

        double wanted = Math.Ceiling(budgetMs / perIterationMs);
        if (wanted >= max) return max;
        if (wanted <= min) return min;
        return (int)wanted;
    }

    /// <summary>
    /// <paramref name="ratio"/> is the candidate median over the reference median, so 1.0 is parity.
    /// Callers only classify a row whose two sides were both measured.
    /// </summary>
    internal static Verdict Classify(double ratio, double noiseBand = NoiseBand)
    {
        if (ratio < 1.0 - noiseBand) return Verdict.Faster;
        if (ratio > 1.0 + noiseBand) return Verdict.Slower;
        return Verdict.WithinNoise;
    }

    internal static double Ratio(TimingSummary candidate, TimingSummary reference)
        => candidate.Measured && reference.Measured && reference.MedianMs > 0
            ? candidate.MedianMs / reference.MedianMs
            : double.NaN;

    internal enum Verdict
    {
        Faster,
        Slower,
        WithinNoise
    }
}
