using Nivara;

namespace NivaraChat.Helpers;

/// <summary>Max-abs parity between two backend embeddings for the same corpus.</summary>
public sealed record SilkParityReport(
    double MaxAbsoluteDifference,
    int Rows,
    int MismatchedRows,
    bool WithinTolerance);

/// <summary>
/// Compares two embedding frames column by column. Pure so it can be tested with hand-built frames.
/// </summary>
public static class AdaptiveSilkParity
{
    public const double Tolerance = 1e-5;

    public static SilkParityReport Compare(
        NivaraFrame cpu,
        NivaraFrame silk,
        IReadOnlyList<string> columns,
        double tolerance = Tolerance)
    {
        ArgumentNullException.ThrowIfNull(cpu);
        ArgumentNullException.ThrowIfNull(silk);
        ArgumentNullException.ThrowIfNull(columns);
        if (cpu.RowCount != silk.RowCount)
            throw new ArgumentException(
                $"Frame row counts differ: {cpu.RowCount} vs {silk.RowCount}.", nameof(silk));

        var worst = 0d;
        var mismatched = 0;
        for (var row = 0; row < cpu.RowCount; row++)
        {
            var rowWorst = 0d;
            foreach (var column in columns)
            {
                var left = cpu.GetColumn<float[]>(column);
                var right = silk.GetColumn<float[]>(column);
                var a = left.IsNull(row) ? null : left[row];
                var b = right.IsNull(row) ? null : right[row];
                if (a is null || b is null || a.Length != b.Length)
                {
                    mismatched++;
                    rowWorst = Math.Max(rowWorst, tolerance + 1);
                    continue;
                }

                for (var i = 0; i < a.Length; i++)
                {
                    var diff = Math.Abs(a[i] - b[i]);
                    if (diff > rowWorst) rowWorst = diff;
                }
            }

            if (rowWorst > worst) worst = rowWorst;
            if (rowWorst > tolerance) mismatched++;
        }

        return new SilkParityReport(worst, cpu.RowCount, mismatched, worst <= tolerance);
    }

    /// <summary>
    /// Compares two flat Laya feature/score vectors element by element, in the same
    /// max-abs style as <see cref="Compare(NivaraFrame, NivaraFrame, IReadOnlyList{string}, double)"/>.
    /// </summary>
    public static SilkParityReport CompareLaya(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareLayaScores(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareLayaFeatures(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareScores(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareFeatures(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareHidden(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport CompareDecisionHead(
        ReadOnlySpan<float> cpu,
        ReadOnlySpan<float> silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    /// <summary>Overload accepting arrays for convenience.</summary>
    public static SilkParityReport Compare(
        float[] cpu,
        float[] silk,
        double tolerance = Tolerance)
        => CompareFloats(cpu, silk, tolerance);

    public static SilkParityReport Compare(
        ReadOnlySpan<double> cpu,
        ReadOnlySpan<double> silk,
        double tolerance = Tolerance)
        => CompareDoubles(cpu, silk, tolerance);

    private static SilkParityReport CompareFloats(ReadOnlySpan<float> cpu, ReadOnlySpan<float> silk, double tolerance)
    {
        if (cpu.Length != silk.Length)
            throw new ArgumentException(
                $"Vector lengths differ: {cpu.Length} vs {silk.Length}.", nameof(silk));

        var worst = 0d;
        var mismatched = 0;
        for (var i = 0; i < cpu.Length; i++)
        {
            var diff = Math.Abs(cpu[i] - silk[i]);
            if (diff > worst) worst = diff;
            if (diff > tolerance) mismatched++;
        }
        return new SilkParityReport(worst, cpu.Length, mismatched, worst <= tolerance);
    }

    private static SilkParityReport CompareDoubles(ReadOnlySpan<double> cpu, ReadOnlySpan<double> silk, double tolerance)
    {
        if (cpu.Length != silk.Length)
            throw new ArgumentException(
                $"Vector lengths differ: {cpu.Length} vs {silk.Length}.", nameof(silk));

        var worst = 0d;
        var mismatched = 0;
        for (var i = 0; i < cpu.Length; i++)
        {
            var diff = Math.Abs(cpu[i] - silk[i]);
            if (diff > worst) worst = diff;
            if (diff > tolerance) mismatched++;
        }
        return new SilkParityReport(worst, cpu.Length, mismatched, worst <= tolerance);
    }
}
