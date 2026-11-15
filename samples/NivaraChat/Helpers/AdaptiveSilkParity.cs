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
}
