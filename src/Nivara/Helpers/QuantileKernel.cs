namespace Nivara.Helpers;

/// <summary>
/// Shared quantile/median computation for the aggregation domain. Nivara's quantile follows the
/// common linear-interpolation definition (numpy default / polars interpolation="linear",
/// Hyndman-Fan type 7): sort ascending, then for a quantile <c>q</c> the index is
/// <c>h = q * (n - 1)</c> and the result is <c>x[floor(h)] + (h - floor(h)) * (x[ceil(h)] - x[floor(h)])</c>.
/// The median is <c>Quantile(0.5)</c>, so even-length data averages the two middle values.
/// </summary>
internal static class QuantileKernel
{
    /// <summary>
    /// Computes the q-th quantile of an ascending-sorted value span. Callers must validate that
    /// <paramref name="q"/> is in [0, 1] and that the span is non-empty.
    /// </summary>
    public static double Compute(ReadOnlySpan<double> sortedValues, double q)
    {
        int n = sortedValues.Length;
        if (n == 1)
            return sortedValues[0];

        double position = q * (n - 1);
        int lower = (int)position;
        double fraction = position - lower;
        int upper = fraction == 0d ? lower : lower + 1;

        return sortedValues[lower] + fraction * (sortedValues[upper] - sortedValues[lower]);
    }

    /// <summary>
    /// Computes the q-th quantile from boxed valid values, converting each across the full
    /// 17-type numeric aggregation domain to double before sorting.
    /// </summary>
    /// <returns>The quantile, or null when <paramref name="validValues"/> is empty</returns>
    public static double? ComputeFromBoxed(IReadOnlyList<object> validValues, double q)
    {
        if (validValues.Count == 0)
            return null;

        var values = new double[validValues.Count];
        for (int i = 0; i < values.Length; i++)
            values[i] = ToDouble(validValues[i]);

        values.AsSpan().Sort();
        return Compute(values, q);
    }

    /// <summary>
    /// Converts a boxed numeric value to double across the full numeric aggregation domain.
    /// Int128/UInt128 (and therefore nint/nuint, whose values box to the native integer) do not
    /// implement IConvertible, so a typed switch is required instead of Convert.ChangeType.
    /// </summary>
    public static double ToDouble(object value)
    {
        return value switch
        {
            byte v => v,
            sbyte v => v,
            short v => v,
            ushort v => v,
            int v => v,
            uint v => v,
            char v => v,
            long v => v,
            ulong v => v,
            nint v => (double)v,
            nuint v => (double)v,
            Int128 v => (double)v,
            UInt128 v => (double)v,
            float v => v,
            Half v => (double)v,
            double v => v,
            decimal v => (double)v,
            _ => throw new ArgumentException($"Cannot convert value of type '{value.GetType().Name}' to double for quantile computation")
        };
    }
}
