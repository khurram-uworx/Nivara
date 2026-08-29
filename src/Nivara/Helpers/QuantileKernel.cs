using System.Numerics;

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
    /// Computes the q-th quantile from a typed column and group indices, extracting values
    /// directly via the typed <see cref="IColumn{T}"/> indexer to avoid boxing.
    /// </summary>
    /// <returns>The quantile, or null when the group has no valid values</returns>
    public static double? ComputeFromColumn(IColumn column, IReadOnlyList<int> groupIndices, double q)
    {
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        return elementType switch
        {
            Type t when t == typeof(double) => TypedQuantile<double>(column, groupIndices, q, static v => v),
            Type t when t == typeof(float) => TypedQuantile<float>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(int) => TypedQuantile<int>(column, groupIndices, q, static v => v),
            Type t when t == typeof(long) => TypedQuantile<long>(column, groupIndices, q, static v => v),
            Type t when t == typeof(decimal) => TypedQuantile<decimal>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(short) => TypedQuantile<short>(column, groupIndices, q, static v => v),
            Type t when t == typeof(ushort) => TypedQuantile<ushort>(column, groupIndices, q, static v => v),
            Type t when t == typeof(byte) => TypedQuantile<byte>(column, groupIndices, q, static v => v),
            Type t when t == typeof(sbyte) => TypedQuantile<sbyte>(column, groupIndices, q, static v => v),
            Type t when t == typeof(uint) => TypedQuantile<uint>(column, groupIndices, q, static v => v),
            Type t when t == typeof(ulong) => TypedQuantile<ulong>(column, groupIndices, q, static v => v),
            Type t when t == typeof(char) => TypedQuantile<char>(column, groupIndices, q, static v => v),
            Type t when t == typeof(bool) => TypedQuantile<bool>(column, groupIndices, q, static v => v ? 1d : 0d),
            Type t when t == typeof(nint) => TypedQuantile<nint>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(nuint) => TypedQuantile<nuint>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(Int128) => TypedQuantile<Int128>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(UInt128) => TypedQuantile<UInt128>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(Half) => TypedQuantile<Half>(column, groupIndices, q, static v => (double)v),
            Type t when t == typeof(BFloat16) => TypedQuantile<BFloat16>(column, groupIndices, q, static v => (double)v),
            _ => ComputeFromBoxed(ExtractBoxedValues(column, groupIndices), q)
        };
    }

    static double? TypedQuantile<T>(IColumn column, IReadOnlyList<int> groupIndices, double q, Func<T, double> toDouble)
        where T : struct
    {
        // Nullable-element columns (NivaraColumn<T?>) implement IColumn<T?>, not IColumn<T>, so
        // read through that view and rely on the nullable indexer's HasValue.
        if (column.ElementType == typeof(T?) && column is IColumn<T?> nullableTyped)
        {
            int nullableCount = 0;
            foreach (var idx in groupIndices)
                if (nullableTyped[idx].HasValue)
                    nullableCount++;
            if (nullableCount == 0)
                return null;

            var nullableValues = new double[nullableCount];
            int nullablePos = 0;
            foreach (var idx in groupIndices)
            {
                var v = nullableTyped[idx];
                if (v.HasValue)
                    nullableValues[nullablePos++] = toDouble(v.GetValueOrDefault());
            }

            nullableValues.AsSpan().Sort();
            return Compute(nullableValues, q);
        }

        var typed = (IColumn<T>)column;

        int count = 0;
        foreach (var idx in groupIndices)
            if (!column.IsNull(idx)) count++;

        if (count == 0)
            return null;

        var values = new double[count];
        int pos = 0;
        foreach (var idx in groupIndices)
        {
            if (!column.IsNull(idx))
                values[pos++] = toDouble(typed[idx]);
        }

        values.AsSpan().Sort();
        return Compute(values, q);
    }

    static List<object> ExtractBoxedValues(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }
        return validValues;
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
            BFloat16 v => (double)v,
            double v => v,
            decimal v => (double)v,
            _ => throw new ArgumentException($"Cannot convert value of type '{value.GetType().Name}' to double for quantile computation")
        };
    }
}
