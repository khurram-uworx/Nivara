using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Helpers;

/// <summary>
/// Shared variance/standard-deviation computation for the aggregation domain. Both follow the
/// standard definition: variance is the mean squared deviation from the mean and the standard
/// deviation is its square root. The <c>ddof</c> parameter is the delta degrees of freedom
/// (numpy <c>ddof</c> semantics): 0 is population (divide by n), 1 is sample (divide by n - 1).
/// </summary>
internal static class MomentsKernel
{
    /// <summary>
    /// Computes the variance of a value span with the given delta degrees of freedom.
    /// </summary>
    /// <param name="values">The values to compute the variance over (non-empty)</param>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="ddof"/> is negative</exception>
    /// <exception cref="InvalidOperationException">Thrown when the span is empty or has fewer than <c>ddof + 1</c> elements</exception>
    public static double Variance(ReadOnlySpan<double> values, int ddof)
    {
        if (ddof < 0)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be >= 0.");
        if (values.Length <= ddof)
            throw new InvalidOperationException($"Cannot compute variance with ddof={ddof} over {values.Length} value(s); need at least {ddof + 1}.");

        var mean = TensorPrimitives.Average<double>(values);
        double sumSquaredDeviation = 0d;
        foreach (var value in values)
        {
            var deviation = value - mean;
            sumSquaredDeviation += deviation * deviation;
        }
        return sumSquaredDeviation / (values.Length - ddof);
    }

    /// <summary>
    /// Computes the standard deviation of a value span with the given delta degrees of freedom.
    /// </summary>
    public static double StdDev(ReadOnlySpan<double> values, int ddof)
        => Math.Sqrt(Variance(values, ddof));

    /// <summary>
    /// Computes the variance or standard deviation of a typed column over the group indices,
    /// reading values through the typed IColumn&lt;T&gt; indexer (no per-element boxing) and
    /// converting each to double across the full numeric aggregation domain.
    /// </summary>
    /// <returns>The moment, or null when the group has no valid values</returns>
    public static double? ComputeFromColumn(IColumn column, IReadOnlyList<int> groupIndices, int ddof, bool variance)
    {
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;
        return elementType switch
        {
            Type t when t == typeof(double) => Compute<double>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(float) => Compute<float>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(int) => Compute<int>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(long) => Compute<long>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(decimal) => Compute<decimal>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(short) => Compute<short>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(ushort) => Compute<ushort>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(byte) => Compute<byte>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(sbyte) => Compute<sbyte>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(uint) => Compute<uint>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(ulong) => Compute<ulong>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(char) => Compute<char>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(nint) => Compute<nint>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(nuint) => Compute<nuint>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(Int128) => Compute<Int128>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(UInt128) => Compute<UInt128>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(Half) => Compute<Half>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            Type t when t == typeof(BFloat16) => Compute<BFloat16>(column, groupIndices, ddof, variance, static v => double.CreateChecked(v)),
            _ => throw new ArgumentException($"Cannot compute standard deviation/variance for type '{elementType.Name}'")
        };
    }

    static double? Compute<T>(IColumn column, IReadOnlyList<int> groupIndices, int ddof, bool variance, Func<T, double> toDouble)
        where T : struct, INumberBase<T>
    {
        // Reuse the nullable-aware typed extraction so NivaraColumn<T?> (which implements
        // IColumn<T?>, not IColumn<T>) is handled without a per-element boxing cast.
        var values = AggregationFunction.ExtractValidTyped<T>(column, groupIndices);
        if (values.Length == 0)
            return null;

        var doubles = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
            doubles[i] = toDouble(values[i]);

        return variance ? Variance(doubles, ddof) : StdDev(doubles, ddof);
    }
}
