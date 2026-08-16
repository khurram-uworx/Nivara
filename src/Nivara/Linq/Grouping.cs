namespace Nivara.Linq;

/// <summary>
/// Marker type for typed LINQ grouping results. Instances are never created: it exists so the C#
/// compiler can infer types for <c>GroupBy(k => ...).Select(g => new { g.Key, ... })</c> and so
/// aggregate calls such as <c>g.Average(p => p.Salary)</c> resolve to concrete signatures. The
/// typed expression translator recognizes these method calls at query-build time and maps them to
/// <see cref="AggregationFunction"/> instances; the bodies are never executed.
/// </summary>
/// <typeparam name="TKey">The group key type</typeparam>
/// <typeparam name="T">The element (row) type being grouped</typeparam>
public sealed class Grouping<TKey, T>
{
    /// <summary>
    /// Gets the group key. Never invoked at runtime.
    /// </summary>
    public TKey Key { get; } = default!;

    /// <summary>
    /// Counts the rows in the group. Translated to a <see cref="RowCountAggregation"/>.
    /// </summary>
    public long Count() => Unreachable<long>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, byte> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, short> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, int> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, long> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, float> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, double> selector) => Unreachable<double>();

    /// <summary>
    /// Averages the selected values in the group. Translated to a <see cref="MeanAggregation"/>.
    /// </summary>
    public double Average(Func<T, decimal> selector) => Unreachable<double>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public long Sum(Func<T, byte> selector) => Unreachable<long>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public long Sum(Func<T, short> selector) => Unreachable<long>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public long Sum(Func<T, int> selector) => Unreachable<long>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public long Sum(Func<T, long> selector) => Unreachable<long>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public double Sum(Func<T, float> selector) => Unreachable<double>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public double Sum(Func<T, double> selector) => Unreachable<double>();

    /// <summary>
    /// Sums the selected values in the group. Translated to a <see cref="SumAggregation"/>.
    /// </summary>
    public decimal Sum(Func<T, decimal> selector) => Unreachable<decimal>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public byte Min(Func<T, byte> selector) => Unreachable<byte>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public short Min(Func<T, short> selector) => Unreachable<short>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public int Min(Func<T, int> selector) => Unreachable<int>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public long Min(Func<T, long> selector) => Unreachable<long>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public float Min(Func<T, float> selector) => Unreachable<float>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public double Min(Func<T, double> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the minimum of the selected values in the group. Translated to a <see cref="MinAggregation"/>.
    /// </summary>
    public decimal Min(Func<T, decimal> selector) => Unreachable<decimal>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public byte Max(Func<T, byte> selector) => Unreachable<byte>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public short Max(Func<T, short> selector) => Unreachable<short>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public int Max(Func<T, int> selector) => Unreachable<int>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public long Max(Func<T, long> selector) => Unreachable<long>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public float Max(Func<T, float> selector) => Unreachable<float>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public double Max(Func<T, double> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the maximum of the selected values in the group. Translated to a <see cref="MaxAggregation"/>.
    /// </summary>
    public decimal Max(Func<T, decimal> selector) => Unreachable<decimal>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>; the result is a double because the median of even-length
    /// data is fractional.
    /// </summary>
    public double Median(Func<T, byte> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, short> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, int> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, long> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, float> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, double> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the median (0.5 quantile) of the selected values in the group. Translated to a
    /// <see cref="MedianAggregation"/>.
    /// </summary>
    public double Median(Func<T, decimal> selector) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation; the result is a double.
    /// </summary>
    public double Quantile(Func<T, byte> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, short> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, int> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, long> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, float> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, double> selector, double q) => Unreachable<double>();

    /// <summary>
    /// Returns the q-th quantile of the selected values in the group. Translated to a
    /// <see cref="QuantileAggregation"/> with linear interpolation.
    /// </summary>
    public double Quantile(Func<T, decimal> selector, double q) => Unreachable<double>();

    static TValue Unreachable<TValue>()
        => throw new NotSupportedException("Grouping<TKey, T> instances are marker types and are never invoked.");
}
