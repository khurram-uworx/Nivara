using Nivara.Tensors;
using System.Globalization;
using System.Numerics;

namespace Nivara;

/// <summary>
/// Eager <see cref="NivaraFrame"/> window-function extensions: rolling aggregates,
/// cumulative aggregates, and shift/lead. These mirror the query-engine window API
/// with the same names, argument order, and null semantics.
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
public static partial class NivaraFrameExtensions
{
    // ── Rolling ──

    /// <summary>
    /// Adds a rolling sum column.
    /// </summary>
    public static NivaraFrame RollingSum(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Sum));

    /// <summary>
    /// Adds a rolling mean column.
    /// </summary>
    public static NivaraFrame RollingMean(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Mean));

    /// <summary>
    /// Adds a rolling minimum column.
    /// </summary>
    public static NivaraFrame RollingMin(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Min));

    /// <summary>
    /// Adds a rolling maximum column.
    /// </summary>
    public static NivaraFrame RollingMax(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Max));

    // ── Cumulative ──

    /// <summary>
    /// Adds a cumulative sum column.
    /// </summary>
    public static NivaraFrame CumulativeSum(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateCumulative(c, nullHandler, CumulativeKind.Sum));

    /// <summary>
    /// Adds a cumulative maximum column.
    /// </summary>
    public static NivaraFrame CumulativeMax(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateCumulative(c, nullHandler, CumulativeKind.Max));

    /// <summary>
    /// Adds a cumulative minimum column.
    /// </summary>
    public static NivaraFrame CumulativeMin(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateCumulative(c, nullHandler, CumulativeKind.Min));

    /// <summary>
    /// Adds a cumulative product column.
    /// </summary>
    public static NivaraFrame CumulativeProduct(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateCumulative(c, nullHandler, CumulativeKind.Product));

    /// <summary>
    /// Adds a running count-of-non-null column.
    /// </summary>
    public static NivaraFrame CumulativeCount(this NivaraFrame frame, string source, string resultColumn)
        => addWindowColumn(frame, source, resultColumn, calculateCumulativeCount);

    // ── Shift / Lead ──

    /// <summary>
    /// Adds a shifted (lag) column. Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Shift(this NivaraFrame frame, string source, string resultColumn, int periods, object? fillValue = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateShift(c, periods, fillValue));

    /// <summary>
    /// Adds a lead column (negative shift). Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Lead(this NivaraFrame frame, string source, string resultColumn, int periods, object? fillValue = null)
        => addWindowColumn(frame, source, resultColumn, c => calculateShift(c, -periods, fillValue));

    // ── Shared dispatch ──

    static NivaraFrame addWindowColumn(NivaraFrame frame, string source, string resultColumn, Func<IColumn, IColumn> computation)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var sourceColumn = frame.GetColumn(source);
        var result = computation(sourceColumn);
        return frame.WithColumn(resultColumn, result);
    }

    static IColumn calculateRolling(IColumn column, int windowSize, int? minPeriods, Func<object?>? nullHandler, RollingKind kind)
        => column switch
        {
            NivaraColumn<int> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<long> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<float> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<double> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<decimal> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            _ => throw new NotSupportedException($"Rolling window requires a numeric column, but {column.ElementType.Name} is not supported")
        };

    static IColumn calculateCumulative(IColumn column, Func<object?>? nullHandler, CumulativeKind kind)
        => column switch
        {
            NivaraColumn<int> c => cumulative(c, nullHandler, kind),
            NivaraColumn<long> c => cumulative(c, nullHandler, kind),
            NivaraColumn<float> c => cumulative(c, nullHandler, kind),
            NivaraColumn<double> c => cumulative(c, nullHandler, kind),
            NivaraColumn<decimal> c => cumulative(c, nullHandler, kind),
            _ => throw new NotSupportedException($"Cumulative requires a numeric column, but {column.ElementType.Name} is not supported")
        };

    static IColumn calculateCumulativeCount(IColumn column)
        => column switch
        {
            NivaraColumn<int> c => c.CumulativeCount(),
            NivaraColumn<long> c => c.CumulativeCount(),
            NivaraColumn<float> c => c.CumulativeCount(),
            NivaraColumn<double> c => c.CumulativeCount(),
            NivaraColumn<decimal> c => c.CumulativeCount(),
            NivaraColumn<string> c => c.CumulativeCount(),
            NivaraColumn<bool> c => c.CumulativeCount(),
            _ => throw new NotSupportedException($"CumulativeCount does not support column type {column.ElementType.Name}")
        };

    static IColumn calculateShift(IColumn column, int periods, object? fillValue)
        => column switch
        {
            NivaraColumn<int> c => shift(c, periods, fillValue),
            NivaraColumn<long> c => shift(c, periods, fillValue),
            NivaraColumn<float> c => shift(c, periods, fillValue),
            NivaraColumn<double> c => shift(c, periods, fillValue),
            NivaraColumn<decimal> c => shift(c, periods, fillValue),
            NivaraColumn<string> c => shift(c, periods, fillValue),
            NivaraColumn<bool> c => shift(c, periods, fillValue),
            _ => throw new NotSupportedException($"Shift does not support column type {column.ElementType.Name}")
        };

    static IColumn rolling<T>(NivaraColumn<T> column, int windowSize, int? minPeriods, Func<object?>? nullHandler, RollingKind kind)
        where T : struct, INumber<T>
    {
        var typedHandler = adaptNullHandler<T>(nullHandler);
        return kind switch
        {
            RollingKind.Sum => column.RollingSum(windowSize, minPeriods, typedHandler),
            RollingKind.Mean => column.RollingMean(windowSize, minPeriods, typedHandler),
            RollingKind.Min => column.RollingMin(windowSize, minPeriods, typedHandler),
            _ => column.RollingMax(windowSize, minPeriods, typedHandler)
        };
    }

    static IColumn cumulative<T>(NivaraColumn<T> column, Func<object?>? nullHandler, CumulativeKind kind)
        where T : struct, INumber<T>
    {
        var typedHandler = adaptNullHandler<T>(nullHandler);
        return kind switch
        {
            CumulativeKind.Sum => column.CumulativeSum(typedHandler),
            CumulativeKind.Max => column.CumulativeMax(typedHandler),
            CumulativeKind.Min => column.CumulativeMin(typedHandler),
            _ => column.CumulativeProduct(typedHandler)
        };
    }

    static IColumn shift<T>(NivaraColumn<T> column, int periods, object? fillValue)
        => fillValue is null
            ? column.Shift(periods)
            : column.Shift(periods, convertFillValue<T>(fillValue));

    static Func<T>? adaptNullHandler<T>(Func<object?>? nullHandler)
        where T : struct
        => nullHandler is null
            ? null
            : () => (T)Convert.ChangeType(nullHandler() ?? default(T), typeof(T), CultureInfo.InvariantCulture);

    static T convertFillValue<T>(object? fillValue)
        => (T)Convert.ChangeType(fillValue, typeof(T), CultureInfo.InvariantCulture)!;

    enum RollingKind { Sum, Mean, Min, Max }

    enum CumulativeKind { Sum, Max, Min, Product }
}
