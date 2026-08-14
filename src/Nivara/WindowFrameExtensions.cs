using Nivara.Tensors;
using System.Collections.Concurrent;
using System.Globalization;
using System.Numerics;
using System.Reflection;
using SortKey = Nivara.Operations.SortKey;
using WindowSpec = Nivara.Operations.WindowSpec;

namespace Nivara;

/// <summary>
/// Eager <see cref="NivaraFrame"/> window-function extensions: rolling aggregates,
/// cumulative aggregates, and shift/lead. These mirror the query-engine window API
/// with the same names, argument order, and null semantics.
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
public static partial class NivaraFrameExtensions
{
    // ── Window specification ──

    /// <summary>
    /// Creates a reusable window specification (<see cref="WindowSpec"/>) for the window-function
    /// extensions on <see cref="NivaraFrame"/>. The spec captures partition-by and order-by keys
    /// and can be reused across multiple window methods.
    /// </summary>
    /// <param name="frame">The source frame (unused by the builder; present for API discoverability)</param>
    /// <returns>An empty window specification to be configured via <see cref="WindowSpec.PartitionBy"/> and <see cref="WindowSpec.OrderBy"/></returns>
    /// <remarks>Added as part of issue #162 Over/WindowSpec builder delivery.</remarks>
    public static WindowSpec Over(this NivaraFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        return new WindowSpec();
    }

    // ── Rolling ──

    /// <summary>
    /// Adds a rolling sum column.
    /// </summary>
    public static NivaraFrame RollingSum(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Sum));

    /// <summary>
    /// Adds a rolling mean column.
    /// </summary>
    public static NivaraFrame RollingMean(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Mean));

    /// <summary>
    /// Adds a rolling minimum column.
    /// </summary>
    public static NivaraFrame RollingMin(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Min));

    /// <summary>
    /// Adds a rolling maximum column.
    /// </summary>
    public static NivaraFrame RollingMax(this NivaraFrame frame, string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Max));

    /// <summary>
    /// Adds a rolling sum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame RollingSum(this NivaraFrame frame, string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Sum));

    /// <summary>
    /// Adds a rolling mean column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame RollingMean(this NivaraFrame frame, string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Mean));

    /// <summary>
    /// Adds a rolling minimum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame RollingMin(this NivaraFrame frame, string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Min));

    /// <summary>
    /// Adds a rolling maximum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame RollingMax(this NivaraFrame frame, string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateRolling(c, windowSize, minPeriods, nullHandler, RollingKind.Max));

    // ── Cumulative ──

    /// <summary>
    /// Adds a cumulative sum column.
    /// </summary>
    public static NivaraFrame CumulativeSum(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateCumulative(c, nullHandler, CumulativeKind.Sum));

    /// <summary>
    /// Adds a cumulative maximum column.
    /// </summary>
    public static NivaraFrame CumulativeMax(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateCumulative(c, nullHandler, CumulativeKind.Max));

    /// <summary>
    /// Adds a cumulative minimum column.
    /// </summary>
    public static NivaraFrame CumulativeMin(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateCumulative(c, nullHandler, CumulativeKind.Min));

    /// <summary>
    /// Adds a cumulative product column.
    /// </summary>
    public static NivaraFrame CumulativeProduct(this NivaraFrame frame, string source, string resultColumn, Func<object?>? nullHandler = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateCumulative(c, nullHandler, CumulativeKind.Product));

    /// <summary>
    /// Adds a running count-of-non-null column.
    /// </summary>
    public static NivaraFrame CumulativeCount(this NivaraFrame frame, string source, string resultColumn)
        => addWindowColumn(frame, source, resultColumn, CalculateCumulativeCount);

    /// <summary>
    /// Adds a cumulative sum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame CumulativeSum(this NivaraFrame frame, string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateCumulative(c, nullHandler, CumulativeKind.Sum));

    /// <summary>
    /// Adds a cumulative maximum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame CumulativeMax(this NivaraFrame frame, string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateCumulative(c, nullHandler, CumulativeKind.Max));

    /// <summary>
    /// Adds a cumulative minimum column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame CumulativeMin(this NivaraFrame frame, string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateCumulative(c, nullHandler, CumulativeKind.Min));

    /// <summary>
    /// Adds a cumulative product column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame CumulativeProduct(this NivaraFrame frame, string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateCumulative(c, nullHandler, CumulativeKind.Product));

    /// <summary>
    /// Adds a running count-of-non-null column over a partitioned/ordered window (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame CumulativeCount(this NivaraFrame frame, string source, string resultColumn, WindowSpec spec)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, CalculateCumulativeCount);

    // ── Shift / Lead ──

    /// <summary>
    /// Adds a shifted (lag) column. Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Shift(this NivaraFrame frame, string source, string resultColumn, int periods, object? fillValue = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateShift(c, periods, fillValue));

    /// <summary>
    /// Adds a lead column (negative shift). Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Lead(this NivaraFrame frame, string source, string resultColumn, int periods, object? fillValue = null)
        => addWindowColumn(frame, source, resultColumn, c => CalculateShift(c, -periods, fillValue));

    /// <summary>
    /// Adds a shifted (lag) column over a partitioned/ordered window (see <see cref="WindowSpec"/>). Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Shift(this NivaraFrame frame, string source, string resultColumn, int periods, WindowSpec spec, object? fillValue = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateShift(c, periods, fillValue));

    /// <summary>
    /// Adds a lead column (negative shift) over a partitioned/ordered window (see <see cref="WindowSpec"/>). Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    public static NivaraFrame Lead(this NivaraFrame frame, string source, string resultColumn, int periods, WindowSpec spec, object? fillValue = null)
        => addPartitionedWindowColumn(frame, source, resultColumn, spec, c => CalculateShift(c, -periods, fillValue));

    // ── Rank family ──

    /// <summary>
    /// Adds a row-number column. With no partition keys the numbering is sequential over all rows;
    /// with no order keys the numbering follows row order.
    /// </summary>
    public static NivaraFrame RowNumber(this NivaraFrame frame, string resultColumn, string[]? partitionBy = null, IReadOnlyList<SortKey>? orderBy = null)
        => addRankColumn(frame, resultColumn, RankKind.RowNumber, partitionBy ?? Array.Empty<string>(), orderBy ?? Array.Empty<SortKey>());

    /// <summary>
    /// Adds a standard rank column (gaps on ties).
    /// </summary>
    public static NivaraFrame Rank(this NivaraFrame frame, string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => addRankColumn(frame, resultColumn, RankKind.Rank, partitionBy, orderBy);

    /// <summary>
    /// Adds a dense-rank column (no gaps on ties).
    /// </summary>
    public static NivaraFrame DenseRank(this NivaraFrame frame, string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => addRankColumn(frame, resultColumn, RankKind.DenseRank, partitionBy, orderBy);

    /// <summary>
    /// Adds a percent-rank column: (rank - 1) / (partitionSize - 1).
    /// </summary>
    public static NivaraFrame PercentRank(this NivaraFrame frame, string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => addRankColumn(frame, resultColumn, RankKind.PercentRank, partitionBy, orderBy);

    /// <summary>
    /// Adds a row-number column from a window specification (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame RowNumber(this NivaraFrame frame, string resultColumn, WindowSpec spec)
        => addRankColumn(frame, resultColumn, RankKind.RowNumber, spec);

    /// <summary>
    /// Adds a standard rank column (gaps on ties) from a window specification (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame Rank(this NivaraFrame frame, string resultColumn, WindowSpec spec)
        => addRankColumn(frame, resultColumn, RankKind.Rank, spec);

    /// <summary>
    /// Adds a dense-rank column (no gaps on ties) from a window specification (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame DenseRank(this NivaraFrame frame, string resultColumn, WindowSpec spec)
        => addRankColumn(frame, resultColumn, RankKind.DenseRank, spec);

    /// <summary>
    /// Adds a percent-rank column from a window specification (see <see cref="WindowSpec"/>).
    /// </summary>
    public static NivaraFrame PercentRank(this NivaraFrame frame, string resultColumn, WindowSpec spec)
        => addRankColumn(frame, resultColumn, RankKind.PercentRank, spec);

    static NivaraFrame addRankColumn(NivaraFrame frame, string resultColumn, RankKind kind, WindowSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return addRankColumn(frame, resultColumn, kind, spec.PartitionColumns.ToArray(), spec.OrderKeys);
    }

    static NivaraFrame addRankColumn(NivaraFrame frame, string resultColumn, RankKind kind, string[] partitionBy, IReadOnlyList<SortKey> orderBy)
    {
        ArgumentNullException.ThrowIfNull(frame);

        if (kind != RankKind.RowNumber && orderBy.Count == 0)
            throw new ArgumentException($"'{kind}' requires at least one order key", nameof(orderBy));

        foreach (var partition in partitionBy)
            if (!frame.HasColumn(partition))
                throw new ArgumentException($"Partition column '{partition}' not found", nameof(partitionBy));

        foreach (var key in orderBy)
            if (!frame.HasColumn(key.ColumnName))
                throw new ArgumentException($"Order column '{key.ColumnName}' not found", nameof(orderBy));

        var columns = frame.ColumnNames.ToDictionary(n => n, n => frame.GetColumn(n), StringComparer.OrdinalIgnoreCase);
        var result = RankKernel.Compute(columns, partitionBy, orderBy, kind);
        return frame.WithColumn(resultColumn, result);
    }

    // ── Shared dispatch ──

    static NivaraFrame addWindowColumn(NivaraFrame frame, string source, string resultColumn, Func<IColumn, IColumn> computation)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var sourceColumn = frame.GetColumn(source);
        var result = computation(sourceColumn);
        return frame.WithColumn(resultColumn, result);
    }

    static NivaraFrame addPartitionedWindowColumn(NivaraFrame frame, string source, string resultColumn, WindowSpec spec, Func<IColumn, IColumn> computation)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(spec);

        var sourceColumn = frame.GetColumn(source);
        IColumn result;
        if (spec.IsEmpty)
        {
            result = computation(sourceColumn);
        }
        else
        {
            var columns = frame.ColumnNames.ToDictionary(n => n, n => frame.GetColumn(n), StringComparer.OrdinalIgnoreCase);
            result = PartitionedWindowEngine.Compute(columns, sourceColumn, spec, computation);
        }

        return frame.WithColumn(resultColumn, result);
    }

    internal static IColumn CalculateRolling(IColumn column, int windowSize, int? minPeriods, Func<object?>? nullHandler, RollingKind kind)
        => column switch
        {
            NivaraColumn<int> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<long> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<float> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<double> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<decimal> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<byte> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<sbyte> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<short> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<ushort> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<uint> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<ulong> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<char> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<nint> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<nuint> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<Int128> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<UInt128> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            NivaraColumn<Half> c => rolling(c, windowSize, minPeriods, nullHandler, kind),
            _ => throw new NotSupportedException($"Rolling window requires a numeric column, but {column.ElementType.Name} is not supported")
        };

    internal static IColumn CalculateCumulative(IColumn column, Func<object?>? nullHandler, CumulativeKind kind)
        => column switch
        {
            NivaraColumn<int> c => cumulative(c, nullHandler, kind),
            NivaraColumn<long> c => cumulative(c, nullHandler, kind),
            NivaraColumn<float> c => cumulative(c, nullHandler, kind),
            NivaraColumn<double> c => cumulative(c, nullHandler, kind),
            NivaraColumn<decimal> c => cumulative(c, nullHandler, kind),
            NivaraColumn<byte> c => cumulative(c, nullHandler, kind),
            NivaraColumn<sbyte> c => cumulative(c, nullHandler, kind),
            NivaraColumn<short> c => cumulative(c, nullHandler, kind),
            NivaraColumn<ushort> c => cumulative(c, nullHandler, kind),
            NivaraColumn<uint> c => cumulative(c, nullHandler, kind),
            NivaraColumn<ulong> c => cumulative(c, nullHandler, kind),
            NivaraColumn<char> c => cumulative(c, nullHandler, kind),
            NivaraColumn<nint> c => cumulative(c, nullHandler, kind),
            NivaraColumn<nuint> c => cumulative(c, nullHandler, kind),
            NivaraColumn<Int128> c => cumulative(c, nullHandler, kind),
            NivaraColumn<UInt128> c => cumulative(c, nullHandler, kind),
            NivaraColumn<Half> c => cumulative(c, nullHandler, kind),
            _ => throw new NotSupportedException($"Cumulative requires a numeric column, but {column.ElementType.Name} is not supported")
        };

    internal static IColumn CalculateCumulativeCount(IColumn column)
        => column switch
        {
            NivaraColumn<int> c => c.CumulativeCount(),
            NivaraColumn<long> c => c.CumulativeCount(),
            NivaraColumn<float> c => c.CumulativeCount(),
            NivaraColumn<double> c => c.CumulativeCount(),
            NivaraColumn<decimal> c => c.CumulativeCount(),
            NivaraColumn<byte> c => c.CumulativeCount(),
            NivaraColumn<sbyte> c => c.CumulativeCount(),
            NivaraColumn<short> c => c.CumulativeCount(),
            NivaraColumn<ushort> c => c.CumulativeCount(),
            NivaraColumn<uint> c => c.CumulativeCount(),
            NivaraColumn<ulong> c => c.CumulativeCount(),
            NivaraColumn<char> c => c.CumulativeCount(),
            NivaraColumn<nint> c => c.CumulativeCount(),
            NivaraColumn<nuint> c => c.CumulativeCount(),
            NivaraColumn<Int128> c => c.CumulativeCount(),
            NivaraColumn<UInt128> c => c.CumulativeCount(),
            NivaraColumn<Half> c => c.CumulativeCount(),
            NivaraColumn<string> c => c.CumulativeCount(),
            NivaraColumn<bool> c => c.CumulativeCount(),
            _ => throw new NotSupportedException($"CumulativeCount does not support column type {column.ElementType.Name}")
        };

    internal static IColumn CalculateShift(IColumn column, int periods, object? fillValue)
        => column switch
        {
            NivaraColumn<int> c => shift(c, periods, fillValue),
            NivaraColumn<long> c => shift(c, periods, fillValue),
            NivaraColumn<float> c => shift(c, periods, fillValue),
            NivaraColumn<double> c => shift(c, periods, fillValue),
            NivaraColumn<decimal> c => shift(c, periods, fillValue),
            NivaraColumn<byte> c => shift(c, periods, fillValue),
            NivaraColumn<sbyte> c => shift(c, periods, fillValue),
            NivaraColumn<short> c => shift(c, periods, fillValue),
            NivaraColumn<ushort> c => shift(c, periods, fillValue),
            NivaraColumn<uint> c => shift(c, periods, fillValue),
            NivaraColumn<ulong> c => shift(c, periods, fillValue),
            NivaraColumn<char> c => shift(c, periods, fillValue),
            NivaraColumn<nint> c => shift(c, periods, fillValue),
            NivaraColumn<nuint> c => shift(c, periods, fillValue),
            NivaraColumn<Int128> c => shift(c, periods, fillValue),
            NivaraColumn<UInt128> c => shift(c, periods, fillValue),
            NivaraColumn<Half> c => shift(c, periods, fillValue),
            NivaraColumn<string> c => shift(c, periods, fillValue),
            NivaraColumn<bool> c => shift(c, periods, fillValue),
            _ => throw new NotSupportedException($"Shift does not support column type {column.ElementType.Name}")
        };

    static readonly ConcurrentDictionary<Type, MethodInfo?> parseCache = new();

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
            : () => convertValue<T>(nullHandler() ?? default(T));

    static T convertFillValue<T>(object? fillValue)
        => convertValue<T>(fillValue!);

    static T convertValue<T>(object value)
    {
        if (value is T typed)
            return typed;

        if (value is string text)
        {
            var parse = parseCache.GetOrAdd(typeof(T), static t =>
                t.GetMethod(nameof(int.TryParse), BindingFlags.Public | BindingFlags.Static, null,
                    new[] { typeof(string), t.MakeByRefType() }, null));
            if (parse is not null)
            {
                var args = new object?[] { text, null };
                if ((bool)parse.Invoke(null, args)!)
                    return (T)args[1]!;
                throw new InvalidOperationException($"Cannot convert '{text}' to {typeof(T).Name}");
            }
        }

        return (T)Convert.ChangeType(value, typeof(T), CultureInfo.InvariantCulture)!;
    }

    internal enum RollingKind { Sum, Mean, Min, Max }

    internal enum CumulativeKind { Sum, Max, Min, Product }
}
