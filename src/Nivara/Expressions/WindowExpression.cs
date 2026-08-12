using Nivara.Operations;
using Nivara.Tensors;

namespace Nivara.Expressions;

/// <summary>
/// Identifies a window-function expression kind.
/// </summary>
internal enum WindowFunctionKind
{
    RollingSum,
    RollingMean,
    RollingMin,
    RollingMax,
    CumulativeSum,
    CumulativeMax,
    CumulativeMin,
    CumulativeProduct,
    CumulativeCount,
    Shift,
    Lead,
    RowNumber,
    Rank,
    DenseRank,
    PercentRank
}

/// <summary>
/// Shared kind mapping / result-type rules for window expressions, used by both the
/// expression AST and the pipeline operations so schema and execution always agree.
/// </summary>
internal static class WindowFunctionHelpers
{
    /// <summary>
    /// Determines whether a window kind is a rank-family function computed over
    /// partition + order-by keys rather than a single source column.
    /// </summary>
    public static bool IsRankFamily(WindowFunctionKind kind)
        => kind is WindowFunctionKind.RowNumber
            or WindowFunctionKind.Rank
            or WindowFunctionKind.DenseRank
            or WindowFunctionKind.PercentRank;

    /// <summary>
    /// Computes the result type for a window kind given the source column type.
    /// </summary>
    public static Type GetResultType(WindowFunctionKind kind, Type sourceType)
    {
        return kind switch
        {
            WindowFunctionKind.RollingMean => typeof(double),
            WindowFunctionKind.CumulativeCount => typeof(long),
            WindowFunctionKind.RowNumber => typeof(long),
            WindowFunctionKind.Rank => typeof(long),
            WindowFunctionKind.DenseRank => typeof(long),
            WindowFunctionKind.PercentRank => typeof(double),
            _ => sourceType
        };
    }

    /// <summary>
    /// Maps a window kind to the eager rolling aggregate kind.
    /// </summary>
    public static NivaraFrameExtensions.RollingKind ToRollingKind(WindowFunctionKind kind)
    {
        return kind switch
        {
            WindowFunctionKind.RollingSum => NivaraFrameExtensions.RollingKind.Sum,
            WindowFunctionKind.RollingMean => NivaraFrameExtensions.RollingKind.Mean,
            WindowFunctionKind.RollingMin => NivaraFrameExtensions.RollingKind.Min,
            WindowFunctionKind.RollingMax => NivaraFrameExtensions.RollingKind.Max,
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a rolling aggregate kind")
        };
    }

    /// <summary>
    /// Maps a window kind to the eager cumulative aggregate kind.
    /// </summary>
    public static NivaraFrameExtensions.CumulativeKind ToCumulativeKind(WindowFunctionKind kind)
    {
        return kind switch
        {
            WindowFunctionKind.CumulativeSum => NivaraFrameExtensions.CumulativeKind.Sum,
            WindowFunctionKind.CumulativeMax => NivaraFrameExtensions.CumulativeKind.Max,
            WindowFunctionKind.CumulativeMin => NivaraFrameExtensions.CumulativeKind.Min,
            WindowFunctionKind.CumulativeProduct => NivaraFrameExtensions.CumulativeKind.Product,
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a cumulative aggregate kind")
        };
    }

    /// <summary>
    /// Maps a window kind to the rank kernel kind.
    /// </summary>
    public static RankKind ToRankKind(WindowFunctionKind kind)
    {
        return kind switch
        {
            WindowFunctionKind.RowNumber => RankKind.RowNumber,
            WindowFunctionKind.Rank => RankKind.Rank,
            WindowFunctionKind.DenseRank => RankKind.DenseRank,
            WindowFunctionKind.PercentRank => RankKind.PercentRank,
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a rank-family kind")
        };
    }
    /// <summary>
    /// Maps a rank kernel kind to the window-function kind.
    /// </summary>
    public static WindowFunctionKind ToWindowFunctionKind(RankKind kind)
    {
        return kind switch
        {
            RankKind.RowNumber => WindowFunctionKind.RowNumber,
            RankKind.Rank => WindowFunctionKind.Rank,
            RankKind.DenseRank => WindowFunctionKind.DenseRank,
            RankKind.PercentRank => WindowFunctionKind.PercentRank,
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a rank-family kind")
        };
    }
}

/// <summary>
/// A window-function expression: rolling / cumulative / shift / lead aggregates over a
/// source sub-expression, or a rank-family function over partition + order-by expressions.
/// Evaluated by the fused evaluator by materializing the computed window result and fusing
/// the surrounding elementwise expression over it.
/// </summary>
internal sealed class WindowExpression : ColumnExpression
{
    /// <summary>
    /// Initializes a rolling-window expression.
    /// </summary>
    public WindowExpression(WindowFunctionKind kind, ColumnExpression source, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
    {
        if (!IsRollingKind(kind))
            throw new ArgumentException($"'{kind}' is not a rolling aggregate kind", nameof(kind));

        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive");

        Kind = kind;
        Source = source ?? throw new ArgumentNullException(nameof(source));
        WindowSize = windowSize;
        MinPeriods = minPeriods;
        NullHandler = nullHandler;
        ResultType = WindowFunctionHelpers.GetResultType(kind, Source.ResultType);
    }

    /// <summary>
    /// Initializes a cumulative-aggregate expression (including the running count).
    /// </summary>
    public WindowExpression(WindowFunctionKind kind, ColumnExpression source, Func<object?>? nullHandler = null)
    {
        if (kind is not (WindowFunctionKind.CumulativeSum or WindowFunctionKind.CumulativeMax
            or WindowFunctionKind.CumulativeMin or WindowFunctionKind.CumulativeProduct
            or WindowFunctionKind.CumulativeCount))
        {
            throw new ArgumentException($"'{kind}' is not a cumulative aggregate kind", nameof(kind));
        }

        Kind = kind;
        Source = source ?? throw new ArgumentNullException(nameof(source));
        NullHandler = nullHandler;
        ResultType = WindowFunctionHelpers.GetResultType(kind, Source.ResultType);
    }

    /// <summary>
    /// Initializes a shift or lead expression.
    /// </summary>
    public WindowExpression(WindowFunctionKind kind, ColumnExpression source, int periods, object? fillValue = null)
    {
        if (kind is not (WindowFunctionKind.Shift or WindowFunctionKind.Lead))
            throw new ArgumentException($"'{kind}' is not a shift/lead kind", nameof(kind));

        Kind = kind;
        Source = source ?? throw new ArgumentNullException(nameof(source));
        Periods = periods;
        FillValue = fillValue;
        ResultType = WindowFunctionHelpers.GetResultType(kind, Source.ResultType);
    }

    /// <summary>
    /// Initializes a rank-family expression over partition + order-by key expressions.
    /// </summary>
    public WindowExpression(WindowFunctionKind kind, IReadOnlyList<ColumnExpression>? partitionBy, IReadOnlyList<SortExpressionKey> orderBy)
    {
        if (!WindowFunctionHelpers.IsRankFamily(kind))
            throw new ArgumentException($"'{kind}' is not a rank-family kind", nameof(kind));

        ArgumentNullException.ThrowIfNull(orderBy);

        if (kind != WindowFunctionKind.RowNumber && orderBy.Count == 0)
            throw new ArgumentException($"'{kind}' requires at least one order key", nameof(orderBy));

        Kind = kind;
        PartitionBy = partitionBy ?? Array.Empty<ColumnExpression>();
        OrderBy = orderBy;
        ResultType = WindowFunctionHelpers.GetResultType(kind, typeof(object));
    }

    /// <summary>
    /// Gets the window function kind.
    /// </summary>
    public WindowFunctionKind Kind { get; }

    /// <summary>
    /// Gets the source sub-expression (null for rank-family kinds).
    /// </summary>
    public ColumnExpression? Source { get; }

    /// <summary>
    /// Gets the rolling window size (rolling kinds only).
    /// </summary>
    public int? WindowSize { get; }

    /// <summary>
    /// Gets the minimum number of valid observations required (rolling kinds only).
    /// </summary>
    public int? MinPeriods { get; }

    /// <summary>
    /// Gets the optional null-replacement handler (rolling / cumulative kinds only).
    /// </summary>
    public Func<object?>? NullHandler { get; }

    /// <summary>
    /// Gets the partition key expressions (rank-family kinds only).
    /// </summary>
    public IReadOnlyList<ColumnExpression> PartitionBy { get; } = Array.Empty<ColumnExpression>();

    /// <summary>
    /// Gets the order-by key expressions (rank-family kinds only).
    /// </summary>
    public IReadOnlyList<SortExpressionKey> OrderBy { get; } = Array.Empty<SortExpressionKey>();

    /// <summary>
    /// Gets the number of positions to shift by (shift/lead kinds only; lead is negative).
    /// </summary>
    public int? Periods { get; }

    /// <summary>
    /// Gets the optional fill value for boundary positions (shift/lead kinds only).
    /// </summary>
    public object? FillValue { get; }

    /// <inheritdoc />
    public override Type ResultType { get; protected set; }

    /// <inheritdoc />
    public override string Name
    {
        get
        {
            if (WindowFunctionHelpers.IsRankFamily(Kind))
            {
                var orderStr = string.Join(", ", OrderBy.Select(k => k.Key.Name));
                var partitionStr = PartitionBy.Count > 0 ? $" OVER (PARTITION BY {string.Join(", ", PartitionBy.Select(p => p.Name))})" : "";
                return $"{Kind}({orderStr}){partitionStr}";
            }

            var sourceName = Source!.Name;
            return Kind switch
            {
                WindowFunctionKind.RollingSum or WindowFunctionKind.RollingMean
                    or WindowFunctionKind.RollingMin or WindowFunctionKind.RollingMax
                    => $"{Kind}({sourceName}, {WindowSize})",
                WindowFunctionKind.Shift or WindowFunctionKind.Lead
                    => $"{Kind}({sourceName}, {Periods})",
                _ => $"{Kind}({sourceName})"
            };
        }
    }

    /// <inheritdoc />
    public override void Validate(Schema schema)
    {
        if (WindowFunctionHelpers.IsRankFamily(Kind))
        {
            foreach (var key in OrderBy)
                key.Key.Validate(schema);

            foreach (var partition in PartitionBy)
                partition.Validate(schema);

            return;
        }

        Source!.Validate(schema);
        ResultType = WindowFunctionHelpers.GetResultType(Kind, Source.ResultType);
    }

    /// <inheritdoc />
    public override string ToString() => Name;

    static bool IsRollingKind(WindowFunctionKind kind)
        => kind is WindowFunctionKind.RollingSum or WindowFunctionKind.RollingMean
            or WindowFunctionKind.RollingMin or WindowFunctionKind.RollingMax;
}
