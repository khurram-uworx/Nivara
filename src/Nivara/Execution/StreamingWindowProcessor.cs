using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Execution;

/// <summary>
/// Streams a first-boundary window operation per chunk with true cross-chunk state,
/// replacing the raw <see cref="WindowOverlapBuffer"/> blocks previously duplicated
/// across the streaming strategy's sync, async, and chunk-enumeration paths.
/// </summary>
/// <remarks>
/// Two streaming mechanisms are combined, keyed by window kind:
/// <list type="bullet">
/// <item>Bounded-lookback kinds (rolling aggregates, lag shift) prepend the last
/// <c>N</c> input rows from the previous chunk via <see cref="WindowOverlapBuffer"/>
/// and trim the overlap prefix from the result.</item>
/// <item>Cumulative kinds (sum/max/min/product/count) carry a single running-aggregate
/// value across chunks. Each chunk's column is recomputed from the raw chunk input and
/// seeded with the carried aggregate, so results are exact for any chunk size. This
/// replaces the earlier overlap=1 approximation, which dropped all history except the
/// immediately preceding row.</item>
/// </list>
/// Rank-family and broadcast windows do not stream here and remain boundary
/// materializations.
/// </remarks>
internal sealed class StreamingWindowProcessor
{
    /// <summary>
    /// Carried running-aggregate state for one cumulative window column.
    /// </summary>
    sealed class CarrySlot
    {
        public required string OutputName { get; init; }
        public required string? SourceColumnName { get; init; }
        public required ColumnExpression? SourceExpression { get; init; }
        public required Func<object?>? NullHandler { get; init; }
        public required NivaraFrameExtensions.CumulativeKind Kind { get; init; }
        public required bool IsCount { get; init; }

        public bool HasState { get; set; }
        public object? State { get; set; }
    }

    readonly IQueryOperation boundaryOp;
    readonly SelectOperation? boundarySelect;
    readonly WindowOverlapBuffer? overlapBuffer;
    readonly int overlapSize;
    readonly List<CarrySlot> carrySlots = [];
    readonly FusedExpressionEvaluator expressionEvaluator = new();

    StreamingWindowProcessor(IQueryOperation boundaryOp, SelectOperation? boundarySelect, CumulativeOperation? cumulative, int overlapSize)
    {
        this.boundaryOp = boundaryOp;
        this.boundarySelect = boundarySelect;
        this.overlapSize = overlapSize;
        overlapBuffer = overlapSize > 0 ? new WindowOverlapBuffer(overlapSize) : null;
        if (boundarySelect != null)
            CollectCarrySlots(boundarySelect);
        else if (cumulative != null)
            CollectCarrySlots(cumulative);
    }

    /// <summary>
    /// Gets the number of overlap rows prepended to each chunk (0 when only carry-state
    /// windows are present).
    /// </summary>
    public int OverlapSize => overlapSize;

    /// <summary>
    /// Creates a processor for a boundary operation containing window expressions, or
    /// null when the operation cannot stream per-chunk (no windows, rank/broadcast
    /// windows only, partitioned standalone windows, or non-window operations).
    /// </summary>
    public static StreamingWindowProcessor? TryCreate(IQueryOperation? boundaryOp)
    {
        if (boundaryOp == null)
            return null;

        if (boundaryOp is SelectOperation select && WindowExpressionInspector.HasWindowExpression(select))
        {
            if (!hasOnlyStreamableWindows(select))
                return null;

            var overlap = WindowOverlapBuffer.DetermineOverlapSize(select);
            return new StreamingWindowProcessor(select, select, null, overlap);
        }

        if (boundaryOp is RollingOperation rolling && rolling.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(rolling, null, null, Math.Max(0, rolling.WindowSize - 1));

        if (boundaryOp is ShiftOperation shift && shift.Periods >= 0 && shift.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(shift, null, null, shift.Periods);

        if (boundaryOp is CumulativeOperation cumulative && cumulative.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(cumulative, null, cumulative, 0);

        return null;
    }

    static bool hasOnlyStreamableWindows(SelectOperation select)
    {
        foreach (var column in select.Columns)
        {
            if (!isStreamableNode(column))
                return false;
        }

        return true;
    }

    static bool isStreamableNode(ColumnExpression node)
        => node switch
        {
            WindowExpression window => window.Kind switch
            {
                WindowFunctionKind.RollingSum or WindowFunctionKind.RollingMean
                    or WindowFunctionKind.RollingMin or WindowFunctionKind.RollingMax
                    or WindowFunctionKind.CumulativeSum or WindowFunctionKind.CumulativeMax
                    or WindowFunctionKind.CumulativeMin or WindowFunctionKind.CumulativeProduct
                    or WindowFunctionKind.CumulativeCount
                    => true,
                WindowFunctionKind.Shift => (window.Periods ?? 0) >= 0,

                // Lead, negative shift, rank family, and broadcast aggregates need data
                // beyond the lookback context and must materialize.
                _ => false
            },
            ScalarExpression scalar => isStreamableNode(scalar.Column),
            BinaryExpression binary => isStreamableNode(binary.Left) && isStreamableNode(binary.Right),
            ComparisonExpression comparison => isStreamableNode(comparison.Left) && isStreamableNode(comparison.Right),
            NotExpression not => isStreamableNode(not.Operand),
            ConditionalExpression conditional => isStreamableNode(conditional.Test)
                && isStreamableNode(conditional.TrueValue)
                && isStreamableNode(conditional.FalseValue),
            _ => true
        };

    void CollectCarrySlots(SelectOperation select)
    {
        for (var i = 0; i < select.Columns.Count; i++)
        {
            var expr = select.Columns[i];
            if (expr is not WindowExpression { Kind: WindowFunctionKind.CumulativeSum or WindowFunctionKind.CumulativeMax or WindowFunctionKind.CumulativeMin or WindowFunctionKind.CumulativeProduct or WindowFunctionKind.CumulativeCount } window)
                continue;

            carrySlots.Add(new CarrySlot
            {
                OutputName = select.OutputNames is not null ? select.OutputNames[i] : expr.Name,
                SourceColumnName = null,
                SourceExpression = window.Source,
                NullHandler = window.NullHandler,
                Kind = window.Kind == WindowFunctionKind.CumulativeProduct
                    ? NivaraFrameExtensions.CumulativeKind.Product
                    : window.Kind == WindowFunctionKind.CumulativeMax
                        ? NivaraFrameExtensions.CumulativeKind.Max
                        : window.Kind == WindowFunctionKind.CumulativeMin
                            ? NivaraFrameExtensions.CumulativeKind.Min
                            : NivaraFrameExtensions.CumulativeKind.Sum,
                IsCount = window.Kind == WindowFunctionKind.CumulativeCount,
            });
        }
    }

    void CollectCarrySlots(CumulativeOperation cumulative)
    {
        carrySlots.Add(new CarrySlot
        {
            OutputName = cumulative.ResultColumn,
            SourceColumnName = cumulative.Source,
            SourceExpression = cumulative.SourceExpression,
            NullHandler = cumulative.NullHandler,
            Kind = cumulative.Kind,
            IsCount = cumulative.IsCount,
        });
    }

    /// <summary>
    /// Processes one chunk of pre-boundary data: runs the boundary window operation with
    /// overlap context when needed, corrects cumulative columns with carried state, and
    /// updates the cross-chunk buffers.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> ProcessChunk(IReadOnlyDictionary<string, IColumn> processedChunk)
    {
        var hasOverlapContext = overlapBuffer is { HasData: true };
        var extended = hasOverlapContext ? overlapBuffer!.PrependToChunk(processedChunk) : processedChunk;
        var result = boundaryOp.Execute(extended);
        var final = hasOverlapContext ? WindowOverlapBuffer.TrimFirstN(result, overlapSize) : result;

        if (carrySlots.Count > 0)
        {
            var mutable = new Dictionary<string, IColumn>(final, StringComparer.OrdinalIgnoreCase);
            foreach (var slot in carrySlots)
                mutable[slot.OutputName] = ComputeCarryColumn(slot, processedChunk);
            final = mutable;
        }

        overlapBuffer?.UpdateFromChunk(processedChunk);
        return final;
    }

    IColumn ComputeCarryColumn(CarrySlot slot, IReadOnlyDictionary<string, IColumn> rawChunk)
    {
        var sourceColumn = slot.SourceExpression is not null
            ? expressionEvaluator.Evaluate(slot.SourceExpression, rawChunk)
            : rawChunk.TryGetValue(slot.SourceColumnName!, out var found)
                ? found
                : throw new ColumnNotFoundException(slot.SourceColumnName!, rawChunk.Keys);

        IColumn corrected;
        if (slot.IsCount)
        {
            // Running count emits long regardless of source type, so the seed-row trick
            // cannot reuse the source column's element type; add the carried total instead.
            corrected = NivaraFrameExtensions.CalculateCumulativeCount(sourceColumn);
            if (slot.HasState)
                corrected = AddConstant(corrected, Convert.ToInt64(slot.State));
        }
        else if (slot.HasState)
        {
            var seed = ColumnFactory.Create(sourceColumn.ElementType, [slot.State]);
            var seeded = ColumnFilterHelper.ConcatenateColumns([seed, sourceColumn]);
            corrected = ComputeCumulative(seeded, slot).Slice(1, seeded.Length - 1);
        }
        else
            corrected = ComputeCumulative(sourceColumn, slot);

        UpdateCarryState(slot, corrected);
        return corrected;
    }

    static IColumn AddConstant(IColumn column, long constant)
    {
        var values = new object?[column.Length];
        for (var i = 0; i < column.Length; i++)
            values[i] = column.IsNull(i) ? null : Convert.ToInt64(column.GetValue(i)) + constant;

        return ColumnFactory.Create(column.ElementType, values);
    }

    static IColumn ComputeCumulative(IColumn column, CarrySlot slot)
        => slot.IsCount
            ? NivaraFrameExtensions.CalculateCumulativeCount(column)
            : NivaraFrameExtensions.CalculateCumulative(column, slot.NullHandler, slot.Kind);

    static void UpdateCarryState(CarrySlot slot, IColumn computed)
    {
        for (var i = computed.Length - 1; i >= 0; i--)
        {
            if (computed.IsNull(i))
                continue;

            slot.State = computed.GetValue(i);
            slot.HasState = true;
            return;
        }
    }
}
