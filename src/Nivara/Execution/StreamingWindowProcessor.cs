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
/// Three streaming mechanisms are combined, keyed by window kind:
/// <list type="bullet">
/// <item>Bounded-lookback kinds (rolling aggregates, lag shift) prepend the last
/// <c>N</c> input rows from the previous chunk via <see cref="WindowOverlapBuffer"/>
/// and trim the overlap prefix from the result.</item>
/// <item>Cumulative kinds (sum/max/min/product/count) carry a single running-aggregate
/// value across chunks. Each chunk's column is recomputed from the fresh chunk input and
/// seeded with the carried aggregate, so results are exact for any chunk size.</item>
/// <item>Lookahead kinds (lead, negative-period shift) stream via delayed emission:
/// each round emits only the rows whose lookahead distance is satisfied by data seen so
/// far and holds the last <c>leadDistance</c> pre-boundary input rows back; held rows are
/// recomputed once the next chunk arrives. <see cref="Flush"/> finalizes whatever remains
/// after the source drains, letting the boundary op apply its natural tail semantics
/// (nulls or fill values). Memory stays bounded by <c>max(leadPeriods)</c> plus the
/// overlap window rather than frame size.</item>
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

        // Delayed-emission queue of cumulative values computed for rows whose results are
        // not yet emitted (used only when leadDistance > 0). Null entries mark null cells.
        public Queue<object?> PendingValues { get; } = new();
        public Type? ElementType { get; set; }
    }

    readonly IQueryOperation boundaryOp;
    readonly SelectOperation? boundarySelect;
    readonly WindowOverlapBuffer? overlapBuffer;
    readonly int overlapSize;
    readonly int leadDistance;
    readonly List<CarrySlot> carrySlots = [];
    readonly FusedExpressionEvaluator expressionEvaluator = new();
    Dictionary<string, IColumn>? pendingRows;

    StreamingWindowProcessor(IQueryOperation boundaryOp, SelectOperation? boundarySelect, CumulativeOperation? cumulative, int overlapSize, int leadDistance = 0)
    {
        this.boundaryOp = boundaryOp;
        this.boundarySelect = boundarySelect;
        this.overlapSize = overlapSize;
        this.leadDistance = Math.Max(0, leadDistance);
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
            var lead = determineLeadDistance(select);
            return new StreamingWindowProcessor(select, select, null, overlap, lead);
        }

        if (boundaryOp is RollingOperation rolling && rolling.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(rolling, null, null, Math.Max(0, rolling.WindowSize - 1));

        if (boundaryOp is ShiftOperation shift && shift.Spec is null or { IsEmpty: true })
        {
            return shift.Periods >= 0
                ? new StreamingWindowProcessor(shift, null, null, shift.Periods)
                : new StreamingWindowProcessor(shift, null, null, 0, -shift.Periods);
        }

        if (boundaryOp is CumulativeOperation cumulative && cumulative.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(cumulative, null, cumulative, 0);

        return null;
    }

    /// <summary>
    /// Processes one chunk of pre-boundary data: runs the boundary window operation with
    /// overlap context when needed, corrects cumulative columns with carried state, and
    /// updates the cross-chunk buffers. With lookahead windows present, only the prefix
    /// whose lead distance is satisfied by data seen so far is returned; call
    /// <see cref="Flush"/> after the source drains to obtain the remaining rows.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> ProcessChunk(IReadOnlyDictionary<string, IColumn> processedChunk)
    {
        if (leadDistance == 0)
            return processChunkImmediate(processedChunk);

        IReadOnlyDictionary<string, IColumn> combined;
        if (pendingRows != null && pendingRows.Count > 0)
        {
            // Contiguous run: held rows from previous rounds directly precede this chunk.
            var concatenated = new Dictionary<string, IColumn>(pendingRows.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in pendingRows)
                concatenated[kvp.Key] = ColumnFilterHelper.ConcatenateColumns([kvp.Value, processedChunk[kvp.Key]]);
            combined = concatenated;
        }
        else
            combined = processedChunk;
        var combinedLength = getRowLength(combined);
        var emitCount = Math.Max(0, combinedLength - leadDistance);

        var hasOverlapContext = overlapBuffer is { HasData: true };
        var extended = hasOverlapContext ? overlapBuffer!.PrependToChunk(combined) : combined;
        var result = boundaryOp.Execute(extended);
        var final = hasOverlapContext ? WindowOverlapBuffer.TrimFirstN(result, overlapSize) : result;

        var emitted = new Dictionary<string, IColumn>(final.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in final)
            emitted[kvp.Key] = slicePrefix(kvp.Value, emitCount);

        foreach (var slot in carrySlots)
            emitted[slot.OutputName] = computeDeferredCarryColumn(slot, processedChunk, emitCount);

        // Lookback context for the next round must include the held rows.
        overlapBuffer?.UpdateFromChunk(combined);
        var holdCount = Math.Min(leadDistance, combinedLength);
        setPendingRows(combined, holdCount);

        return emitted;
    }

    /// <summary>
    /// Finalizes rows still held back by delayed emission after the source drained. The
    /// boundary operation runs over the pending rows alone, so their tail positions
    /// receive the operation's natural end-of-data semantics (nulls or fill values).
    /// Returns null when nothing is pending.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn>? Flush()
    {
        if (pendingRows == null || pendingRows.Count == 0)
            return null;

        var hasOverlapContext = overlapBuffer is { HasData: true };
        var extended = hasOverlapContext ? overlapBuffer!.PrependToChunk(pendingRows) : pendingRows;
        var result = boundaryOp.Execute(extended);
        var final = hasOverlapContext ? WindowOverlapBuffer.TrimFirstN(result, overlapSize) : result;

        if (carrySlots.Count > 0)
        {
            var mutable = new Dictionary<string, IColumn>(final, StringComparer.OrdinalIgnoreCase);
            foreach (var slot in carrySlots)
                mutable[slot.OutputName] = buildDeferredColumn(slot);
            return mutable;
        }

        return final;
    }

    IReadOnlyDictionary<string, IColumn> processChunkImmediate(IReadOnlyDictionary<string, IColumn> processedChunk)
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
                WindowFunctionKind.Shift => true,
                WindowFunctionKind.Lead => true,

                // Rank family and broadcast aggregates need data beyond the lookback/
                // lookahead context and must materialize.
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

    static int determineLeadDistance(SelectOperation select)
    {
        int maxLead = 0;
        foreach (var col in select.Columns)
        {
            var lead = getMaxLeadFromExpression(col);
            maxLead = Math.Max(maxLead, lead);
        }
        return maxLead;
    }

    static int getMaxLeadFromExpression(ColumnExpression node)
    {
        return node switch
        {
            WindowExpression window => Math.Max(0, window.Kind switch
            {
                WindowFunctionKind.Lead => window.Periods ?? 0,
                WindowFunctionKind.Shift => -(window.Periods ?? 0),
                _ => 0
            }),
            ScalarExpression scalar => getMaxLeadFromExpression(scalar.Column),
            BinaryExpression binary => Math.Max(
                getMaxLeadFromExpression(binary.Left),
                getMaxLeadFromExpression(binary.Right)),
            ComparisonExpression comparison => Math.Max(
                getMaxLeadFromExpression(comparison.Left),
                getMaxLeadFromExpression(comparison.Right)),
            NotExpression not => getMaxLeadFromExpression(not.Operand),
            ConditionalExpression conditional => Math.Max(
                Math.Max(
                    getMaxLeadFromExpression(conditional.Test),
                    getMaxLeadFromExpression(conditional.TrueValue)),
                getMaxLeadFromExpression(conditional.FalseValue)),
            _ => 0
        };
    }

    void setPendingRows(IReadOnlyDictionary<string, IColumn> combined, int holdCount)
    {
        if (holdCount <= 0)
        {
            pendingRows = null;
            return;
        }

        var tail = new Dictionary<string, IColumn>(combined.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in combined)
        {
            var length = kvp.Value.Length;
            var start = Math.Max(0, length - holdCount);
            tail[kvp.Key] = length - start > 0 ? kvp.Value.Slice(start, length - start) : kvp.Value;
        }
        pendingRows = tail;
    }

    IColumn computeDeferredCarryColumn(CarrySlot slot, IReadOnlyDictionary<string, IColumn> freshChunk, int emitCount)
    {
        // Cumulative values are computed over the fresh rows only: held rows were already
        // counted into the carried state when they were fresh. Values wait in the slot's
        // queue until their row enters the emitted prefix.
        var computed = ComputeCarryColumn(slot, freshChunk);
        slot.ElementType ??= computed.ElementType;
        for (var i = 0; i < computed.Length; i++)
            slot.PendingValues.Enqueue(computed.IsNull(i) ? null : computed.GetValue(i));

        return buildDeferredPrefix(slot, emitCount);
    }

    IColumn buildDeferredPrefix(CarrySlot slot, int count)
    {
        var values = new object?[count];
        for (var i = 0; i < count; i++)
            values[i] = slot.PendingValues.Dequeue();

        return ColumnFactory.Create(slot.ElementType ?? typeof(long), values);
    }

    IColumn buildDeferredColumn(CarrySlot slot)
    {
        var values = new object?[slot.PendingValues.Count];
        for (var i = 0; i < values.Length; i++)
            values[i] = slot.PendingValues.Dequeue();

        return ColumnFactory.Create(slot.ElementType ?? typeof(long), values);
    }

    static IColumn slicePrefix(IColumn column, int count)
    {
        if (count <= 0)
            return ColumnFilterHelper.CreateEmptyColumn(column.ElementType);
        if (count < column.Length)
            return column.Slice(0, count);
        return column;
    }

    static int getRowLength(IReadOnlyDictionary<string, IColumn> data)
        => data.Count > 0 ? data.Values.First().Length : 0;

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
