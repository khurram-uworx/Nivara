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
/// Every round executes the boundary operation over one contiguous run:
/// the last <c>contextSize</c> input rows (the sum of the largest lookback and lookahead
/// distances, so re-run rows keep both their history and their future) followed by the
/// fresh chunk. Re-running over carried context means every window kind sees correct
/// history:
/// <list type="bullet">
/// <item>Bounded-lookback kinds (rolling aggregates, lag shift) read their history from
/// the prepended context rows.</item>
/// <item>Cumulative kinds (sum/max/min/product/count) ignore re-run rows: their columns
/// are recomputed per round from the fresh chunk only, seeded with a carried
/// running-aggregate value.</item>
/// <item>Lookahead kinds (lead, negative-period shift) rely on delayed emission: only the
/// rows farther than <c>leadDistance</c> from the end of seen data are final and emitted;
/// the rest are re-computed in later runs. <see cref="Flush"/> finalizes whatever remains
/// after the source drains by emitting the premature-boundary rows of the last run, which
/// become exact once no further data exists. Memory stays bounded by
/// <c>max(lookback, leadPeriods)</c> rather than frame size.</item>
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
    readonly int contextSize;
    readonly int leadDistance;
    readonly List<CarrySlot> carrySlots = [];
    readonly FusedExpressionEvaluator expressionEvaluator = new();

    Dictionary<string, IColumn>? lastRunInput;
    IReadOnlyDictionary<string, IColumn>? lastRunResult;
    long lastRunStart;
    long totalRowsSeen;
    long emittedCount;

    StreamingWindowProcessor(IQueryOperation boundaryOp, SelectOperation? boundarySelect, CumulativeOperation? cumulative, int overlapSize, int leadDistance)
    {
        this.boundaryOp = boundaryOp;
        // Re-run rows need their own lookback history behind them AND their lookahead
        // ahead of them, so the carried tail must span the sum of both distances.
        this.contextSize = Math.Max(0, overlapSize) + Math.Max(0, leadDistance);
        this.leadDistance = leadDistance;
        if (boundarySelect != null)
            CollectCarrySlots(boundarySelect);
        else if (cumulative != null)
            CollectCarrySlots(cumulative);
    }

    /// <summary>
    /// Gets the number of context rows carried across chunks (the sum of the largest
    /// lookback and lookahead distances).
    /// </summary>
    public int OverlapSize => contextSize;

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

            var lookback = WindowOverlapBuffer.DetermineOverlapSize(select);
            var lead = determineLeadDistance(select);
            return new StreamingWindowProcessor(select, select, null, lookback, lead);
        }

        if (boundaryOp is RollingOperation rolling && rolling.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(rolling, null, null, Math.Max(0, rolling.WindowSize - 1), 0);

        if (boundaryOp is ShiftOperation shift && shift.Spec is null or { IsEmpty: true })
        {
            return shift.Periods >= 0
                ? new StreamingWindowProcessor(shift, null, null, shift.Periods, 0)
                : new StreamingWindowProcessor(shift, null, null, 0, -shift.Periods);
        }

        if (boundaryOp is CumulativeOperation cumulative && cumulative.Spec is null or { IsEmpty: true })
            return new StreamingWindowProcessor(cumulative, null, cumulative, 0, 0);

        return null;
    }

    /// <summary>
    /// Processes one chunk of pre-boundary data: runs the boundary window operation over
    /// the carried context plus the fresh chunk, emits the rows whose window contexts are
    /// fully satisfied by data seen so far, and updates the cross-chunk buffers. With
    /// lookahead windows present, call <see cref="Flush"/> after the source drains to
    /// obtain the remaining rows.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> ProcessChunk(IReadOnlyDictionary<string, IColumn> processedChunk)
    {
        var chunkLength = getRowLength(processedChunk);
        var contextLength = lastRunInput != null && lastRunInput.Count > 0 ? getRowLength(lastRunInput) : 0;

        IReadOnlyDictionary<string, IColumn> run;
        if (contextLength > 0)
        {
            var concatenated = new Dictionary<string, IColumn>(lastRunInput!.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in lastRunInput)
                concatenated[kvp.Key] = ColumnFilterHelper.ConcatenateColumns([kvp.Value, processedChunk[kvp.Key]]);
            run = concatenated;
        }
        else
            run = processedChunk;

        var runLength = contextLength + chunkLength;
        var runStart = totalRowsSeen - contextLength;

        var result = boundaryOp.Execute(run);

        // Rows farther than leadDistance from the newest row have their entire lookahead
        // inside this run and are final; the tail rows are re-computed in a later run.
        var finalEnd = Math.Min(totalRowsSeen + chunkLength - leadDistance, runStart + runLength);
        var from = (int)Math.Max(0, emittedCount - runStart);
        var to = (int)Math.Max(from, finalEnd - runStart);
        var emitCount = to - from;

        var emitted = new Dictionary<string, IColumn>(result.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in result)
            emitted[kvp.Key] = sliceRange(kvp.Value, from, emitCount);

        foreach (var slot in carrySlots)
            emitted[slot.OutputName] = carryColumnForEmission(slot, processedChunk, emitCount);

        lastRunResult = result;
        lastRunStart = runStart;
        lastRunInput = contextSize > 0 ? takeLastRows(run, contextSize) : null;
        totalRowsSeen += chunkLength;
        emittedCount += emitCount;

        return emitted;
    }

    /// <summary>
    /// Finalizes rows still held back by delayed emission after the source drained. Their
    /// values come from the last executed run, where positions beyond the run's end
    /// already received the operation's end-of-data semantics (nulls or fill values),
    /// which are exact once no further data exists. Returns null when nothing is pending.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn>? Flush()
    {
        if (lastRunResult == null || emittedCount >= totalRowsSeen)
            return null;

        var from = (int)(emittedCount - lastRunStart);
        var runLength = getRowLength(lastRunResult);
        var flushCount = runLength - from;

        var flushed = new Dictionary<string, IColumn>(lastRunResult.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in lastRunResult)
            flushed[kvp.Key] = sliceRange(kvp.Value, from, flushCount);

        foreach (var slot in carrySlots)
            flushed[slot.OutputName] = buildDeferredColumn(slot);

        emittedCount = totalRowsSeen;
        lastRunResult = null;
        lastRunInput = null;

        return flushed;
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

                // Rank family and broadcast aggregates need data beyond the carried
                // context and must materialize.
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

    IColumn carryColumnForEmission(CarrySlot slot, IReadOnlyDictionary<string, IColumn> freshChunk, int emitCount)
    {
        var corrected = ComputeCarryColumn(slot, freshChunk);
        if (leadDistance == 0)
            return corrected;

        // Cumulative values are computed over the fresh rows only: re-run context rows
        // were already counted into the carried state when they were fresh. Values wait in
        // the slot's queue until their row enters the emitted range.
        slot.ElementType ??= corrected.ElementType;
        for (var i = 0; i < corrected.Length; i++)
            slot.PendingValues.Enqueue(corrected.IsNull(i) ? null : corrected.GetValue(i));

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

    static IColumn sliceRange(IColumn column, int start, int count)
    {
        if (count <= 0)
            return ColumnFilterHelper.CreateEmptyColumn(column.ElementType);
        if (start == 0 && count == column.Length)
            return column;
        return column.Slice(start, count);
    }

    static Dictionary<string, IColumn> takeLastRows(IReadOnlyDictionary<string, IColumn> data, int count)
    {
        var tail = new Dictionary<string, IColumn>(data.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in data)
        {
            var length = kvp.Value.Length;
            var start = Math.Max(0, length - count);
            tail[kvp.Key] = length - start > 0 ? kvp.Value.Slice(start, length - start) : kvp.Value;
        }
        return tail;
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
