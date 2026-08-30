using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using System.Numerics;
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

        // Delayed-emission staging of cumulative values computed for rows whose results
        // are not yet emitted (used only when leadDistance > 0). Created lazily on the
        // first carry; null cells are preserved via the buffer's null mask.
        public PendingColumnBuffer? PendingBuffer { get; set; }
        public Type? ElementType { get; set; }
    }

    /// <summary>
    /// Typed staging for cumulative values whose emission is delayed by a lookahead
    /// window. Subclasses remove the per-element boxing the previous boxed queue
    /// incurred while preserving null cells through a null mask.
    /// </summary>
    abstract class PendingColumnBuffer
    {
        public abstract int Count { get; }
        public abstract void Enqueue(IColumn corrected);
        public abstract IColumn Dequeue(int count);
    }

    /// <summary>
    /// Null-aware staging for a numeric <see cref="NivaraColumn{T}"/>: values queue as
    /// <see cref="Nullable{T}"/> (no boxing) and dequeue into caller-owned arrays plus a
    /// null mask, materialized without an object[] round trip.
    /// </summary>
    sealed class TypedPendingColumnBuffer<T> : PendingColumnBuffer
        where T : struct
    {
        readonly Queue<T?> values = new();

        public override int Count => values.Count;

        public override void Enqueue(IColumn column)
        {
            var typed = (NivaraColumn<T>)column;
            for (var i = 0; i < typed.Length; i++)
                values.Enqueue(typed.IsNull(i) ? null : typed[i]);
        }

        public override IColumn Dequeue(int count)
        {
            var data = new T[count];
            var nullMask = new bool[count];
            var hasNulls = false;
            for (var i = 0; i < count; i++)
            {
                var value = values.Dequeue();
                if (value.HasValue)
                    data[i] = value.Value;
                else
                {
                    nullMask[i] = true;
                    hasNulls = true;
                }
            }

            return hasNulls
                ? NivaraColumn<T>.CreateFromOwnedArrays(data, nullMask)
                : NivaraColumn<T>.CreateFromOwnedArray(data);
        }
    }

    /// <summary>
    /// Fallback staging for reference-typed or unknown columns, preserving the prior
    /// boxed object[] behavior.
    /// </summary>
    sealed class BoxedPendingBuffer : PendingColumnBuffer
    {
        readonly Queue<object?> values = new();
        readonly Type elementType;

        public BoxedPendingBuffer(Type elementType)
        {
            this.elementType = elementType;
        }

        public override int Count => values.Count;

        public override void Enqueue(IColumn column)
        {
            for (var i = 0; i < column.Length; i++)
                values.Enqueue(column.IsNull(i) ? null : column.GetValue(i));
        }

        public override IColumn Dequeue(int count)
        {
            var boxed = new object?[count];
            for (var i = 0; i < count; i++)
                boxed[i] = values.Dequeue();

            return ColumnFactory.Create(elementType, boxed);
        }
    }

    /// <summary>
    /// Echoes the input columns unchanged, mirroring <c>WindowOperationBase.Execute</c>'s source
    /// passthrough for standalone window operations whose only computed output is a carry slot
    /// (issue #358). Lets the streaming path keep the source columns in the emitted frame without
    /// re-materializing the (overflow-prone, mid-run) cumulative over the boundary run.
    /// </summary>
    sealed class PassthroughOperation : IQueryOperation
    {
        public string OperationType => Query.OperationType.Select;

        public Schema TransformSchema(Schema inputSchema) => inputSchema;

        public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input) => input;

        public ValueTask<IReadOnlyDictionary<string, IColumn>> ExecuteAsync(
            IReadOnlyDictionary<string, IColumn> input, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            return new(input);
        }
    }

    // Per-run op evaluating only the non-carry boundary columns. Null only when every select column
    // is a carry window (all-carry SelectOperation), where no boundary re-computation is needed and
    // a SelectOperation passes no source columns through. For a standalone CumulativeOperation it is
    // a source-column passthrough (its only window output is a carry slot, but WindowOperationBase
    // still passes the source columns through).
    readonly IQueryOperation? reRunBoundaryOp;
    readonly int contextSize;
    readonly int leadDistance;
    readonly List<CarrySlot> carrySlots = [];
    readonly HashSet<string> carryOutputNames = new(StringComparer.OrdinalIgnoreCase);
    readonly FusedExpressionEvaluator expressionEvaluator = new();

    Dictionary<string, IColumn>? lastRunInput;
    IReadOnlyDictionary<string, IColumn>? lastRunResult;
    long lastRunStart;
    long totalRowsSeen;
    long emittedCount;

    StreamingWindowProcessor(IQueryOperation boundaryOp, SelectOperation? boundarySelect, CumulativeOperation? cumulative, int overlapSize, int leadDistance)
    {
        // Re-run rows need their own lookback history behind them AND their lookahead
        // ahead of them, so the carried tail must span the sum of both distances.
        this.contextSize = Math.Max(0, overlapSize) + Math.Max(0, leadDistance);
        this.leadDistance = leadDistance;
        if (boundarySelect != null)
        {
            CollectCarrySlots(boundarySelect);
            this.reRunBoundaryOp = BuildReducedSelect(boundarySelect);
        }
        else if (cumulative != null)
        {
            CollectCarrySlots(cumulative);
            // WindowOperationBase semantics pass the source columns through; a passthrough yields
            // them without re-materializing the cumulative over the run (which could overflow, #358).
            this.reRunBoundaryOp = new PassthroughOperation();
        }
        else
        {
            this.reRunBoundaryOp = boundaryOp;
        }
    }

    /// <summary>
    /// Builds the per-run boundary select that evaluates only the non-carry columns: each top-level
    /// carried cumulative window is replaced by its source projection, because its emitted value is
    /// always overwritten by the carry slot. Re-materializing the cumulative product over a re-run
    /// (which starts mid-column, not at dataset row 0) would trip the checked long accumulator in
    /// <c>cumulativeScan</c> (issue #358). Returns null when every select column is a carry window.
    /// </summary>
    static SelectOperation? BuildReducedSelect(SelectOperation select)
    {
        var columns = new List<ColumnExpression>(select.Columns.Count);
        var outputNames = new List<string>(select.Columns.Count);
        var anyRealColumn = false;

        for (var i = 0; i < select.Columns.Count; i++)
        {
            var expr = select.Columns[i];
            if (expr is WindowExpression { Kind: WindowFunctionKind.CumulativeSum or WindowFunctionKind.CumulativeMax or WindowFunctionKind.CumulativeMin or WindowFunctionKind.CumulativeProduct or WindowFunctionKind.CumulativeCount } window)
            {
                columns.Add(window.Source!);
            }
            else
            {
                columns.Add(expr);
                anyRealColumn = true;
            }

            outputNames.Add(select.OutputNames is not null ? select.OutputNames[i] : expr.Name);
        }

        return anyRealColumn ? new SelectOperation(columns.ToArray(), outputNames.ToArray()) : null;
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

        // Re-run the boundary op over only the non-carry columns. Carry slots compute their
        // columns from the carried state over the fresh chunk (never re-materialized over the
        // mid-run start), which both avoids overflow (issue #358) and is the authoritative value.
        var result = reRunBoundaryOp is not null
            ? reRunBoundaryOp.Execute(run)
            : new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

        // Rows farther than leadDistance from the newest row have their entire lookahead
        // inside this run and are final; the tail rows are re-computed in a later run.
        var finalEnd = Math.Min(totalRowsSeen + chunkLength - leadDistance, runStart + runLength);
        var from = (int)Math.Max(0, emittedCount - runStart);
        var to = (int)Math.Max(from, finalEnd - runStart);
        var emitCount = to - from;

        var emitted = new Dictionary<string, IColumn>(result.Count + carrySlots.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in result)
        {
            if (!carryOutputNames.Contains(kvp.Key))
                emitted[kvp.Key] = sliceRange(kvp.Value, from, emitCount);
        }

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
        // the slot's buffer until their row enters the emitted range.
        slot.ElementType ??= corrected.ElementType;
        slot.PendingBuffer ??= createPendingBuffer(corrected);
        slot.PendingBuffer.Enqueue(corrected);

        return buildDeferredPrefix(slot, emitCount);
    }

    IColumn buildDeferredPrefix(CarrySlot slot, int count)
        => slot.PendingBuffer!.Dequeue(count);

    IColumn buildDeferredColumn(CarrySlot slot)
    {
        if (slot.PendingBuffer is { } buffer && buffer.Count > 0)
            return buffer.Dequeue(buffer.Count);

        return ColumnFactory.Create(slot.ElementType ?? typeof(long), []);
    }

    /// <summary>
    /// Creates the null-aware typed staging buffer matching the corrected column's
    /// element type, falling back to boxed staging for any non-numeric column.
    /// </summary>
    static PendingColumnBuffer createPendingBuffer(IColumn corrected)
    {
        return corrected switch
        {
            NivaraColumn<int> => new TypedPendingColumnBuffer<int>(),
            NivaraColumn<long> => new TypedPendingColumnBuffer<long>(),
            NivaraColumn<float> => new TypedPendingColumnBuffer<float>(),
            NivaraColumn<double> => new TypedPendingColumnBuffer<double>(),
            NivaraColumn<decimal> => new TypedPendingColumnBuffer<decimal>(),
            NivaraColumn<byte> => new TypedPendingColumnBuffer<byte>(),
            NivaraColumn<sbyte> => new TypedPendingColumnBuffer<sbyte>(),
            NivaraColumn<short> => new TypedPendingColumnBuffer<short>(),
            NivaraColumn<ushort> => new TypedPendingColumnBuffer<ushort>(),
            NivaraColumn<uint> => new TypedPendingColumnBuffer<uint>(),
            NivaraColumn<ulong> => new TypedPendingColumnBuffer<ulong>(),
            NivaraColumn<char> => new TypedPendingColumnBuffer<char>(),
            NivaraColumn<nint> => new TypedPendingColumnBuffer<nint>(),
            NivaraColumn<nuint> => new TypedPendingColumnBuffer<nuint>(),
            NivaraColumn<Int128> => new TypedPendingColumnBuffer<Int128>(),
            NivaraColumn<UInt128> => new TypedPendingColumnBuffer<UInt128>(),
            NivaraColumn<Half> => new TypedPendingColumnBuffer<Half>(),
            NivaraColumn<BFloat16> => new TypedPendingColumnBuffer<BFloat16>(),
            _ => new BoxedPendingBuffer(corrected.ElementType)
        };
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
            carryOutputNames.Add(select.OutputNames is not null ? select.OutputNames[i] : expr.Name);
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
        carryOutputNames.Add(cumulative.ResultColumn);
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
        return column switch
        {
            NivaraColumn<int> c => addConstant(c, constant),
            NivaraColumn<long> c => addConstant(c, constant),
            NivaraColumn<float> c => addConstant(c, constant),
            NivaraColumn<double> c => addConstant(c, constant),
            NivaraColumn<decimal> c => addConstant(c, constant),
            NivaraColumn<byte> c => addConstant(c, constant),
            NivaraColumn<sbyte> c => addConstant(c, constant),
            NivaraColumn<short> c => addConstant(c, constant),
            NivaraColumn<ushort> c => addConstant(c, constant),
            NivaraColumn<uint> c => addConstant(c, constant),
            NivaraColumn<ulong> c => addConstant(c, constant),
            NivaraColumn<char> c => addConstant(c, constant),
            NivaraColumn<nint> c => addConstant(c, constant),
            NivaraColumn<nuint> c => addConstant(c, constant),
            NivaraColumn<Int128> c => addConstant(c, constant),
            NivaraColumn<UInt128> c => addConstant(c, constant),
            NivaraColumn<Half> c => addConstant(c, constant),
            NivaraColumn<BFloat16> c => addConstant(c, constant),
            _ => addConstantBoxed(column, constant)
        };
    }

    static IColumn addConstant<T>(NivaraColumn<T> column, long constant)
        where T : struct, INumber<T>
    {
        var result = new T[column.Length];
        if (column.TryGetSpan(out var span))
        {
            NumericTensorKernels<T>.Add(span, T.CreateChecked(constant), result);
            return NivaraColumn<T>.CreateFromOwnedArray(result);
        }

        var nullMask = new bool[column.Length];
        var hasNulls = false;
        var offset = T.CreateChecked(constant);
        for (var i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
            {
                nullMask[i] = true;
                hasNulls = true;
            }
            else
            {
                result[i] = column[i] + offset;
            }
        }

        return hasNulls
            ? NivaraColumn<T>.CreateFromSpans(result, nullMask)
            : NivaraColumn<T>.CreateFromOwnedArray(result);
    }

    static IColumn addConstantBoxed(IColumn column, long constant)
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
