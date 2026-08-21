using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Execution;

/// <summary>
/// Manages a trailing-row overlap buffer for streaming window operations. When a window
/// boundary op follows streamable ops, the overlap buffer stores the last
/// <see cref="OverlapSize"/> rows from the previous chunk's processed data and prepends
/// them to the current chunk before running the boundary op. The result is then trimmed
/// to remove the overlap prefix, yielding correct per-chunk windowed results.
/// </summary>
/// <remarks>
/// This enables true per-chunk streaming for bounded-lookback window ops (rolling
/// aggregates, lag shift) without modifying any window kernels — the boundary op sees a
/// slightly longer column and produces correct results because the overlap provides the
/// necessary lookback context. Cumulative kinds require full running history and are
/// streamed by <see cref="StreamingWindowProcessor"/> carry state instead.
/// </remarks>
internal sealed class WindowOverlapBuffer
{
    readonly int overlapSize;
    Dictionary<string, IColumn>? tailColumns;

    public WindowOverlapBuffer(int overlapSize)
    {
        if (overlapSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(overlapSize), "Overlap size must be positive");

        this.overlapSize = overlapSize;
    }

    public int OverlapSize => overlapSize;

    public bool HasData => tailColumns != null;

    /// <summary>
    /// Extracts the last <see cref="OverlapSize"/> rows from each column in
    /// <paramref name="processedData"/> and stores them as the overlap tail.
    /// Call this AFTER the boundary op produces results but using the PRE-boundary
    /// processed data (the boundary op's input), so the next chunk gets the correct
    /// lookback context.
    /// </summary>
    public void UpdateFromChunk(IReadOnlyDictionary<string, IColumn> processedData)
    {
        var newTail = new Dictionary<string, IColumn>(processedData.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in processedData)
        {
            var col = kvp.Value;
            var colLength = col.Length;
            var start = Math.Max(0, colLength - overlapSize);
            var take = colLength - start;
            newTail[kvp.Key] = take > 0 ? col.Slice(start, take) : col;
        }
        tailColumns = newTail;
    }

    /// <summary>
    /// Prepends the stored overlap rows to each column in <paramref name="chunkData"/>,
    /// returning an extended dictionary with <see cref="OverlapSize"/> extra leading rows
    /// per column.
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> PrependToChunk(
        IReadOnlyDictionary<string, IColumn> chunkData)
    {
        if (tailColumns == null)
            throw new InvalidOperationException("No overlap data available. Call UpdateFromChunk first.");

        var result = new Dictionary<string, IColumn>(chunkData.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in chunkData)
        {
            if (tailColumns.TryGetValue(kvp.Key, out var tail) && tail.Length > 0)
                result[kvp.Key] = ColumnFilterHelper.ConcatenateColumns(new List<IColumn> { tail, kvp.Value });
            else
                result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    /// <summary>
    /// Removes the first <paramref name="n"/> rows from every column in
    /// <paramref name="data"/>, returning a new dictionary with the overlap prefix trimmed.
    /// </summary>
    public static IReadOnlyDictionary<string, IColumn> TrimFirstN(
        IReadOnlyDictionary<string, IColumn> data, int n)
    {
        if (n <= 0)
            return data;

        var result = new Dictionary<string, IColumn>(data.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in data)
        {
            var colLength = kvp.Value.Length;
            var remaining = colLength - n;
            result[kvp.Key] = remaining > 0 ? kvp.Value.Slice(n, remaining) : CreateEmptyColumn(kvp.Value.ElementType);
        }
        return result;
    }

    /// <summary>
    /// Inspects a boundary operation to determine the overlap size needed for streaming.
    /// Returns 0 when the operation does not contain any overlapable window expressions.
    /// </summary>
    public static int DetermineOverlapSize(IQueryOperation? boundaryOp)
    {
        if (boundaryOp == null)
            return 0;

        return boundaryOp switch
        {
            SelectOperation select => determineOverlapFromSelect(select),
            _ => 0
        };
    }

    static int determineOverlapFromSelect(SelectOperation select)
    {
        int maxOverlap = 0;
        foreach (var col in select.Columns)
        {
            var overlap = getMaxOverlapFromExpression(col);
            maxOverlap = Math.Max(maxOverlap, overlap);
        }
        return maxOverlap;
    }

    static int getMaxOverlapFromExpression(ColumnExpression node)
    {
        return node switch
        {
            WindowExpression window => getOverlapForWindowExpression(window),
            ScalarExpression scalar => getMaxOverlapFromExpression(scalar.Column),
            BinaryExpression binary => Math.Max(
                getMaxOverlapFromExpression(binary.Left),
                getMaxOverlapFromExpression(binary.Right)),
            ComparisonExpression comparison => Math.Max(
                getMaxOverlapFromExpression(comparison.Left),
                getMaxOverlapFromExpression(comparison.Right)),
            NotExpression not => getMaxOverlapFromExpression(not.Operand),
            ConditionalExpression conditional => Math.Max(
                Math.Max(
                    getMaxOverlapFromExpression(conditional.Test),
                    getMaxOverlapFromExpression(conditional.TrueValue)),
                getMaxOverlapFromExpression(conditional.FalseValue)),
            _ => 0
        };
    }

    static int getOverlapForWindowExpression(WindowExpression window)
    {
        return window.Kind switch
        {
            WindowFunctionKind.RollingSum or WindowFunctionKind.RollingMean
                or WindowFunctionKind.RollingMin or WindowFunctionKind.RollingMax
                => (window.WindowSize ?? 1) - 1,

            WindowFunctionKind.Shift => Math.Max(0, window.Periods ?? 0),
            WindowFunctionKind.Lead => 0,

            // Cumulative kinds stream via StreamingWindowProcessor carry state instead:
            // an overlap prefix cannot reproduce a running aggregate (all prior history
            // is required, not just the last N rows).
            _ => 0
        };
    }

    static IColumn CreateEmptyColumn(Type elementType)
    {
        return ColumnFilterHelper.CreateEmptyColumn(elementType);
    }
}
