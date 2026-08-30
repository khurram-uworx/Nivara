using Nivara.Expressions;
using Nivara.Query;

namespace Nivara.Execution;

/// <summary>
/// Determines the lookback context size needed to stream window boundaries per chunk.
/// Formerly an instance buffer that prepended overlap rows to each chunk and trimmed the
/// results; that mechanism was superseded by <see cref="StreamingWindowProcessor"/>'s
/// sliding context run, which subsumes both lookback and lookahead distances.
/// </summary>
internal static class WindowOverlapBuffer
{
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

            // Lead and negative-period shifts need lookahead rather than lookback; they
            // stream via StreamingWindowProcessor delayed emission instead of overlap.
            _ => 0
        };
    }
}
