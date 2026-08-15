using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Expressions;

/// <summary>
/// Detects window expressions inside query operations. Strategies that execute operations per
/// chunk/slice (streaming, parallel) must not partition a plan whose expressions embed a
/// <see cref="WindowExpression"/>: windows are whole-column by construction, so evaluating them
/// per chunk would produce different results (issue #245). Operations that hold no expressions
/// (or only whole-column materialized ones, like expression sort keys) return false.
/// </summary>
internal static class WindowExpressionInspector
{
    /// <summary>
    /// Gets whether the operation carries a window expression anywhere in its expression surface.
    /// </summary>
    /// <param name="operation">The operation to inspect</param>
    /// <returns>True when the operation evaluates a window expression</returns>
    public static bool HasWindowExpression(IQueryOperation operation)
    {
        return operation switch
        {
            SelectOperation select => select.Columns.Any(FusedExpressionEvaluator.ContainsWindowExpression),
            FilterOperation filter => FusedExpressionEvaluator.ContainsWindowExpression(filter.Condition),
            _ => false
        };
    }
}
