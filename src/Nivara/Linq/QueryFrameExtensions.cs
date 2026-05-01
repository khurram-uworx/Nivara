using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Linq;

/// <summary>
/// Extension methods for QueryFrame to support LINQ-like syntax
/// </summary>
public static class QueryFrameExtensions
{
    /// <summary>
    /// Filters the query frame using a lambda expression
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="predicate">A function that returns a boolean column expression</param>
    /// <returns>A filtered query frame</returns>
    public static QueryFrame Where(this QueryFrame source, Func<RowExpressionBuilder, ColumnExpression> predicate)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(predicate);

        var expression = predicate(RowExpressionBuilder.Instance);
        return source.Filter(expression);
    }

    /// <summary>
    /// Projects the query frame using lambda expressions
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="selectors">Functions that return column expressions to select</param>
    /// <returns>A projected query frame</returns>
    public static QueryFrame Select(this QueryFrame source, params Func<RowExpressionBuilder, ColumnExpression>[] selectors)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(selectors);

        if (selectors.Length == 0)
            throw new ArgumentException("Must specify at least one selector", nameof(selectors));

        var expressions = selectors.Select(s => s(RowExpressionBuilder.Instance)).ToArray();
        return source.Select(expressions);
    }

    /// <summary>
    /// Projects the query frame using column names (alias for Select)
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="columnNames">Names of columns to select</param>
    /// <returns>A projected query frame</returns>
    public static QueryFrame Select(this QueryFrame source, params string[] columnNames)
    {
        ArgumentNullException.ThrowIfNull(source);
        return source.Select(columnNames);
    }

    /// <summary>
    /// Sorts the query frame
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="keySelector">Function to select the sort key</param>
    /// <param name="descending">Whether to sort in descending order</param>
    /// <returns>A sorted query frame</returns>
    public static QueryFrame OrderBy(this QueryFrame source, Func<RowExpressionBuilder, ColumnExpression> keySelector, bool descending = false)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(keySelector);

        var expression = keySelector(RowExpressionBuilder.Instance);

        // At the moment, SortOperation expects a column name. 
        // If the expression is a ColumnReference, we can extract the name.
        if (expression is ColumnReference colRef)
        {
            return source.Sort(colRef.ColumnName, descending ? SortDirection.Descending : SortDirection.Ascending);
        }

        // If it's not a direct column reference (e.g. calculated), existing Sort might not support it directly 
        // unless we project it first. For now, assuming direct column reference or basic name resolution.
        // If the expression has a name (e.g. from an alias or base impl), we try that.

        if (!string.IsNullOrEmpty(expression.Name) && !expression.Name.Contains("(")) // Simplistic check for now
        {
            return source.Sort(expression.Name, descending ? SortDirection.Descending : SortDirection.Ascending);
        }

        throw new NotSupportedException("OrderBy currently supports only direct column references or named expressions. Complex expressions must be selected/computed first.");
    }

    /// <summary>
    /// Sorts the query frame in descending order
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="keySelector">Function to select the sort key</param>
    /// <returns>A sorted query frame</returns>
    public static QueryFrame OrderByDescending(this QueryFrame source, Func<RowExpressionBuilder, ColumnExpression> keySelector)
    {
        return source.OrderBy(keySelector, descending: true);
    }

    /// <summary>
    /// Executes the query and returns a materialized NivaraFrame (Alias for Collect)
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <returns>A materialized NivaraFrame</returns>
    public static NivaraFrame ToNivaraFrame(this QueryFrame source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return source.Collect();
    }

    /// <summary>
    /// Executes the query and returns a materialized NivaraFrame (Alias for Collect, matching LINQ's ToList)
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <returns>A materialized NivaraFrame</returns>
    public static NivaraFrame ToList(this QueryFrame source)
    {
        return source.ToNivaraFrame();
    }
}
