using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Linq;

/// <summary>
/// Extension methods for QueryFrame to support LINQ-like syntax
/// </summary>
internal static class NivaraLinqExtensions
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

        // Direct column references sort via the column-name based SortOperation.
        if (expression is ColumnReference colRef)
        {
            return source.Sort(colRef.ColumnName, descending ? SortDirection.Descending : SortDirection.Ascending);
        }

        // Computed keys are materialized into a column at execution time and sorted on.
        return source.SortByExpression(expression, descending ? SortDirection.Descending : SortDirection.Ascending);
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
    /// Applies a stable secondary sort to the query frame using a lambda expression, merging into the
    /// primary sort from a preceding <see cref="OrderBy(QueryFrame, Func{RowExpressionBuilder, ColumnExpression}, bool)"/>
    /// so the ordering composes lexicographically (primary key first, then this secondary key). Computed
    /// keys are supported. Without a preceding sort, acts as a primary sort.
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="keySelector">Function to select the secondary sort key</param>
    /// <param name="descending">Whether to sort in descending order</param>
    /// <returns>A sorted query frame</returns>
    public static QueryFrame ThenBy(this QueryFrame source, Func<RowExpressionBuilder, ColumnExpression> keySelector, bool descending = false)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(keySelector);

        var expression = keySelector(RowExpressionBuilder.Instance);
        return source.ThenBy(expression, descending ? SortDirection.Descending : SortDirection.Ascending);
    }

    /// <summary>
    /// Applies a stable secondary descending sort to the query frame using a lambda expression, merging
    /// into the primary sort from a preceding <see cref="OrderBy(QueryFrame, Func{RowExpressionBuilder, ColumnExpression}, bool)"/>.
    /// Computed keys are supported. Without a preceding sort, acts as a primary descending sort.
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <param name="keySelector">Function to select the secondary sort key</param>
    /// <returns>A sorted query frame</returns>
    public static QueryFrame ThenByDescending(this QueryFrame source, Func<RowExpressionBuilder, ColumnExpression> keySelector)
    {
        return source.ThenBy(keySelector, descending: true);
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
    /// Executes the query and materializes each result row as a typed <see cref="NivaraRow"/> view
    /// </summary>
    /// <param name="source">The source query frame</param>
    /// <returns>A read-only list of row views over the materialized frame</returns>
    public static IReadOnlyList<NivaraRow> ToRowList(this QueryFrame source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var frame = source.Collect();
        var columns = frame.ColumnNames.Select(name => frame.GetColumn(name)).ToArray();
        var map = new Dictionary<string, int>(frame.ColumnNames.Count, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < frame.ColumnNames.Count; i++)
            map[frame.ColumnNames[i]] = i;

        var rows = new List<NivaraRow>(frame.RowCount);
        for (int i = 0; i < frame.RowCount; i++)
            rows.Add(new NivaraRow(columns, map, i));

        return rows;
    }
}
