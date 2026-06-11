using Nivara.Expressions;

namespace Nivara.Linq;

/// <summary>
/// Helper class to build column expressions using lambda syntax.
/// Allows accessing columns via indexer or methods.
/// </summary>
public sealed class RowExpressionBuilder
{
    static readonly RowExpressionBuilder instance = new();

    /// <summary>
    /// gets a singleton instance of the builder
    /// </summary>
    public static RowExpressionBuilder Instance => instance;

    RowExpressionBuilder() { }

    /// <summary>
    /// Creates a column reference for the specified column name
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>A column expression</returns>
    public ColumnExpression this[string columnName] => ColumnExpressions.Col(columnName);

    /// <summary>
    /// Creates a column reference for the specified column name
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>A column expression</returns>
    public ColumnExpression Col(string columnName) => ColumnExpressions.Col(columnName);

    /// <summary>
    /// Creates a literal expression
    /// </summary>
    /// <param name="value">The literal value</param>
    /// <returns>A literal expression</returns>
    public ColumnExpression Lit(object? value) => ColumnExpressions.Lit(value);
}
