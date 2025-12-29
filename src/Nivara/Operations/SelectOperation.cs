using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Represents a select (projection) operation that selects specific columns or expressions
/// </summary>
internal sealed class SelectOperation : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of SelectOperation
    /// </summary>
    /// <param name="columns">The column expressions to select</param>
    /// <exception cref="ArgumentNullException">Thrown when columns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public SelectOperation(ColumnExpression[] columns)
    {
        if (columns == null)
            throw new ArgumentNullException(nameof(columns));

        if (columns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression", nameof(columns));

        Columns = columns.ToArray(); // Create a defensive copy
    }

    /// <summary>
    /// Gets the column expressions to select
    /// </summary>
    public IReadOnlyList<ColumnExpression> Columns { get; }

    /// <inheritdoc />
    public string OperationType => "Select";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate all column expressions against the schema
        foreach (var column in Columns)
        {
            try
            {
                column.Validate(inputSchema);
            }
            catch (SchemaValidationException ex)
            {
                throw new SchemaValidationException($"Select column validation failed for '{column.Name}': {ex.Message}");
            }
        }

        // Build the new schema with selected columns
        var selectedColumns = new List<(string Name, Type Type)>();

        foreach (var column in Columns)
        {
            var columnName = GetColumnName(column, inputSchema);
            var columnType = GetColumnType(column, inputSchema);
            selectedColumns.Add((columnName, columnType));
        }

        return new Schema(selectedColumns);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        try
        {
            var selectedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            var evaluator = new ExpressionEvaluator();

            foreach (var columnExpr in Columns)
            {
                var columnName = GetColumnName(columnExpr, input);
                var resultColumn = evaluator.Evaluate(columnExpr, input);
                selectedColumns[columnName] = resultColumn;
            }

            return selectedColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Select operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Gets the name for a column expression in the result schema
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="inputSchema">The input schema (for validation)</param>
    /// <returns>The column name to use in the result</returns>
    private static string GetColumnName(ColumnExpression expression, Schema inputSchema)
    {
        // For simple column references, use the original column name
        if (expression is ColumnReference columnRef)
        {
            return columnRef.ColumnName;
        }

        // For complex expressions, use the expression's display name
        return expression.Name;
    }

    /// <summary>
    /// Gets the name for a column expression in the result (runtime version)
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>The column name to use in the result</returns>
    private static string GetColumnName(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        // For simple column references, use the original column name
        if (expression is ColumnReference columnRef)
        {
            return columnRef.ColumnName;
        }

        // For complex expressions, use the expression's display name
        return expression.Name;
    }

    /// <summary>
    /// Gets the type for a column expression in the result schema
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="inputSchema">The input schema</param>
    /// <returns>The column type in the result</returns>
    private static Type GetColumnType(ColumnExpression expression, Schema inputSchema)
    {
        // For simple column references, get the type from the schema
        if (expression is ColumnReference columnRef)
        {
            return inputSchema.GetColumnType(columnRef.ColumnName);
        }

        // For other expressions, use the expression's result type
        return expression.ResultType;
    }

    /// <summary>
    /// Returns a string representation of the select operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var columnNames = Columns.Select(c => c.Name);
        return $"Select({string.Join(", ", columnNames)})";
    }
}