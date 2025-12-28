using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Represents a group by operation that groups rows by specified columns
/// </summary>
internal sealed class GroupByOperation : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of GroupByOperation
    /// </summary>
    /// <param name="groupByColumns">The column expressions to group by</param>
    /// <exception cref="ArgumentNullException">Thrown when groupByColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public GroupByOperation(ColumnExpression[] groupByColumns)
    {
        if (groupByColumns == null)
            throw new ArgumentNullException(nameof(groupByColumns));

        if (groupByColumns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression for grouping", nameof(groupByColumns));

        GroupByColumns = groupByColumns.ToArray(); // Create a defensive copy
    }

    /// <summary>
    /// Gets the column expressions to group by
    /// </summary>
    public IReadOnlyList<ColumnExpression> GroupByColumns { get; }

    /// <inheritdoc />
    public string OperationType => "GroupBy";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate all group by column expressions against the schema
        foreach (var column in GroupByColumns)
        {
            try
            {
                column.Validate(inputSchema);
            }
            catch (SchemaValidationException ex)
            {
                throw new SchemaValidationException($"GroupBy column validation failed for '{column.Name}': {ex.Message}");
            }
        }

        // For now, GroupBy returns only the grouped columns
        // In a full implementation, this would include aggregation columns
        var groupedColumns = new List<(string Name, Type Type)>();
        
        foreach (var column in GroupByColumns)
        {
            var columnName = GetColumnName(column, inputSchema);
            var columnType = GetColumnType(column, inputSchema);
            groupedColumns.Add((columnName, columnType));
        }

        return new Schema(groupedColumns);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (input.Count == 0)
            return input;

        try
        {
            // For this basic implementation, we'll return distinct values of the grouped columns
            // A full implementation would support aggregation functions
            var evaluator = new ExpressionEvaluator();
            var groupedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

            // Evaluate each group by column
            var groupByValues = new List<(string Name, IColumn Column)>();
            
            foreach (var columnExpr in GroupByColumns)
            {
                var columnName = GetColumnName(columnExpr, input);
                var column = evaluator.Evaluate(columnExpr, input);
                groupByValues.Add((columnName, column));
            }

            // Create a composite key for each row and find unique combinations
            var rowCount = groupByValues.First().Column.Length;
            var uniqueRows = new Dictionary<string, int>();
            var firstOccurrences = new List<int>();

            for (int i = 0; i < rowCount; i++)
            {
                // Create a composite key from all group by column values
                var keyParts = groupByValues.Select(gv => gv.Column.GetValue(i)?.ToString() ?? "null");
                var compositeKey = string.Join("|", keyParts);

                if (!uniqueRows.ContainsKey(compositeKey))
                {
                    uniqueRows[compositeKey] = i;
                    firstOccurrences.Add(i);
                }
            }

            // Create result columns with only the first occurrence of each unique combination
            foreach (var (name, column) in groupByValues)
            {
                var distinctColumn = CreateDistinctColumn(column, firstOccurrences);
                groupedColumns[name] = distinctColumn;
            }

            return groupedColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"GroupBy operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Creates a new column containing only the values at the specified indices
    /// </summary>
    /// <param name="column">The source column</param>
    /// <param name="indices">The indices of values to include</param>
    /// <returns>A new column with distinct values</returns>
    private static IColumn CreateDistinctColumn(IColumn column, List<int> indices)
    {
        var elementType = column.ElementType;
        
        // Use dynamic dispatch to create the appropriate column type
        return elementType switch
        {
            Type t when t == typeof(int) => CreateDistinctColumnTyped<int>(column, indices),
            Type t when t == typeof(double) => CreateDistinctColumnTyped<double>(column, indices),
            Type t when t == typeof(float) => CreateDistinctColumnTyped<float>(column, indices),
            Type t when t == typeof(long) => CreateDistinctColumnTyped<long>(column, indices),
            Type t when t == typeof(string) => CreateDistinctColumnTyped<string>(column, indices),
            Type t when t == typeof(bool) => CreateDistinctColumnTyped<bool>(column, indices),
            Type t when t == typeof(decimal) => CreateDistinctColumnTyped<decimal>(column, indices),
            Type t when t == typeof(byte) => CreateDistinctColumnTyped<byte>(column, indices),
            Type t when t == typeof(short) => CreateDistinctColumnTyped<short>(column, indices),
            Type t when t == typeof(DateTime) => CreateDistinctColumnTyped<DateTime>(column, indices),
            _ => CreateDistinctColumnGeneric(column, indices)
        };
    }

    /// <summary>
    /// Creates a distinct column for a specific type
    /// </summary>
    private static IColumn CreateDistinctColumnTyped<T>(IColumn column, List<int> indices)
    {
        var distinctArray = new T[indices.Count];
        
        for (int i = 0; i < indices.Count; i++)
        {
            var value = column.GetValue(indices[i]);
            distinctArray[i] = (T)value!;
        }

        return NivaraColumn<T>.Create(distinctArray);
    }

    /// <summary>
    /// Creates a distinct column for unknown types using object column
    /// </summary>
    private static IColumn CreateDistinctColumnGeneric(IColumn column, List<int> indices)
    {
        var distinctArray = new object[indices.Count];
        
        for (int i = 0; i < indices.Count; i++)
        {
            distinctArray[i] = column.GetValue(indices[i])!;
        }

        return NivaraColumn<object>.Create(distinctArray);
    }

    /// <summary>
    /// Gets the name for a column expression in the result schema
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="inputSchema">The input schema</param>
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
    /// Returns a string representation of the group by operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var columnNames = GroupByColumns.Select(c => c.Name);
        return $"GroupBy({string.Join(", ", columnNames)})";
    }
}