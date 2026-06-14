using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents a filter operation that applies a condition to filter rows
/// </summary>
sealed class FilterOperation : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of FilterOperation
    /// </summary>
    /// <param name="condition">The condition to filter by</param>
    /// <exception cref="ArgumentNullException">Thrown when condition is null</exception>
    public FilterOperation(ColumnExpression condition)
    {
        Condition = condition ?? throw new ArgumentNullException(nameof(condition));
    }

    /// <summary>
    /// Gets the filter condition
    /// </summary>
    public ColumnExpression Condition { get; }

    public string OperationType => Query.OperationType.Filter;

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate the condition against the schema
        try
        {
            Condition.Validate(inputSchema);
        }
        catch (SchemaValidationException ex)
        {
            throw new SchemaValidationException($"Filter condition validation failed: {ex.Message}");
        }

        // Filter doesn't change the schema structure, only the number of rows
        return inputSchema;
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
            // Evaluate the condition to get a boolean mask
            var evaluator = new ExpressionEvaluator();
            var mask = evaluator.EvaluateBoolean(Condition, input);

            // Apply the mask to all columns
            var filteredColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

            foreach (var kvp in input)
            {
                var filteredColumn = ApplyMask(kvp.Value, mask);
                filteredColumns[kvp.Key] = filteredColumn;
            }

            return filteredColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Filter operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Applies a boolean mask to a column to filter its values
    /// </summary>
    /// <param name="column">The column to filter</param>
    /// <param name="mask">The boolean mask indicating which rows to keep</param>
    /// <returns>A new column with filtered values</returns>
    static IColumn ApplyMask(IColumn column, NivaraColumn<bool> mask)
    {
        if (column.Length != mask.Length)
            throw new ArgumentException("Column and mask must have the same length");

        var filteredIndices = new List<int>();

        for (int i = 0; i < mask.Length; i++)
            if (mask[i] == true) // Only include rows where mask is true
                filteredIndices.Add(i);

        // Create a new column with only the filtered values
        return ColumnFilterHelper.CreateFilteredColumn(column, filteredIndices);
    }

    public override string ToString()
        => $"Filter({Condition})";
}
