using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents a sort operation that orders rows by a computed column expression.
/// The key expression is materialized into a column at execution time, then the
/// existing <see cref="SortOperation"/> machinery (comparer, null ordering, reorder)
/// is reused to sort every column by those indices.
/// </summary>
sealed class SortByExpressionOperation : IQueryOperation
{
    const string SyntheticKeyName = "__nivara_sort_key";

    readonly ColumnExpression key;
    readonly SortDirection direction;
    readonly NullOrdering nullOrdering;
    readonly bool stable;

    /// <summary>
    /// Initializes a new instance of SortByExpressionOperation
    /// </summary>
    /// <param name="key">The key expression to sort by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <param name="stable">Whether to use stable sorting</param>
    /// <exception cref="ArgumentNullException">Thrown when key is null</exception>
    public SortByExpressionOperation(ColumnExpression key, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast, bool stable = true)
    {
        this.key = key ?? throw new ArgumentNullException(nameof(key));
        this.direction = direction;
        this.nullOrdering = nullOrdering;
        this.stable = stable;
    }

    /// <summary>
    /// Gets the key expression used for sorting
    /// </summary>
    public ColumnExpression Key => key;

    /// <summary>
    /// Gets the sort direction
    /// </summary>
    public SortDirection Direction => direction;

    /// <summary>
    /// Gets the null ordering strategy
    /// </summary>
    public NullOrdering NullOrdering => nullOrdering;

    /// <summary>
    /// Gets whether stable sorting is used
    /// </summary>
    public bool IsStable => stable;

    /// <inheritdoc />
    public string OperationType => Query.OperationType.SortByExpression;

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        try
        {
            key.Validate(inputSchema);
        }
        catch (SchemaValidationException ex)
        {
            throw new SchemaValidationException($"Sort key expression validation failed: {ex.Message}");
        }

        // Sort doesn't change the schema structure, only the row order
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
            var evaluator = new ExpressionEvaluator();
            var keyColumn = evaluator.Evaluate(key, input);

            var synthetic = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
                synthetic[kvp.Key] = kvp.Value;
            synthetic[SyntheticKeyName] = keyColumn;

            var sortOperation = new SortOperation(SyntheticKeyName, direction, nullOrdering, stable);
            var sorted = sortOperation.Execute(synthetic);

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in sorted)
                if (!string.Equals(kvp.Key, SyntheticKeyName, StringComparison.OrdinalIgnoreCase))
                    result[kvp.Key] = kvp.Value;

            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Sort operation failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var directionStr = direction == SortDirection.Ascending ? "ASC" : "DESC";
        var nullStr = nullOrdering == NullOrdering.NullsFirst ? "NULLS FIRST" : "NULLS LAST";
        return $"SortByExpression({key.Name} {directionStr} {nullStr})";
    }
}
