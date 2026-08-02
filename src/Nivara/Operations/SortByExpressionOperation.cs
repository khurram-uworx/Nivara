using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Describes a single computed sort key: a column expression, direction, and null ordering.
/// </summary>
internal readonly struct SortExpressionKey
{
    /// <summary>
    /// Initializes a new instance of SortExpressionKey
    /// </summary>
    /// <param name="key">The key expression to sort by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <exception cref="ArgumentNullException">Thrown when key is null</exception>
    public SortExpressionKey(ColumnExpression key, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast)
    {
        Key = key ?? throw new ArgumentNullException(nameof(key));
        Direction = direction;
        NullOrdering = nullOrdering;
    }

    /// <summary>
    /// Gets the key expression to sort by
    /// </summary>
    public ColumnExpression Key { get; }

    /// <summary>
    /// Gets the sort direction
    /// </summary>
    public SortDirection Direction { get; }

    /// <summary>
    /// Gets the null ordering strategy
    /// </summary>
    public NullOrdering NullOrdering { get; }
}

/// <summary>
/// Represents a sort operation that orders rows by one or more computed column expressions.
/// The key expressions are materialized into columns at execution time, then the
/// existing <see cref="SortOperation"/> machinery (comparer, null ordering, reorder)
/// is reused to sort every column by those indices.
/// </summary>
sealed class SortByExpressionOperation : IQueryOperation
{
    const string SyntheticKeyNamePrefix = "__nivara_sort_key_";

    readonly IReadOnlyList<SortExpressionKey> keys;
    readonly bool stable;

    /// <summary>
    /// Initializes a new instance with one or more computed sort keys
    /// </summary>
    /// <param name="keys">The sort keys defining the sort order and priority</param>
    /// <param name="stable">Whether to use stable sorting</param>
    /// <exception cref="ArgumentNullException">Thrown when keys is null</exception>
    /// <exception cref="ArgumentException">Thrown when no keys are provided</exception>
    public SortByExpressionOperation(IEnumerable<SortExpressionKey> keys, bool stable = true)
    {
        if (keys == null)
            throw new ArgumentNullException(nameof(keys));

        var keyList = keys.ToList();
        if (keyList.Count == 0)
            throw new ArgumentException("At least one sort key expression is required", nameof(keys));

        this.keys = keyList;
        this.stable = stable;
    }

    /// <summary>
    /// Initializes a new instance with a single computed sort key
    /// </summary>
    /// <param name="key">The key expression to sort by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <param name="stable">Whether to use stable sorting</param>
    /// <exception cref="ArgumentNullException">Thrown when key is null</exception>
    public SortByExpressionOperation(ColumnExpression key, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast, bool stable = true)
        : this(new[] { new SortExpressionKey(key, direction, nullOrdering) }, stable)
    {
    }

    /// <summary>
    /// Gets the sort key expressions, directions, and null orderings
    /// </summary>
    public IReadOnlyList<SortExpressionKey> Keys => keys;

    /// <summary>
    /// Gets the primary key expression used for sorting
    /// </summary>
    public ColumnExpression Key => keys[0].Key;

    /// <summary>
    /// Gets the primary sort direction
    /// </summary>
    public SortDirection Direction => keys[0].Direction;

    /// <summary>
    /// Gets the primary null ordering strategy
    /// </summary>
    public NullOrdering NullOrdering => keys[0].NullOrdering;

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

        foreach (var sortKey in keys)
        {
            try
            {
                sortKey.Key.Validate(inputSchema);
            }
            catch (SchemaValidationException ex)
            {
                throw new SchemaValidationException($"Sort key expression validation failed: {ex.Message}");
            }
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
            var (syntheticInput, syntheticSortKeys) = MaterializeKeys(input, evaluator);
            var sortOperation = new SortOperation(syntheticSortKeys, stable);
            return StripSyntheticKeys(sortOperation.Execute(syntheticInput));
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Sort operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Evaluates the computed key expressions against the input columns, producing a copy of the
    /// input with synthetic key columns added and the sort keys referencing them. Exposed for the
    /// parallel execution strategy so it can materialize keys once, then reuse the chunked sort path.
    /// </summary>
    /// <param name="input">The input columns</param>
    /// <param name="evaluator">The evaluator used to materialize the key expressions</param>
    /// <returns>The synthetic input dictionary and the sort keys referencing the synthetic columns</returns>
    internal (IReadOnlyDictionary<string, IColumn> SyntheticInput, IReadOnlyList<SortKey> SortKeys)
        MaterializeKeys(IReadOnlyDictionary<string, IColumn> input, ExpressionEvaluator evaluator)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(evaluator);

        var synthetic = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in input)
            synthetic[kvp.Key] = kvp.Value;

        var syntheticSortKeys = new SortKey[keys.Count];
        for (int i = 0; i < keys.Count; i++)
        {
            var syntheticName = SyntheticKeyNamePrefix + i;
            synthetic[syntheticName] = evaluator.Evaluate(keys[i].Key, input);
            syntheticSortKeys[i] = new SortKey(syntheticName, keys[i].Direction, keys[i].NullOrdering);
        }

        return (synthetic, syntheticSortKeys);
    }

    /// <summary>
    /// Removes the synthetic key columns from a sorted result dictionary.
    /// </summary>
    /// <param name="sorted">The sorted columns, including synthetic key columns</param>
    /// <returns>A dictionary without the synthetic key columns</returns>
    internal static IReadOnlyDictionary<string, IColumn> StripSyntheticKeys(IReadOnlyDictionary<string, IColumn> sorted)
    {
        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in sorted)
            if (!kvp.Key.StartsWith(SyntheticKeyNamePrefix, StringComparison.OrdinalIgnoreCase))
                result[kvp.Key] = kvp.Value;

        return result;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var keyDescriptions = keys.Select(k =>
        {
            var directionStr = k.Direction == SortDirection.Ascending ? "ASC" : "DESC";
            var nullStr = k.NullOrdering == NullOrdering.NullsFirst ? "NULLS FIRST" : "NULLS LAST";
            return $"{k.Key.Name} {directionStr} {nullStr}";
        });
        return $"SortByExpression({string.Join(", ", keyDescriptions)})";
    }
}
