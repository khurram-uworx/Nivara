using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Query;
using Nivara.Tensors;

namespace Nivara.Operations;

/// <summary>
/// Rank-family window operation (row_number / rank / dense_rank / percent_rank).
/// Appends a result column while preserving all input columns. Unlike
/// <see cref="WindowOperationBase"/> there is no single source column: ranks are computed
/// over partition + order-by keys. Keys may be named columns or computed expressions.
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery.</remarks>
sealed class RankOperation : IQueryOperation
{
    readonly string[] partitionBy;
    readonly IReadOnlyList<SortKey> orderBy;
    readonly IReadOnlyList<ColumnExpression> partitionByExpressions;
    readonly IReadOnlyList<SortExpressionKey> orderByExpressions;
    readonly bool expressionBased;

    /// <summary>
    /// Initializes a new instance of RankOperation over named partition/order columns
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="kind">The rank function kind</param>
    /// <param name="orderBy">The order keys</param>
    /// <param name="partitionBy">The partition key column names (null = single partition)</param>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace, or when
    /// <paramref name="kind"/> is not RowNumber and no order keys are provided</exception>
    public RankOperation(string resultColumn, RankKind kind, IReadOnlyList<SortKey> orderBy, string[]? partitionBy = null)
    {
        if (string.IsNullOrWhiteSpace(resultColumn))
            throw new ArgumentException("Result column name cannot be null or whitespace", nameof(resultColumn));

        ArgumentNullException.ThrowIfNull(orderBy);

        if (kind != RankKind.RowNumber && orderBy.Count == 0)
            throw new ArgumentException($"'{kind}' requires at least one order key", nameof(orderBy));

        ResultColumn = resultColumn;
        Kind = kind;
        this.orderBy = orderBy.ToList();
        this.partitionBy = partitionBy ?? Array.Empty<string>();
        partitionByExpressions = Array.Empty<ColumnExpression>();
        orderByExpressions = Array.Empty<SortExpressionKey>();
        expressionBased = false;
    }

    /// <summary>
    /// Initializes a new instance of RankOperation over computed partition/order key expressions
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="kind">The rank function kind</param>
    /// <param name="orderBy">The order key expressions</param>
    /// <param name="partitionBy">The partition key expressions (null = single partition)</param>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace, or when
    /// <paramref name="kind"/> is not RowNumber and no order keys are provided</exception>
    public RankOperation(string resultColumn, RankKind kind, IReadOnlyList<SortExpressionKey> orderBy, IReadOnlyList<ColumnExpression>? partitionBy = null)
    {
        if (string.IsNullOrWhiteSpace(resultColumn))
            throw new ArgumentException("Result column name cannot be null or whitespace", nameof(resultColumn));

        ArgumentNullException.ThrowIfNull(orderBy);

        if (kind != RankKind.RowNumber && orderBy.Count == 0)
            throw new ArgumentException($"'{kind}' requires at least one order key", nameof(orderBy));

        ResultColumn = resultColumn;
        Kind = kind;
        this.orderBy = Array.Empty<SortKey>();
        this.partitionBy = Array.Empty<string>();
        orderByExpressions = orderBy.ToList();
        partitionByExpressions = partitionBy ?? Array.Empty<ColumnExpression>();
        expressionBased = true;
    }

    /// <summary>
    /// Initializes a new instance of RankOperation over a window specification (partition/order keys)
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="kind">The rank function kind</param>
    /// <param name="spec">The window specification</param>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or spec is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace, or when
    /// <paramref name="kind"/> is not RowNumber and the spec has no order keys</exception>
    public RankOperation(string resultColumn, RankKind kind, WindowSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);

        if (string.IsNullOrWhiteSpace(resultColumn))
            throw new ArgumentException("Result column name cannot be null or whitespace", nameof(resultColumn));

        if (kind != RankKind.RowNumber && spec.OrderKeys.Count == 0)
            throw new ArgumentException($"'{kind}' requires at least one order key", nameof(spec));

        ResultColumn = resultColumn;
        Kind = kind;
        orderBy = spec.OrderKeys.ToList();
        partitionBy = spec.PartitionColumns.ToArray();
        partitionByExpressions = Array.Empty<ColumnExpression>();
        orderByExpressions = Array.Empty<SortExpressionKey>();
        expressionBased = false;
    }

    /// <summary>
    /// Gets the name of the appended result column
    /// </summary>
    public string ResultColumn { get; }

    /// <summary>
    /// Gets the rank function kind
    /// </summary>
    public RankKind Kind { get; }

    /// <summary>
    /// Gets the partition key column names (empty when computed expressions are used)
    /// </summary>
    public IReadOnlyList<string> PartitionBy => partitionBy;

    /// <summary>
    /// Gets the order keys (empty when computed expressions are used)
    /// </summary>
    public IReadOnlyList<SortKey> OrderBy => orderBy;

    /// <summary>
    /// Gets a value indicating whether the keys are computed expressions rather than named columns
    /// </summary>
    public bool IsExpressionBased => expressionBased;

    /// <summary>
    /// Gets the partition key expressions (empty when named columns are used)
    /// </summary>
    public IReadOnlyList<ColumnExpression> PartitionByExpressions => partitionByExpressions;

    /// <summary>
    /// Gets the order key expressions (empty when named columns are used)
    /// </summary>
    public IReadOnlyList<SortExpressionKey> OrderByExpressions => orderByExpressions;

    public string OperationType => Query.OperationType.Rank;

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);

        if (inputSchema.HasColumn(ResultColumn))
            throw new ArgumentException($"Result column '{ResultColumn}' already exists in the schema", nameof(ResultColumn));

        if (expressionBased)
        {
            foreach (var key in orderByExpressions)
            {
                try
                {
                    key.Key.Validate(inputSchema);
                }
                catch (SchemaValidationException ex)
                {
                    throw new SchemaValidationException($"Rank order expression validation failed: {ex.Message}");
                }

                if (!SortOperation.IsComparableType(key.Key.ResultType))
                {
                    throw new SchemaValidationException(
                        $"Rank order expression '{key.Key.Name}' of type '{key.Key.ResultType.Name}' is not comparable and cannot be used for ranking");
                }
            }

            foreach (var partition in partitionByExpressions)
            {
                try
                {
                    partition.Validate(inputSchema);
                }
                catch (SchemaValidationException ex)
                {
                    throw new SchemaValidationException($"Rank partition expression validation failed: {ex.Message}");
                }
            }

            return inputSchema.WithColumn(ResultColumn, Kind == RankKind.PercentRank ? typeof(double) : typeof(long));
        }

        foreach (var partition in partitionBy)
        {
            if (!inputSchema.HasColumn(partition))
            {
                throw new SchemaValidationException(
                    $"Partition column '{partition}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");
            }
        }

        foreach (var sortKey in orderBy)
        {
            if (!inputSchema.HasColumn(sortKey.ColumnName))
            {
                throw new SchemaValidationException(
                    $"Order column '{sortKey.ColumnName}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");
            }

            var columnType = inputSchema.GetColumnType(sortKey.ColumnName);
            if (!SortOperation.IsComparableType(columnType))
            {
                throw new SchemaValidationException(
                    $"Column '{sortKey.ColumnName}' of type '{columnType.Name}' is not comparable and cannot be used for ranking");
            }
        }

        return inputSchema.WithColumn(ResultColumn, Kind == RankKind.PercentRank ? typeof(double) : typeof(long));
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        try
        {
            IColumn resultColumn;
            if (expressionBased)
            {
                var window = new WindowExpression(WindowFunctionHelpers.ToWindowFunctionKind(Kind), partitionByExpressions, orderByExpressions);
                resultColumn = new FusedExpressionEvaluator().Evaluate(window, input);
            }
            else
            {
                resultColumn = RankKernel.Compute(input, partitionBy, orderBy, Kind);
            }

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
                result[kvp.Key] = kvp.Value;
            result[ResultColumn] = resultColumn;
            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Rank operation failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var orderStr = string.Join(", ", expressionBased
            ? orderByExpressions.Select(k => k.Key.Name)
            : orderBy.Select(k => k.ColumnName));
        var partitionStr = (expressionBased
            ? partitionByExpressions.Count > 0
            : partitionBy.Length > 0)
            ? $" OVER (PARTITION BY {string.Join(", ", expressionBased ? partitionByExpressions.Select(p => p.Name) : partitionBy)})"
            : "";
        return $"{Kind}({orderStr} -> {ResultColumn}){partitionStr}";
    }
}
