using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Tensors;

namespace Nivara.Query;

/// <summary>
/// Represents a lazy query frame that builds query plans without immediate execution.
/// Provides a fluent API for constructing complex queries that are executed only when Collect() is called.
/// </summary>
public sealed class QueryFrame : IDisposable
{
    readonly IQuerySource source;
    readonly List<IQueryOperation> operations;
    bool disposed;

    /// <summary>
    /// Initializes a new instance of QueryFrame with the specified data source
    /// </summary>
    /// <param name="source">The data source for the query</param>
    /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
    internal QueryFrame(IQuerySource source)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        operations = new List<IQueryOperation>();

        // Track lazy queries for abandoned resource cleanup
        if (source.IsLazy)
        {
            NivaraResourceManager.TrackResource(this, "LazyQueryFrame", 0, () =>
            {
                // Cleanup action for abandoned lazy queries
                try
                {
                    source?.Dispose();
                }
                catch
                {
                    // Ignore disposal errors for abandoned resources
                }
            });
        }
    }

    /// <summary>
    /// Initializes a new instance of QueryFrame with the specified data source and operations
    /// </summary>
    /// <param name="source">The data source for the query</param>
    /// <param name="operations">The existing operations</param>
    /// <exception cref="ArgumentNullException">Thrown when source or operations is null</exception>
    internal QueryFrame(IQuerySource source, IEnumerable<IQueryOperation> operations)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        this.operations = operations?.ToList() ?? throw new ArgumentNullException(nameof(operations));

        // Track lazy queries for abandoned resource cleanup
        if (source.IsLazy)
        {
            NivaraResourceManager.TrackResource(this, "LazyQueryFrame", 0, () =>
            {
                // Cleanup action for abandoned lazy queries
                try
                {
                    source?.Dispose();
                }
                catch
                {
                    // Ignore disposal errors for abandoned resources
                }
            });
        }
    }

    /// <summary>
    /// Gets the schema that will result from executing this query
    /// </summary>
    public Schema Schema
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            var plan = new QueryPlan(source, operations);
            return plan.ResultSchema;
        }
    }

    /// <summary>
    /// Gets a value indicating whether this query uses a lazy data source
    /// </summary>
    public bool IsLazy
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return source.IsLazy;
        }
    }

    /// <summary>
    /// Adds a filter operation to the query chain
    /// </summary>
    /// <param name="condition">The condition to filter by</param>
    /// <returns>A new QueryFrame with the filter operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when condition is null</exception>
    public QueryFrame Filter(ColumnExpression condition)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        ArgumentNullException.ThrowIfNull(condition);

        var filterOperation = new FilterOperation(condition);
        var newOperations = operations.Concat(new[] { filterOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a select (projection) operation to the query chain
    /// </summary>
    /// <param name="columns">The column expressions to select</param>
    /// <returns>A new QueryFrame with the select operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when columns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public QueryFrame Select(params ColumnExpression[] columns)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columns == null)
            throw new ArgumentNullException(nameof(columns));

        if (columns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression", nameof(columns));

        var selectOperation = new SelectOperation(columns);
        var newOperations = operations.Concat(new[] { selectOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a select (projection) operation to the query chain using column names
    /// </summary>
    /// <param name="columnNames">The names of the columns to select</param>
    /// <returns>A new QueryFrame with the select operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no column names are specified</exception>
    public QueryFrame Select(params string[] columnNames)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        if (columnNames.Length == 0)
            throw new ArgumentException("Must specify at least one column name", nameof(columnNames));

        var columnExpressions = columnNames.Select(name => ColumnExpressions.Col(name)).ToArray();
        return Select(columnExpressions);
    }

    /// <summary>
    /// Adds a group by operation to the query chain
    /// </summary>
    /// <param name="columnNames">The names of the columns to group by</param>
    /// <returns>A new QueryFrame with the group by operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no column names are specified</exception>
    public QueryFrame GroupBy(params string[] columnNames)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        if (columnNames.Length == 0)
            throw new ArgumentException("Must specify at least one column name", nameof(columnNames));

        var expressions = columnNames.Select(name => ColumnExpressions.Col(name)).ToArray();
        var groupByOperation = new GroupByOperation(expressions);
        var newOperations = operations.Concat(new[] { groupByOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a group by operation to the query chain using column expressions
    /// </summary>
    /// <param name="columns">The column expressions to group by</param>
    /// <returns>A new QueryFrame with the group by operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when columns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public QueryFrame GroupBy(params ColumnExpression[] columns)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columns == null)
            throw new ArgumentNullException(nameof(columns));

        if (columns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression", nameof(columns));

        var groupByOperation = new GroupByOperation(columns);
        var newOperations = operations.Concat(new[] { groupByOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a distinct operation that removes duplicate rows from the result.
    /// </summary>
    /// <returns>A new QueryFrame with the distinct operation added</returns>
    public QueryFrame Distinct()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var distinctOp = new DistinctOperation();
        var newOperations = operations.Concat(new[] { distinctOp });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a distinct operation that removes rows duplicate in the specified columns.
    /// </summary>
    /// <param name="columnNames">The column names to use for deduplication</param>
    /// <returns>A new QueryFrame with the distinct operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when columnNames is null</exception>
    /// <exception cref="ArgumentException">Thrown when no column names are specified</exception>
    public QueryFrame Distinct(params string[] columnNames)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (columnNames == null)
            throw new ArgumentNullException(nameof(columnNames));

        if (columnNames.Length == 0)
            throw new ArgumentException("Must specify at least one column name", nameof(columnNames));

        var distinctOp = new DistinctOperation(columnNames);
        var newOperations = operations.Concat(new[] { distinctOp });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a sort operation to the query pipeline for single column sorting
    /// </summary>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <param name="direction">The sort direction (ascending or descending)</param>
    /// <param name="nullOrdering">How to order null values (nulls first or nulls last)</param>
    /// <param name="stable">Whether to use stable sorting (preserves relative order of equal elements)</param>
    /// <returns>A new QueryFrame with the sort operation added</returns>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    public QueryFrame Sort(string columnName, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast, bool stable = true)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        var sortOperation = new SortOperation(columnName, direction, nullOrdering, stable);
        var newOperations = operations.Concat(new[] { sortOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a sort operation to the query pipeline for multi-column sorting
    /// </summary>
    /// <param name="sortKeys">The sort keys defining the sort order and priority</param>
    /// <param name="stable">Whether to use stable sorting (preserves relative order of equal elements)</param>
    /// <returns>A new QueryFrame with the sort operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when sortKeys is null</exception>
    /// <exception cref="ArgumentException">Thrown when no sort keys are provided</exception>
    public QueryFrame Sort(IEnumerable<SortKey> sortKeys, bool stable = true)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (sortKeys == null)
            throw new ArgumentNullException(nameof(sortKeys));

        var sortOperation = new SortOperation(sortKeys, stable);
        var newOperations = operations.Concat(new[] { sortOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a sort operation to the query pipeline for multi-column sorting
    /// </summary>
    /// <param name="sortKeys">The sort keys defining the sort order and priority</param>
    /// <returns>A new QueryFrame with the sort operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when sortKeys is null</exception>
    /// <exception cref="ArgumentException">Thrown when no sort keys are provided</exception>
    public QueryFrame Sort(params SortKey[] sortKeys)
    {
        return Sort(sortKeys, stable: true);
    }

    /// <summary>
    /// Adds a sort operation to the query pipeline using a computed column expression as the sort key.
    /// The expression is materialized into a column at execution time and every input column is
    /// reordered by the resulting sort order.
    /// </summary>
    /// <param name="keyExpression">The key expression to sort by</param>
    /// <param name="direction">The sort direction (ascending or descending)</param>
    /// <param name="nullOrdering">How to order null values (nulls first or nulls last)</param>
    /// <param name="stable">Whether to use stable sorting (preserves relative order of equal elements)</param>
    /// <returns>A new QueryFrame with the sort operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when keyExpression is null</exception>
    public QueryFrame SortByExpression(ColumnExpression keyExpression, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast, bool stable = true)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        ArgumentNullException.ThrowIfNull(keyExpression);

        var sortOperation = new SortByExpressionOperation(keyExpression, direction, nullOrdering, stable);
        var newOperations = operations.Concat(new[] { sortOperation });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Appends a secondary sort key to the query, merging it into the preceding sort operation when
    /// one is present so the ordering composes lexicographically (primary key first, then secondary).
    /// When the preceding operation is not a sort, the key acts as a primary sort key.
    /// </summary>
    /// <param name="key">The secondary key expression</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <returns>A new QueryFrame with the secondary sort key added</returns>
    /// <exception cref="ArgumentNullException">Thrown when key is null</exception>
    internal QueryFrame ThenBy(ColumnExpression key, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        ArgumentNullException.ThrowIfNull(key);

        var newOperations = new List<IQueryOperation>(operations);

        if (operations.Count > 0)
        {
            var last = operations[operations.Count - 1];

            if (last is SortOperation sortOp)
            {
                if (key is ColumnReference colRef)
                {
                    var mergedKeys = sortOp.SortKeys
                        .Concat(new[] { new SortKey(colRef.ColumnName, direction, nullOrdering) })
                        .ToArray();
                    newOperations[^1] = new SortOperation(mergedKeys, sortOp.IsStable);
                }
                else
                {
                    var mergedKeys = sortOp.SortKeys
                        .Select(k => new SortExpressionKey(ColumnExpressions.Col(k.ColumnName), k.Direction, k.NullOrdering))
                        .Concat(new[] { new SortExpressionKey(key, direction, nullOrdering) })
                        .ToArray();
                    newOperations[^1] = new SortByExpressionOperation(mergedKeys, sortOp.IsStable);
                }

                return new QueryFrame(source, newOperations);
            }

            if (last is SortByExpressionOperation sortExprOp)
            {
                var mergedKeys = sortExprOp.Keys
                    .Concat(new[] { new SortExpressionKey(key, direction, nullOrdering) })
                    .ToArray();
                newOperations[^1] = new SortByExpressionOperation(mergedKeys, sortExprOp.IsStable);

                return new QueryFrame(source, newOperations);
            }
        }

        if (key is ColumnReference columnRef)
            newOperations.Add(new SortOperation(columnRef.ColumnName, direction, nullOrdering));
        else
            newOperations.Add(new SortByExpressionOperation(key, direction, nullOrdering));

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Executes the query and returns a materialized NivaraFrame
    /// This is the execution barrier that triggers lazy query evaluation
    /// </summary>
    /// <returns>A materialized NivaraFrame with the query results</returns>
    /// <exception cref="QueryExecutionException">Thrown when query execution fails</exception>
    public NivaraFrame Collect()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        try
        {
            var queryPlan = new QueryPlan(source, operations);
            var executor = new QueryExecutor();
            return executor.Execute(queryPlan);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Query execution failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Returns a string representation of the query plan for debugging
    /// </summary>
    /// <returns>A formatted string describing the query plan</returns>
    public string ExplainPlan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var queryPlan = new QueryPlan(source, operations);
        return QueryPlanAnalyzer.Explain(queryPlan);
    }

    /// <summary>
    /// Returns diagnostic information about the query plan based on the specified mode
    /// </summary>
    /// <param name="mode">The diagnostic mode to use</param>
    /// <returns>Diagnostic information formatted according to the mode</returns>
    public string GetDiagnosticInfo(QueryDiagnosticMode mode = QueryDiagnosticMode.Basic)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var queryPlan = new QueryPlan(source, operations);
        return QueryDiagnostics.GetDiagnosticInfo(queryPlan, mode);
    }

    /// <summary>
    /// Analyzes the query plan for potential optimization opportunities
    /// </summary>
    /// <returns>A list of optimization suggestions</returns>
    public IReadOnlyList<string> AnalyzeOptimizations()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var queryPlan = new QueryPlan(source, operations);
        return QueryPlanAnalyzer.AnalyzeOptimizations(queryPlan);
    }

    /// <summary>
    /// Analyzes the query plan for potential issues and provides recommendations
    /// </summary>
    /// <returns>A list of diagnostic recommendations</returns>
    public IReadOnlyList<string> AnalyzeQueryPlan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var queryPlan = new QueryPlan(source, operations);
        return QueryDiagnostics.AnalyzeQueryPlan(queryPlan);
    }

    /// <summary>
    /// Returns a string representation of the query frame
    /// </summary>
    /// <returns>A formatted string describing the query frame</returns>
    public override string ToString()
    {
        if (disposed)
            return "QueryFrame [Disposed]";

        var operationNames = operations.Select(op => op.OperationType);
        var pipeline = string.Join(" -> ", operationNames);

        if (string.IsNullOrEmpty(pipeline))
            return $"QueryFrame {{ Source: {source.GetType().Name}, Operations: None }}";

        return $"QueryFrame {{ Source: {source.GetType().Name}, Pipeline: {pipeline} }}";
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current query frame
    /// </summary>
    /// <param name="obj">The object to compare</param>
    /// <returns>True if the objects are equal, false otherwise</returns>
    public override bool Equals(object? obj)
    {
        return ReferenceEquals(this, obj);
    }

    /// <summary>
    /// Returns a hash code for the query frame
    /// </summary>
    /// <returns>A hash code for the query frame</returns>
    public override int GetHashCode()
    {
        return base.GetHashCode();
    }

    /// <summary>
    /// Adds a row-selection operation that extracts rows by index.
    /// </summary>
    /// <param name="indices">The row indices to select</param>
    /// <returns>A new QueryFrame with the select-rows operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when indices is null</exception>
    /// <exception cref="ArgumentException">Thrown when indices is empty</exception>
    public QueryFrame SelectRows(params int[] indices)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var selectRowsOp = new SelectRowsOperation(indices);
        var newOperations = operations.Concat(new[] { selectRowsOp });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a skip operation that omits the first N rows.
    /// </summary>
    /// <param name="count">The number of rows to skip (must be non-negative)</param>
    /// <returns>A new QueryFrame with the skip operation added</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when count is negative</exception>
    public QueryFrame Skip(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var sliceOp = new SliceOperation(skip: count);
        var newOperations = operations.Concat(new[] { sliceOp });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a take operation that keeps only the first N rows.
    /// </summary>
    /// <param name="count">The number of rows to take (must be non-negative)</param>
    /// <returns>A new QueryFrame with the take operation added</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when count is negative</exception>
    public QueryFrame Take(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var sliceOp = new SliceOperation(skip: 0, take: count);
        var newOperations = operations.Concat(new[] { sliceOp });

        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Adds a combined skip-and-take (page) operation.
    /// </summary>
    /// <param name="skip">The number of rows to skip (must be non-negative)</param>
    /// <param name="take">The number of rows to take after skipping (must be non-negative)</param>
    /// <returns>A new QueryFrame with the slice operation added</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when skip or take are negative</exception>
    public QueryFrame Slice(int skip, int take)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var sliceOp = new SliceOperation(skip, take);
        var newOperations = operations.Concat(new[] { sliceOp });

        return new QueryFrame(source, newOperations);
    }

    // ── Window functions ──

    /// <summary>
    /// Adds a rolling-sum window operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required (defaults to the full window)</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the rolling-sum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame RollingSum(string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => AddWindowOperation(new RollingOperation(source, resultColumn, windowSize, minPeriods, nullHandler, NivaraFrameExtensions.RollingKind.Sum));

    /// <summary>
    /// Adds a rolling-mean window operation that appends a double result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required (defaults to the full window)</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the rolling-mean operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame RollingMean(string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => AddWindowOperation(new RollingOperation(source, resultColumn, windowSize, minPeriods, nullHandler, NivaraFrameExtensions.RollingKind.Mean));

    /// <summary>
    /// Adds a rolling-minimum window operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required (defaults to the full window)</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the rolling-minimum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame RollingMin(string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => AddWindowOperation(new RollingOperation(source, resultColumn, windowSize, minPeriods, nullHandler, NivaraFrameExtensions.RollingKind.Min));

    /// <summary>
    /// Adds a rolling-maximum window operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required (defaults to the full window)</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the rolling-maximum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame RollingMax(string source, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null)
        => AddWindowOperation(new RollingOperation(source, resultColumn, windowSize, minPeriods, nullHandler, NivaraFrameExtensions.RollingKind.Max));

    /// <summary>
    /// Adds a cumulative-sum operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the cumulative-sum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame CumulativeSum(string source, string resultColumn, Func<object?>? nullHandler = null)
        => AddWindowOperation(new CumulativeOperation(source, resultColumn, nullHandler, NivaraFrameExtensions.CumulativeKind.Sum));

    /// <summary>
    /// Adds a cumulative-maximum operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the cumulative-maximum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame CumulativeMax(string source, string resultColumn, Func<object?>? nullHandler = null)
        => AddWindowOperation(new CumulativeOperation(source, resultColumn, nullHandler, NivaraFrameExtensions.CumulativeKind.Max));

    /// <summary>
    /// Adds a cumulative-minimum operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the cumulative-minimum operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame CumulativeMin(string source, string resultColumn, Func<object?>? nullHandler = null)
        => AddWindowOperation(new CumulativeOperation(source, resultColumn, nullHandler, NivaraFrameExtensions.CumulativeKind.Min));

    /// <summary>
    /// Adds a cumulative-product operation that appends a result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <returns>A new QueryFrame with the cumulative-product operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame CumulativeProduct(string source, string resultColumn, Func<object?>? nullHandler = null)
        => AddWindowOperation(new CumulativeOperation(source, resultColumn, nullHandler, NivaraFrameExtensions.CumulativeKind.Product));

    /// <summary>
    /// Adds a running count-of-non-null operation that appends a long result column.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <returns>A new QueryFrame with the cumulative-count operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame CumulativeCount(string source, string resultColumn)
        => AddWindowOperation(new CumulativeOperation(source, resultColumn, null, NivaraFrameExtensions.CumulativeKind.Sum, isCount: true));

    /// <summary>
    /// Adds a shift (lag) operation that appends a result column. Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="periods">The number of positions to shift by</param>
    /// <param name="fillValue">Optional fill value for boundary positions</param>
    /// <returns>A new QueryFrame with the shift operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame Shift(string source, string resultColumn, int periods, object? fillValue = null)
        => AddWindowOperation(new ShiftOperation(source, resultColumn, periods, fillValue));

    /// <summary>
    /// Adds a lead operation that appends a result column. Boundary positions are null, or <paramref name="fillValue"/> when provided.
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="periods">The number of positions to lead by</param>
    /// <param name="fillValue">Optional fill value for boundary positions</param>
    /// <returns>A new QueryFrame with the lead operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when source or resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when source or resultColumn is whitespace</exception>
    public QueryFrame Lead(string source, string resultColumn, int periods, object? fillValue = null)
        => AddWindowOperation(new ShiftOperation(source, resultColumn, -periods, fillValue));

    /// <summary>
    /// Adds a row-number operation that appends a long result column. With no partition keys the
    /// numbering is sequential over all rows; with no order keys the numbering follows row order.
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="partitionBy">The partition key column names (null = single partition)</param>
    /// <param name="orderBy">The order keys (null = row order within each partition)</param>
    /// <returns>A new QueryFrame with the row-number operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace</exception>
    public QueryFrame RowNumber(string resultColumn, string[]? partitionBy = null, IReadOnlyList<SortKey>? orderBy = null)
        => AddWindowOperation(new RankOperation(resultColumn, RankKind.RowNumber, orderBy ?? Array.Empty<SortKey>(), partitionBy));

    /// <summary>
    /// Adds a standard rank operation (gaps on ties) that appends a long result column.
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="orderBy">The order keys (at least one is required)</param>
    /// <param name="partitionBy">The partition key column names</param>
    /// <returns>A new QueryFrame with the rank operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace or no order keys are provided</exception>
    public QueryFrame Rank(string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => AddWindowOperation(new RankOperation(resultColumn, RankKind.Rank, orderBy, partitionBy));

    /// <summary>
    /// Adds a dense-rank operation (no gaps on ties) that appends a long result column.
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="orderBy">The order keys (at least one is required)</param>
    /// <param name="partitionBy">The partition key column names</param>
    /// <returns>A new QueryFrame with the dense-rank operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace or no order keys are provided</exception>
    public QueryFrame DenseRank(string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => AddWindowOperation(new RankOperation(resultColumn, RankKind.DenseRank, orderBy, partitionBy));

    /// <summary>
    /// Adds a percent-rank operation that appends a double result column: (rank - 1) / (partitionSize - 1).
    /// </summary>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="orderBy">The order keys (at least one is required)</param>
    /// <param name="partitionBy">The partition key column names</param>
    /// <returns>A new QueryFrame with the percent-rank operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when resultColumn or orderBy is null</exception>
    /// <exception cref="ArgumentException">Thrown when resultColumn is whitespace or no order keys are provided</exception>
    public QueryFrame PercentRank(string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
        => AddWindowOperation(new RankOperation(resultColumn, RankKind.PercentRank, orderBy, partitionBy));

    QueryFrame AddWindowOperation(IQueryOperation operation)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (operation is WindowOperationBase windowOp)
        {
            if (string.IsNullOrWhiteSpace(windowOp.Source))
                throw new ArgumentException("Source column name cannot be null or whitespace", nameof(windowOp.Source));
            if (string.IsNullOrWhiteSpace(windowOp.ResultColumn))
                throw new ArgumentException("Result column name cannot be null or whitespace", nameof(windowOp.ResultColumn));
        }

        var newOperations = operations.Concat(new[] { operation });
        return new QueryFrame(source, newOperations);
    }

    /// <summary>
    /// Extracts the query plan for inspection or custom execution via <see cref="Execution.ExecutionEngine"/>.
    /// </summary>
    /// <returns>A QueryPlan representing this query's source and operations</returns>
    public QueryPlan ToQueryPlan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new QueryPlan(source, operations);
    }

    /// <summary>
    /// Appends a single operation to the pipeline, returning a new QueryFrame. Used by the typed
    /// LINQ layer to compose group-by aggregations that depend on deferred grouping state.
    /// </summary>
    /// <param name="operation">The operation to append</param>
    /// <returns>A new QueryFrame with the operation added</returns>
    /// <exception cref="ArgumentNullException">Thrown when operation is null</exception>
    internal QueryFrame WithOperation(IQueryOperation operation)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        ArgumentNullException.ThrowIfNull(operation);

        return new QueryFrame(source, operations.Concat(new[] { operation }));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Untrack from resource manager
            NivaraResourceManager.UntrackResource(this);

            // QueryFrame doesn't own the source in most cases, so we don't dispose it
            // The source is typically owned by the caller or factory methods
            // Operations are value types or immutable, no disposal needed
            disposed = true;
        }
    }
}
