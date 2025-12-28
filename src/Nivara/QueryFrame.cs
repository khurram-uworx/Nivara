using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara;

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

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // QueryFrame doesn't own the source, so we don't dispose it
            // Operations are value types or immutable, no disposal needed
            disposed = true;
        }
    }
}