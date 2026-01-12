namespace Nivara.Query;

/// <summary>
/// Abstract base class for query plan nodes that represent operations in a query tree
/// </summary>
public abstract class QueryNode
{
    /// <summary>
    /// Initializes a new instance of QueryNode
    /// </summary>
    /// <param name="outputSchema">The schema produced by this node</param>
    protected QueryNode(Schema outputSchema)
    {
        OutputSchema = outputSchema ?? throw new ArgumentNullException(nameof(outputSchema));
        Children = new List<QueryNode>();
        EstimatedRowCount = -1; // Unknown by default
        EstimatedExecutionTime = TimeSpan.Zero;
    }

    /// <summary>
    /// Gets the child nodes of this query node
    /// </summary>
    public List<QueryNode> Children { get; }

    /// <summary>
    /// Gets the schema that this node will produce
    /// </summary>
    public Schema OutputSchema { get; }

    /// <summary>
    /// Gets or sets the estimated number of rows this node will produce
    /// </summary>
    public long EstimatedRowCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated execution time for this node
    /// </summary>
    public TimeSpan EstimatedExecutionTime { get; set; }

    /// <summary>
    /// Gets the type of this query node
    /// </summary>
    public abstract string NodeType { get; }

    /// <summary>
    /// Accepts a visitor for traversing the query tree
    /// </summary>
    /// <param name="visitor">The visitor to accept</param>
    public abstract void Accept(IQueryNodeVisitor visitor);

    /// <summary>
    /// Accepts a visitor for transforming the query tree
    /// </summary>
    /// <typeparam name="T">The result type of the transformation</typeparam>
    /// <param name="visitor">The visitor to accept</param>
    /// <returns>The result of the transformation</returns>
    public abstract T Accept<T>(IQueryNodeVisitor<T> visitor);

    /// <summary>
    /// Creates a copy of this node with the specified children
    /// </summary>
    /// <param name="newChildren">The new children for the copied node</param>
    /// <returns>A copy of this node with the new children</returns>
    public abstract QueryNode WithChildren(IEnumerable<QueryNode> newChildren);

    /// <summary>
    /// Returns a string representation of this query node
    /// </summary>
    /// <returns>A formatted string describing the node</returns>
    public override string ToString()
    {
        var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
        return $"{NodeType} [Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
    }
}

/// <summary>
/// Represents a data source node in the query plan
/// </summary>
public sealed class SourceNode : QueryNode
{
    /// <summary>
    /// Initializes a new instance of SourceNode
    /// </summary>
    /// <param name="source">The data source</param>
    public SourceNode(IQuerySource source) : base(source?.Schema ?? throw new ArgumentNullException(nameof(source)))
    {
        Source = source;
    }

    /// <summary>
    /// Gets the data source for this node
    /// </summary>
    public IQuerySource Source { get; }

    /// <inheritdoc />
    public override string NodeType => "Source";

    /// <inheritdoc />
    public override void Accept(IQueryNodeVisitor visitor)
    {
        visitor.Visit(this);
    }

    /// <inheritdoc />
    public override T Accept<T>(IQueryNodeVisitor<T> visitor)
    {
        return visitor.Visit(this);
    }

    /// <inheritdoc />
    public override QueryNode WithChildren(IEnumerable<QueryNode> newChildren)
    {
        // Source nodes don't have children
        return this;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
        return $"Source [{Source.GetType().Name}, Lazy: {Source.IsLazy}, Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
    }
}

/// <summary>
/// Represents a filter operation node in the query plan
/// </summary>
public sealed class FilterNode : QueryNode
{
    /// <summary>
    /// Initializes a new instance of FilterNode
    /// </summary>
    /// <param name="inputSchema">The input schema</param>
    /// <param name="predicate">The filter predicate</param>
    /// <param name="selectivity">The estimated selectivity (fraction of rows that pass the filter)</param>
    public FilterNode(Schema inputSchema, string predicate, double selectivity = 0.5) : base(inputSchema)
    {
        Predicate = predicate ?? throw new ArgumentNullException(nameof(predicate));
        Selectivity = Math.Clamp(selectivity, 0.0, 1.0);
    }

    /// <summary>
    /// Gets the filter predicate
    /// </summary>
    public string Predicate { get; }

    /// <summary>
    /// Gets the estimated selectivity (fraction of rows that pass the filter)
    /// </summary>
    public double Selectivity { get; }

    /// <inheritdoc />
    public override string NodeType => "Filter";

    /// <inheritdoc />
    public override void Accept(IQueryNodeVisitor visitor)
    {
        visitor.Visit(this);
    }

    /// <inheritdoc />
    public override T Accept<T>(IQueryNodeVisitor<T> visitor)
    {
        return visitor.Visit(this);
    }

    /// <inheritdoc />
    public override QueryNode WithChildren(IEnumerable<QueryNode> newChildren)
    {
        var newNode = new FilterNode(OutputSchema, Predicate, Selectivity)
        {
            EstimatedRowCount = EstimatedRowCount,
            EstimatedExecutionTime = EstimatedExecutionTime
        };
        newNode.Children.AddRange(newChildren);
        return newNode;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
        return $"Filter [Predicate: {Predicate}, Selectivity: {Selectivity:P1}, Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
    }
}

/// <summary>
/// Represents a projection (select) operation node in the query plan
/// </summary>
public sealed class ProjectionNode : QueryNode
{
    /// <summary>
    /// Initializes a new instance of ProjectionNode
    /// </summary>
    /// <param name="outputSchema">The output schema after projection</param>
    /// <param name="projections">The list of column projections</param>
    public ProjectionNode(Schema outputSchema, IReadOnlyList<string> projections) : base(outputSchema)
    {
        Projections = projections ?? throw new ArgumentNullException(nameof(projections));
    }

    /// <summary>
    /// Gets the list of column projections
    /// </summary>
    public IReadOnlyList<string> Projections { get; }

    /// <summary>
    /// Gets a value indicating whether this projection eliminates columns
    /// </summary>
    public bool EliminatesColumns => Children.Count > 0 && OutputSchema.ColumnNames.Count < Children[0].OutputSchema.ColumnNames.Count;

    /// <inheritdoc />
    public override string NodeType => "Projection";

    /// <inheritdoc />
    public override void Accept(IQueryNodeVisitor visitor)
    {
        visitor.Visit(this);
    }

    /// <inheritdoc />
    public override T Accept<T>(IQueryNodeVisitor<T> visitor)
    {
        return visitor.Visit(this);
    }

    /// <inheritdoc />
    public override QueryNode WithChildren(IEnumerable<QueryNode> newChildren)
    {
        var newNode = new ProjectionNode(OutputSchema, Projections)
        {
            EstimatedRowCount = EstimatedRowCount,
            EstimatedExecutionTime = EstimatedExecutionTime
        };
        newNode.Children.AddRange(newChildren);
        return newNode;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var projectionStr = string.Join(", ", Projections);
        var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
        return $"Projection [Columns: {projectionStr}, Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
    }
}

/// <summary>
/// Represents a group by operation node in the query plan
/// </summary>
public sealed class GroupByNode : QueryNode
{
    /// <summary>
    /// Initializes a new instance of GroupByNode
    /// </summary>
    /// <param name="outputSchema">The output schema after grouping</param>
    /// <param name="groupByColumns">The columns to group by</param>
    /// <param name="aggregations">The aggregation functions to apply</param>
    public GroupByNode(Schema outputSchema, IReadOnlyList<string> groupByColumns, IReadOnlyList<string> aggregations) : base(outputSchema)
    {
        GroupByColumns = groupByColumns ?? throw new ArgumentNullException(nameof(groupByColumns));
        Aggregations = aggregations ?? throw new ArgumentNullException(nameof(aggregations));
    }

    /// <summary>
    /// Gets the columns to group by
    /// </summary>
    public IReadOnlyList<string> GroupByColumns { get; }

    /// <summary>
    /// Gets the aggregation functions to apply
    /// </summary>
    public IReadOnlyList<string> Aggregations { get; }

    /// <inheritdoc />
    public override string NodeType => "GroupBy";

    /// <inheritdoc />
    public override void Accept(IQueryNodeVisitor visitor)
    {
        visitor.Visit(this);
    }

    /// <inheritdoc />
    public override T Accept<T>(IQueryNodeVisitor<T> visitor)
    {
        return visitor.Visit(this);
    }

    /// <inheritdoc />
    public override QueryNode WithChildren(IEnumerable<QueryNode> newChildren)
    {
        var newNode = new GroupByNode(OutputSchema, GroupByColumns, Aggregations)
        {
            EstimatedRowCount = EstimatedRowCount,
            EstimatedExecutionTime = EstimatedExecutionTime
        };
        newNode.Children.AddRange(newChildren);
        return newNode;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var groupByStr = string.Join(", ", GroupByColumns);
        var aggregationStr = string.Join(", ", Aggregations);
        var rowCountStr = EstimatedRowCount >= 0 ? EstimatedRowCount.ToString("N0") : "Unknown";
        return $"GroupBy [Keys: {groupByStr}, Aggregations: {aggregationStr}, Rows: {rowCountStr}, Schema: {OutputSchema.ColumnNames.Count} columns]";
    }
}

/// <summary>
/// Visitor interface for traversing query node trees
/// </summary>
public interface IQueryNodeVisitor
{
    /// <summary>
    /// Visits a source node
    /// </summary>
    /// <param name="node">The source node to visit</param>
    void Visit(SourceNode node);

    /// <summary>
    /// Visits a filter node
    /// </summary>
    /// <param name="node">The filter node to visit</param>
    void Visit(FilterNode node);

    /// <summary>
    /// Visits a projection node
    /// </summary>
    /// <param name="node">The projection node to visit</param>
    void Visit(ProjectionNode node);

    /// <summary>
    /// Visits a group by node
    /// </summary>
    /// <param name="node">The group by node to visit</param>
    void Visit(GroupByNode node);
}

/// <summary>
/// Generic visitor interface for transforming query node trees
/// </summary>
/// <typeparam name="T">The result type of the transformation</typeparam>
public interface IQueryNodeVisitor<T>
{
    /// <summary>
    /// Visits a source node and returns a result
    /// </summary>
    /// <param name="node">The source node to visit</param>
    /// <returns>The result of visiting the node</returns>
    T Visit(SourceNode node);

    /// <summary>
    /// Visits a filter node and returns a result
    /// </summary>
    /// <param name="node">The filter node to visit</param>
    /// <returns>The result of visiting the node</returns>
    T Visit(FilterNode node);

    /// <summary>
    /// Visits a projection node and returns a result
    /// </summary>
    /// <param name="node">The projection node to visit</param>
    /// <returns>The result of visiting the node</returns>
    T Visit(ProjectionNode node);

    /// <summary>
    /// Visits a group by node and returns a result
    /// </summary>
    /// <param name="node">The group by node to visit</param>
    /// <returns>The result of visiting the node</returns>
    T Visit(GroupByNode node);
}