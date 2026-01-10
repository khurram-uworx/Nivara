namespace Nivara;

/// <summary>
/// Visitor interface for traversing query plans
/// </summary>
public interface IQueryPlanVisitor
{
    /// <summary>
    /// Visits a query plan
    /// </summary>
    /// <param name="plan">The query plan to visit</param>
    void Visit(QueryPlan plan);

    /// <summary>
    /// Visits a query operation
    /// </summary>
    /// <param name="operation">The operation to visit</param>
    void Visit(IQueryOperation operation);
}

/// <summary>
/// Generic visitor interface for transforming query plans
/// </summary>
/// <typeparam name="T">The result type of the transformation</typeparam>
public interface IQueryPlanVisitor<T>
{
    /// <summary>
    /// Visits a query plan and returns a result
    /// </summary>
    /// <param name="plan">The query plan to visit</param>
    /// <returns>The result of visiting the plan</returns>
    T Visit(QueryPlan plan);

    /// <summary>
    /// Visits a query operation and returns a result
    /// </summary>
    /// <param name="operation">The operation to visit</param>
    /// <returns>The result of visiting the operation</returns>
    T Visit(IQueryOperation operation);
}

/// <summary>
/// Base class for query plan visitors that provides default traversal behavior
/// </summary>
public abstract class QueryPlanVisitorBase : IQueryPlanVisitor
{
    /// <inheritdoc />
    public virtual void Visit(QueryPlan plan)
    {
        if (plan == null)
            return;

        VisitSource(plan.Source);

        foreach (var operation in plan.Operations)
        {
            Visit(operation);
        }
    }

    /// <inheritdoc />
    public virtual void Visit(IQueryOperation operation)
    {
        if (operation == null)
            return;

        switch (operation.OperationType)
        {
            case "Filter":
                VisitFilter(operation);
                break;
            case "Select":
                VisitSelect(operation);
                break;
            case "GroupBy":
                VisitGroupBy(operation);
                break;
            case "Sort":
                VisitSort(operation);
                break;
            case "Join":
                VisitJoin(operation);
                break;
            case "Projection":
                VisitProjection(operation);
                break;
            case "Concatenation":
                VisitConcatenation(operation);
                break;
            default:
                VisitUnknownOperation(operation);
                break;
        }
    }

    /// <summary>
    /// Visits a query source
    /// </summary>
    /// <param name="source">The source to visit</param>
    protected virtual void VisitSource(IQuerySource source)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a filter operation
    /// </summary>
    /// <param name="operation">The filter operation to visit</param>
    protected virtual void VisitFilter(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a select operation
    /// </summary>
    /// <param name="operation">The select operation to visit</param>
    protected virtual void VisitSelect(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a group by operation
    /// </summary>
    /// <param name="operation">The group by operation to visit</param>
    protected virtual void VisitGroupBy(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a sort operation
    /// </summary>
    /// <param name="operation">The sort operation to visit</param>
    protected virtual void VisitSort(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a join operation
    /// </summary>
    /// <param name="operation">The join operation to visit</param>
    protected virtual void VisitJoin(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a projection operation
    /// </summary>
    /// <param name="operation">The projection operation to visit</param>
    protected virtual void VisitProjection(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits a concatenation operation
    /// </summary>
    /// <param name="operation">The concatenation operation to visit</param>
    protected virtual void VisitConcatenation(IQueryOperation operation)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Visits an unknown operation type
    /// </summary>
    /// <param name="operation">The unknown operation to visit</param>
    protected virtual void VisitUnknownOperation(IQueryOperation operation)
    {
        // Default implementation does nothing
    }
}

/// <summary>
/// Base class for query plan transformers that can modify query plans
/// </summary>
/// <typeparam name="T">The result type of the transformation</typeparam>
public abstract class QueryPlanTransformerBase<T> : IQueryPlanVisitor<T>
{
    /// <inheritdoc />
    public virtual T Visit(QueryPlan plan)
    {
        if (plan == null)
            return default(T)!;

        var transformedOperations = new List<IQueryOperation>();

        foreach (var operation in plan.Operations)
        {
            var transformedOperation = Visit(operation);
            if (transformedOperation is IQueryOperation op)
            {
                transformedOperations.Add(op);
            }
        }

        var transformedPlan = new QueryPlan(plan.Source, transformedOperations);
        return (T)(object)transformedPlan;
    }

    /// <inheritdoc />
    public virtual T Visit(IQueryOperation operation)
    {
        if (operation == null)
            return default(T)!;

        return operation.OperationType switch
        {
            "Filter" => VisitFilter(operation),
            "Select" => VisitSelect(operation),
            "GroupBy" => VisitGroupBy(operation),
            "Sort" => VisitSort(operation),
            "Join" => VisitJoin(operation),
            "Projection" => VisitProjection(operation),
            "Concatenation" => VisitConcatenation(operation),
            _ => VisitUnknownOperation(operation)
        };
    }

    /// <summary>
    /// Visits a filter operation
    /// </summary>
    /// <param name="operation">The filter operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitFilter(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a select operation
    /// </summary>
    /// <param name="operation">The select operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitSelect(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a group by operation
    /// </summary>
    /// <param name="operation">The group by operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitGroupBy(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a sort operation
    /// </summary>
    /// <param name="operation">The sort operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitSort(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a join operation
    /// </summary>
    /// <param name="operation">The join operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitJoin(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a projection operation
    /// </summary>
    /// <param name="operation">The projection operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitProjection(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits a concatenation operation
    /// </summary>
    /// <param name="operation">The concatenation operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitConcatenation(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }

    /// <summary>
    /// Visits an unknown operation type
    /// </summary>
    /// <param name="operation">The unknown operation to visit</param>
    /// <returns>The transformed result</returns>
    protected virtual T VisitUnknownOperation(IQueryOperation operation)
    {
        return (T)(object)operation; // Default: return unchanged
    }
}

/// <summary>
/// Visitor that collects statistics about a query plan
/// </summary>
public sealed class QueryPlanStatisticsVisitor : QueryPlanVisitorBase
{
    private readonly Dictionary<string, int> _operationCounts = new();
    private int _totalOperations;
    private int _maxDepth;
    private int _currentDepth;

    /// <summary>
    /// Gets the operation counts by type
    /// </summary>
    public IReadOnlyDictionary<string, int> OperationCounts => _operationCounts;

    /// <summary>
    /// Gets the total number of operations
    /// </summary>
    public int TotalOperations => _totalOperations;

    /// <summary>
    /// Gets the maximum depth of the query plan
    /// </summary>
    public int MaxDepth => _maxDepth;

    /// <inheritdoc />
    public override void Visit(IQueryOperation operation)
    {
        if (operation == null)
            return;

        _totalOperations++;
        _currentDepth++;
        _maxDepth = Math.Max(_maxDepth, _currentDepth);

        var operationType = operation.OperationType;
        _operationCounts[operationType] = _operationCounts.GetValueOrDefault(operationType, 0) + 1;

        base.Visit(operation);

        _currentDepth--;
    }

    /// <summary>
    /// Resets the statistics
    /// </summary>
    public void Reset()
    {
        _operationCounts.Clear();
        _totalOperations = 0;
        _maxDepth = 0;
        _currentDepth = 0;
    }

    /// <summary>
    /// Generates a statistics report
    /// </summary>
    /// <returns>A formatted statistics report</returns>
    public string GenerateReport()
    {
        var report = new System.Text.StringBuilder();

        report.AppendLine("Query Plan Statistics:");
        report.AppendLine($"  Total Operations: {TotalOperations}");
        report.AppendLine($"  Max Depth: {MaxDepth}");
        report.AppendLine();

        if (OperationCounts.Count > 0)
        {
            report.AppendLine("Operation Counts:");
            foreach (var (operationType, count) in OperationCounts.OrderByDescending(kvp => kvp.Value))
            {
                report.AppendLine($"  {operationType}: {count}");
            }
        }

        return report.ToString();
    }
}

/// <summary>
/// Visitor that validates a query plan for correctness
/// </summary>
public sealed class QueryPlanValidationVisitor : QueryPlanVisitorBase
{
    private readonly List<string> _errors = new();
    private Schema _currentSchema = null!;

    /// <summary>
    /// Gets the validation errors found
    /// </summary>
    public IReadOnlyList<string> Errors => _errors;

    /// <summary>
    /// Gets a value indicating whether the query plan is valid
    /// </summary>
    public bool IsValid => _errors.Count == 0;

    /// <inheritdoc />
    public override void Visit(QueryPlan plan)
    {
        _errors.Clear();

        if (plan == null)
        {
            _errors.Add("Query plan is null");
            return;
        }

        if (plan.Source == null)
        {
            _errors.Add("Query plan source is null");
            return;
        }

        _currentSchema = plan.Source.Schema;

        base.Visit(plan);
    }

    /// <inheritdoc />
    protected override void VisitFilter(IQueryOperation operation)
    {
        // For filter operations, we'd need to access the condition
        // Since the operation is internal, we can't access it directly
        // This is a limitation of the current design
    }

    /// <inheritdoc />
    protected override void VisitSelect(IQueryOperation operation)
    {
        // Update current schema based on selection
        try
        {
            _currentSchema = operation.TransformSchema(_currentSchema);
        }
        catch (Exception ex)
        {
            _errors.Add($"Select operation schema transformation failed: {ex.Message}");
        }
    }

    /// <inheritdoc />
    protected override void VisitGroupBy(IQueryOperation operation)
    {
        // Update current schema based on grouping
        try
        {
            _currentSchema = operation.TransformSchema(_currentSchema);
        }
        catch (Exception ex)
        {
            _errors.Add($"GroupBy operation schema transformation failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Generates a validation report
    /// </summary>
    /// <returns>A formatted validation report</returns>
    public string GenerateReport()
    {
        if (IsValid)
        {
            return "Query plan is valid.";
        }

        var report = new System.Text.StringBuilder();
        report.AppendLine($"Query plan validation failed with {Errors.Count} error(s):");

        for (int i = 0; i < Errors.Count; i++)
        {
            report.AppendLine($"  {i + 1}. {Errors[i]}");
        }

        return report.ToString();
    }
}