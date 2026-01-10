namespace Nivara;

/// <summary>
/// Optimizes query plans by applying various optimization techniques
/// </summary>
internal sealed class QueryOptimizer
{
    private readonly OptimizationEngine _engine;

    /// <summary>
    /// Initializes a new instance of QueryOptimizer with default optimization rules
    /// </summary>
    public QueryOptimizer()
    {
        _engine = OptimizationEngine.CreateDefault();
    }

    /// <summary>
    /// Initializes a new instance of QueryOptimizer with custom optimization rules
    /// </summary>
    /// <param name="engine">The optimization engine to use</param>
    public QueryOptimizer(OptimizationEngine engine)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
    }

    /// <summary>
    /// Gets the optimization engine used by this optimizer
    /// </summary>
    public OptimizationEngine Engine => _engine;

    /// <summary>
    /// Optimizes a query plan by applying optimization rules
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    /// <exception cref="ArgumentNullException">Thrown when plan is null</exception>
    public QueryPlan Optimize(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        try
        {
            var result = _engine.Optimize(plan);
            return result.OptimizedPlan;
        }
        catch (Exception)
        {
            // If optimization fails, return the original plan
            // Optimization should never break a valid query
            return plan;
        }
    }

    /// <summary>
    /// Optimizes a query plan and returns detailed optimization results
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>The optimization result with statistics</returns>
    /// <exception cref="ArgumentNullException">Thrown when plan is null</exception>
    public OptimizationResult OptimizeWithStatistics(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        try
        {
            return _engine.Optimize(plan);
        }
        catch (Exception)
        {
            // If optimization fails, return a result with the original plan
            return new OptimizationResult
            {
                OriginalPlan = plan,
                OptimizedPlan = plan,
                Statistics = new List<OptimizationStatistics>(),
                TotalOptimizationTime = TimeSpan.Zero
            };
        }
    }

    /// <summary>
    /// Analyzes a query plan and provides optimization suggestions
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>A list of optimization suggestions</returns>
    public static IReadOnlyList<string> AnalyzeOptimizationOpportunities(QueryPlan plan)
    {
        if (plan == null)
            return Array.Empty<string>();

        var suggestions = new List<string>();

        // Check for multiple filter operations
        var filterCount = plan.Operations.Count(op => op.OperationType == "Filter");
        if (filterCount > 1)
        {
            suggestions.Add($"Found {filterCount} filter operations - consider combining them for better performance");
        }

        // Check for multiple select operations
        var selectCount = plan.Operations.Count(op => op.OperationType == "Select");
        if (selectCount > 1)
        {
            suggestions.Add($"Found {selectCount} select operations - consider combining projections");
        }

        // Check for predicate pushdown opportunities
        if (plan.Source.IsLazy && filterCount > 0)
        {
            suggestions.Add("Filter operations on lazy source detected - predicate pushdown optimization available");
        }

        // Check for column elimination opportunities
        var sourceColumnCount = plan.Source.Schema.ColumnNames.Count;
        var resultColumnCount = plan.ResultSchema.ColumnNames.Count;

        if (resultColumnCount < sourceColumnCount && selectCount == 0)
        {
            suggestions.Add($"Query uses {resultColumnCount} of {sourceColumnCount} columns - consider adding explicit column selection");
        }

        return suggestions;
    }
}