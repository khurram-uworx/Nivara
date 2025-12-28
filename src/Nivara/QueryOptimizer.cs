using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Optimizes query plans by applying various optimization techniques
/// </summary>
internal sealed class QueryOptimizer
{
    /// <summary>
    /// Optimizes a query plan by applying various optimization passes
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
            var optimizedPlan = plan;

            // Apply optimization passes in order
            optimizedPlan = ApplyPredicatePushdown(optimizedPlan);
            optimizedPlan = ApplyOperationFusion(optimizedPlan);
            optimizedPlan = ApplyColumnElimination(optimizedPlan);
            optimizedPlan = ApplyOperationReordering(optimizedPlan);

            return optimizedPlan;
        }
        catch (Exception)
        {
            // If optimization fails, return the original plan
            // Optimization should never break a valid query
            return plan;
        }
    }

    /// <summary>
    /// Applies predicate pushdown optimization to move filter operations closer to data sources
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyPredicatePushdown(QueryPlan plan)
    {
        // For now, this is a placeholder implementation
        // A full implementation would analyze filter operations and move them earlier in the pipeline
        // when it's safe to do so (i.e., when they don't depend on columns created by later operations)
        
        var operations = plan.Operations.ToList();
        var optimizedOperations = new List<IQueryOperation>();
        var filterOperations = new List<IQueryOperation>();

        // Separate filter operations from other operations
        foreach (var operation in operations)
        {
            if (operation.OperationType == "Filter")
            {
                filterOperations.Add(operation);
            }
            else
            {
                // Add any accumulated filters before this operation
                optimizedOperations.AddRange(filterOperations);
                filterOperations.Clear();
                optimizedOperations.Add(operation);
            }
        }

        // Add any remaining filters at the end
        optimizedOperations.AddRange(filterOperations);

        return new QueryPlan(plan.Source, optimizedOperations);
    }

    /// <summary>
    /// Applies operation fusion to combine compatible operations
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyOperationFusion(QueryPlan plan)
    {
        // For now, this is a placeholder implementation
        // A full implementation would identify adjacent operations that can be fused
        // For example, multiple Select operations could be combined into one
        
        var operations = plan.Operations.ToList();
        var fusedOperations = new List<IQueryOperation>();

        for (int i = 0; i < operations.Count; i++)
        {
            var currentOp = operations[i];
            
            // Check if we can fuse with the next operation
            if (i + 1 < operations.Count && CanFuseOperations(currentOp, operations[i + 1]))
            {
                // Skip the current operation and let the fusion happen in the next iteration
                // This is a simplified approach - a full implementation would actually create fused operations
                continue;
            }
            
            fusedOperations.Add(currentOp);
        }

        return new QueryPlan(plan.Source, fusedOperations);
    }

    /// <summary>
    /// Applies column elimination to remove unused columns early in the pipeline
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyColumnElimination(QueryPlan plan)
    {
        // For now, this is a placeholder implementation
        // A full implementation would analyze which columns are actually used
        // and add Select operations to eliminate unused columns early
        
        return plan; // No optimization applied yet
    }

    /// <summary>
    /// Applies operation reordering when it improves performance without changing semantics
    /// </summary>
    /// <param name="plan">The query plan to optimize</param>
    /// <returns>An optimized query plan</returns>
    private static QueryPlan ApplyOperationReordering(QueryPlan plan)
    {
        // For now, this is a placeholder implementation
        // A full implementation would reorder operations when safe to do so
        // For example, moving cheap operations before expensive ones
        
        return plan; // No optimization applied yet
    }

    /// <summary>
    /// Determines if two operations can be fused together
    /// </summary>
    /// <param name="first">The first operation</param>
    /// <param name="second">The second operation</param>
    /// <returns>True if the operations can be fused, false otherwise</returns>
    private static bool CanFuseOperations(IQueryOperation first, IQueryOperation second)
    {
        // Simple fusion rules - can be expanded
        return first.OperationType == "Select" && second.OperationType == "Select";
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