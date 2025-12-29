using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Executes query plans by applying operations to data sources
/// </summary>
internal sealed class QueryExecutor
{
    /// <summary>
    /// Executes a query plan and returns the materialized result
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <returns>A materialized NivaraFrame with the query results</returns>
    /// <exception cref="QueryExecutionException">Thrown when execution fails</exception>
    public NivaraFrame Execute(QueryPlan plan)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        try
        {
            // Execute the data source to get initial columns
            var currentColumns = plan.Source.Execute();

            if (currentColumns == null)
                throw new QueryExecutionException("Data source returned null columns");

            // Apply each operation in sequence
            foreach (var operation in plan.Operations)
            {
                try
                {
                    currentColumns = operation.Execute(currentColumns);

                    if (currentColumns == null)
                        throw new QueryExecutionException($"Operation '{operation.OperationType}' returned null columns");
                }
                catch (Exception ex) when (ex is not QueryExecutionException)
                {
                    throw new QueryExecutionException($"Operation '{operation.OperationType}' failed: {ex.Message}", ex);
                }
            }

            // Create the final frame from the result columns
            if (currentColumns.Count == 0)
            {
                throw new QueryExecutionException("Query execution resulted in no columns");
            }

            var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(namedColumns);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Query execution failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Executes a query plan with optimization
    /// </summary>
    /// <param name="plan">The query plan to execute</param>
    /// <param name="optimizer">The query optimizer to use (optional)</param>
    /// <returns>A materialized NivaraFrame with the query results</returns>
    /// <exception cref="QueryExecutionException">Thrown when execution fails</exception>
    public NivaraFrame ExecuteOptimized(QueryPlan plan, QueryOptimizer? optimizer = null)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        try
        {
            // Apply optimization if optimizer is provided
            var optimizedPlan = optimizer?.Optimize(plan) ?? plan;

            // Execute the optimized plan
            return Execute(optimizedPlan);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Optimized query execution failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Validates that a query plan can be executed without actually executing it
    /// </summary>
    /// <param name="plan">The query plan to validate</param>
    /// <returns>True if the plan is valid, false otherwise</returns>
    public bool ValidatePlan(QueryPlan plan)
    {
        if (plan == null)
            return false;

        try
        {
            // Validate that the result schema can be computed
            var resultSchema = plan.ResultSchema;

            // Validate that each operation can transform the schema correctly
            var currentSchema = plan.Source.Schema;

            foreach (var operation in plan.Operations)
            {
                currentSchema = operation.TransformSchema(currentSchema);
            }

            // Ensure the computed schema matches the plan's result schema
            return currentSchema.Equals(resultSchema);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Estimates the execution cost of a query plan
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>An estimated execution cost (higher values indicate more expensive operations)</returns>
    public int EstimateExecutionCost(QueryPlan plan)
    {
        if (plan == null)
            return int.MaxValue;

        try
        {
            int cost = 0;

            // Base cost for data source
            cost += plan.Source.IsLazy ? 10 : 5; // Lazy sources have higher initial cost

            // Add cost for each operation
            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 5,    // Relatively cheap
                    "Select" => 3,    // Very cheap
                    "GroupBy" => 20,  // Expensive due to sorting/grouping
                    _ => 10           // Default cost for unknown operations
                };
            }

            return cost;
        }
        catch
        {
            return int.MaxValue; // Return maximum cost if estimation fails
        }
    }
}