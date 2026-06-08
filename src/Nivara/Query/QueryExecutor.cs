using Nivara.Exceptions;

namespace Nivara.Query;

/// <summary>
/// Executes query plans by applying operations to data sources
/// </summary>
sealed class QueryExecutor
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
            // Validate the plan before execution
            if (!ValidatePlan(plan))
            {
                var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
                throw new QueryExecutionException($"Query plan validation failed. {diagnosticInfo}");
            }

            // Execute the data source to get initial columns
            IReadOnlyDictionary<string, IColumn> currentColumns;

            try
            {
                currentColumns = plan.Source.Execute();
            }
            catch (Exception ex)
            {
                var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
                throw new QueryExecutionException($"Data source execution failed: {ex.Message}. {diagnosticInfo}", ex);
            }

            if (currentColumns == null)
            {
                var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
                throw new QueryExecutionException($"Data source returned null columns. {diagnosticInfo}");
            }

            // Apply each operation in sequence
            for (int i = 0; i < plan.Operations.Count; i++)
            {
                var operation = plan.Operations[i];

                try
                {
                    currentColumns = operation.Execute(currentColumns);

                    if (currentColumns == null)
                    {
                        var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
                        throw new QueryExecutionException($"Operation '{operation.OperationType}' at position {i + 1} returned null columns. {diagnosticInfo}");
                    }
                }
                catch (Exception ex) when (ex is not QueryExecutionException)
                {
                    var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
                    throw new QueryExecutionException(
                        $"Operation '{operation.OperationType}' at position {i + 1} failed: {ex.Message}. {diagnosticInfo}",
                        operation.OperationType,
                        ex);
                }
            }

            // Create the final frame from the result columns
            if (currentColumns.Count == 0)
            {
                var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);
                throw new QueryExecutionException($"Query execution resulted in no columns. {diagnosticInfo}");
            }

            var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(namedColumns);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            var diagnosticInfo = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, ex);
            throw new QueryExecutionException($"Query execution failed: {ex.Message}. {diagnosticInfo}", ex);
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
    public long EstimateExecutionCost(QueryPlan plan)
    {
        if (plan == null)
            return long.MaxValue;

        try
        {
            long cost = 0;

            cost += plan.Source.IsLazy ? 10 : 5;

            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    OperationType.Filter => 5,
                    OperationType.Select => 3,
                    OperationType.GroupBy => 20,
                    _ => 10
                };
            }

            return cost;
        }
        catch
        {
            return long.MaxValue;
        }
    }
}
