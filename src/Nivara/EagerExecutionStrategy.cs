using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Implements eager execution strategy that executes operations immediately without building query plans.
/// This strategy provides immediate results but may miss optimization opportunities.
/// </summary>
internal sealed class EagerExecutionStrategy : IExecutionStrategy
{
    readonly QueryExecutor executor;

    /// <summary>
    /// Initializes a new instance of EagerExecutionStrategy
    /// </summary>
    public EagerExecutionStrategy()
    {
        executor = new QueryExecutor();
    }

    /// <inheritdoc />
    public NivaraFrame Execute(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        try
        {
            // Check for cancellation before starting
            context.CancellationToken.ThrowIfCancellationRequested();

            // Report progress
            ReportProgress(context, "Starting eager execution", 0, plan.Operations.Count + 1);

            // Execute the data source immediately
            var currentColumns = plan.Source.Execute();
            ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

            // Apply each operation immediately without building intermediate plans
            for (int i = 0; i < plan.Operations.Count; i++)
            {
                var operation = plan.Operations[i];

                // Check for cancellation before each operation
                context.CancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // Execute the operation immediately
                    currentColumns = operation.Execute(currentColumns);

                    // Report progress
                    ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
                }
                catch (Exception ex) when (ex is not QueryExecutionException)
                {
                    throw new QueryExecutionException(
                        $"Eager execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                        operation.OperationType,
                        ex);
                }
            }

            // Create the final frame from the result columns
            if (currentColumns.Count == 0)
            {
                throw new QueryExecutionException("Eager execution resulted in no columns");
            }

            var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(namedColumns);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Eager execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public async Task<NivaraFrame> ExecuteAsync(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        try
        {
            // For eager execution, we execute each operation as a separate async task
            // This allows for better cancellation support and progress reporting

            // Check for cancellation before starting
            context.CancellationToken.ThrowIfCancellationRequested();

            // Report progress
            ReportProgress(context, "Starting async eager execution", 0, plan.Operations.Count + 1);

            // Execute the data source
            var currentColumns = await Task.Run(() => plan.Source.Execute(), context.CancellationToken);
            ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

            // Apply each operation asynchronously
            for (int i = 0; i < plan.Operations.Count; i++)
            {
                var operation = plan.Operations[i];

                // Check for cancellation before each operation
                context.CancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // Execute the operation asynchronously
                    currentColumns = await Task.Run(() => operation.Execute(currentColumns), context.CancellationToken);

                    // Report progress
                    ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
                }
                catch (Exception ex) when (ex is not QueryExecutionException)
                {
                    throw new QueryExecutionException(
                        $"Async eager execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                        operation.OperationType,
                        ex);
                }
            }

            // Create the final frame from the result columns
            if (currentColumns.Count == 0)
            {
                throw new QueryExecutionException("Async eager execution resulted in no columns");
            }

            var namedColumns = currentColumns.Select(kvp => (kvp.Key, kvp.Value));
            return new NivaraFrame(namedColumns);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Async eager execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public bool ValidatePlan(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            // For eager execution, we need to validate that all operations can be executed immediately
            // This is the same validation as lazy execution since operations are the same
            return executor.ValidatePlan(plan);
        }
        catch
        {
            return false;
        }
    }

    /// <inheritdoc />
    public long EstimateExecutionCost(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null || context == null)
            return long.MaxValue;

        try
        {
            // Base cost for eager execution (higher than lazy due to immediate materialization)
            long cost = 200;

            // Add cost for data source (higher for eager since we materialize immediately)
            cost += plan.Source.IsLazy ? 200 : 150; // Lazy sources are more expensive in eager execution

            // Add cost for each operation (higher than lazy since no optimization)
            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 300,      // Higher than lazy due to no pushdown optimization
                    "Select" => 150,      // Slightly higher
                    "Sort" => 1200,       // Higher due to immediate materialization
                    "GroupBy" => 1800,    // Higher due to no optimization
                    "Join" => 2500,       // Much higher due to immediate materialization
                    "Concatenation" => 400, // Higher due to immediate copying
                    _ => 600              // Default cost for unknown operations
                };
            }

            // Eager execution has overhead for immediate materialization
            var materializationOverhead = plan.Operations.Count * 100;
            cost += materializationOverhead;

            return Math.Max(cost, 200); // Minimum cost
        }
        catch
        {
            return long.MaxValue;
        }
    }

    /// <summary>
    /// Reports progress for the execution
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <param name="operationName">The name of the current operation</param>
    /// <param name="completedWork">The amount of work completed</param>
    /// <param name="totalWork">The total amount of work</param>
    private static void ReportProgress(ExecutionContext context, string operationName, long completedWork, long totalWork)
    {
        if (context.Progress != null)
        {
            var progress = new ExecutionProgress(operationName, completedWork, totalWork);
            context.Progress.Report(progress);
        }
    }
}