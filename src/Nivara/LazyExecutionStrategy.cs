using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Implements lazy execution strategy that builds query plans and executes on demand.
/// This is the default strategy that provides optimal memory usage and allows for query optimization.
/// </summary>
internal sealed class LazyExecutionStrategy : IExecutionStrategy
{
    readonly QueryExecutor executor;

    /// <summary>
    /// Initializes a new instance of LazyExecutionStrategy
    /// </summary>
    public LazyExecutionStrategy()
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
            ReportProgress(context, "Starting lazy execution", 0, 1);

            // Execute the plan using the existing QueryExecutor
            var result = executor.Execute(plan);

            // Report completion
            ReportProgress(context, "Lazy execution completed", 1, 1);

            return result;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Lazy execution failed: {ex.Message}", ex);
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
            // For lazy execution, we can run synchronously on a background thread
            // This maintains the lazy semantics while providing async interface
            return await Task.Run(() => Execute(plan, context), context.CancellationToken);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Async lazy execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public bool ValidatePlan(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            // Use the existing QueryExecutor validation
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
            // Base cost for lazy execution (lower than eager due to optimization opportunities)
            long cost = 100;

            // Add cost for data source
            cost += plan.Source.IsLazy ? 50 : 100; // Lazy sources are cheaper in lazy execution

            // Add cost for each operation
            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 200,      // Relatively cheap
                    "Select" => 100,      // Very cheap
                    "Sort" => 1000,       // Expensive
                    "GroupBy" => 1500,    // Most expensive
                    "Join" => 2000,       // Very expensive
                    "Concatenation" => 300, // Moderate cost
                    _ => 500              // Default cost for unknown operations
                };
            }

            // Lazy execution benefits from potential optimizations
            var optimizationDiscount = Math.Min(cost * 0.2, 1000);
            cost -= (long)optimizationDiscount;

            return Math.Max(cost, 100); // Minimum cost
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