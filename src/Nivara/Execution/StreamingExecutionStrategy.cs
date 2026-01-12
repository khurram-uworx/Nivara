using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Execution;

/// <summary>
/// Implements streaming execution strategy that processes data in chunks for large datasets.
/// This strategy manages memory usage by processing data incrementally.
/// </summary>
internal sealed class StreamingExecutionStrategy : IExecutionStrategy
{
    readonly QueryExecutor executor;
    const int DefaultChunkSize = 10000; // Default number of rows per chunk

    /// <summary>
    /// Initializes a new instance of StreamingExecutionStrategy
    /// </summary>
    public StreamingExecutionStrategy()
    {
        executor = new QueryExecutor();
    }

    /// <inheritdoc />
    public NivaraFrame Execute(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        try
        {
            // Check for cancellation before starting
            context.CancellationToken.ThrowIfCancellationRequested();

            // For now, streaming execution falls back to regular execution
            // In a full implementation, this would process data in chunks
            // based on the memory budget and estimated data size

            // Calculate chunk size based on memory budget
            var chunkSize = CalculateChunkSize(context.MemoryBudget);

            ReportProgress(context, "Starting streaming execution", 0, 1);

            // Check if the plan is suitable for streaming
            if (!IsSuitableForStreaming(plan))
            {
                // Fall back to lazy execution for plans that can't be streamed
                var lazyStrategy = new LazyExecutionStrategy();
                return lazyStrategy.Execute(plan, context);
            }

            // For operations that can be streamed, we would implement chunk-based processing here
            // For now, we use the existing executor with progress reporting
            var result = executor.Execute(plan);

            ReportProgress(context, "Streaming execution completed", 1, 1);
            return result;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Streaming execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public async Task<NivaraFrame> ExecuteAsync(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));
        if (context == null)
            throw new ArgumentNullException(nameof(context));

        try
        {
            // Check for cancellation before starting
            context.CancellationToken.ThrowIfCancellationRequested();

            // Calculate chunk size based on memory budget
            var chunkSize = CalculateChunkSize(context.MemoryBudget);

            ReportProgress(context, "Starting async streaming execution", 0, 1);

            // Check if the plan is suitable for streaming
            if (!IsSuitableForStreaming(plan))
            {
                // Fall back to lazy execution for plans that can't be streamed
                var lazyStrategy = new LazyExecutionStrategy();
                return await lazyStrategy.ExecuteAsync(plan, context);
            }

            // Execute with streaming semantics on a background thread
            var result = await Task.Run(() => Execute(plan, context), context.CancellationToken);

            ReportProgress(context, "Async streaming execution completed", 1, 1);
            return result;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Async streaming execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public bool ValidatePlan(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            // Validate the plan using the existing executor
            if (!executor.ValidatePlan(plan))
                return false;

            // Additional validation for streaming execution
            // Check if memory budget is reasonable
            if (context.MemoryBudget <= 0)
                return false;

            // Check if operations are streamable (for future implementation)
            return IsSuitableForStreaming(plan);
        }
        catch
        {
            return false;
        }
    }

    /// <inheritdoc />
    public long EstimateExecutionCost(QueryPlan plan, NivaraExecutionContext context)
    {
        if (plan == null || context == null)
            return long.MaxValue;

        try
        {
            // Base cost for streaming execution (moderate, between lazy and eager)
            long cost = 150;

            // Add cost for data source
            cost += plan.Source.IsLazy ? 100 : 120; // Streaming works well with both

            // Add cost for each operation
            foreach (var operation in plan.Operations)
            {
                cost += operation.OperationType switch
                {
                    "Filter" => 250,      // Good for streaming
                    "Select" => 120,      // Excellent for streaming
                    "Sort" => 2000,       // Poor for streaming (needs all data)
                    "GroupBy" => 2500,    // Poor for streaming (needs all data)
                    "Join" => 3000,       // Very poor for streaming
                    "Concatenation" => 200, // Good for streaming
                    _ => 400              // Default cost
                };
            }

            // Apply streaming benefits for suitable operations
            if (IsSuitableForStreaming(plan))
            {
                var streamingDiscount = Math.Min(cost * 0.15, 800);
                cost -= (long)streamingDiscount;
            }
            else
            {
                // Add penalty for non-streamable plans
                cost += 1000;
            }

            return Math.Max(cost, 150); // Minimum cost
        }
        catch
        {
            return long.MaxValue;
        }
    }

    /// <summary>
    /// Determines if a query plan is suitable for streaming execution
    /// </summary>
    /// <param name="plan">The query plan to analyze</param>
    /// <returns>True if the plan can be executed with streaming, false otherwise</returns>
    private static bool IsSuitableForStreaming(QueryPlan plan)
    {
        // Operations that work well with streaming
        var streamableOperations = new HashSet<string>
        {
            "Filter",
            "Select",
            "Concatenation"
        };

        // Operations that require all data and are not suitable for streaming
        var nonStreamableOperations = new HashSet<string>
        {
            "Sort",
            "GroupBy",
            "Join"
        };

        // Check if any operations are explicitly non-streamable
        foreach (var operation in plan.Operations)
        {
            if (nonStreamableOperations.Contains(operation.OperationType))
            {
                return false;
            }
        }

        // If all operations are streamable or unknown (which we assume can be streamed), return true
        return true;
    }

    /// <summary>
    /// Calculates the appropriate chunk size based on memory budget
    /// </summary>
    /// <param name="memoryBudget">The available memory budget in bytes</param>
    /// <returns>The number of rows to process per chunk</returns>
    private static int CalculateChunkSize(long memoryBudget)
    {
        // Estimate memory per row (this is a rough estimate)
        const long estimatedBytesPerRow = 100; // Assume 100 bytes per row on average

        // Use 10% of memory budget for chunk processing
        var chunkMemory = memoryBudget / 10;
        var calculatedChunkSize = (int)(chunkMemory / estimatedBytesPerRow);

        // Ensure chunk size is within reasonable bounds
        return Math.Max(1000, Math.Min(calculatedChunkSize, 100000));
    }

    /// <summary>
    /// Reports progress for the execution
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <param name="operationName">The name of the current operation</param>
    /// <param name="completedWork">The amount of work completed</param>
    /// <param name="totalWork">The total amount of work</param>
    private static void ReportProgress(NivaraExecutionContext context, string operationName, long completedWork, long totalWork)
    {
        if (context.Progress != null)
        {
            var progress = new ExecutionProgress(operationName, completedWork, totalWork);
            context.Progress.Report(progress);
        }
    }
}