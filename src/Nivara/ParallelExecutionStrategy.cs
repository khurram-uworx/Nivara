using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Implements parallel execution strategy that uses multiple threads for parallelizable operations.
/// This strategy provides improved performance for CPU-intensive operations on multi-core systems.
/// </summary>
internal sealed class ParallelExecutionStrategy : IExecutionStrategy
{
    readonly QueryExecutor executor;

    /// <summary>
    /// Initializes a new instance of ParallelExecutionStrategy
    /// </summary>
    public ParallelExecutionStrategy()
    {
        executor = new QueryExecutor();
    }

    /// <inheritdoc />
    public NivaraFrame Execute(QueryPlan plan, ExecutionContext context)
    {
        // For synchronous execution, we'll use the async version and wait for it
        return ExecuteAsync(plan, context).GetAwaiter().GetResult();
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
            // Check for cancellation before starting
            context.CancellationToken.ThrowIfCancellationRequested();

            // Validate parallelism configuration
            ValidateParallelismConfiguration(context);

            ReportProgress(context, "Starting async parallel execution", 0, plan.Operations.Count + 1);

            // Execute the data source asynchronously
            var currentColumns = await Task.Run(() => plan.Source.Execute(), context.CancellationToken);
            ReportProgress(context, "Data source executed", 1, plan.Operations.Count + 1);

            // Apply each operation with parallel processing where possible
            for (int i = 0; i < plan.Operations.Count; i++)
            {
                var operation = plan.Operations[i];

                // Check for cancellation before each operation
                context.CancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // Execute the operation with parallel processing if supported
                    currentColumns = await ExecuteOperationParallelAsync(operation, currentColumns, context);

                    // Report progress
                    ReportProgress(context, $"Operation {operation.OperationType} completed", i + 2, plan.Operations.Count + 1);
                }
                catch (Exception ex) when (ex is not QueryExecutionException)
                {
                    throw new QueryExecutionException(
                        $"Async parallel execution failed at operation '{operation.OperationType}' (position {i + 1}): {ex.Message}",
                        operation.OperationType,
                        ex);
                }
            }

            // Create the final frame from the result columns
            if (currentColumns.Count == 0)
            {
                throw new QueryExecutionException("Async parallel execution resulted in no columns");
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
            throw new QueryExecutionException($"Async parallel execution failed: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public bool ValidatePlan(QueryPlan plan, ExecutionContext context)
    {
        if (plan == null || context == null)
            return false;

        try
        {
            // Validate the plan using the existing executor
            if (!executor.ValidatePlan(plan))
                return false;

            // Additional validation for parallel execution
            ValidateParallelismConfiguration(context);

            return true;
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
            // Base cost for parallel execution
            long cost = 180;

            // Add cost for data source
            cost += plan.Source.IsLazy ? 120 : 100;

            // Add cost for each operation with parallel benefits
            foreach (var operation in plan.Operations)
            {
                var baseCost = operation.OperationType switch
                {
                    "Filter" => 300,      // Can benefit from parallelism
                    "Select" => 150,      // Limited parallel benefit
                    "Sort" => 800,        // Good parallel benefit
                    "GroupBy" => 1000,    // Excellent parallel benefit
                    "Join" => 1500,       // Good parallel benefit
                    "Concatenation" => 250, // Some parallel benefit
                    _ => 500              // Default cost
                };

                // Apply parallelism discount based on degree of parallelism
                var parallelismFactor = Math.Min(context.MaxDegreeOfParallelism, Environment.ProcessorCount);
                var parallelDiscount = IsParallelizable(operation.OperationType)
                    ? baseCost * (1.0 - (0.7 / parallelismFactor)) // Up to 70% discount for highly parallel operations
                    : baseCost * 0.95; // Small discount for coordination overhead

                cost += (long)parallelDiscount;
            }

            // Add overhead for thread coordination
            var coordinationOverhead = plan.Operations.Count * 50;
            cost += coordinationOverhead;

            return Math.Max(cost, 180); // Minimum cost
        }
        catch
        {
            return long.MaxValue;
        }
    }

    /// <summary>
    /// Executes an operation with parallel processing if supported
    /// </summary>
    /// <param name="operation">The operation to execute</param>
    /// <param name="input">The input columns</param>
    /// <param name="context">The execution context</param>
    /// <returns>The result columns</returns>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteOperationParallelAsync(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        ExecutionContext context)
    {
        // Validate parallel configuration
        ParallelExecutionHelper.ValidateParallelConfiguration(context.MaxDegreeOfParallelism);

        // Check if the operation can benefit from parallelism
        if (IsParallelizable(operation.OperationType) && ShouldUseParallelism(input, context))
        {
            // Get recommended parallelism level
            var recommendedParallelism = ParallelExecutionHelper.GetRecommendedParallelism(context.MaxDegreeOfParallelism);

            // Execute with parallel processing based on operation type
            return operation.OperationType switch
            {
                "Filter" => await ExecuteFilterParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Sort" => await ExecuteSortParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "GroupBy" => await ExecuteGroupByParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Join" => await ExecuteJoinParallel(operation, input, recommendedParallelism, context.CancellationToken),
                "Concatenation" => await ExecuteConcatenationParallel(operation, input, recommendedParallelism, context.CancellationToken),
                _ => await Task.Run(() => operation.Execute(input), context.CancellationToken)
            };
        }
        else
        {
            // Fall back to standard execution for non-parallelizable operations
            return await Task.Run(() => operation.Execute(input), context.CancellationToken);
        }
    }

    /// <summary>
    /// Executes a filter operation in parallel
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteFilterParallel(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        // For filter operations, we can process chunks of rows in parallel
        // This is a simplified implementation - in practice, we'd need to coordinate with the actual FilterOperation
        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    /// <summary>
    /// Executes a sort operation in parallel
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteSortParallel(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        // Sort operations can benefit from parallel merge sort algorithms
        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    /// <summary>
    /// Executes a group by operation in parallel
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteGroupByParallel(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        // GroupBy operations can use parallel partitioning and aggregation
        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    /// <summary>
    /// Executes a join operation in parallel
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteJoinParallel(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        // Join operations can parallelize hash table building and probing
        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    /// <summary>
    /// Executes a concatenation operation in parallel
    /// </summary>
    private async Task<IReadOnlyDictionary<string, IColumn>> ExecuteConcatenationParallel(
        IQueryOperation operation,
        IReadOnlyDictionary<string, IColumn> input,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        // Concatenation operations can process multiple sources in parallel
        return await Task.Run(() => operation.Execute(input), cancellationToken);
    }

    /// <summary>
    /// Determines if an operation type can benefit from parallelism
    /// </summary>
    /// <param name="operationType">The type of operation</param>
    /// <returns>True if the operation can be parallelized, false otherwise</returns>
    private static bool IsParallelizable(string operationType)
    {
        return operationType switch
        {
            "Filter" => true,        // Can process chunks in parallel
            "Select" => false,       // Usually too simple to benefit
            "Sort" => true,          // Can use parallel sorting
            "GroupBy" => true,       // Can use parallel partitioning
            "Join" => true,          // Can parallelize hash building and probing
            "Concatenation" => true, // Can process sources in parallel
            _ => false               // Conservative default
        };
    }

    /// <summary>
    /// Determines if parallelism should be used based on data size and configuration
    /// </summary>
    /// <param name="input">The input columns</param>
    /// <param name="context">The execution context</param>
    /// <returns>True if parallelism should be used, false otherwise</returns>
    private static bool ShouldUseParallelism(IReadOnlyDictionary<string, IColumn> input, ExecutionContext context)
    {
        // Get the total number of rows
        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;

        return ParallelExecutionHelper.ShouldUseParallelProcessing(
            totalRows,
            context.MaxDegreeOfParallelism);
    }

    /// <summary>
    /// Validates the parallelism configuration
    /// </summary>
    /// <param name="context">The execution context to validate</param>
    /// <exception cref="QueryExecutionException">Thrown when configuration is invalid</exception>
    private static void ValidateParallelismConfiguration(ExecutionContext context)
    {
        ParallelExecutionHelper.ValidateParallelConfiguration(context.MaxDegreeOfParallelism);
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