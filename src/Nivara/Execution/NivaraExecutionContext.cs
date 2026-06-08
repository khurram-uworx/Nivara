using Nivara.Diagnostics;

namespace Nivara.Execution;

/// <summary>
/// Provides execution context and configuration for query operations
/// </summary>
public sealed class NivaraExecutionContext
{
    /// <summary>
    /// Initializes a new instance of ExecutionContext with default settings
    /// </summary>
    public NivaraExecutionContext()
    {
        Strategy = ExecutionStrategy.Lazy;
        MaxDegreeOfParallelism = Environment.ProcessorCount;
        MemoryBudget = 1024 * 1024 * 1024; // 1GB default
        CancellationToken = CancellationToken.None;
    }

    /// <summary>
    /// Initializes a new instance of ExecutionContext with specified strategy
    /// </summary>
    /// <param name="strategy">The execution strategy to use</param>
    public NivaraExecutionContext(ExecutionStrategy strategy) : this()
    {
        Strategy = strategy;
    }

    /// <summary>
    /// Gets or sets the execution strategy
    /// </summary>
    public ExecutionStrategy Strategy { get; set; }

    /// <summary>
    /// Gets or sets the maximum degree of parallelism for parallel execution
    /// </summary>
    public int MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Gets or sets the memory budget in bytes for operations
    /// </summary>
    public long MemoryBudget { get; set; }

    /// <summary>
    /// Gets or sets the cancellation token for operation cancellation
    /// </summary>
    public CancellationToken CancellationToken { get; set; }

    /// <summary>
    /// Gets or sets the progress reporter for long-running operations
    /// </summary>
    public IProgress<ExecutionProgress>? Progress { get; set; }

    /// <summary>
    /// Gets or sets the execution diagnostics instance for tracking performance metrics
    /// </summary>
    public ExecutionDiagnostics? ExecutionDiagnostics { get; set; }

    /// <summary>
    /// Creates a copy of this execution context
    /// </summary>
    /// <returns>A new ExecutionContext with the same settings</returns>
    public NivaraExecutionContext Clone()
        => new NivaraExecutionContext
        {
            Strategy = Strategy,
            MaxDegreeOfParallelism = MaxDegreeOfParallelism,
            MemoryBudget = MemoryBudget,
            CancellationToken = CancellationToken,
            Progress = Progress,
            ExecutionDiagnostics = ExecutionDiagnostics
        };

    /// <summary>
    /// Creates an execution context with the specified strategy
    /// </summary>
    /// <param name="strategy">The execution strategy</param>
    /// <returns>A new ExecutionContext with the specified strategy</returns>
    public static NivaraExecutionContext WithStrategy(ExecutionStrategy strategy)
        => new NivaraExecutionContext(strategy);

    /// <summary>
    /// Creates an execution context for parallel execution with specified degree of parallelism
    /// </summary>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <returns>A new ExecutionContext configured for parallel execution</returns>
    public static NivaraExecutionContext WithParallelism(int maxDegreeOfParallelism)
        => new NivaraExecutionContext(ExecutionStrategy.Parallel)
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism
        };

    /// <summary>
    /// Creates an execution context with the specified memory budget
    /// </summary>
    /// <param name="memoryBudgetBytes">The memory budget in bytes</param>
    /// <returns>A new ExecutionContext with the specified memory budget</returns>
    public static NivaraExecutionContext WithMemoryBudget(long memoryBudgetBytes)
        => new NivaraExecutionContext
        {
            MemoryBudget = memoryBudgetBytes
        };

    /// <summary>
    /// Creates an execution context with the specified cancellation token
    /// </summary>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>A new ExecutionContext with the specified cancellation token</returns>
    public static NivaraExecutionContext WithCancellation(CancellationToken cancellationToken)
        => new NivaraExecutionContext
        {
            CancellationToken = cancellationToken
        };

    /// <summary>
    /// Returns a string representation of the execution context
    /// </summary>
    /// <returns>A formatted string describing the execution context</returns>
    public override string ToString()
        => $"ExecutionContext {{ Strategy: {Strategy}, MaxDegreeOfParallelism: {MaxDegreeOfParallelism}, MemoryBudget: {MemoryBudget:N0} bytes, Diagnostics: {(ExecutionDiagnostics != null ? "Set" : "None")} }}";
}

/// <summary>
/// Represents progress information for long-running operations
/// </summary>
public sealed class ExecutionProgress
{
    /// <summary>
    /// Initializes a new instance of ExecutionProgress
    /// </summary>
    /// <param name="operationName">The name of the current operation</param>
    /// <param name="completedWork">The amount of work completed</param>
    /// <param name="totalWork">The total amount of work</param>
    public ExecutionProgress(string operationName, long completedWork, long totalWork)
    {
        OperationName = operationName ?? throw new ArgumentNullException(nameof(operationName));
        CompletedWork = completedWork;
        TotalWork = totalWork;
    }

    /// <summary>
    /// Gets the name of the current operation
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Gets the amount of work completed
    /// </summary>
    public long CompletedWork { get; }

    /// <summary>
    /// Gets the total amount of work
    /// </summary>
    public long TotalWork { get; }

    /// <summary>
    /// Gets the completion percentage (0.0 to 1.0)
    /// </summary>
    public double PercentComplete => TotalWork > 0 ? (double)CompletedWork / TotalWork : 0.0;

    /// <summary>
    /// Gets a value indicating whether the operation is complete
    /// </summary>
    public bool IsComplete => CompletedWork >= TotalWork;

    /// <summary>
    /// Returns a string representation of the execution progress
    /// </summary>
    /// <returns>A formatted string describing the progress</returns>
    public override string ToString()
    {
        var percentage = (PercentComplete * 100).ToString("F1");
        return $"{OperationName}: {CompletedWork:N0}/{TotalWork:N0} ({percentage}%)";
    }
}
