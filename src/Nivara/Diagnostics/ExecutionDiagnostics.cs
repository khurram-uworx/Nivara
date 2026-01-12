using Nivara.Execution;
using System.Diagnostics;

namespace Nivara.Diagnostics;

/// <summary>
/// Provides comprehensive diagnostic information about query execution,
/// including timing, memory usage, and performance analysis.
/// </summary>
public sealed class ExecutionDiagnostics
{
    private readonly List<OperationTiming> operationTimings = new();
    private readonly List<PerformanceWarning> warnings = new();
    private readonly List<OptimizationApplied> optimizationsApplied = new();
    private readonly Stopwatch totalTimer = new();
    private long initialMemory;
    private long peakMemoryUsage;

    /// <summary>
    /// Initializes a new instance of ExecutionDiagnostics
    /// </summary>
    public ExecutionDiagnostics()
    {
        StartTime = DateTime.UtcNow;
        initialMemory = GC.GetTotalMemory(false);
        peakMemoryUsage = initialMemory;
    }

    /// <summary>
    /// Gets the start time of execution
    /// </summary>
    public DateTime StartTime { get; }

    /// <summary>
    /// Gets the end time of execution, if completed
    /// </summary>
    public DateTime? EndTime { get; private set; }

    /// <summary>
    /// Gets the total execution time
    /// </summary>
    public TimeSpan TotalExecutionTime => totalTimer.Elapsed;

    /// <summary>
    /// Gets the peak memory usage during execution
    /// </summary>
    public long PeakMemoryUsage => peakMemoryUsage;

    /// <summary>
    /// Gets the memory allocated during execution
    /// </summary>
    public long MemoryAllocated => PeakMemoryUsage - initialMemory;

    /// <summary>
    /// Gets the degree of parallelism used during execution
    /// </summary>
    public int ParallelismDegree { get; internal set; } = 1;

    /// <summary>
    /// Gets the execution strategy used
    /// </summary>
    public ExecutionStrategy ExecutionStrategy { get; internal set; } = ExecutionStrategy.Eager;

    /// <summary>
    /// Gets the operation timings recorded during execution
    /// </summary>
    public IReadOnlyList<OperationTiming> OperationTimings => operationTimings;

    /// <summary>
    /// Gets the performance warnings detected during execution
    /// </summary>
    public IReadOnlyList<PerformanceWarning> Warnings => warnings;

    /// <summary>
    /// Gets the optimizations applied during execution
    /// </summary>
    public IReadOnlyList<OptimizationApplied> OptimizationsApplied => optimizationsApplied;

    /// <summary>
    /// Gets a value indicating whether execution is currently in progress
    /// </summary>
    public bool IsExecuting => totalTimer.IsRunning;

    /// <summary>
    /// Starts execution timing
    /// </summary>
    internal void StartExecution()
    {
        totalTimer.Start();
        UpdateMemoryUsage();
    }

    /// <summary>
    /// Ends execution timing
    /// </summary>
    internal void EndExecution()
    {
        totalTimer.Stop();
        EndTime = DateTime.UtcNow;
        UpdateMemoryUsage();
    }

    /// <summary>
    /// Records timing for a specific operation
    /// </summary>
    /// <param name="operationType">The type of operation</param>
    /// <param name="duration">The duration of the operation</param>
    /// <param name="rowsProcessed">The number of rows processed</param>
    /// <param name="memoryUsed">The memory used by the operation</param>
    internal void RecordOperationTiming(string operationType, TimeSpan duration, long rowsProcessed, long memoryUsed)
    {
        operationTimings.Add(new OperationTiming(operationType, duration, rowsProcessed, memoryUsed));
        UpdateMemoryUsage();
    }

    /// <summary>
    /// Records a performance warning
    /// </summary>
    /// <param name="warning">The performance warning to record</param>
    internal void RecordWarning(PerformanceWarning warning)
    {
        warnings.Add(warning);
    }

    /// <summary>
    /// Records an applied optimization
    /// </summary>
    /// <param name="optimization">The optimization that was applied</param>
    internal void RecordOptimization(OptimizationApplied optimization)
    {
        optimizationsApplied.Add(optimization);
    }

    /// <summary>
    /// Updates the peak memory usage tracking
    /// </summary>
    private void UpdateMemoryUsage()
    {
        var currentMemory = GC.GetTotalMemory(false);
        if (currentMemory > peakMemoryUsage)
        {
            peakMemoryUsage = currentMemory;
        }
    }

    /// <summary>
    /// Generates a comprehensive human-readable performance report
    /// </summary>
    /// <returns>A detailed performance report with optimization suggestions</returns>
    public string GenerateReport()
    {
        var report = new System.Text.StringBuilder();

        // Header
        report.AppendLine("Execution Diagnostics Report");
        report.AppendLine("===========================");
        report.AppendLine($"Execution Time: {TotalExecutionTime.TotalMilliseconds:F2} ms");
        report.AppendLine($"Memory Usage: {MemoryAllocated / 1024.0 / 1024.0:F2} MB allocated, {PeakMemoryUsage / 1024.0 / 1024.0:F2} MB peak");
        report.AppendLine($"Execution Strategy: {ExecutionStrategy}");
        report.AppendLine($"Parallelism Degree: {ParallelismDegree}");
        report.AppendLine();

        // Operation Breakdown
        if (operationTimings.Count > 0)
        {
            report.AppendLine("Operation Breakdown:");
            report.AppendLine("-------------------");

            var totalOperationTime = operationTimings.Sum(t => t.Duration.TotalMilliseconds);

            foreach (var timing in operationTimings.OrderByDescending(t => t.Duration))
            {
                var percentage = totalOperationTime > 0 ? (timing.Duration.TotalMilliseconds / totalOperationTime) * 100 : 0;
                var throughput = timing.Duration.TotalSeconds > 0 ? timing.RowsProcessed / timing.Duration.TotalSeconds : 0;

                report.AppendLine($"  {timing.OperationType,-15} {timing.Duration.TotalMilliseconds,8:F2} ms ({percentage,5:F1}%) " +
                                $"{timing.RowsProcessed,10:N0} rows {throughput,10:F0} rows/sec " +
                                $"{timing.MemoryUsed / 1024.0 / 1024.0,8:F2} MB");
            }
            report.AppendLine();
        }

        // Optimizations Applied
        if (optimizationsApplied.Count > 0)
        {
            report.AppendLine("Optimizations Applied:");
            report.AppendLine("--------------------");
            foreach (var optimization in optimizationsApplied)
            {
                report.AppendLine($"  • {optimization.OptimizationName}: {optimization.Description}");
                if (optimization.EstimatedImprovement.HasValue)
                {
                    report.AppendLine($"    Estimated improvement: {optimization.EstimatedImprovement.Value:F1}%");
                }
            }
            report.AppendLine();
        }

        // Performance Warnings
        if (warnings.Count > 0)
        {
            report.AppendLine("Performance Warnings:");
            report.AppendLine("--------------------");
            foreach (var warning in warnings.OrderByDescending(w => w.Severity))
            {
                report.AppendLine($"  {warning.Severity}: {warning.Message}");
                if (!string.IsNullOrEmpty(warning.Suggestion))
                {
                    report.AppendLine($"    Suggestion: {warning.Suggestion}");
                }
            }
            report.AppendLine();
        }

        // Performance Analysis
        report.AppendLine("Performance Analysis:");
        report.AppendLine("--------------------");

        // Analyze execution efficiency
        if (TotalExecutionTime.TotalMilliseconds > 1000)
        {
            report.AppendLine("  • Long execution time detected - consider optimization opportunities");
        }

        if (MemoryAllocated > 100 * 1024 * 1024) // 100MB
        {
            report.AppendLine("  • High memory usage detected - consider streaming or chunked processing");
        }

        if (ParallelismDegree == 1 && operationTimings.Any(t => t.RowsProcessed > 10000))
        {
            report.AppendLine("  • Large dataset processed sequentially - consider parallel execution");
        }

        // Suggest optimizations based on operation patterns
        var filterOperations = operationTimings.Where(t => t.OperationType.Contains("Filter")).ToList();
        var sortOperations = operationTimings.Where(t => t.OperationType.Contains("Sort")).ToList();

        if (sortOperations.Count > 0 && filterOperations.Count > 0)
        {
            var lastFilter = filterOperations.LastOrDefault();
            var firstSort = sortOperations.FirstOrDefault();

            if (lastFilter != null && firstSort != null &&
                operationTimings.IndexOf(firstSort) < operationTimings.IndexOf(lastFilter))
            {
                report.AppendLine("  • Consider moving filter operations before sort operations for better performance");
            }
        }

        return report.ToString();
    }

    /// <summary>
    /// Gets a summary of the execution performance
    /// </summary>
    /// <returns>A concise performance summary</returns>
    public ExecutionSummary GetSummary()
    {
        var totalRows = operationTimings.Sum(t => t.RowsProcessed);
        var averageThroughput = TotalExecutionTime.TotalSeconds > 0 ? totalRows / TotalExecutionTime.TotalSeconds : 0;

        return new ExecutionSummary(
            TotalExecutionTime,
            MemoryAllocated,
            PeakMemoryUsage,
            totalRows,
            averageThroughput,
            operationTimings.Count,
            warnings.Count,
            optimizationsApplied.Count,
            ExecutionStrategy,
            ParallelismDegree);
    }
}

/// <summary>
/// Represents timing information for a specific operation
/// </summary>
public sealed class OperationTiming
{
    /// <summary>
    /// Initializes a new instance of OperationTiming
    /// </summary>
    /// <param name="operationType">The type of operation</param>
    /// <param name="duration">The duration of the operation</param>
    /// <param name="rowsProcessed">The number of rows processed</param>
    /// <param name="memoryUsed">The memory used by the operation</param>
    public OperationTiming(string operationType, TimeSpan duration, long rowsProcessed, long memoryUsed)
    {
        OperationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
        Duration = duration;
        RowsProcessed = rowsProcessed;
        MemoryUsed = memoryUsed;
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets the type of operation
    /// </summary>
    public string OperationType { get; }

    /// <summary>
    /// Gets the duration of the operation
    /// </summary>
    public TimeSpan Duration { get; }

    /// <summary>
    /// Gets the number of rows processed
    /// </summary>
    public long RowsProcessed { get; }

    /// <summary>
    /// Gets the memory used by the operation
    /// </summary>
    public long MemoryUsed { get; }

    /// <summary>
    /// Gets the timestamp when the operation completed
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets the throughput in rows per second
    /// </summary>
    public double Throughput => Duration.TotalSeconds > 0 ? RowsProcessed / Duration.TotalSeconds : 0;
}

/// <summary>
/// Represents a performance warning detected during execution
/// </summary>
public sealed class PerformanceWarning
{
    /// <summary>
    /// Initializes a new instance of PerformanceWarning
    /// </summary>
    /// <param name="severity">The severity of the warning</param>
    /// <param name="message">The warning message</param>
    /// <param name="suggestion">An optional suggestion for improvement</param>
    public PerformanceWarning(PerformanceWarningSeverity severity, string message, string? suggestion = null)
    {
        Severity = severity;
        Message = message ?? throw new ArgumentNullException(nameof(message));
        Suggestion = suggestion;
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets the severity of the warning
    /// </summary>
    public PerformanceWarningSeverity Severity { get; }

    /// <summary>
    /// Gets the warning message
    /// </summary>
    public string Message { get; }

    /// <summary>
    /// Gets an optional suggestion for improvement
    /// </summary>
    public string? Suggestion { get; }

    /// <summary>
    /// Gets the timestamp when the warning was generated
    /// </summary>
    public DateTime Timestamp { get; }
}

/// <summary>
/// Defines the severity levels for performance warnings
/// </summary>
public enum PerformanceWarningSeverity
{
    /// <summary>
    /// Informational message about performance characteristics
    /// </summary>
    Info,

    /// <summary>
    /// Warning about potential performance issues
    /// </summary>
    Warning,

    /// <summary>
    /// Critical performance issue that should be addressed
    /// </summary>
    Critical
}

/// <summary>
/// Represents an optimization that was applied during execution
/// </summary>
public sealed class OptimizationApplied
{
    /// <summary>
    /// Initializes a new instance of OptimizationApplied
    /// </summary>
    /// <param name="optimizationName">The name of the optimization</param>
    /// <param name="description">A description of what was optimized</param>
    /// <param name="estimatedImprovement">The estimated performance improvement percentage</param>
    public OptimizationApplied(string optimizationName, string description, double? estimatedImprovement = null)
    {
        OptimizationName = optimizationName ?? throw new ArgumentNullException(nameof(optimizationName));
        Description = description ?? throw new ArgumentNullException(nameof(description));
        EstimatedImprovement = estimatedImprovement;
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets the name of the optimization
    /// </summary>
    public string OptimizationName { get; }

    /// <summary>
    /// Gets a description of what was optimized
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets the estimated performance improvement percentage, if available
    /// </summary>
    public double? EstimatedImprovement { get; }

    /// <summary>
    /// Gets the timestamp when the optimization was applied
    /// </summary>
    public DateTime Timestamp { get; }
}

/// <summary>
/// Provides a summary of execution performance
/// </summary>
public sealed class ExecutionSummary
{
    /// <summary>
    /// Initializes a new instance of ExecutionSummary
    /// </summary>
    internal ExecutionSummary(
        TimeSpan totalExecutionTime,
        long memoryAllocated,
        long peakMemoryUsage,
        long totalRowsProcessed,
        double averageThroughput,
        int operationCount,
        int warningCount,
        int optimizationCount,
        ExecutionStrategy executionStrategy,
        int parallelismDegree)
    {
        TotalExecutionTime = totalExecutionTime;
        MemoryAllocated = memoryAllocated;
        PeakMemoryUsage = peakMemoryUsage;
        TotalRowsProcessed = totalRowsProcessed;
        AverageThroughput = averageThroughput;
        OperationCount = operationCount;
        WarningCount = warningCount;
        OptimizationCount = optimizationCount;
        ExecutionStrategy = executionStrategy;
        ParallelismDegree = parallelismDegree;
    }

    /// <summary>
    /// Gets the total execution time
    /// </summary>
    public TimeSpan TotalExecutionTime { get; }

    /// <summary>
    /// Gets the memory allocated during execution
    /// </summary>
    public long MemoryAllocated { get; }

    /// <summary>
    /// Gets the peak memory usage during execution
    /// </summary>
    public long PeakMemoryUsage { get; }

    /// <summary>
    /// Gets the total number of rows processed
    /// </summary>
    public long TotalRowsProcessed { get; }

    /// <summary>
    /// Gets the average throughput in rows per second
    /// </summary>
    public double AverageThroughput { get; }

    /// <summary>
    /// Gets the number of operations executed
    /// </summary>
    public int OperationCount { get; }

    /// <summary>
    /// Gets the number of performance warnings generated
    /// </summary>
    public int WarningCount { get; }

    /// <summary>
    /// Gets the number of optimizations applied
    /// </summary>
    public int OptimizationCount { get; }

    /// <summary>
    /// Gets the execution strategy used
    /// </summary>
    public ExecutionStrategy ExecutionStrategy { get; }

    /// <summary>
    /// Gets the degree of parallelism used
    /// </summary>
    public int ParallelismDegree { get; }

    /// <summary>
    /// Returns a string representation of the execution summary
    /// </summary>
    /// <returns>A formatted string with summary statistics</returns>
    public override string ToString()
    {
        return $"ExecutionSummary {{ " +
               $"Time: {TotalExecutionTime.TotalMilliseconds:F2}ms, " +
               $"Memory: {MemoryAllocated / 1024.0 / 1024.0:F2}MB, " +
               $"Rows: {TotalRowsProcessed:N0}, " +
               $"Throughput: {AverageThroughput:F0} rows/sec, " +
               $"Operations: {OperationCount}, " +
               $"Warnings: {WarningCount}, " +
               $"Optimizations: {OptimizationCount} " +
               $"}}";
    }
}