using System.Diagnostics;

namespace Nivara.Diagnostics;

/// <summary>
/// Provides helper methods for collecting execution diagnostics during query operations
/// </summary>
public static class DiagnosticHelper
{
    /// <summary>
    /// Executes an operation with diagnostic tracking
    /// </summary>
    /// <typeparam name="T">The return type of the operation</typeparam>
    /// <param name="diagnostics">The diagnostics instance to record to</param>
    /// <param name="operationType">The type of operation being executed</param>
    /// <param name="operation">The operation to execute</param>
    /// <param name="rowCount">The number of rows being processed</param>
    /// <returns>The result of the operation</returns>
    public static T ExecuteWithDiagnostics<T>(ExecutionDiagnostics diagnostics, string operationType,
        Func<T> operation, long rowCount = 0)
    {
        var stopwatch = Stopwatch.StartNew();
        var initialMemory = GC.GetTotalMemory(false);

        try
        {
            var result = operation();
            stopwatch.Stop();

            var finalMemory = GC.GetTotalMemory(false);
            var memoryUsed = Math.Max(0, finalMemory - initialMemory);

            diagnostics.RecordOperationTiming(operationType, stopwatch.Elapsed, rowCount, memoryUsed);

            // Check for performance warnings
            CheckPerformanceWarnings(diagnostics, operationType, stopwatch.Elapsed, rowCount, memoryUsed);

            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            // Record the failed operation
            var finalMemory = GC.GetTotalMemory(false);
            var memoryUsed = Math.Max(0, finalMemory - initialMemory);
            diagnostics.RecordOperationTiming($"{operationType} (Failed)", stopwatch.Elapsed, rowCount, memoryUsed);

            // Record a critical warning about the failure
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Critical,
                $"Operation {operationType} failed: {ex.Message}",
                "Check input data and operation parameters"));

            throw;
        }
    }

    /// <summary>
    /// Executes an async operation with diagnostic tracking
    /// </summary>
    /// <typeparam name="T">The return type of the operation</typeparam>
    /// <param name="diagnostics">The diagnostics instance to record to</param>
    /// <param name="operationType">The type of operation being executed</param>
    /// <param name="operation">The async operation to execute</param>
    /// <param name="rowCount">The number of rows being processed</param>
    /// <returns>The result of the operation</returns>
    public static async Task<T> ExecuteWithDiagnosticsAsync<T>(ExecutionDiagnostics diagnostics, string operationType,
        Func<Task<T>> operation, long rowCount = 0)
    {
        var stopwatch = Stopwatch.StartNew();
        var initialMemory = GC.GetTotalMemory(false);

        try
        {
            var result = await operation();
            stopwatch.Stop();

            var finalMemory = GC.GetTotalMemory(false);
            var memoryUsed = Math.Max(0, finalMemory - initialMemory);

            diagnostics.RecordOperationTiming(operationType, stopwatch.Elapsed, rowCount, memoryUsed);

            // Check for performance warnings
            CheckPerformanceWarnings(diagnostics, operationType, stopwatch.Elapsed, rowCount, memoryUsed);

            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            // Record the failed operation
            var finalMemory = GC.GetTotalMemory(false);
            var memoryUsed = Math.Max(0, finalMemory - initialMemory);
            diagnostics.RecordOperationTiming($"{operationType} (Failed)", stopwatch.Elapsed, rowCount, memoryUsed);

            // Record a critical warning about the failure
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Critical,
                $"Operation {operationType} failed: {ex.Message}",
                "Check input data and operation parameters"));

            throw;
        }
    }

    /// <summary>
    /// Checks for common performance warnings based on operation characteristics
    /// </summary>
    /// <param name="diagnostics">The diagnostics instance to record warnings to</param>
    /// <param name="operationType">The type of operation</param>
    /// <param name="duration">The duration of the operation</param>
    /// <param name="rowCount">The number of rows processed</param>
    /// <param name="memoryUsed">The memory used by the operation</param>
    private static void CheckPerformanceWarnings(ExecutionDiagnostics diagnostics, string operationType,
        TimeSpan duration, long rowCount, long memoryUsed)
    {
        // Check for slow operations
        if (duration.TotalMilliseconds > 5000) // 5 seconds
        {
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Warning,
                $"Slow {operationType} operation: {duration.TotalMilliseconds:F0}ms",
                "Consider optimizing the operation or using parallel execution"));
        }

        // Check for high memory usage
        if (memoryUsed > 100 * 1024 * 1024) // 100MB
        {
            diagnostics.RecordWarning(new PerformanceWarning(
                PerformanceWarningSeverity.Warning,
                $"High memory usage in {operationType}: {memoryUsed / 1024.0 / 1024.0:F1}MB",
                "Consider streaming or chunked processing for large datasets"));
        }

        // Check for low throughput
        if (rowCount > 0 && duration.TotalSeconds > 0)
        {
            var throughput = rowCount / duration.TotalSeconds;
            if (throughput < 1000 && rowCount > 10000) // Less than 1K rows/sec for large datasets
            {
                diagnostics.RecordWarning(new PerformanceWarning(
                    PerformanceWarningSeverity.Info,
                    $"Low throughput in {operationType}: {throughput:F0} rows/sec",
                    "Consider parallel execution or operation optimization"));
            }
        }

        // Check for specific operation patterns
        CheckOperationSpecificWarnings(diagnostics, operationType, duration, rowCount, memoryUsed);
    }

    /// <summary>
    /// Checks for operation-specific performance warnings
    /// </summary>
    /// <param name="diagnostics">The diagnostics instance to record warnings to</param>
    /// <param name="operationType">The type of operation</param>
    /// <param name="duration">The duration of the operation</param>
    /// <param name="rowCount">The number of rows processed</param>
    /// <param name="memoryUsed">The memory used by the operation</param>
    private static void CheckOperationSpecificWarnings(ExecutionDiagnostics diagnostics, string operationType,
        TimeSpan duration, long rowCount, long memoryUsed)
    {
        switch (operationType.ToLowerInvariant())
        {
            case "sort":
            case "sortoperation":
                if (rowCount > 1000000) // 1M rows
                {
                    diagnostics.RecordWarning(new PerformanceWarning(
                        PerformanceWarningSeverity.Info,
                        "Large sort operation detected",
                        "Consider filtering data before sorting or using external sort for very large datasets"));
                }
                break;

            case "groupby":
            case "groupbyoperation":
                if (duration.TotalMilliseconds > 2000 && rowCount > 0)
                {
                    diagnostics.RecordWarning(new PerformanceWarning(
                        PerformanceWarningSeverity.Info,
                        "Slow grouping operation",
                        "Consider pre-sorting data by group keys or using parallel grouping"));
                }
                break;

            case "join":
            case "joinoperation":
                if (memoryUsed > 50 * 1024 * 1024) // 50MB
                {
                    diagnostics.RecordWarning(new PerformanceWarning(
                        PerformanceWarningSeverity.Warning,
                        "High memory usage in join operation",
                        "Consider using streaming join or ensuring smaller table is on the right side"));
                }
                break;

            case "filter":
            case "filteroperation":
                if (rowCount > 0 && duration.TotalSeconds > 0)
                {
                    var throughput = rowCount / duration.TotalSeconds;
                    if (throughput < 100000) // Less than 100K rows/sec
                    {
                        diagnostics.RecordWarning(new PerformanceWarning(
                            PerformanceWarningSeverity.Info,
                            "Slow filter operation",
                            "Consider optimizing filter predicates or using vectorized operations"));
                    }
                }
                break;
        }
    }

    /// <summary>
    /// Creates a scoped diagnostic context for tracking nested operations
    /// </summary>
    /// <param name="diagnostics">The parent diagnostics instance</param>
    /// <param name="operationType">The type of operation being tracked</param>
    /// <returns>A disposable scope that will record timing when disposed</returns>
    public static DiagnosticScope CreateScope(ExecutionDiagnostics diagnostics, string operationType)
    {
        return new DiagnosticScope(diagnostics, operationType);
    }
}

/// <summary>
/// Provides a scoped context for tracking operation diagnostics
/// </summary>
public sealed class DiagnosticScope : IDisposable
{
    private readonly ExecutionDiagnostics diagnostics;
    private readonly string operationType;
    private readonly Stopwatch stopwatch;
    private readonly long initialMemory;
    private long rowCount;
    private bool disposed;

    /// <summary>
    /// Initializes a new instance of DiagnosticScope
    /// </summary>
    /// <param name="diagnostics">The diagnostics instance to record to</param>
    /// <param name="operationType">The type of operation being tracked</param>
    internal DiagnosticScope(ExecutionDiagnostics diagnostics, string operationType)
    {
        this.diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        this.operationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
        this.stopwatch = Stopwatch.StartNew();
        this.initialMemory = GC.GetTotalMemory(false);
    }

    /// <summary>
    /// Sets the number of rows processed by this operation
    /// </summary>
    /// <param name="count">The number of rows processed</param>
    public void SetRowCount(long count)
    {
        rowCount = count;
    }

    /// <summary>
    /// Records a performance warning within this scope
    /// </summary>
    /// <param name="warning">The warning to record</param>
    public void RecordWarning(PerformanceWarning warning)
    {
        diagnostics.RecordWarning(warning);
    }

    /// <summary>
    /// Records an optimization applied within this scope
    /// </summary>
    /// <param name="optimization">The optimization to record</param>
    public void RecordOptimization(OptimizationApplied optimization)
    {
        diagnostics.RecordOptimization(optimization);
    }

    /// <summary>
    /// Disposes the scope and records the operation timing
    /// </summary>
    public void Dispose()
    {
        if (!disposed)
        {
            stopwatch.Stop();
            var finalMemory = GC.GetTotalMemory(false);
            var memoryUsed = Math.Max(0, finalMemory - initialMemory);

            diagnostics.RecordOperationTiming(operationType, stopwatch.Elapsed, rowCount, memoryUsed);
            disposed = true;
        }
    }
}