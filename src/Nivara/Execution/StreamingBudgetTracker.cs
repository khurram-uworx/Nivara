using Nivara.Diagnostics;

namespace Nivara.Execution;

/// <summary>
/// Tracks accumulated frame memory during streaming chunk processing and emits
/// a diagnostic warning when the estimated usage exceeds the configured budget.
/// </summary>
/// <remarks>
/// This tracker lives in the core assembly and uses
/// <see cref="NivaraFrame.estimateFrameMemoryUsage"/> for per-frame memory
/// estimation. It replaces the rudimentary row-count heuristic previously used
/// in <see cref="StreamingExecutionStrategy.StreamChunksAsync"/>.
/// </remarks>
sealed class StreamingBudgetTracker : IDisposable
{
    readonly long memoryBudget;
    readonly double budgetMultiplier;
    long estimatedMemoryUsage;
    bool warningEmitted;

    /// <summary>
    /// Initializes a new tracker.
    /// </summary>
    /// <param name="memoryBudget">The memory budget in bytes from <see cref="NivaraExecutionContext.MemoryBudget"/>.</param>
    /// <param name="budgetMultiplier">
    /// Multiplier applied to <paramref name="memoryBudget"/> to define the
    /// warning threshold. Default 2.0 accounts for boundary-concatenation overhead.
    /// </param>
    public StreamingBudgetTracker(long memoryBudget, double budgetMultiplier = 2.0)
    {
        this.memoryBudget = memoryBudget;
        this.budgetMultiplier = budgetMultiplier;
    }

    public long EstimatedMemoryUsage => Interlocked.Read(ref estimatedMemoryUsage);

    public long WarningThreshold => (long)(memoryBudget * budgetMultiplier);

    public bool IsBudgetExceeded => EstimatedMemoryUsage > WarningThreshold;

    /// <summary>
    /// Adds a frame's estimated memory to the running total.
    /// </summary>
    public void RecordFrame(NivaraFrame frame)
    {
        Interlocked.Add(ref estimatedMemoryUsage, frame.estimateFrameMemoryUsage());
    }

    /// <summary>
    /// Emits a <see cref="PerformanceWarning"/> once if the accumulated memory
    /// has exceeded the budget threshold.
    /// </summary>
    public void RecordWarningIfExceeded(ExecutionDiagnostics? diagnostics)
    {
        if (warningEmitted || diagnostics == null)
            return;

        var usage = EstimatedMemoryUsage;
        if (usage <= WarningThreshold)
            return;

        warningEmitted = true;
        diagnostics.RecordWarning(new PerformanceWarning(
            PerformanceWarningSeverity.Warning,
            $"Streaming memory budget exceeded: accumulated ~{usage:N0} bytes exceeds threshold of {WarningThreshold:N0} bytes (budget: {memoryBudget:N0}, multiplier: {budgetMultiplier}x)",
            "Consider increasing MemoryBudget or reducing chunk count"));
    }

    public void Reset()
    {
        Interlocked.Exchange(ref estimatedMemoryUsage, 0);
        warningEmitted = false;
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref estimatedMemoryUsage, 0);
    }
}
