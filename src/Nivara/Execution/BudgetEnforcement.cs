namespace Nivara.Execution;

/// <summary>
/// Controls how <see cref="NivaraExecutionContext.MemoryBudget"/> is enforced by the
/// streaming execution strategy.
/// </summary>
public enum BudgetEnforcement
{
    /// <summary>
    /// Budget is tracked by <see cref="StreamingBudgetTracker"/> and a
    /// <see cref="PerformanceWarning"/> is emitted when accumulated memory exceeds
    /// the threshold. Reads are never blocked. Default for backward compatibility.
    /// </summary>
    Advisory = 0,

    /// <summary>
    /// Budget is a hard cap on in-flight data. Source reads are gated by a
    /// <see cref="MemoryBudget"/>: the producer cannot advance until the consumer
    /// has released enough capacity.
    /// </summary>
    Enforced = 1
}
