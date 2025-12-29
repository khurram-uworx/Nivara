namespace Nivara;

/// <summary>
/// Defines the execution strategy for query operations
/// </summary>
public enum ExecutionStrategy
{
    /// <summary>
    /// Build query plan, execute on Collect() - default strategy
    /// </summary>
    Lazy,

    /// <summary>
    /// Execute operations immediately without building query plans
    /// </summary>
    Eager,

    /// <summary>
    /// Process data in chunks for large datasets to manage memory usage
    /// </summary>
    Streaming,

    /// <summary>
    /// Use multiple threads for parallelizable operations
    /// </summary>
    Parallel
}