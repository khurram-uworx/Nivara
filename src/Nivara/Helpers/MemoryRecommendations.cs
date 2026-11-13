namespace Nivara.Helpers;

/// <summary>
/// Memory management recommendations for large dataset operations
/// </summary>
public class MemoryRecommendations
{
    /// <summary>
    /// The warning level for memory usage
    /// </summary>
    public MemoryWarningLevel WarningLevel { get; init; }

    /// <summary>
    /// List of recommendations for memory optimization
    /// </summary>
    public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Estimated memory usage in bytes
    /// </summary>
    public long EstimatedMemoryUsage { get; init; }

    /// <summary>
    /// Available memory in bytes. Defaults to <see cref="GC.GetGCMemoryInfo"/>'s
    /// <c>TotalAvailableMemoryBytes</c> (a process/container-aware figure), or the caller-supplied
    /// budget when <see cref="NivaraFrame.GetMemoryRecommendations(string, long?)"/>
    /// is invoked with one. This is <em>not</em> the managed heap size.
    /// </summary>
    public long AvailableMemory { get; init; }
}

/// <summary>
/// Warning levels for memory usage
/// </summary>
public enum MemoryWarningLevel
{
    /// <summary>
    /// No memory concerns
    /// </summary>
    None,

    /// <summary>
    /// Moderate memory usage - consider optimizations
    /// </summary>
    Medium,

    /// <summary>
    /// High memory usage - optimizations strongly recommended
    /// </summary>
    High
}
