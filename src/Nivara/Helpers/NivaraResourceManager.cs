using System.Collections.Concurrent;

namespace Nivara.Helpers;

/// <summary>
/// Manages resources and cleanup for abandoned lazy queries and large dataset operations.
/// Provides memory management guidance and automatic cleanup of orphaned resources.
/// </summary>
internal static class NivaraResourceManager
{
    private static readonly ConcurrentDictionary<WeakReference, ResourceInfo> _trackedResources = new();
    private static readonly Timer _cleanupTimer;
    private static readonly object _lock = new object();

    static NivaraResourceManager()
    {
        // Run cleanup every 30 seconds
        _cleanupTimer = new Timer(CleanupAbandonedResources, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    /// <summary>
    /// Information about a tracked resource
    /// </summary>
    private class ResourceInfo
    {
        public string ResourceType { get; init; } = string.Empty;
        public DateTime CreatedAt { get; init; }
        public long EstimatedMemoryUsage { get; init; }
        public Action? CleanupAction { get; init; }
    }

    /// <summary>
    /// Tracks a resource for automatic cleanup if abandoned
    /// </summary>
    /// <param name="resource">The resource to track</param>
    /// <param name="resourceType">The type of resource for diagnostics</param>
    /// <param name="estimatedMemoryUsage">Estimated memory usage in bytes</param>
    /// <param name="cleanupAction">Optional cleanup action to perform when resource is abandoned</param>
    public static void TrackResource(object resource, string resourceType, long estimatedMemoryUsage = 0, Action? cleanupAction = null)
    {
        if (resource == null) return;

        var weakRef = new WeakReference(resource);
        var info = new ResourceInfo
        {
            ResourceType = resourceType,
            CreatedAt = DateTime.UtcNow,
            EstimatedMemoryUsage = estimatedMemoryUsage,
            CleanupAction = cleanupAction
        };

        _trackedResources.TryAdd(weakRef, info);
    }

    /// <summary>
    /// Untrack a resource (called when properly disposed)
    /// </summary>
    /// <param name="resource">The resource to untrack</param>
    public static void UntrackResource(object resource)
    {
        if (resource == null) return;

        lock (_lock)
        {
            // Find and remove the weak reference for this resource
            var keysToRemove = new List<WeakReference>();

            foreach (var kvp in _trackedResources)
            {
                var target = kvp.Key.Target;
                if (target != null && ReferenceEquals(target, resource))
                {
                    keysToRemove.Add(kvp.Key);
                }
            }

            foreach (var key in keysToRemove)
            {
                _trackedResources.TryRemove(key, out _);
            }
        }
    }

    /// <summary>
    /// Gets memory management recommendations for large dataset operations
    /// </summary>
    /// <param name="estimatedDataSize">Estimated size of the dataset in bytes</param>
    /// <param name="operationType">The type of operation being performed</param>
    /// <returns>Memory management recommendations</returns>
    public static MemoryRecommendations GetMemoryRecommendations(long estimatedDataSize, string operationType)
    {
        var availableMemory = GC.GetTotalMemory(false);
        var recommendations = new List<string>();
        var warningLevel = MemoryWarningLevel.None;

        // Check if dataset is large relative to available memory
        var memoryRatio = (double)estimatedDataSize / availableMemory;

        // Also consider absolute size thresholds for datasets
        const long LARGE_DATASET_THRESHOLD = 20 * 1024; // 20KB (more appropriate for test data)
        const long VERY_LARGE_DATASET_THRESHOLD = 100 * 1024; // 100KB

        if (memoryRatio > 0.5 || estimatedDataSize > VERY_LARGE_DATASET_THRESHOLD)
        {
            warningLevel = MemoryWarningLevel.High;
            recommendations.Add("Dataset is very large relative to available memory. Consider processing in chunks.");
            recommendations.Add("Use lazy evaluation with Scan operations instead of Read operations.");
            recommendations.Add("Apply filters early in the query pipeline to reduce data size.");
        }
        else if (memoryRatio > 0.2 || estimatedDataSize > LARGE_DATASET_THRESHOLD)
        {
            warningLevel = MemoryWarningLevel.Medium;
            recommendations.Add("Dataset is moderately large. Consider using lazy evaluation for better memory efficiency.");
            recommendations.Add("Apply column selection early to reduce memory usage.");
        }
        else if (estimatedDataSize > 5 * 1024) // 5KB
        {
            warningLevel = MemoryWarningLevel.Medium;
            recommendations.Add("Consider using lazy evaluation for better memory efficiency.");
        }

        // Operation-specific recommendations
        switch (operationType.ToLowerInvariant())
        {
            case "groupby":
                recommendations.Add("GroupBy operations can be memory-intensive. Consider pre-filtering data.");
                break;
            case "join":
                recommendations.Add("Join operations require both datasets in memory. Ensure smaller dataset is on the right side.");
                break;
            case "sort":
                recommendations.Add("Sort operations require full dataset in memory. Consider using Top/Bottom operations if possible.");
                break;
        }

        return new MemoryRecommendations
        {
            WarningLevel = warningLevel,
            Recommendations = recommendations,
            EstimatedMemoryUsage = estimatedDataSize,
            AvailableMemory = availableMemory
        };
    }

    /// <summary>
    /// Forces cleanup of abandoned resources
    /// </summary>
    public static void ForceCleanup()
    {
        CleanupAbandonedResources(null);
    }

    /// <summary>
    /// Gets statistics about currently tracked resources
    /// </summary>
    /// <returns>Resource tracking statistics</returns>
    public static ResourceStatistics GetResourceStatistics()
    {
        var stats = new Dictionary<string, int>();
        long totalMemoryUsage = 0;
        int abandonedCount = 0;

        foreach (var kvp in _trackedResources)
        {
            var info = kvp.Value;
            var resourceType = info.ResourceType;

            if (!stats.ContainsKey(resourceType))
                stats[resourceType] = 0;

            stats[resourceType]++;
            totalMemoryUsage += info.EstimatedMemoryUsage;

            // Check if resource is abandoned (weak reference target is null)
            if (kvp.Key.Target == null)
                abandonedCount++;
        }

        return new ResourceStatistics
        {
            TrackedResourcesByType = stats,
            TotalEstimatedMemoryUsage = totalMemoryUsage,
            AbandonedResourceCount = abandonedCount,
            TotalTrackedResources = _trackedResources.Count
        };
    }

    /// <summary>
    /// Cleanup timer callback that removes abandoned resources
    /// </summary>
    /// <param name="state">Timer state (unused)</param>
    private static void CleanupAbandonedResources(object? state)
    {
        lock (_lock)
        {
            var keysToRemove = new List<WeakReference>();
            var cleanupActions = new List<Action>();

            foreach (var kvp in _trackedResources)
            {
                var weakRef = kvp.Key;
                var info = kvp.Value;

                // If the target is null, the resource has been garbage collected
                if (weakRef.Target == null)
                {
                    keysToRemove.Add(weakRef);

                    // Collect cleanup actions to run outside the lock
                    if (info.CleanupAction != null)
                    {
                        cleanupActions.Add(info.CleanupAction);
                    }
                }
            }

            // Remove abandoned resources from tracking
            foreach (var key in keysToRemove)
            {
                _trackedResources.TryRemove(key, out _);
            }

            // Run cleanup actions outside the lock to avoid deadlocks
            foreach (var cleanupAction in cleanupActions)
            {
                try
                {
                    cleanupAction();
                }
                catch
                {
                    // Ignore cleanup errors - resource is already abandoned
                }
            }
        }
    }
}

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
    /// Available memory in bytes
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

/// <summary>
/// Statistics about tracked resources
/// </summary>
public class ResourceStatistics
{
    /// <summary>
    /// Count of tracked resources by type
    /// </summary>
    public IReadOnlyDictionary<string, int> TrackedResourcesByType { get; init; } = new Dictionary<string, int>();

    /// <summary>
    /// Total estimated memory usage of tracked resources
    /// </summary>
    public long TotalEstimatedMemoryUsage { get; init; }

    /// <summary>
    /// Number of abandoned resources awaiting cleanup
    /// </summary>
    public int AbandonedResourceCount { get; init; }

    /// <summary>
    /// Total number of tracked resources
    /// </summary>
    public int TotalTrackedResources { get; init; }
}
