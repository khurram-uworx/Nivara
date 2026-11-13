namespace Nivara.Helpers;

/// <summary>
/// Provides memory-management advice for large dataset operations.
/// </summary>
/// <remarks>
/// This helper is internal; the public entry point is
/// <see cref="Nivara.NivaraFrame.GetMemoryRecommendations(string, long?)"/>.
/// </remarks>
internal static class MemoryAdvisor
{
    /// <summary>
    /// Gets memory management recommendations for large dataset operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <paramref name="estimatedDataSize"/> is compared against an available-memory figure and against
    /// absolute thresholds scaled to megabytes. The available-memory figure defaults to
    /// <see cref="GC.GetGCMemoryInfo"/>'s <c>TotalAvailableMemoryBytes</c> (a process/container-aware value,
    /// <em>not</em> the managed heap size reported by <see cref="GC.GetTotalMemory(bool)"/>); pass an explicit
    /// <paramref name="memoryBudgetBytes"/> to tie the advice to a caller-supplied budget such as
    /// <see cref="Nivara.Execution.NivaraExecutionContext.MemoryBudget"/> (the streaming governor).
    /// </para>
    /// <para>
    /// This is advisory heuristics only. The real, enforceable governor is
    /// <see cref="Nivara.Execution.MemoryBudget"/> / <see cref="Nivara.Execution.NivaraExecutionContext.MemoryBudget"/>.
    /// </para>
    /// </remarks>
    /// <param name="estimatedDataSize">Estimated size of the dataset in bytes</param>
    /// <param name="operationType">The type of operation being performed</param>
    /// <param name="memoryBudgetBytes">Optional caller-supplied memory budget in bytes. When null, the runtime's available memory is used.</param>
    /// <returns>Memory management recommendations</returns>
    internal static MemoryRecommendations GetMemoryRecommendations(long estimatedDataSize, string operationType, long? memoryBudgetBytes = null)
    {
        long availableMemory;
        if (memoryBudgetBytes.HasValue && memoryBudgetBytes.Value > 0)
        {
            availableMemory = memoryBudgetBytes.Value;
        }
        else
        {
            var gcInfo = GC.GetGCMemoryInfo();
            availableMemory = gcInfo.TotalAvailableMemoryBytes;
            if (availableMemory <= 0)
            {
                availableMemory = 1024L * 1024 * 1024; // 1 GB fallback
            }
        }

        var recommendations = new List<string>();
        var warningLevel = MemoryWarningLevel.None;

        // Check if dataset is large relative to available memory
        var memoryRatio = (double)estimatedDataSize / availableMemory;

        // Absolute size thresholds scaled to megabytes (production scale, not test-data scale).
        const long SMALL_DATASET_THRESHOLD = 5 * 1024 * 1024; // 5 MB
        const long LARGE_DATASET_THRESHOLD = 20 * 1024 * 1024; // 20 MB
        const long VERY_LARGE_DATASET_THRESHOLD = 100 * 1024 * 1024; // 100 MB

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
        else if (estimatedDataSize > SMALL_DATASET_THRESHOLD)
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
}