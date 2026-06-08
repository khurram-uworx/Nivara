using Nivara.Exceptions;
using Nivara.Operations;

namespace Nivara.Execution;

/// <summary>
/// Provides helper methods for parallel execution of DataFrame operations
/// </summary>
static class ParallelExecutionHelper
{
    /// <summary>
    /// Determines the optimal chunk size for parallel processing based on data size and available cores
    /// </summary>
    /// <param name="totalItems">The total number of items to process</param>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <param name="minChunkSize">The minimum chunk size</param>
    /// <param name="maxChunkSize">The maximum chunk size</param>
    /// <returns>The optimal chunk size</returns>
    public static int CalculateOptimalChunkSize(
        int totalItems,
        int maxDegreeOfParallelism,
        int minChunkSize = 100,
        int maxChunkSize = 10000)
    {
        if (totalItems <= 0)
            return minChunkSize;

        // Aim for 2-4 chunks per core to allow for load balancing
        var targetChunks = maxDegreeOfParallelism * 3;
        var calculatedChunkSize = totalItems / targetChunks;

        // Ensure chunk size is within bounds
        return Math.Max(minChunkSize, Math.Min(calculatedChunkSize, maxChunkSize));
    }

    /// <summary>
    /// Checks if parallel processing would be beneficial for the given data size
    /// </summary>
    /// <param name="itemCount">The number of items to process</param>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <param name="minItemsForParallelism">The minimum number of items to justify parallel processing</param>
    /// <returns>True if parallel processing is recommended, false otherwise</returns>
    public static bool ShouldUseParallelProcessing(
        int itemCount,
        int maxDegreeOfParallelism,
        int minItemsForParallelism = 1000)
    {
        return maxDegreeOfParallelism > 1 && itemCount >= minItemsForParallelism;
    }

    /// <summary>
    /// Validates parallel execution configuration and throws appropriate exceptions
    /// </summary>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism to validate</param>
    /// <exception cref="QueryExecutionException">Thrown when configuration is invalid</exception>
    public static void ValidateParallelConfiguration(int maxDegreeOfParallelism)
    {
        if (maxDegreeOfParallelism <= 0)
        {
            throw new QueryExecutionException("MaxDegreeOfParallelism must be greater than zero");
        }

        if (maxDegreeOfParallelism > Environment.ProcessorCount * 4)
        {
            throw new QueryExecutionException(
                $"MaxDegreeOfParallelism ({maxDegreeOfParallelism}) is too high. " +
                $"Consider using a value closer to the number of processor cores ({Environment.ProcessorCount})");
        }
    }

    /// <summary>
    /// Gets the recommended degree of parallelism for the current system
    /// </summary>
    /// <param name="requestedParallelism">The requested degree of parallelism</param>
    /// <returns>The recommended degree of parallelism</returns>
    public static int GetRecommendedParallelism(int requestedParallelism)
    {
        var processorCount = Environment.ProcessorCount;

        // Cap at 2x processor count to avoid excessive context switching
        var maxRecommended = processorCount * 2;

        return Math.Min(requestedParallelism, maxRecommended);
    }

    /// <summary>
    /// Creates a subset of columns for a specific row range by slicing each column
    /// </summary>
    public static IReadOnlyDictionary<string, IColumn> CreateRowSubset(
        IReadOnlyDictionary<string, IColumn> source, int start, int length)
    {
        var result = new Dictionary<string, IColumn>(source.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var kvp in source)
            result[kvp.Key] = SliceColumn(kvp.Value, start, length);

        return result;
    }

    /// <summary>
    /// Slices a single column by delegating to <see cref="IColumn.Slice"/>
    /// </summary>
    public static IColumn SliceColumn(IColumn column, int start, int length)
        => column.Slice(start, length);

    /// <summary>
    /// Creates a list of chunk ranges for splitting work across parallel workers
    /// </summary>
    public static List<(int Start, int Length)> CreateChunkRanges(int totalRows, int chunkSize)
    {
        var ranges = new List<(int, int)>();
        for (int start = 0; start < totalRows; start += chunkSize)
        {
            int length = Math.Min(chunkSize, totalRows - start);
            ranges.Add((start, length));
        }

        return ranges;
    }

    /// <summary>
    /// Vertically concatenates multiple column dictionaries (row-stack)
    /// </summary>
    public static IReadOnlyDictionary<string, IColumn> ConcatenateColumnDictionaries(
        IReadOnlyList<IReadOnlyDictionary<string, IColumn>> chunks)
    {
        if (chunks.Count == 1)
            return chunks[0];

        var nonEmpty = chunks.Where(c => c.Values.FirstOrDefault()?.Length > 0).ToList();
        if (nonEmpty.Count == 0) return chunks[0];
        if (nonEmpty.Count == 1) return nonEmpty[0];

        var allColumnNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var chunk in chunks)
            foreach (var key in chunk.Keys)
                allColumnNames.Add(key);

        var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        foreach (var columnName in allColumnNames)
        {
            var columnsToConcat = new List<IColumn>();
            foreach (var chunk in chunks)
                if (chunk.TryGetValue(columnName, out var column))
                    columnsToConcat.Add(column);

            if (columnsToConcat.Count > 0)
                result[columnName] = ConcatenationOperation.ConcatenateColumns(columnsToConcat);
        }
        return result;
    }

    /// <summary>
    /// Merges partial GroupBy dictionaries: for matching GroupKey, concatenates index lists
    /// </summary>
    public static Dictionary<GroupKey, List<int>> MergeGroupByDictionaries(
        IReadOnlyList<Dictionary<GroupKey, List<int>>> partialMaps)
    {
        if (partialMaps.Count == 1)
            return new Dictionary<GroupKey, List<int>>(partialMaps[0]);

        var merged = new Dictionary<GroupKey, List<int>>();
        foreach (var partial in partialMaps)
            foreach (var kvp in partial)
                if (merged.TryGetValue(kvp.Key, out var existing))
                    existing.AddRange(kvp.Value);
                else
                    merged[kvp.Key] = new List<int>(kvp.Value);

        return merged;
    }

    /// <summary>
    /// Performs a k-way merge of sorted chunk indices into global sort order
    /// </summary>
    public static int[] MergeSortedChunks(
        IReadOnlyDictionary<string, IColumn> input,
        int[][] sortedLocalIndices,
        int[] chunkStarts,
        List<(int Start, int Length)> ranges,
        IReadOnlyList<SortKey> sortKeys)
    {
        var totalRows = input.Values.FirstOrDefault()?.Length ?? 0;
        var globalIndices = new int[totalRows];
        var currentPtrs = new int[ranges.Count];
        var mergeComparer = new MultiColumnComparer(input, sortKeys);

        for (int outputPos = 0; outputPos < totalRows; outputPos++)
        {
            int bestChunk = -1;
            int bestOriginalRow = -1;

            for (int c = 0; c < ranges.Count; c++)
            {
                if (currentPtrs[c] >= ranges[c].Length)
                    continue;

                var originalRow = chunkStarts[c] + sortedLocalIndices[c][currentPtrs[c]];

                if (bestChunk == -1)
                {
                    bestChunk = c;
                    bestOriginalRow = originalRow;
                }
                else
                {
                    var cmp = mergeComparer.Compare(originalRow, bestOriginalRow);
                    if (cmp < 0 || (cmp == 0 && c < bestChunk))
                    {
                        bestChunk = c;
                        bestOriginalRow = originalRow;
                    }
                }
            }

            globalIndices[outputPos] = bestOriginalRow;
            currentPtrs[bestChunk]++;
        }

        return globalIndices;
    }

    /// <summary>
    /// Merges partial Join hash maps: for matching CompositeKey, concatenates index lists
    /// </summary>
    public static Dictionary<CompositeKey, List<int>> MergeJoinHashMaps(
        IReadOnlyList<Dictionary<CompositeKey, List<int>>> partialMaps)
    {
        if (partialMaps.Count == 1)
            return new Dictionary<CompositeKey, List<int>>(partialMaps[0]);

        var merged = new Dictionary<CompositeKey, List<int>>();
        foreach (var partial in partialMaps)
            foreach (var kvp in partial)
                if (merged.TryGetValue(kvp.Key, out var existing))
                    existing.AddRange(kvp.Value);
                else
                    merged[kvp.Key] = new List<int>(kvp.Value);

        return merged;
    }
}


