using Nivara.Exceptions;
using System.Collections.Concurrent;

namespace Nivara.Execution;

/// <summary>
/// Provides helper methods for parallel execution of DataFrame operations
/// </summary>
static class ParallelExecutionHelper
{
    /// <summary>
    /// Executes a function in parallel across multiple chunks of data
    /// </summary>
    /// <typeparam name="T">The type of input data</typeparam>
    /// <typeparam name="TResult">The type of result data</typeparam>
    /// <param name="source">The source data to process</param>
    /// <param name="processor">The function to apply to each chunk</param>
    /// <param name="chunkSize">The size of each chunk</param>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>The combined results from all chunks</returns>
    public static async Task<IEnumerable<TResult>> ProcessInParallelAsync<T, TResult>(
        IEnumerable<T> source,
        Func<IEnumerable<T>, TResult> processor,
        int chunkSize,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken = default)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (processor == null)
            throw new ArgumentNullException(nameof(processor));

        var chunks = source.Chunk(chunkSize);
        var results = new ConcurrentBag<TResult>();

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism,
            CancellationToken = cancellationToken
        };

        await Parallel.ForEachAsync(chunks, parallelOptions, async (chunk, ct) =>
        {
            ct.ThrowIfCancellationRequested();
            var result = await Task.Run(() => processor(chunk), ct);
            results.Add(result);
        });

        return results;
    }

    /// <summary>
    /// Processes columns in parallel using the specified operation
    /// </summary>
    /// <param name="columns">The columns to process</param>
    /// <param name="operation">The operation to apply to each column</param>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>The processed columns</returns>
    public static async Task<IReadOnlyDictionary<string, IColumn>> ProcessColumnsInParallelAsync(
        IReadOnlyDictionary<string, IColumn> columns,
        Func<IColumn, IColumn> operation,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken = default)
    {
        if (columns == null)
            throw new ArgumentNullException(nameof(columns));
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        var results = new ConcurrentDictionary<string, IColumn>();

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism,
            CancellationToken = cancellationToken
        };

        await Parallel.ForEachAsync(columns, parallelOptions, async (kvp, ct) =>
        {
            ct.ThrowIfCancellationRequested();
            var processedColumn = await Task.Run(() => operation(kvp.Value), ct);
            results.TryAdd(kvp.Key, processedColumn);
        });

        return results;
    }

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
    /// Executes a parallel aggregation operation
    /// </summary>
    /// <typeparam name="T">The type of items to aggregate</typeparam>
    /// <typeparam name="TResult">The type of the aggregation result</typeparam>
    /// <param name="source">The source data</param>
    /// <param name="aggregator">The aggregation function</param>
    /// <param name="combiner">The function to combine partial results</param>
    /// <param name="maxDegreeOfParallelism">The maximum degree of parallelism</param>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>The aggregated result</returns>
    public static async Task<TResult> ParallelAggregateAsync<T, TResult>(
        IEnumerable<T> source,
        Func<IEnumerable<T>, TResult> aggregator,
        Func<TResult, TResult, TResult> combiner,
        int maxDegreeOfParallelism,
        CancellationToken cancellationToken = default)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (aggregator == null)
            throw new ArgumentNullException(nameof(aggregator));
        if (combiner == null)
            throw new ArgumentNullException(nameof(combiner));

        var sourceList = source.ToList();
        if (sourceList.Count == 0)
            throw new ArgumentException("Source cannot be empty", nameof(source));

        if (!ShouldUseParallelProcessing(sourceList.Count, maxDegreeOfParallelism))
        {
            // Use sequential processing for small datasets
            return aggregator(sourceList);
        }

        var chunkSize = CalculateOptimalChunkSize(sourceList.Count, maxDegreeOfParallelism);
        var chunks = sourceList.Chunk(chunkSize);
        var partialResults = new ConcurrentBag<TResult>();

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism,
            CancellationToken = cancellationToken
        };

        await Parallel.ForEachAsync(chunks, parallelOptions, async (chunk, ct) =>
        {
            ct.ThrowIfCancellationRequested();
            var partialResult = await Task.Run(() => aggregator(chunk), ct);
            partialResults.Add(partialResult);
        });

        // Combine all partial results
        return partialResults.Aggregate(combiner);
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
}

/// <summary>
/// Provides extension methods for parallel processing
/// </summary>
static class ParallelExtensions
{
    /// <summary>
    /// Splits an enumerable into chunks of the specified size
    /// </summary>
    /// <typeparam name="T">The type of elements</typeparam>
    /// <param name="source">The source enumerable</param>
    /// <param name="chunkSize">The size of each chunk</param>
    /// <returns>An enumerable of chunks</returns>
    public static IEnumerable<IEnumerable<T>> Chunk<T>(this IEnumerable<T> source, int chunkSize)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (chunkSize <= 0)
            throw new ArgumentException("Chunk size must be greater than zero", nameof(chunkSize));

        return ChunkIterator(source, chunkSize);
    }

    private static IEnumerable<IEnumerable<T>> ChunkIterator<T>(IEnumerable<T> source, int chunkSize)
    {
        using var enumerator = source.GetEnumerator();

        while (enumerator.MoveNext())
        {
            var chunk = new List<T>(chunkSize) { enumerator.Current };

            for (int i = 1; i < chunkSize && enumerator.MoveNext(); i++)
            {
                chunk.Add(enumerator.Current);
            }

            yield return chunk;
        }
    }
}
