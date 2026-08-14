using System.Collections;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

/// <summary>
/// Enumerates batches of a <see cref="TensorDataset{T}"/> with configurable batch size,
/// shuffling, and RNG seeding. Each enumeration and each epoch re-shuffles with a fresh
/// seed when shuffling is enabled.
/// </summary>
public sealed class DataLoader<T> : IEnumerable<Batch<T>> where T : struct, IFloatingPointIeee754<T>
{
    readonly TensorDataset<T> dataset;
    readonly int batchSize;
    readonly bool shuffle;
    readonly int? seed;

    /// <summary>The underlying dataset.</summary>
    public TensorDataset<T> Dataset => dataset;

    /// <summary>The number of rows per batch (the last batch may be smaller).</summary>
    public int BatchSize => batchSize;

    /// <summary>Whether rows are shuffled per epoch.</summary>
    public bool Shuffle => shuffle;

    /// <summary>The RNG seed; null selects a seed derived from the epoch.</summary>
    public int? Seed => seed;

    /// <summary>
    /// Creates a data loader over a dataset.
    /// </summary>
    /// <param name="dataset">The source dataset</param>
    /// <param name="batchSize">The number of rows per batch (must be positive)</param>
    /// <param name="shuffle">Whether to shuffle rows per epoch</param>
    /// <param name="seed">Optional RNG seed for deterministic shuffling</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataset"/> is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="batchSize"/> is not positive</exception>
    public DataLoader(TensorDataset<T> dataset, int batchSize, bool shuffle = true, int? seed = null)
    {
        this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive.", nameof(batchSize));

        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.seed = seed;
    }

    /// <summary>The number of rows in the underlying dataset.</summary>
    public int Count => dataset.Count;

    int enumerationCount;

    /// <summary>Enumerates all batches in a single pass.</summary>
    public IEnumerator<Batch<T>> GetEnumerator()
    {
        return GetBatches(Interlocked.Increment(ref enumerationCount), 0).GetEnumerator();
    }

    /// <summary>
    /// Produces the batches for one epoch.
    /// </summary>
    /// <param name="epoch">Epoch number; combined with the seed to derive shuffling</param>
    /// <param name="skipBatches">Number of leading batches to skip</param>
    /// <returns>The sequence of batches for the epoch</returns>
    public IEnumerable<Batch<T>> GetBatches(int epoch = 0, int skipBatches = 0)
    {
        int count = dataset.Count;
        int[] indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = i;

        if (shuffle)
        {
            var rng = seed.HasValue
                ? new Random(seed.Value + epoch)
                : new Random(epoch);
            for (int i = count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int batchIndex = 0;
        for (int i = 0; i < count; i += batchSize)
        {
            if (batchIndex < skipBatches)
            {
                batchIndex++;
                continue;
            }

            int remaining = count - i;
            int batchLen = remaining < batchSize ? remaining : batchSize;
            yield return dataset.GetBatch(indices.AsSpan(i, batchLen));
            batchIndex++;
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
