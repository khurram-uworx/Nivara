using System.Collections;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class DataLoader<T> : IEnumerable<Batch<T>> where T : struct, IFloatingPointIeee754<T>
{
    readonly TensorDataset<T> dataset;
    readonly int batchSize;
    readonly bool shuffle;
    readonly int? seed;

    public TensorDataset<T> Dataset => dataset;
    public int BatchSize => batchSize;
    public bool Shuffle => shuffle;
    public int? Seed => seed;

    public DataLoader(TensorDataset<T> dataset, int batchSize, bool shuffle = true, int? seed = null)
    {
        this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive.", nameof(batchSize));

        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.seed = seed;
    }

    public int Count => dataset.Count;

    int enumerationCount;

    public IEnumerator<Batch<T>> GetEnumerator()
    {
        return GetBatches(Interlocked.Increment(ref enumerationCount), 0).GetEnumerator();
    }

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
