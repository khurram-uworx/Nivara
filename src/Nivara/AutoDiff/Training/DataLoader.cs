using System.Collections;
using System.Numerics;
using System.Threading;

namespace Nivara.AutoDiff.Training;

public sealed class DataLoader<T> : IEnumerable<Batch<T>> where T : struct, IFloatingPointIeee754<T>
{
    readonly TensorDataset<T> _dataset;
    readonly int _batchSize;
    readonly bool _shuffle;
    readonly int? _seed;

    public TensorDataset<T> Dataset => _dataset;
    public int BatchSize => _batchSize;
    public bool Shuffle => _shuffle;
    public int? Seed => _seed;

    public DataLoader(TensorDataset<T> dataset, int batchSize, bool shuffle = true, int? seed = null)
    {
        _dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));

        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive.", nameof(batchSize));

        _batchSize = batchSize;
        _shuffle = shuffle;
        _seed = seed;
    }

    public int Count => _dataset.Count;

    int enumerationCount;

    public IEnumerator<Batch<T>> GetEnumerator()
    {
        return GetBatches(Interlocked.Increment(ref enumerationCount), 0).GetEnumerator();
    }

    public IEnumerable<Batch<T>> GetBatches(int epoch = 0, int skipBatches = 0)
    {
        int count = _dataset.Count;
        int[] indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = i;

        if (_shuffle)
        {
            var rng = _seed.HasValue
                ? new Random(_seed.Value + epoch)
                : new Random(epoch);
            for (int i = count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int batchIndex = 0;
        for (int i = 0; i < count; i += _batchSize)
        {
            if (batchIndex < skipBatches)
            {
                batchIndex++;
                continue;
            }

            int remaining = count - i;
            int batchLen = remaining < _batchSize ? remaining : _batchSize;
            yield return _dataset.GetBatch(indices.AsSpan(i, batchLen));
            batchIndex++;
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
