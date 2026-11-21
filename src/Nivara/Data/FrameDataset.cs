namespace Nivara;

/// <summary>
/// The result of splitting a frame into a training and a test part.
/// Supports both property access (<c>Train</c>/<c>Test</c>) and tuple deconstruction.
/// </summary>
public sealed class TrainTestSplit
{
    public TrainTestSplit(NivaraFrame train, NivaraFrame test)
    {
        Train = train ?? throw new ArgumentNullException(nameof(train));
        Test = test ?? throw new ArgumentNullException(nameof(test));
    }

    public NivaraFrame Train { get; }
    public NivaraFrame Test { get; }

    // Convenience aliases
    public NivaraFrame Training => Train;
    public NivaraFrame Testing => Test;
    public NivaraFrame TrainFrame => Train;
    public NivaraFrame TestFrame => Test;

    public void Deconstruct(out NivaraFrame train, out NivaraFrame test)
    {
        train = Train;
        test = Test;
    }
}

/// <summary>
/// Frame-level dataset operations (split / shuffle / batch / chunk / stratified split).
/// All operations are pure: they return new frames and never mutate the source.
/// </summary>
internal static class FrameDatasetCore
{
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, double fraction = 0.2, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (double.IsNaN(fraction) || fraction < 0 || fraction > 1)
            throw new ArgumentOutOfRangeException(nameof(fraction), "Fraction must be between 0 and 1.");

        int n = frame.RowCount;
        int testCount = FractionToTestCount(n, fraction);
        return SplitByCounts(frame, n - testCount, testCount, seed);
    }

    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, int testCount, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (testCount < 0)
            throw new ArgumentOutOfRangeException(nameof(testCount), "Test count cannot be negative.");

        int n = frame.RowCount;
        if (testCount > n)
            throw new ArgumentOutOfRangeException(nameof(testCount), "Test count cannot exceed row count.");

        return SplitByCounts(frame, n - testCount, testCount, seed);
    }

    static TrainTestSplit SplitByCounts(NivaraFrame frame, int trainCount, int testCount, int? seed)
    {
        int n = frame.RowCount;
        if (trainCount < 0) trainCount = 0;
        if (testCount < 0) testCount = 0;
        if (trainCount + testCount > n)
        {
            testCount = Math.Min(testCount, n);
            trainCount = Math.Min(trainCount, n - testCount);
        }

        int[] indices = Enumerable.Range(0, n).ToArray();
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var trainIndices = indices.Take(trainCount).ToArray();
        var testIndices = indices.Skip(trainCount).Take(testCount).ToArray();

        if (trainCount == 0 && testCount == 0)
        {
            return new TrainTestSplit(frame.Slice(0, 0), frame.Slice(0, 0));
        }

        // ReorderByIndices requires a full-length permutation; build one and slice.
        var permuted = frame.ReorderByIndices(indices);
        var trainFrame = trainCount == 0 ? frame.Slice(0, 0) : permuted.Slice(0, trainCount);
        var testFrame = testCount == 0 ? frame.Slice(0, 0) : permuted.Slice(trainCount, testCount);

        return new TrainTestSplit(trainFrame, testFrame);
    }

    public static NivaraFrame Shuffle(NivaraFrame frame, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        int n = frame.RowCount;
        if (n <= 1)
            return n == 0 ? frame.Slice(0, 0) : frame.ReorderByIndices(Enumerable.Range(0, n).ToArray());

        int[] indices = Enumerable.Range(0, n).ToArray();
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        return frame.ReorderByIndices(indices);
    }

    public static IReadOnlyList<NivaraFrame> Batch(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");

        NivaraFrame source = shuffle ? Shuffle(frame, seed) : frame;
        int n = source.RowCount;
        var result = new List<NivaraFrame>();
        for (int i = 0; i < n; i += batchSize)
        {
            int len = Math.Min(batchSize, n - i);
            result.Add(source.Slice(i, len));
        }
        if (n == 0)
            result.Add(source.Slice(0, 0));
        return result;
    }

    public static IReadOnlyList<NivaraFrame> Chunk(NivaraFrame frame, int chunkSize)
        => Batch(frame, chunkSize, false, null);

    public static TrainTestSplit StratifiedSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (string.IsNullOrWhiteSpace(labelColumn))
            throw new ArgumentException("Label column cannot be null or whitespace.", nameof(labelColumn));
        if (!frame.HasColumn(labelColumn))
            throw new ArgumentException($"Column '{labelColumn}' not found in frame.", nameof(labelColumn));
        if (double.IsNaN(fraction) || fraction < 0 || fraction > 1)
            throw new ArgumentOutOfRangeException(nameof(fraction), "Fraction must be between 0 and 1.");

        int n = frame.RowCount;
        IColumn labelCol = frame.GetColumn(labelColumn);

        // Group row indices by label value in first-occurrence order.
        var groups = new Dictionary<object, List<int>>();
        for (int i = 0; i < n; i++)
        {
            object? key = labelCol.IsNull(i) ? null : labelCol.GetValue(i);
            if (!groups.TryGetValue(key!, out var list))
            {
                list = new List<int>();
                groups[key!] = list;
            }
            list.Add(i);
        }

        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var group in groups)
        {
            var idx = group.Value.ToArray();
            for (int i = idx.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (idx[i], idx[j]) = (idx[j], idx[i]);
            }
            int groupTest = idx.Length == 0
                ? 0
                : (int)Math.Round(idx.Length * fraction);
            if (groupTest > idx.Length) groupTest = idx.Length;
            if (fraction > 0 && idx.Length > 0 && groupTest == 0)
                groupTest = 1;
            if (fraction < 1 && groupTest == idx.Length && idx.Length > 0)
                groupTest = Math.Max(0, idx.Length - 1);

            for (int i = 0; i < idx.Length - groupTest; i++)
                trainIndices.Add(idx[i]);
            for (int i = idx.Length - groupTest; i < idx.Length; i++)
                testIndices.Add(idx[i]);
        }

        trainIndices.Sort();
        testIndices.Sort();

        if (trainIndices.Count == 0 && testIndices.Count == 0)
        {
            return new TrainTestSplit(frame.Slice(0, 0), frame.Slice(0, 0));
        }

        var allIndices = trainIndices.Concat(testIndices).ToArray();
        var permuted = frame.ReorderByIndices(allIndices);
        var trainFrame = trainIndices.Count == 0 ? frame.Slice(0, 0) : permuted.Slice(0, trainIndices.Count);
        var testFrame = testIndices.Count == 0 ? frame.Slice(0, 0) : permuted.Slice(trainIndices.Count, testIndices.Count);

        return new TrainTestSplit(trainFrame, testFrame);
    }

    static int FractionToTestCount(int n, double fraction)
    {
        if (n == 0) return 0;
        if (fraction <= 0.5)
            return (int)Math.Round(n * fraction);
        return (int)Math.Round(n * (1 - fraction));
    }
}

/// <summary>
/// Extension methods exposing frame-level dataset operations directly on <see cref="NivaraFrame"/>.
/// </summary>
public static class DatasetExtensions
{
    public static TrainTestSplit SplitTrainTest(this NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, fraction, seed);

    public static TrainTestSplit SplitTrainTest(this NivaraFrame frame, int testCount, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, testCount, seed);

    public static NivaraFrame Shuffle(this NivaraFrame frame, int? seed = null)
        => FrameDatasetCore.Shuffle(frame, seed);

    public static IReadOnlyList<NivaraFrame> Batch(this NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);

    public static IReadOnlyList<NivaraFrame> Chunk(this NivaraFrame frame, int chunkSize)
        => FrameDatasetCore.Chunk(frame, chunkSize);

    public static TrainTestSplit StratifiedSplit(this NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);

    public static TrainTestSplit TrainTestSplit(this NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => SplitTrainTest(frame, fraction, seed);

    public static TrainTestSplit StratifiedTrainTestSplit(this NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => StratifiedSplit(frame, labelColumn, fraction, seed);

    public static IReadOnlyList<NivaraFrame> Batches(this NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Batch(frame, batchSize, shuffle, seed);

    public static IReadOnlyList<NivaraFrame> Chunks(this NivaraFrame frame, int chunkSize)
        => Chunk(frame, chunkSize);
}

/// <summary>Static entry point mirroring <see cref="DatasetExtensions"/>.</summary>
public static class Dataset
{
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, int testCount, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static NivaraFrame Shuffle(NivaraFrame frame, int? seed = null)
        => FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<NivaraFrame> Batch(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunk(NivaraFrame frame, int chunkSize)
        => FrameDatasetCore.Chunk(frame, chunkSize);
    public static TrainTestSplit StratifiedSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
    public static TrainTestSplit TrainTestSplit(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit StratifiedTrainTestSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => StratifiedSplit(frame, labelColumn, fraction, seed);
    public static IReadOnlyList<NivaraFrame> Batches(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunks(NivaraFrame frame, int chunkSize)
        => Chunk(frame, chunkSize);
}

/// <summary>Alias static entry point.</summary>
public static class Datasets
{
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, int testCount, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static NivaraFrame Shuffle(NivaraFrame frame, int? seed = null)
        => FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<NivaraFrame> Batch(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunk(NivaraFrame frame, int chunkSize)
        => FrameDatasetCore.Chunk(frame, chunkSize);
    public static TrainTestSplit StratifiedSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
    public static TrainTestSplit TrainTestSplit(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit StratifiedTrainTestSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => StratifiedSplit(frame, labelColumn, fraction, seed);
    public static IReadOnlyList<NivaraFrame> Batches(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunks(NivaraFrame frame, int chunkSize)
        => Chunk(frame, chunkSize);
}

/// <summary>Alias static entry point.</summary>
public static class NivaraDataset
{
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, int testCount, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static NivaraFrame Shuffle(NivaraFrame frame, int? seed = null)
        => FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<NivaraFrame> Batch(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunk(NivaraFrame frame, int chunkSize)
        => FrameDatasetCore.Chunk(frame, chunkSize);
    public static TrainTestSplit StratifiedSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
    public static TrainTestSplit TrainTestSplit(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit StratifiedTrainTestSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => StratifiedSplit(frame, labelColumn, fraction, seed);
    public static IReadOnlyList<NivaraFrame> Batches(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunks(NivaraFrame frame, int chunkSize)
        => Chunk(frame, chunkSize);
}

/// <summary>Alias static entry point.</summary>
public static class FrameDataset
{
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit SplitTrainTest(NivaraFrame frame, int testCount, int? seed = null)
        => FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static NivaraFrame Shuffle(NivaraFrame frame, int? seed = null)
        => FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<NivaraFrame> Batch(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunk(NivaraFrame frame, int chunkSize)
        => FrameDatasetCore.Chunk(frame, chunkSize);
    public static TrainTestSplit StratifiedSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
    public static TrainTestSplit TrainTestSplit(NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => SplitTrainTest(frame, fraction, seed);
    public static TrainTestSplit StratifiedTrainTestSplit(NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => StratifiedSplit(frame, labelColumn, fraction, seed);
    public static IReadOnlyList<NivaraFrame> Batches(NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<NivaraFrame> Chunks(NivaraFrame frame, int chunkSize)
        => Chunk(frame, chunkSize);
}
