namespace Nivara.Data;

/// <summary>
/// Mirror of <see cref="Nivara.DatasetExtensions"/> in the <c>Nivara.Data</c> namespace.
/// </summary>
public static class DatasetExtensions
{
    public static Nivara.TrainTestSplit SplitTrainTest(this Nivara.NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static Nivara.TrainTestSplit SplitTrainTest(this Nivara.NivaraFrame frame, int testCount, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static Nivara.NivaraFrame Shuffle(this Nivara.NivaraFrame frame, int? seed = null)
        => Nivara.FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Batch(this Nivara.NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Nivara.FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Chunk(this Nivara.NivaraFrame frame, int chunkSize)
        => Nivara.FrameDatasetCore.Chunk(frame, chunkSize);
    public static Nivara.TrainTestSplit StratifiedSplit(this Nivara.NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
}

/// <summary>Static entry point in <c>Nivara.Data</c>.</summary>
public static class Dataset
{
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, int testCount, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static Nivara.NivaraFrame Shuffle(Nivara.NivaraFrame frame, int? seed = null)
        => Nivara.FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Batch(Nivara.NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Nivara.FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Chunk(Nivara.NivaraFrame frame, int chunkSize)
        => Nivara.FrameDatasetCore.Chunk(frame, chunkSize);
    public static Nivara.TrainTestSplit StratifiedSplit(Nivara.NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
}

/// <summary>Alias static entry point in <c>Nivara.Data</c>.</summary>
public static class Datasets
{
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, int testCount, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static Nivara.NivaraFrame Shuffle(Nivara.NivaraFrame frame, int? seed = null)
        => Nivara.FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Batch(Nivara.NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Nivara.FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Chunk(Nivara.NivaraFrame frame, int chunkSize)
        => Nivara.FrameDatasetCore.Chunk(frame, chunkSize);
    public static Nivara.TrainTestSplit StratifiedSplit(Nivara.NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
}

/// <summary>Alias static entry point in <c>Nivara.Data</c>.</summary>
public static class NivaraDataset
{
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, int testCount, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static Nivara.NivaraFrame Shuffle(Nivara.NivaraFrame frame, int? seed = null)
        => Nivara.FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Batch(Nivara.NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Nivara.FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Chunk(Nivara.NivaraFrame frame, int chunkSize)
        => Nivara.FrameDatasetCore.Chunk(frame, chunkSize);
    public static Nivara.TrainTestSplit StratifiedSplit(Nivara.NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
}

/// <summary>Alias static entry point in <c>Nivara.Data</c>.</summary>
public static class FrameDataset
{
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, fraction, seed);
    public static Nivara.TrainTestSplit SplitTrainTest(Nivara.NivaraFrame frame, int testCount, int? seed = null)
        => Nivara.FrameDatasetCore.SplitTrainTest(frame, testCount, seed);
    public static Nivara.NivaraFrame Shuffle(Nivara.NivaraFrame frame, int? seed = null)
        => Nivara.FrameDatasetCore.Shuffle(frame, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Batch(Nivara.NivaraFrame frame, int batchSize, bool shuffle = false, int? seed = null)
        => Nivara.FrameDatasetCore.Batch(frame, batchSize, shuffle, seed);
    public static IReadOnlyList<Nivara.NivaraFrame> Chunk(Nivara.NivaraFrame frame, int chunkSize)
        => Nivara.FrameDatasetCore.Chunk(frame, chunkSize);
    public static Nivara.TrainTestSplit StratifiedSplit(Nivara.NivaraFrame frame, string labelColumn, double fraction = 0.2, int? seed = null)
        => Nivara.FrameDatasetCore.StratifiedSplit(frame, labelColumn, fraction, seed);
}
