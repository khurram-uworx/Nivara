using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class DataParallelTrainingResult<T> where T : struct, INumber<T>
{
    public IReadOnlyList<DataParallelEpochResult<T>> Epochs { get; }
    public TimeSpan TotalElapsed { get; }

    public DataParallelTrainingResult(IReadOnlyList<DataParallelEpochResult<T>> epochs, TimeSpan totalElapsed)
    {
        Epochs = epochs;
        TotalElapsed = totalElapsed;
    }

    public void PrintSummary()
    {
        Console.WriteLine($"Data-parallel training completed in {TotalElapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Epochs: {Epochs.Count}");
        Console.WriteLine();

        foreach (var epoch in Epochs)
        {
            Console.WriteLine(
                $"Epoch {epoch.Epoch,3} | Loss: {epoch.Loss,10:F6} | " +
                $"Workers: {epoch.Workers,2} | Chunks: {epoch.Chunks,4} | " +
                $"Grad Norm: {epoch.GradientNorm,10:F6} | Time: {epoch.Elapsed.TotalSeconds:F2}s");
        }
    }
}

public sealed class DataParallelEpochResult<T> where T : struct, INumber<T>
{
    public int Epoch { get; init; }
    public T Loss { get; init; }
    public TimeSpan Elapsed { get; init; }
    public int Workers { get; init; }
    public int Chunks { get; init; }
    public double GradientNorm { get; init; }
}
