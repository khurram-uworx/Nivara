using System.Numerics;

namespace Nivara.AutoDiff.Training;

/// <summary>
/// The result of a data-parallel training run: per-epoch results plus total elapsed time.
/// </summary>
public sealed class DataParallelTrainingResult<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Per-epoch results in run order.</summary>
    public IReadOnlyList<DataParallelEpochResult<T>> Epochs { get; }

    /// <summary>Total wall-clock time for the run.</summary>
    public TimeSpan TotalElapsed { get; }

    /// <summary>
    /// Creates a data-parallel training result.
    /// </summary>
    /// <param name="epochs">Per-epoch results</param>
    /// <param name="totalElapsed">Total elapsed time</param>
    public DataParallelTrainingResult(IReadOnlyList<DataParallelEpochResult<T>> epochs, TimeSpan totalElapsed)
    {
        Epochs = epochs;
        TotalElapsed = totalElapsed;
    }

    /// <summary>Prints a summary of the run to the console.</summary>
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

/// <summary>Per-epoch data-parallel training metrics.</summary>
public sealed class DataParallelEpochResult<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>The 1-based epoch number.</summary>
    public int Epoch { get; init; }

    /// <summary>Average loss across batches in the epoch.</summary>
    public T Loss { get; init; }

    /// <summary>Wall-clock time for the epoch.</summary>
    public TimeSpan Elapsed { get; init; }

    /// <summary>Number of worker threads used.</summary>
    public int Workers { get; init; }

    /// <summary>Number of data chunks the epoch was split into.</summary>
    public int Chunks { get; init; }

    /// <summary>L2 norm of the summed gradients before the optimizer step.</summary>
    public double GradientNorm { get; init; }
}
