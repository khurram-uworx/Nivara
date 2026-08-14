using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

/// <summary>
/// The result of a training run: per-epoch results plus total elapsed time.
/// </summary>
public sealed class TrainingResult<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Per-epoch results in run order.</summary>
    public IReadOnlyList<EpochResult<T>> Epochs { get; }

    /// <summary>Total wall-clock time for the run.</summary>
    public TimeSpan TotalElapsed { get; }

    /// <summary>
    /// Creates a training result.
    /// </summary>
    /// <param name="epochs">Per-epoch results</param>
    /// <param name="totalElapsed">Total elapsed time</param>
    public TrainingResult(IReadOnlyList<EpochResult<T>> epochs, TimeSpan totalElapsed)
    {
        Epochs = epochs;
        TotalElapsed = totalElapsed;
    }

    /// <summary>Prints a summary of the run to the console.</summary>
    public void PrintSummary()
    {
        Console.WriteLine($"Training completed in {TotalElapsed.TotalSeconds:F2}s");
        Console.WriteLine($"Epochs: {Epochs.Count}");
        Console.WriteLine();

        foreach (var epoch in Epochs)
        {
            Console.WriteLine(
                $"Epoch {epoch.Epoch,3} | Loss: {epoch.Loss,10:F6} | " +
                $"Batches: {epoch.Batches,4} | Time: {epoch.Elapsed.TotalSeconds:F2}s");
        }
    }
}

/// <summary>Per-epoch training metrics.</summary>
public sealed class EpochResult<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>The 1-based epoch number.</summary>
    public int Epoch { get; }

    /// <summary>Average loss across batches in the epoch.</summary>
    public T Loss { get; }

    /// <summary>Wall-clock time for the epoch.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Number of batches processed in the epoch.</summary>
    public int Batches { get; }

    /// <summary>
    /// Creates an epoch result.
    /// </summary>
    /// <param name="epoch">The epoch number</param>
    /// <param name="loss">Average loss</param>
    /// <param name="elapsed">Elapsed time</param>
    /// <param name="batches">Batch count</param>
    public EpochResult(int epoch, T loss, TimeSpan elapsed, int batches)
    {
        Epoch = epoch;
        Loss = loss;
        Elapsed = elapsed;
        Batches = batches;
    }
}

/// <summary>
/// Sequential training loop: for each epoch, iterates batches, computes forward pass and loss
/// inside a <see cref="GradientUtils.Grad"/> scope, backpropagates, steps the optimizer, and
/// zeroes gradients. Provides checkpoint save/load and epoch lifecycle hooks for subclassing.
/// </summary>
public class TrainingLoop<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    readonly Module<T> model;
    readonly DataLoader<T> loader;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn;
    readonly Optimizer.Optimizer<T> optimizer;
    int maxEpoch;
    bool disposed;

    /// <summary>The model being trained.</summary>
    public Module<T> Model => model;

    /// <summary>The data loader providing batches.</summary>
    public DataLoader<T> Loader => loader;

    /// <summary>The highest epoch reached by the loop.</summary>
    public int MaxEpoch => maxEpoch;

    /// <summary>
    /// Creates a training loop.
    /// </summary>
    /// <param name="model">The model to train</param>
    /// <param name="loader">The data loader supplying batches</param>
    /// <param name="lossFn">Loss function of (output, labels)</param>
    /// <param name="optimizer">The optimizer used to update parameters</param>
    /// <param name="epochs">Number of epochs to run</param>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="epochs"/> is not positive</exception>
    public TrainingLoop(
        Module<T> model,
        DataLoader<T> loader,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn,
        Optimizer.Optimizer<T> optimizer,
        int epochs)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
        this.loader = loader ?? throw new ArgumentNullException(nameof(loader));
        this.lossFn = lossFn ?? throw new ArgumentNullException(nameof(lossFn));
        this.optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive.", nameof(epochs));

        maxEpoch = epochs;
    }

    /// <summary>
    /// Runs the training loop from the current epoch counter up to <see cref="MaxEpoch"/>.
    /// </summary>
    /// <param name="startEpoch">The first epoch to run (default 1)</param>
    /// <returns>A summary of the training run</returns>
    public TrainingResult<T> Run(int startEpoch = 1)
    {
        var epochResults = new List<EpochResult<T>>(maxEpoch - startEpoch + 1);
        var totalSw = Stopwatch.StartNew();

        for (int epoch = startEpoch; epoch <= maxEpoch; epoch++)
        {
            OnEpochStart(epoch);

            var epochSw = Stopwatch.StartNew();
            var totalLoss = T.Zero;
            int batchCount = 0;

            foreach (var batch in loader.GetBatches(epoch))
            {
                using var gradScope = GradientUtils.Grad();

                var output = model.Forward(batch.Features);
                var loss = lossFn(output, batch.Labels);
                loss.Backward();

                optimizer.Step();
                optimizer.ZeroGrad();

                totalLoss += loss[0];
                batchCount++;
                OnBatchEnd(epoch, batchCount, loss[0]);
            }

            epochSw.Stop();

            T avgLoss = batchCount > 0
                ? totalLoss / T.CreateChecked(batchCount)
                : T.Zero;

            var epochResult = new EpochResult<T>(epoch, avgLoss, epochSw.Elapsed, batchCount);
            epochResults.Add(epochResult);
            OnEpochEnd(epoch, epochResult);
        }

        totalSw.Stop();
        return new TrainingResult<T>(epochResults, totalSw.Elapsed);
    }

    /// <summary>
    /// Extends the loop by additional epochs, resuming from the next epoch after
    /// <see cref="MaxEpoch"/>.
    /// </summary>
    /// <param name="additionalEpochs">The number of additional epochs to run</param>
    /// <returns>A summary of the continued run</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="additionalEpochs"/> is not positive</exception>
    public TrainingResult<T> Continue(int additionalEpochs)
    {
        if (additionalEpochs <= 0)
            throw new ArgumentException("Additional epochs must be positive.", nameof(additionalEpochs));

        int startEpoch = maxEpoch + 1;
        maxEpoch += additionalEpochs;
        return Run(startEpoch);
    }

    /// <summary>
    /// Saves a checkpoint of the model and optimizer state.
    /// </summary>
    /// <param name="path">The destination file path</param>
    /// <param name="epoch">The epoch to record in the checkpoint</param>
    /// <param name="loss">The loss to record in the checkpoint</param>
    public void SaveCheckpoint(string path, int epoch, T loss)
    {
        var epochResult = new EpochResult<T>(epoch, loss, TimeSpan.Zero, 0);
        var optimizerState = optimizer.StateDict();
        Serialization.ModelSerializer.SaveCheckpoint(model, epochResult, path, optimizerState);
    }

    /// <summary>
    /// Loads model weights, optimizer state, and the recorded epoch from a checkpoint,
    /// setting <see cref="MaxEpoch"/> to the checkpoint epoch.
    /// </summary>
    /// <param name="path">The checkpoint file path</param>
    public void LoadCheckpoint(string path)
    {
        var checkpoint = Serialization.ModelSerializer.LoadCheckpoint<T>(path);
        model.LoadStateDict(checkpoint.Parameters.ToDictionary(
            kv => kv.Key,
            kv => new ReverseGradTensor<T>(
                NivaraColumn<T>.CreateFromOwnedArray(kv.Value.Values),
                requiresGrad: true,
                kv.Value.Shape)));
        optimizer.LoadStateDict(new Dictionary<string, T[]>(checkpoint.OptimizerState));
        maxEpoch = checkpoint.Epoch;
    }

    /// <summary>Called at the start of each epoch. Override to hook epoch boundaries.</summary>
    /// <param name="epoch">The epoch number about to start</param>
    protected virtual void OnEpochStart(int epoch)
    {
    }

    /// <summary>Called after each batch is processed. Override to observe batch-level loss.</summary>
    /// <param name="epoch">The current epoch number</param>
    /// <param name="batch">The 1-based batch index</param>
    /// <param name="lossValue">The loss of the batch</param>
    protected virtual void OnBatchEnd(int epoch, int batch, T lossValue)
    {
    }

    /// <summary>Called at the end of each epoch. Override to hook epoch completion.</summary>
    /// <param name="epoch">The epoch number that finished</param>
    /// <param name="result">The epoch result</param>
    protected virtual void OnEpochEnd(int epoch, EpochResult<T> result)
    {
    }

    /// <summary>
    /// Disposes the model and optimizer.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the model and optimizer.
    /// </summary>
    /// <param name="disposing">Whether managed resources should be released</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposed) return;
        if (disposing)
        {
            model.Dispose();
            optimizer.Dispose();
        }
        disposed = true;
    }
}
