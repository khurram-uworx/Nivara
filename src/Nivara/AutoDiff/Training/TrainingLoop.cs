using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class TrainingResult<T> where T : struct, IFloatingPointIeee754<T>
{
    public IReadOnlyList<EpochResult<T>> Epochs { get; }
    public TimeSpan TotalElapsed { get; }

    public TrainingResult(IReadOnlyList<EpochResult<T>> epochs, TimeSpan totalElapsed)
    {
        Epochs = epochs;
        TotalElapsed = totalElapsed;
    }

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

public sealed class EpochResult<T> where T : struct, IFloatingPointIeee754<T>
{
    public int Epoch { get; }
    public T Loss { get; }
    public TimeSpan Elapsed { get; }
    public int Batches { get; }

    public EpochResult(int epoch, T loss, TimeSpan elapsed, int batches)
    {
        Epoch = epoch;
        Loss = loss;
        Elapsed = elapsed;
        Batches = batches;
    }
}

public class TrainingLoop<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    readonly Module<T> model;
    readonly DataLoader<T> loader;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn;
    readonly Optimizer.Optimizer<T> optimizer;
    int maxEpoch;
    bool disposed;

    public Module<T> Model => model;
    public DataLoader<T> Loader => loader;
    public int MaxEpoch => maxEpoch;

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

    public TrainingResult<T> Continue(int additionalEpochs)
    {
        if (additionalEpochs <= 0)
            throw new ArgumentException("Additional epochs must be positive.", nameof(additionalEpochs));

        int startEpoch = maxEpoch + 1;
        maxEpoch += additionalEpochs;
        return Run(startEpoch);
    }

    public void SaveCheckpoint(string path, int epoch, T loss)
    {
        var epochResult = new EpochResult<T>(epoch, loss, TimeSpan.Zero, 0);
        var optimizerState = optimizer.StateDict();
        Serialization.ModelSerializer.SaveCheckpoint(model, epochResult, path, optimizerState);
    }

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

    protected virtual void OnEpochStart(int epoch)
    {
    }

    protected virtual void OnBatchEnd(int epoch, int batch, T lossValue)
    {
    }

    protected virtual void OnEpochEnd(int epoch, EpochResult<T> result)
    {
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

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
