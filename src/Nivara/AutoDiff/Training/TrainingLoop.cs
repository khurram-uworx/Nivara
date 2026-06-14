using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff.Training;

public sealed class TrainingResult<T> where T : struct, INumber<T>
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

public sealed class EpochResult<T> where T : struct, INumber<T>
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

public class TrainingLoop<T> : IDisposable where T : struct, INumber<T>
{
    readonly Module<T> _model;
    readonly DataLoader<T> _loader;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> _lossFn;
    readonly Optimizer.Optimizer<T> _optimizer;
    readonly int _epochs;
    bool disposed;

    public Module<T> Model => _model;
    public DataLoader<T> Loader => _loader;
    public int Epochs => _epochs;

    public TrainingLoop(
        Module<T> model,
        DataLoader<T> loader,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn,
        Optimizer.Optimizer<T> optimizer,
        int epochs)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _lossFn = lossFn ?? throw new ArgumentNullException(nameof(lossFn));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive.", nameof(epochs));

        _epochs = epochs;
    }

    public TrainingResult<T> Run()
    {
        var epochResults = new List<EpochResult<T>>(_epochs);
        var totalSw = Stopwatch.StartNew();

        for (int epoch = 1; epoch <= _epochs; epoch++)
        {
            OnEpochStart(epoch);

            var epochSw = Stopwatch.StartNew();
            var totalLoss = T.Zero;
            int batchCount = 0;

            foreach (var batch in _loader)
            {
                using var gradScope = GradientUtils.Grad();

                var output = _model.Forward(batch.Features);
                var loss = _lossFn(output, batch.Labels);
                loss.Backward();

                _optimizer.Step();
                _optimizer.ZeroGrad();

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

    protected virtual void OnEpochStart(int epoch)
    {
    }

    protected virtual void OnBatchEnd(int epoch, int batch, T lossValue)
    {
    }

    protected virtual void OnEpochEnd(int epoch, EpochResult<T> result)
    {
    }

    protected void SaveCheckpoint(string path, int epoch, EpochResult<T> result)
    {
        var checkpoint = new
        {
            epoch,
            loss = result.Loss,
            elapsed = result.Elapsed.TotalSeconds,
            batches = result.Batches
        };

        var json = System.Text.Json.JsonSerializer.Serialize(checkpoint, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });

        File.WriteAllText(path, json);
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
            _model.Dispose();
            _optimizer.Dispose();
        }
        disposed = true;
    }
}
