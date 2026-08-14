using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using Nivara.Execution;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Training;

/// <summary>
/// Data-parallel training loop: splits each epoch's rows into chunks, computes forward passes
/// and losses for all chunks in parallel inside <see cref="GradientUtils.Grad"/> scopes,
/// backpropagates each chunk, sums the gradients, applies a single optimizer step, and reports
/// per-epoch metrics including the summed-gradient norm.
/// </summary>
public class DataParallelTrainer<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    readonly Module<T> model;
    readonly DataLoader<T> loader;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn;
    readonly Optimizer.Optimizer<T> optimizer;
    readonly int epochs;
    readonly int? maxDegreeOfParallelism;
    bool disposed;

    /// <summary>
    /// Creates a data-parallel trainer.
    /// </summary>
    /// <param name="model">The model to train</param>
    /// <param name="loader">The data loader supplying the dataset</param>
    /// <param name="lossFn">Loss function of (output, labels)</param>
    /// <param name="optimizer">The optimizer used to update parameters</param>
    /// <param name="epochs">Number of epochs to run</param>
    /// <param name="maxDegreeOfParallelism">Optional cap on parallel workers; defaults to the processor count</param>
    /// <exception cref="ArgumentNullException">Thrown when any required argument is null</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="epochs"/> is not positive</exception>
    public DataParallelTrainer(
        Module<T> model,
        DataLoader<T> loader,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn,
        Optimizer.Optimizer<T> optimizer,
        int epochs,
        int? maxDegreeOfParallelism = null)
    {
        this.model = model ?? throw new ArgumentNullException(nameof(model));
        this.loader = loader ?? throw new ArgumentNullException(nameof(loader));
        this.lossFn = lossFn ?? throw new ArgumentNullException(nameof(lossFn));
        this.optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive.", nameof(epochs));

        this.epochs = epochs;
        this.maxDegreeOfParallelism = maxDegreeOfParallelism;
    }

    /// <summary>
    /// Runs the data-parallel training loop for all configured epochs.
    /// </summary>
    /// <returns>A summary of the training run</returns>
    public DataParallelTrainingResult<T> Run()
    {
        var epochResults = new List<DataParallelEpochResult<T>>(epochs);
        var totalSw = Stopwatch.StartNew();
        var dataset = loader.Dataset;
        int totalRows = dataset.Count;
        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(
            maxDegreeOfParallelism ?? Environment.ProcessorCount);
        int chunkSize = loader.BatchSize;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            OnEpochStart(epoch);
            var epochSw = Stopwatch.StartNew();

            var indices = CreateShuffledIndices(totalRows, epoch);
            var ranges = ParallelExecutionHelper.CreateChunkRanges(totalRows, chunkSize);

            var allGradients = new List<Dictionary<string, T[]>>(ranges.Count);
            var totalLoss = T.Zero;
            int batchCount = 0;

            var forwardResults = new (ReverseGradTensor<T> output, ReverseGradTensor<T> loss)?[ranges.Count];

            if (maxDop <= 1 || ranges.Count <= 1)
            {
                for (int i = 0; i < ranges.Count; i++)
                {
                    var range = ranges[i];
                    var chunkIndices = indices.AsSpan(range.Start, range.Length);
                    var batch = dataset.GetBatch(chunkIndices);

                    using var gradScope = GradientUtils.Grad();
                    var output = model.Forward(batch.Features);
                    var loss = lossFn(output, batch.Labels);

                    forwardResults[i] = (output, loss);
                }
            }
            else
            {
                Parallel.For(0, ranges.Count, new ParallelOptions { MaxDegreeOfParallelism = maxDop }, i =>
                {
                    var range = ranges[i];
                    var chunkIndices = indices.AsSpan(range.Start, range.Length);
                    var batch = dataset.GetBatch(chunkIndices);

                    using var gradScope = GradientUtils.Grad();
                    var output = model.Forward(batch.Features);
                    var loss = lossFn(output, batch.Labels);

                    forwardResults[i] = (output, loss);
                });
            }

            for (int i = 0; i < ranges.Count; i++)
            {
                var result = forwardResults[i];
                if (result is not { } r) continue;

                try
                {
                    using var gradScope = GradientUtils.Grad();
                    r.loss.Backward();
                    batchCount++;

                    T lossVal = r.loss[0];
                    totalLoss += lossVal;

                    var snapshot = CloneGradients();
                    allGradients.Add(snapshot);
                }
                finally
                {
                    r.loss.Dispose();
                    r.output.Dispose();
                }
            }

            if (allGradients.Count > 0)
            {
                SumAndApplyGradients(allGradients);
                double gradNorm = ComputeGradientNorm();
                optimizer.Step();
                optimizer.ZeroGrad();

                epochSw.Stop();

                T avgLoss = batchCount > 0
                    ? totalLoss / T.CreateChecked(batchCount)
                    : T.Zero;

                var epochResult = new DataParallelEpochResult<T>
                {
                    Epoch = epoch,
                    Loss = avgLoss,
                    Elapsed = epochSw.Elapsed,
                    Workers = maxDop,
                    Chunks = ranges.Count,
                    GradientNorm = gradNorm
                };
                epochResults.Add(epochResult);
                OnEpochEnd(epoch, epochResult);
            }
            else
            {
                epochSw.Stop();

                var epochResult = new DataParallelEpochResult<T>
                {
                    Epoch = epoch,
                    Loss = T.Zero,
                    Elapsed = epochSw.Elapsed,
                    Workers = maxDop,
                    Chunks = ranges.Count,
                    GradientNorm = 0
                };
                epochResults.Add(epochResult);
                OnEpochEnd(epoch, epochResult);
            }
        }

        totalSw.Stop();
        return new DataParallelTrainingResult<T>(epochResults, totalSw.Elapsed);
    }

    int[] CreateShuffledIndices(int totalRows, int epoch)
    {
        var indices = new int[totalRows];
        for (int i = 0; i < totalRows; i++)
            indices[i] = i;

        if (loader.Shuffle)
        {
            var seed = loader.Seed ?? (epoch * 397);
            var rng = new Random(seed);
            for (int i = totalRows - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        return indices;
    }

    Dictionary<string, T[]> CloneGradients()
    {
        var snapshot = new Dictionary<string, T[]>();
        var parameters = model.Parameters();

        foreach (var (name, tensor) in parameters)
        {
            if (tensor.Grad != null)
            {
                var length = tensor.Grad.Length;
                var gradData = new T[length];
                tensor.Grad.CopyTo(gradData.AsSpan(), T.Zero);
                snapshot[name] = gradData;
            }
        }

        return snapshot;
    }

    void SumAndApplyGradients(List<Dictionary<string, T[]>> allGradients)
    {
        if (allGradients.Count == 0) return;

        var parameters = model.Parameters();
        int chunkCount = allGradients.Count;

        foreach (var (name, tensor) in parameters)
        {
            if (!allGradients[0].ContainsKey(name)) continue;

            int length = allGradients[0][name].Length;
            var summed = new T[length];

            for (int c = 0; c < chunkCount; c++)
            {
                if (!allGradients[c].TryGetValue(name, out var chunkGrad)) continue;

                TensorPrimitives.Add(summed, chunkGrad, summed);
            }

            tensor.Grad = NivaraColumn<T>.Create(summed);
        }
    }

    double ComputeGradientNorm()
    {
        double sumSq = 0;
        var parameters = model.Parameters();

        foreach (var (_, tensor) in parameters)
        {
            if (tensor.Grad == null) continue;

            var grad = tensor.Grad;
            for (int i = 0; i < grad.Length; i++)
            {
                double val = double.CreateChecked(grad[i]);
                sumSq += val * val;
            }
        }

        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// Disposes the model and optimizer.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    void Dispose(bool disposing)
    {
        if (disposed) return;
        if (disposing)
        {
            model.Dispose();
            optimizer.Dispose();
        }
        disposed = true;
    }

    /// <summary>Called at the start of each epoch. Override to hook epoch boundaries.</summary>
    /// <param name="epoch">The epoch number about to start</param>
    protected virtual void OnEpochStart(int epoch) { }

    /// <summary>Called at the end of each epoch. Override to hook epoch completion.</summary>
    /// <param name="epoch">The epoch number that finished</param>
    /// <param name="result">The epoch result</param>
    protected virtual void OnEpochEnd(int epoch, DataParallelEpochResult<T> result) { }
}
