using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using Nivara.AutoDiff.Nn;
using Nivara.Execution;

namespace Nivara.AutoDiff.Training;

public class DataParallelTrainer<T> : IDisposable where T : struct, INumber<T>
{
    readonly Module<T> _model;
    readonly DataLoader<T> _loader;
    readonly Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> _lossFn;
    readonly Optimizer.Optimizer<T> _optimizer;
    readonly int _epochs;
    readonly int? _maxDegreeOfParallelism;
    bool disposed;

    public DataParallelTrainer(
        Module<T> model,
        DataLoader<T> loader,
        Func<ReverseGradTensor<T>, ReverseGradTensor<T>, ReverseGradTensor<T>> lossFn,
        Optimizer.Optimizer<T> optimizer,
        int epochs,
        int? maxDegreeOfParallelism = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _lossFn = lossFn ?? throw new ArgumentNullException(nameof(lossFn));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));

        if (epochs <= 0)
            throw new ArgumentException("Epochs must be positive.", nameof(epochs));

        _epochs = epochs;
        _maxDegreeOfParallelism = maxDegreeOfParallelism;
    }

    public DataParallelTrainingResult<T> Run()
    {
        var epochResults = new List<DataParallelEpochResult<T>>(_epochs);
        var totalSw = Stopwatch.StartNew();
        var dataset = _loader.Dataset;
        int totalRows = dataset.Count;
        var maxDop = ParallelExecutionHelper.GetRecommendedParallelism(
            _maxDegreeOfParallelism ?? Environment.ProcessorCount);
        int chunkSize = _loader.BatchSize;

        for (int epoch = 1; epoch <= _epochs; epoch++)
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

                    var output = _model.Forward(batch.Features);
                    var loss = _lossFn(output, batch.Labels);

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

                    var output = _model.Forward(batch.Features);
                    var loss = _lossFn(output, batch.Labels);

                    forwardResults[i] = (output, loss);
                });
            }

            for (int i = 0; i < ranges.Count; i++)
            {
                var result = forwardResults[i];
                if (result is not { } r) continue;

                try
                {
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
                _optimizer.Step();
                _optimizer.ZeroGrad();

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

        if (_loader.Shuffle)
        {
            var seed = _loader.Seed ?? (epoch * 397);
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
        var parameters = _model.Parameters();

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

        var parameters = _model.Parameters();
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
        var parameters = _model.Parameters();

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
            _model.Dispose();
            _optimizer.Dispose();
        }
        disposed = true;
    }

    protected virtual void OnEpochStart(int epoch) { }
    protected virtual void OnEpochEnd(int epoch, DataParallelEpochResult<T> result) { }
}
