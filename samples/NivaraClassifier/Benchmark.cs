using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Training;
using Nivara.Samples;
using Nivara.Silk;
using System.Diagnostics;
using System.Globalization;

namespace NivaraClassifier;

/// <summary>
/// Measures the sample's CPU and Silk paths on one machine: forward, predict and one training
/// epoch across a batch sweep, plus the two costs that explain the difference — the per-instance
/// OpenCL program build and the per-call host/device transfer. Protocol follows
/// <c>tests/Nivara.PerformanceTests/LayaForwardProbe</c>: one warmup, three rounds, median primary,
/// and alternated round order so the side measured first does not absorb the drift.
/// </summary>
internal static class Benchmark
{
    const int Rounds = 3;
    const int CalibrationIterations = 5;
    const double RoundBudgetMs = 20.0;
    const int MaxIterations = 2000;
    static readonly int[] DefaultBatches = [1, 8, 32, 128];

    internal static void Run(Options opt)
    {
        if (IsDebugBuild())
        {
            Console.WriteLine("DEBUG BUILD - timings would not be evidence. Not measured.");
            Console.WriteLine("  Nivara.dll is unoptimized in Debug while System.Numerics.Tensors stays");
            Console.WriteLine("  ReadyToRun-optimized, so the two routes are not comparable. Re-run with:");
            Console.WriteLine("    dotnet run -c Release --project samples/NivaraClassifier -- --benchmark [--silk]");
            Environment.ExitCode = 2;
            return;
        }

        bool conv;
        switch (opt.Mode)
        {
            case "linear":
                conv = false;
                break;
            case "conv":
                conv = true;
                break;
            default:
                Console.Error.WriteLine($"Unknown --mode '{opt.Mode}'; expected linear or conv.");
                Environment.ExitCode = 2;
                return;
        }

        int[] batches = opt.BatchSizeExplicit ? [opt.BatchSize] : DefaultBatches;
        int vocab = opt.MaxVocab;
        int seqLen = opt.MaxSeqLen;
        int sampleCount = Math.Max(1, opt.NumSamples);
        int embed = opt.EmbeddingDim;
        int hidden = opt.HiddenDim;

        Console.WriteLine("=== NivaraClassifier benchmark (CPU vs Silk) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Processors: {Environment.ProcessorCount}");
        Console.WriteLine("Build: Release - these numbers are evidence.");
        Console.WriteLine($"Model: {(conv ? "conv" : "linear")}  vocab={vocab}  embed={embed}  hidden={hidden}  seq={seqLen}");
        Console.WriteLine($"Corpus: {sampleCount} rows x {seqLen} tokens (seed {opt.Seed}); batch sizes {string.Join(", ", batches)}");
        Console.WriteLine("Protocol: 1 warmup + calibration per leg, 3 timed rounds of ~20 ms, median primary, "
            + $"{BenchStats.NoiseBand:P0} noise band");

        var corpus = BuildCorpus(sampleCount, seqLen, vocab, opt.Seed);

        Module<float> cpuModel;
        Func<int[], int[]> cpuPredict;
        if (conv)
        {
            var model = new ConvTextClassifierModel<float>(vocab, embed, hidden, numClasses: 2, seqLen);
            cpuModel = model;
            cpuPredict = model.Predict;
        }
        else
        {
            var model = new TextClassifierModel<float>(vocab, embed, hidden, numClasses: 2, seqLen);
            cpuModel = model;
            cpuPredict = model.Predict;
        }
        cpuModel.Eval();

        GpuTextLinearClassifier? linear = null;
        GpuTextConvClassifier? gpuConv = null;
        SilkRuntime? transferRuntime = null;
        double constructRuntimeMs = double.NaN;
        double constructClassifierMs = double.NaN;
        string? silkUnavailable = null;

        if (opt.Silk)
        {
            try
            {
                transferRuntime = TimeOnce(() => new SilkRuntime(), out constructRuntimeMs);
                Console.WriteLine($"Device: {transferRuntime.Describe()}");

                var weights = ParameterArrays(cpuModel);
                if (conv)
                {
                    gpuConv = TimeOnce(
                        () => new GpuTextConvClassifier(vocab, embed, hidden, seqLen, opt.Seed),
                        out constructClassifierMs);
                    gpuConv.SetWeightsCpuLayout(vocab, embed, seqLen,
                        weights[0], weights[1], weights[2], weights[3], weights[4], weights[5],
                        weights[6], weights[7], weights[8], weights[9], weights[10]);
                }
                else
                {
                    linear = TimeOnce(
                        () => new GpuTextLinearClassifier(vocab, embed, hidden, opt.Seed),
                        out constructClassifierMs);
                    linear.SetWeightsCpuLayout(vocab, embed, seqLen,
                        weights[0], weights[1], weights[2], weights[3], weights[4]);
                }
            }
            catch (InvalidOperationException ex)
            {
                silkUnavailable = ex.Message;
                linear?.Dispose();
                gpuConv?.Dispose();
                transferRuntime?.Dispose();
                linear = null;
                gpuConv = null;
                transferRuntime = null;
                Console.WriteLine($"Silk/OpenCL unavailable ({ex.Message}); measuring the CPU path only.");
            }
        }

        if (!double.IsNaN(constructRuntimeMs) || !double.IsNaN(constructClassifierMs))
        {
            Console.WriteLine();
            Console.WriteLine("One-shot (OpenCL program build included):");
            if (!double.IsNaN(constructRuntimeMs))
                Console.WriteLine($"  silk construct (runtime)      {Seconds(constructRuntimeMs)}");
            if (!double.IsNaN(constructClassifierMs))
                Console.WriteLine($"  silk construct (classifier)   {Seconds(constructClassifierMs)}");
        }

        var results = new List<LegResult>();

        if (transferRuntime is not null)
        {
            var device = transferRuntime;
            SilkRuntime.SilkBuffer input = SilkRuntime.SilkBuffer.Empty;
            SilkRuntime.SilkBuffer logits = SilkRuntime.SilkBuffer.Empty;
            foreach (int batch in batches)
            {
                int payload = conv ? batch * embed * seqLen : batch * embed;
                float[] host = new float[payload];
                input = device.Ensure(input, payload);
                logits = device.Ensure(logits, batch * 2);

                var summary = MeasureAdaptive(() =>
                {
                    device.Upload(input.Handle, host);
                    device.Download(logits.Handle, batch * 2);
                });
                results.Add(new LegResult(batch, "silk transfer", summary.Timing, summary.Iterations,
                    (payload + batch * 2) * sizeof(float)));
            }
        }

        var frame = FrameBuilder.BuildDocumentClassificationFrame(
            corpus.Tokens, corpus.Labels, sampleCount, seqLen);
        string[] features = Enumerable.Range(0, seqLen).Select(d => $"tok_{d}").ToArray();
        var dataset = new TensorDataset<float>(frame, features, ["label"]);
        var lossFn = new CrossEntropyLoss<float>();
        var flatTokens = ToFloats(corpus.Tokens);

        foreach (int batch in batches)
        {
            int[] tokens = Leading(corpus.Tokens, batch * seqLen);
            var input = ReverseGradTensor<float>.FromMatrix(
                Leading(flatTokens, batch * seqLen), batch, seqLen, requiresGrad: false);

            Action cpuForward = () => cpuModel.Forward(input);
            Action cpuPredictAction = () => cpuPredict(tokens);
            Action? silkForward = null;
            Action? silkPredict = null;
            if (linear is not null)
            {
                var gpu = linear;
                silkForward = () => gpu.Forward(tokens, batch, seqLen);
                silkPredict = () => gpu.Predict(tokens, seqLen);
            }
            else if (gpuConv is not null)
            {
                var gpu = gpuConv;
                silkForward = () => gpu.Forward(tokens, batch);
                silkPredict = () => gpu.Predict(tokens, seqLen);
            }

            int forwardBytes = conv
                ? (batch * embed * seqLen + batch * 2) * sizeof(float)
                : (batch * embed + batch * 2) * sizeof(float);

            AddPair(results, batch, "forward", cpuForward, silkForward, forwardBytes);
            AddPair(results, batch, "predict", cpuPredictAction, silkPredict, forwardBytes);

            var optimizer = new Adam<float>(learningRate: (float)opt.LearningRate);
            optimizer.AddParameterGroup(cpuModel.GetParameters().Values);
            var loader = new DataLoader<float>(dataset, batch, shuffle: true, seed: opt.Seed);
            var loop = new TrainingLoop<float>(cpuModel, loader,
                (logits, targets) =>
                {
                    int rows = logits.Length / 2;
                    var classes = new int[rows];
                    for (int i = 0; i < rows; i++)
                        classes[i] = int.CreateChecked(targets.Data[i]);
                    return lossFn.Forward(logits, classes);
                },
                optimizer, epochs: 1);

            cpuModel.Train();
            var cpuEpoch = Measure(() => loop.Continue(1), warmup: () => loop.Run(), iterations: 1);
            cpuModel.Eval();
            results.Add(new LegResult(batch, "cpu epoch", cpuEpoch, 1, null));

            if (linear is not null)
            {
                var gpu = linear;
                var silkEpoch = Measure(() => gpu.Train(corpus.Tokens, corpus.Labels, batch, seqLen,
                    1, (float)opt.LearningRate, opt.Seed), iterations: 1, warmups: 1);
                results.Add(new LegResult(batch, "silk epoch", silkEpoch, 1, null));
            }
            else if (gpuConv is not null)
            {
                var gpu = gpuConv;
                var silkEpoch = Measure(() => gpu.Train(corpus.Tokens, corpus.Labels, batch, seqLen,
                    1, (float)opt.LearningRate, opt.Seed), iterations: 1, warmups: 1);
                results.Add(new LegResult(batch, "silk epoch", silkEpoch, 1, null));
            }
        }

        if (silkUnavailable is not null)
            Console.WriteLine($"Silk rows omitted: {silkUnavailable}");

        PrintTable(results);
        PrintRatios(results);

        cpuModel.Dispose();
        linear?.Dispose();
        gpuConv?.Dispose();
        transferRuntime?.Dispose();
    }

    static bool IsDebugBuild()
    {
#if DEBUG
        return true;
#else
        return false;
#endif
    }

    static T TimeOnce<T>(Func<T> factory, out double elapsedMs)
    {
        long start = Stopwatch.GetTimestamp();
        T value = factory();
        elapsedMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        return value;
    }

    static (int[] Tokens, int[] Labels) BuildCorpus(int sampleCount, int seqLen, int vocab, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[sampleCount * seqLen];
        var labels = new int[sampleCount];
        for (int i = 0; i < tokens.Length; i++)
            tokens[i] = rng.Next(vocab);
        for (int i = 0; i < labels.Length; i++)
            labels[i] = rng.Next(2);
        return (tokens, labels);
    }

    static float[] ToFloats(int[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = values[i];
        return result;
    }

    static int[] Leading(int[] values, int length)
    {
        var result = new int[length];
        Array.Copy(values, result, length);
        return result;
    }

    static float[] Leading(float[] values, int length)
    {
        var result = new float[length];
        Array.Copy(values, result, length);
        return result;
    }

    static float[][] ParameterArrays(Module<float> model)
        => model.GetParameters().Values.Select(p => p.Tensor.Data.ToArray()).ToArray();

    static void AddPair(List<LegResult> results, int batch, string leg, Action cpu, Action? silk, int bytes)
    {
        if (silk is null)
        {
            var single = MeasureAdaptive(cpu);
            results.Add(new LegResult(batch, $"cpu {leg}", single.Timing, single.Iterations, null));
            return;
        }

        var (cpuSummary, innerIterations, silkSummary, _) = MeasurePair(cpu, silk);
        results.Add(new LegResult(batch, $"cpu {leg}", cpuSummary, innerIterations, null));
        results.Add(new LegResult(batch, $"silk {leg}", silkSummary, innerIterations, bytes));
    }

    /// <summary>
    /// Calibrates both sides to the same iteration count — the faster side's ratio — so a pair's
    /// rounds finish in comparable time and the alternated order is meaningful.
    /// </summary>
    static (TimingSummary Cpu, int Iterations, TimingSummary Silk, int SilkIterations) MeasurePair(
        Action cpu, Action silk)
    {
        cpu();
        silk();
        int iterations = Math.Min(CalibratedIterations(cpu), CalibratedIterations(silk));

        var cpuSamples = new double[Rounds];
        var silkSamples = new double[Rounds];
        for (int round = 0; round < Rounds; round++)
        {
            if (round % 2 == 0)
            {
                cpuSamples[round] = TimePerOp(cpu, iterations);
                silkSamples[round] = TimePerOp(silk, iterations);
            }
            else
            {
                silkSamples[round] = TimePerOp(silk, iterations);
                cpuSamples[round] = TimePerOp(cpu, iterations);
            }
        }
        return (BenchStats.Summarize(cpuSamples), iterations, BenchStats.Summarize(silkSamples), iterations);
    }

    static (TimingSummary Timing, int Iterations) MeasureAdaptive(Action action)
    {
        action();
        int iterations = CalibratedIterations(action);
        var samples = new double[Rounds];
        for (int round = 0; round < Rounds; round++)
            samples[round] = TimePerOp(action, iterations);
        return (BenchStats.Summarize(samples), iterations);
    }

    static TimingSummary Measure(Action action, Action warmup, int iterations)
    {
        warmup();
        var samples = new double[Rounds];
        for (int round = 0; round < Rounds; round++)
            samples[round] = TimePerOp(action, iterations);
        return BenchStats.Summarize(samples);
    }

    static TimingSummary Measure(Action action, int iterations, int warmups)
    {
        for (int i = 0; i < warmups; i++) action();
        var samples = new double[Rounds];
        for (int round = 0; round < Rounds; round++)
            samples[round] = TimePerOp(action, iterations);
        return BenchStats.Summarize(samples);
    }

    static int CalibratedIterations(Action action)
    {
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < CalibrationIterations; i++) action();
        double perIteration = Stopwatch.GetElapsedTime(start).TotalMilliseconds / CalibrationIterations;
        return BenchStats.Iterations(perIteration, RoundBudgetMs, min: 1, max: MaxIterations);
    }

    static double TimePerOp(Action action, int iterations)
    {
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < iterations; i++) action();
        return Stopwatch.GetElapsedTime(start).TotalMilliseconds / iterations;
    }

    static void PrintTable(List<LegResult> results)
    {
        Console.WriteLine();
        WriteRow("batch", "leg", "median", "min", "max", "iters", "ops/s", "B/op");
        WriteRow("-----", "--------------------------", "--------", "--------", "--------", "-----", "--------", "----");
        foreach (var row in results)
        {
            WriteRow(
                row.Batch.ToString(CultureInfo.InvariantCulture),
                row.Leg,
                Milliseconds(row.Timing.MedianMs),
                Milliseconds(row.Timing.MinMs),
                Milliseconds(row.Timing.MaxMs),
                row.Iterations.ToString(CultureInfo.InvariantCulture),
                row.Timing.Measured
                    ? (1000.0 / row.Timing.MedianMs).ToString("N0", CultureInfo.InvariantCulture)
                    : "-",
                row.BytesPerOp?.ToString(CultureInfo.InvariantCulture) ?? "-");
        }
    }

    static void WriteRow(string batch, string leg, string median, string min, string max, string iters, string ops, string bytes)
        => Console.WriteLine($"{batch,5} | {leg,-26} | {median,9} | {min,7} | {max,7} | {iters,5} | {ops,9} | {bytes,6}");

    static void PrintRatios(List<LegResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("Ratios (silk / cpu, median):");
        foreach (int batch in results.Select(r => r.Batch).Distinct())
        {
            var parts = new List<string>();
            foreach (string leg in new[] { "forward", "predict", "epoch" })
            {
                var cpu = Find(results, batch, $"cpu {leg}");
                var silk = Find(results, batch, $"silk {leg}");
                if (cpu is null || silk is null) continue;
                double ratio = BenchStats.Ratio(silk.Value.Timing, cpu.Value.Timing);
                if (double.IsNaN(ratio)) continue;
                parts.Add($"{leg} {ratio.ToString("F2", CultureInfo.InvariantCulture)}x "
                    + $"({VerdictText(BenchStats.Classify(ratio))})");
            }

            if (parts.Count == 0) continue;
            Console.WriteLine($"  batch {batch,3}: {string.Join(" | ", parts)}");

            var transfer = Find(results, batch, "silk transfer");
            var forward = Find(results, batch, "silk forward");
            if (transfer is not null && forward is not null && forward.Value.Timing.Measured)
            {
                double share = transfer.Value.Timing.MedianMs / forward.Value.Timing.MedianMs;
                Console.WriteLine($"             transfer/forward {share.ToString("P1", CultureInfo.InvariantCulture)}");
            }
        }
    }

    static LegResult? Find(List<LegResult> results, int batch, string leg)
    {
        foreach (var row in results)
            if (row.Batch == batch && row.Leg == leg) return row;
        return null;
    }

    static string VerdictText(BenchStats.Verdict verdict) => verdict switch
    {
        BenchStats.Verdict.Faster => "silk faster",
        BenchStats.Verdict.Slower => "silk slower",
        _ => "within noise"
    };

    static string Milliseconds(double ms)
        => ms < 1.0
            ? (ms * 1000.0).ToString("F1", CultureInfo.InvariantCulture) + "us"
            : ms.ToString("F3", CultureInfo.InvariantCulture) + "ms";

    static string Seconds(double ms)
        => (ms / 1000.0).ToString("F3", CultureInfo.InvariantCulture) + "s";

    readonly record struct LegResult(int Batch, string Leg, TimingSummary Timing, int Iterations, int? BytesPerOp);
}
