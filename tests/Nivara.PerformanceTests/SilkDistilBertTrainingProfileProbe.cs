using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// Phase-1 measurement for issue #961: breaks one Silk DistilBERT training batch
/// down into upload / forward / download / host loss / host softmax-gradient /
/// upload gradient / backward / AdamW. Device-side kernels are not individually
/// separated here; that is the follow-up optimisation work.
/// </summary>
internal static class SilkDistilBertTrainingProfileProbe
{
    const int SeqLen = 128;
    const int Warmup = 1;
    const int Iters = 5;
    const float LearningRate = 2e-5f;
    const float WeightDecay = 0.01f;

    public static int Run()
    {
        Console.WriteLine("=== Silk DistilBERT training batch profile (GPU/OpenCL) ===");
        if (!TryResolveModel(out string modelDir))
        {
            Console.WriteLine("DistilBERT checkpoint absent; place samples/data/distilbert.");
            return 0;
        }

        using var runtime = new SilkRuntime();
        var tensors = SafeTensorsLoader.Read(Path.Combine(modelDir, "model.safetensors"));
        var distilConfig = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var bertConfig = distilConfig.ToBertConfig();

        using var model = new DistilBertForSequenceClassification<float>(bertConfig, numClasses: 2);
        model.LoadWeights(tensors);

        using var trainer = new SilkDistilBertTrainer(runtime, bertConfig, model.ToSilkTensors(), BertGpuNaming.DistilBert, maxBatch: 8, maxSeqLen: 128);
        trainer.ProfilingEnabled = true;
        Console.WriteLine($"Device: {runtime.DeviceName}");
        Console.WriteLine($"Config: layers={bertConfig.NumHiddenLayers}, hidden={bertConfig.HiddenSize}, heads={bertConfig.NumAttentionHeads}, vocab={bertConfig.VocabSize}");
        Console.WriteLine();

        int numClasses = trainer.NumClasses;
        int[] stageOrder =
        [
            0, // upload
            1, // forward
            2, // download
            3, // loss
            4, // input grad
            5, // upload grad
            6, // backward
            7, // adam
        ];
        string[] stageNames =
        [
            "UploadBatch",
            "ForwardCore",
            "SoftmaxCEBackward (device)",
            "Backward chain",
            "AdamUpdateAll",
            "Download(logits)",
            "CrossEntropy (host)",
        ];

        foreach (int batch in new[] { 2, 4, 8 })
        {
            int nRows = batch * SeqLen;
            var tokens = new int[nRows];
            var mask = new float[nRows];
            var labels = new int[batch];
            var rng = new Random(42 + batch);
            for (int i = 0; i < batch; i++)
            {
                labels[i] = i % numClasses;
                for (int s = 0; s < SeqLen; s++)
                {
                    tokens[i * SeqLen + s] = rng.Next(1, distilConfig.VocabSize);
                    mask[i * SeqLen + s] = 1f;
                }
            }

            var batchTokens = tokens;
            var batchMask = mask;
            var batchLabels = labels;
            var gradient = new float[batch * numClasses];

            trainer.ZeroMoments();

            // Warmup (not recorded).
            ProfileOnce(trainer, runtime, batchTokens, batchMask, batchLabels, batch, SeqLen, gradient, numClasses);

            var stageMs = new List<double>[stageNames.Length];
            for (int i = 0; i < stageNames.Length; i++) stageMs[i] = [];
            var totalMs = new List<double>();
            var backwardStageMs = new Dictionary<string, List<double>>();

            for (int iter = 0; iter < Iters; iter++)
            {
                var swTotal = Stopwatch.StartNew();
                var times = ProfileOnce(trainer, runtime, batchTokens, batchMask, batchLabels, batch, SeqLen, gradient, numClasses);
                swTotal.Stop();
                for (int i = 0; i < stageNames.Length; i++)
                    stageMs[i].Add(times[i]);
                foreach (var kv in trainer.LastPhaseMs)
                {
                    if (!backwardStageMs.TryGetValue(kv.Key, out var list))
                    {
                        list = [];
                        backwardStageMs[kv.Key] = list;
                    }
                    list.Add(kv.Value);
                }
                totalMs.Add(swTotal.Elapsed.TotalMilliseconds);
            }

            Array.Sort(totalMs.ToArray());
            double totalMedian = Median(totalMs);

            Console.WriteLine($"--- B={batch}, L={SeqLen}, classes={numClasses} ---");
            Console.WriteLine($"{"Stage",-22} {"ms (med)",10} {"% of total",12}");
            double stageSum = 0;
            for (int i = 0; i < stageNames.Length; i++)
            {
                double m = Median(stageMs[i]);
                stageSum += m;
                double pct = totalMedian > 0 ? 100.0 * m / totalMedian : 0;
                Console.WriteLine($"{stageNames[i],-22} {m,10:F3} {pct,12:F1}");
            }
            Console.WriteLine($"{"Sum of stages",-22} {stageSum,10:F3} {(totalMedian > 0 ? 100.0 * stageSum / totalMedian : 0),12:F1}");
            Console.WriteLine($"{"Measured total",-22} {totalMedian,10:F3} {100.0,12:F1}");
            Console.WriteLine();
            if (backwardStageMs.Count > 0)
            {
                Console.WriteLine("  Backward sub-stages (GPU, ms):");
                foreach (var kv in backwardStageMs.OrderBy(k => k.Key))
                    Console.WriteLine($"    {kv.Key,-22} {Median(kv.Value),10:F3}");
                Console.WriteLine();
            }
        }

        return 0;
    }

    static double[] ProfileOnce(
        SilkDistilBertTrainer trainer, SilkRuntime runtime,
        int[] batchTokens, float[] batchMask, int[] batchLabels,
        int batch, int seqLen, float[] gradient, int numClasses)
    {
        var times = new double[7];
        var sw = new Stopwatch();

        sw.Restart();
        trainer.UploadBatch(batchTokens, batchMask, batch, seqLen);
        sw.Stop();
        times[0] = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        trainer.ForwardCore(batch, seqLen);
        runtime.Finish();
        sw.Stop();
        times[1] = sw.Elapsed.TotalMilliseconds;

        // Labels on device for the softmax CE backward kernel.
        runtime.Upload(trainer.LabelsBuffer, batchLabels.AsSpan(0, batch));

        sw.Restart();
        SilkLauncher.SoftmaxCrossEntropyBackward(runtime, trainer.LogitsBuffer, trainer.LabelsBuffer, trainer.DLogitsBuffer, batch, numClasses, 1f / batch);
        runtime.Finish();
        sw.Stop();
        times[2] = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        trainer.Backward(batch, seqLen);
        runtime.Finish();
        sw.Stop();
        times[3] = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        trainer.AdamUpdateAll(LearningRate, WeightDecay);
        runtime.Finish();
        sw.Stop();
        times[4] = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        var logits = runtime.Download(trainer.LogitsBuffer, batch * numClasses);
        sw.Stop();
        times[5] = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        float loss = SilkLoss.CrossEntropy(logits, batchLabels, batch, numClasses);
        sw.Stop();
        times[6] = sw.Elapsed.TotalMilliseconds;
        _ = loss;

        return times;
    }

    static double Median(List<double> values)
    {
        if (values.Count == 0) return 0;
        var sorted = values.OrderBy(v => v).ToArray();
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    static bool TryResolveModel(out string modelDir)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "samples", "data", "distilbert");
            if (File.Exists(Path.Combine(candidate, "model.safetensors"))
                && File.Exists(Path.Combine(candidate, "config.json")))
            {
                modelDir = candidate;
                return true;
            }
        }
        modelDir = "";
        return false;
    }
}
