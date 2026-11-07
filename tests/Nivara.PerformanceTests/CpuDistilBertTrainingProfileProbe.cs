using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
using Nivara.AI;
using Nivara.Samples;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// CPU counterpart to SilkDistilBertTrainingProfileProbe (#1003): measures
/// reverse-mode (CPU) DistilBERT fine-tuning ms/batch at B=2/4/8, L=128 so the
/// two can be reported side by side.
/// </summary>
internal static class CpuDistilBertTrainingProfileProbe
{
    const int SeqLen = 128;
    const int Warmup = 1;
    const int Iters = 5;
    const float LearningRate = 2e-5f;

    public static int Run()
    {
        Console.WriteLine("=== CPU DistilBERT training batch profile (reverse-mode AutoDiff) ===");
        if (!TryResolveModel(out string modelDir))
        {
            Console.WriteLine("DistilBERT checkpoint absent; place samples/data/distilbert.");
            return 0;
        }

        var distilConfig = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var bertConfig = distilConfig.ToBertConfig();
        Console.WriteLine($"Config: layers={bertConfig.NumHiddenLayers}, hidden={bertConfig.HiddenSize}, heads={bertConfig.NumAttentionHeads}, vocab={bertConfig.VocabSize}");
        Console.WriteLine();

        int numClasses = 2;

        foreach (int batch in new[] { 2, 4, 8 })
        {
            var model = new DistilBertForSequenceClassification<float>(bertConfig, numClasses: numClasses);
            var tensors = SafeTensorsLoader.Read(Path.Combine(modelDir, "model.safetensors"));
            model.LoadWeights(tensors);
            model.Train();

            var allParams = model.GetParameters().Values;
            using var optimizer = new AdamW<float>(LearningRate, beta1: 0.9, beta2: 0.999, eps: 1e-8f);
            optimizer.AddParameterGroup(allParams, learningRate: LearningRate, weightDecay: 0.01f);

            int nRows = batch * SeqLen;
            var tokens = new float[nRows];
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

            var lossFn = new CrossEntropyLoss<float>();

            // Warmup (not recorded).
            ProfileOnce(model, optimizer, lossFn, tokens, mask, labels, batch, SeqLen);

            var totalMs = new List<double>();
            for (int iter = 0; iter < Iters; iter++)
            {
                var sw = Stopwatch.StartNew();
                ProfileOnce(model, optimizer, lossFn, tokens, mask, labels, batch, SeqLen);
                sw.Stop();
                totalMs.Add(sw.Elapsed.TotalMilliseconds);
            }

            totalMs.Sort();
            double median = totalMs.Count % 2 == 1
                ? totalMs[totalMs.Count / 2]
                : (totalMs[totalMs.Count / 2 - 1] + totalMs[totalMs.Count / 2]) / 2.0;

            Console.WriteLine($"--- B={batch}, L={SeqLen}, classes={numClasses} ---");
            Console.WriteLine($"{"Measured total (median)",-28} {median,10:F3}");
            Console.WriteLine($"  (warmup excluded; {Iters} timed batches)");
            Console.WriteLine();
        }

        return 0;
    }

    static void ProfileOnce(
        DistilBertForSequenceClassification<float> model,
        AdamW<float> optimizer,
        CrossEntropyLoss<float> lossFn,
        float[] tokens, float[] mask, int[] labels,
        int batch, int seqLen)
    {
        var inputIds = GradientUtils.Constant(tokens);
        var attnMask = GradientUtils.Constant(mask);

        using (GradientUtils.Grad())
        {
            var logits = model.Forward(inputIds, attnMask, batch, seqLen);
            var loss = lossFn.Forward(logits, labels);
            loss.Backward();
            optimizer.Step();
            optimizer.ZeroGrad();
            _ = loss.Data[0];
        }
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