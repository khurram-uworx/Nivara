using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using Nivara.Tensors;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class PerfTests
{
    [Test]
    public void GeluVsRelu_Throughput_1K()
    {
        RunGeluVsReluBenchmark(1_000);
    }

    [Test]
    public void GeluVsRelu_Throughput_10K()
    {
        RunGeluVsReluBenchmark(10_000);
    }

    [Test]
    public void GeluVsRelu_Throughput_100K()
    {
        RunGeluVsReluBenchmark(100_000);
    }

    static void RunGeluVsReluBenchmark(int size)
    {
        var rng = new Random(42);
        var data = new float[size];
        for (int i = 0; i < size; i++)
            data[i] = (float)(rng.NextDouble() * 4 - 2);

        var column = NivaraColumn<float>.Create(data);

        column.Gelu();
        column.Relu();
        var geluTime = MeasureBestOfFiveMs(() => column.Gelu());
        var reluTime = MeasureBestOfFiveMs(() => column.Relu());

        double geluPerSec = size / (geluTime / 1000.0);
        double reluPerSec = size / (reluTime / 1000.0);

        TestContext.Out.WriteLine(
            $"GELU vs ReLU ({size} elements): " +
            $"GELU={geluTime:F2}ms ({geluPerSec:F0} el/s), " +
            $"ReLU={reluTime:F2}ms ({reluPerSec:F0} el/s), " +
            $"ratio={geluTime / reluTime:F2}x");
    }

    [Test]
    public void EmbeddingGather_OneHotMatMul_Vs_Gather()
    {
        int numTokens = 128;
        int vocabSize = 30522;
        int embedDim = 768;
        var rng = new Random(42);

        var tokenIds = new int[numTokens];
        for (int i = 0; i < numTokens; i++)
            tokenIds[i] = rng.Next(vocabSize);

        var weightData = new float[vocabSize * embedDim];
        for (int i = 0; i < weightData.Length; i++)
            weightData[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        // Measure one-hot + MatMul path
        long oneHotAlloc;
        double oneHotTime;
        using (GradientUtils.Grad())
        {
            var weightTensor = ReverseGradTensor<float>.FromMatrix(weightData, vocabSize, embedDim, requiresGrad: false);

            oneHotTime = MeasureBestOfFiveMs(() =>
            {
                var oneHotData = new float[numTokens * vocabSize];
                Array.Clear(oneHotData);
                for (int i = 0; i < numTokens; i++)
                    oneHotData[i * vocabSize + tokenIds[i]] = 1f;

                var oneHotTensor = ReverseGradTensor<float>.FromMatrix(oneHotData, numTokens, vocabSize, requiresGrad: false);
                var result = ReverseGradOperations.MatMul(oneHotTensor, weightTensor);
                _ = result.Data[0];
            });
            GC.Collect();
            GC.WaitForPendingFinalizers();
            var preAlloc = GC.GetAllocatedBytesForCurrentThread();
            {
                var oneHotData = new float[numTokens * vocabSize];
                Array.Clear(oneHotData);
                for (int i = 0; i < numTokens; i++)
                    oneHotData[i * vocabSize + tokenIds[i]] = 1f;

                var oneHotTensor = ReverseGradTensor<float>.FromMatrix(oneHotData, numTokens, vocabSize, requiresGrad: false);
                var result = ReverseGradOperations.MatMul(oneHotTensor, weightTensor);
                _ = result.Data[0];
            }
            oneHotAlloc = GC.GetAllocatedBytesForCurrentThread() - preAlloc;
        }

        // Measure Gather path
        long gatherAlloc;
        double gatherTime;
        using (GradientUtils.Grad())
        {
            var weightTensor = ReverseGradTensor<float>.FromMatrix(weightData, vocabSize, embedDim, requiresGrad: false);

            gatherTime = MeasureBestOfFiveMs(() =>
            {
                var result = ReverseGradOperations.Gather(weightTensor, tokenIds);
                _ = result.Data[0];
            });
            GC.Collect();
            GC.WaitForPendingFinalizers();
            var preAlloc = GC.GetAllocatedBytesForCurrentThread();
            {
                var result = ReverseGradOperations.Gather(weightTensor, tokenIds);
                _ = result.Data[0];
            }
            gatherAlloc = GC.GetAllocatedBytesForCurrentThread() - preAlloc;
        }

        double mbSaved = (oneHotAlloc - gatherAlloc) / (1024.0 * 1024.0);
        TestContext.Out.WriteLine(
            $"Embedding Gather ({numTokens} tokens, vocab={vocabSize}, dim={embedDim}):\n" +
            $"  One-hot+MatMul: {oneHotTime:F1}ms, {oneHotAlloc / (1024.0 * 1024.0):F1}MB allocated\n" +
            $"  Gather:         {gatherTime:F1}ms, {gatherAlloc / (1024.0 * 1024.0):F1}MB allocated\n" +
            $"  Speedup: {oneHotTime / gatherTime:F1}x, Memory saved: {mbSaved:F1}MB");
    }

    [Test]
    public void MiniLm_Inference_Latency()
    {
        string modelDir = Path.Combine(
            TestContext.CurrentContext.TestDirectory,
            "..", "..", "..", "..", "..",
            "samples", "data", "minilm");

        string safetensorsPath = Path.Combine(modelDir, "model.safetensors");
        string configPath = Path.Combine(modelDir, "config.json");
        string vocabPath = Path.Combine(modelDir, "vocab.txt");

        if (!File.Exists(safetensorsPath) || !File.Exists(configPath) || !File.Exists(vocabPath))
            Assert.Ignore("MiniLM weight files not found; skipping end-to-end latency benchmark.");

        Dictionary<string, (float[] Data, int[] Shape)> tensors;
        try
        {
            tensors = SafeTensorsLoader.Read(safetensorsPath);
        }
        catch (NotSupportedException ex)
        {
            Assert.Ignore($"Cannot load MiniLM weights: {ex.Message}");
            return;
        }

        var config = BertConfig.FromJson(File.ReadAllText(configPath));
        var tokenizer = MiniLMTokenizer.Load(vocabPath);

        using (GradientUtils.Grad())
        {
            var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
            var input = MiniLMTokenizer.Tokenize(tokenizer,
                "This is a sample sentence for benchmarking MiniLM inference latency.", maxLen: 128);

            for (int i = 0; i < 3; i++)
                model.Forward(input);

            var times = new double[5];
            for (int i = 0; i < 5; i++)
            {
                var sw = Stopwatch.StartNew();
                model.Forward(input);
                sw.Stop();
                times[i] = sw.ElapsedMilliseconds;
            }

            double avg = times.Average();
            double min = times.Min();
            double max = times.Max();

            TestContext.Out.WriteLine(
                $"MiniLM end-to-end latency (5 runs):\n" +
                $"  Average: {avg:F1}ms\n" +
                $"  Min:     {min}ms\n" +
                $"  Max:     {max}ms");

            var totalBytes = tensors.Sum(t => t.Value.Data.Length * 4L);
            TestContext.Out.WriteLine(
                $"  Model parameters: {tensors.Count} tensors, " +
                $"{totalBytes / (1024.0 * 1024.0):F1}MB total weight data");
        }
    }

    [Test]
    public void DistilBert_Inference_Latency()
    {
        string modelDir = Path.Combine(
            TestContext.CurrentContext.TestDirectory,
            "..", "..", "..", "..", "..",
            "samples", "data", "distilbert_sst");

        string safetensorsPath = Path.Combine(modelDir, "model.safetensors");
        string configPath = Path.Combine(modelDir, "config.json");
        string vocabPath = Path.Combine(modelDir, "vocab.txt");

        if (!File.Exists(safetensorsPath) || !File.Exists(configPath) || !File.Exists(vocabPath))
            Assert.Ignore("DistilBERT SST-2 weight files not found; skipping end-to-end latency benchmark.");

        Dictionary<string, (float[] Data, int[] Shape)> tensors;
        try
        {
            tensors = SafeTensorsLoader.Read(safetensorsPath);
        }
        catch (NotSupportedException ex)
        {
            Assert.Ignore($"Cannot load DistilBERT weights: {ex.Message}");
            return;
        }

        var config = DistilBertConfig.FromJson(File.ReadAllText(configPath)).ToBertConfig();
        var tokenizer = MiniLMTokenizer.Load(vocabPath);

        // Inference-default path: no Grad() scope, exercises the leaf fast paths.
        var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        model.LoadWeights(tensors);

        var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer,
            "This is a sample sentence for benchmarking DistilBERT SST-2 inference latency.", maxLen: 128);
        var inputIds = GradientUtils.Constant(tokenIds);
        var mask = GradientUtils.Constant(attnMask);

        for (int i = 0; i < 3; i++)
            model.Forward(inputIds, mask, 1, tokenIds.Length);

        var times = new double[5];
        for (int i = 0; i < 5; i++)
        {
            var sw = Stopwatch.StartNew();
            model.Forward(inputIds, mask, 1, tokenIds.Length);
            sw.Stop();
            times[i] = sw.ElapsedMilliseconds;
        }

        double avg = times.Average();
        double min = times.Min();
        double max = times.Max();

        TestContext.Out.WriteLine(
            $"DistilBERT SST-2 end-to-end latency (5 runs):\n" +
            $"  Average: {avg:F1}ms\n" +
            $"  Min:     {min}ms\n" +
            $"  Max:     {max}ms");

        var totalBytes = tensors.Sum(t => t.Value.Data.Length * 4L);
        TestContext.Out.WriteLine(
            $"  Model parameters: {tensors.Count} tensors, " +
            $"{totalBytes / (1024.0 * 1024.0):F1}MB total weight data");
    }

    static double MeasureBestOfFiveMs(Func<NivaraColumn<float>> func)
    {
        var best = double.MaxValue;
        var sw = new Stopwatch();
        for (int i = 0; i < 5; i++)
        {
            sw.Restart();
            var result = func();
            sw.Stop();
            _ = result.Length;
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    static double MeasureBestOfFiveMs(Action action)
    {
        var best = double.MaxValue;
        var sw = new Stopwatch();
        for (int i = 0; i < 5; i++)
        {
            sw.Restart();
            action();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }
}
