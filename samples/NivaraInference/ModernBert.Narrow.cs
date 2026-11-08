using System.Diagnostics;
using System.Numerics;

namespace NivaraInference;

public static partial class ModernBert
{
    public static int RunGpu(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var config = LoadConfig(modelDir);
        var tokenizer = LoadTokenizer(modelDir, config);

        Console.WriteLine($"=== {ModelTypeName} Inference (GPU) ===");
        PrintConfig(config);
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        using var session = Nivara.Silk.SilkModernBertSession.Open(config, tensors);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}, precision BFloat16");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Parameters: {tensors.Values.Sum(t => t.Data.Length):N0} " +
                          $"({tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0):F1} MB as BFloat16)");
        Console.WriteLine();

        Console.WriteLine($"Sentences ({SampleSentences.Length}), last_hidden_state stats per run:");
        Console.WriteLine();
        for (int i = 0; i < SampleSentences.Length; i++)
        {
            var ids = Encode(tokenizer, SampleSentences[i]);
            int validLength = ids.Length;

            var sw = Stopwatch.StartNew();
            var hidden = session.Forward(ids, validLength);
            sw.Stop();

            var stats = SpanStats(hidden.AsSpan(0, validLength * config.HiddenSize));
            Console.WriteLine(
                $"  [{i}] {stats.Min,9:F3} {stats.Max,9:F3} {stats.Mean,8:F4} {stats.StdDev,8:F4}  " +
                $"{sw.ElapsedMilliseconds,5} ms  {validLength,3} tok  {Truncate(SampleSentences[i], 58)}");
        }

        Console.WriteLine();
        Console.WriteLine("The encoder emits no logits — it is the trunk a masked-LM or embedding model " +
                          "sits on. Run 'modernbert --silk compare' for the GPU-vs-CPU parity gate.");
        return 0;
    }

    public static int RunGpu(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var config = LoadConfig(modelDir);
        var tokenizer = LoadTokenizer(modelDir, config);

        Console.WriteLine($"=== {ModelTypeName} Inference (GPU) ===");
        PrintConfig(config);
        Console.WriteLine();

        var buildSw = Stopwatch.StartNew();
        using var session = Nivara.Silk.SilkModernBertSession.Open(config, tensors);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}, precision Half");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"Parameters: {tensors.Values.Sum(t => t.Data.Length):N0} " +
                          $"({tensors.Values.Sum(t => t.Data.Length) * 2.0 / (1024.0 * 1024.0):F1} MB as Half)");
        Console.WriteLine();

        Console.WriteLine($"Sentences ({SampleSentences.Length}), last_hidden_state stats per run:");
        Console.WriteLine();
        for (int i = 0; i < SampleSentences.Length; i++)
        {
            var ids = Encode(tokenizer, SampleSentences[i]);
            int validLength = ids.Length;

            var sw = Stopwatch.StartNew();
            var hidden = session.Forward(ids, validLength);
            sw.Stop();

            var stats = SpanStats(hidden.AsSpan(0, validLength * config.HiddenSize));
            Console.WriteLine(
                $"  [{i}] {stats.Min,9:F3} {stats.Max,9:F3} {stats.Mean,8:F4} {stats.StdDev,8:F4}  " +
                $"{sw.ElapsedMilliseconds,5} ms  {validLength,3} tok  {Truncate(SampleSentences[i], 58)}");
        }

        Console.WriteLine();
        Console.WriteLine("The encoder emits no logits — it is the trunk a masked-LM or embedding model " +
                          "sits on. Run 'modernbert --silk compare' for the GPU-vs-CPU parity gate.");
        return 0;
    }
}
