using Nivara.AI;
using Nivara.Silk;
using System.Diagnostics;
using System.Text.Json;

namespace Nivara.PerformanceTests;

internal static class SilkEncoderProfileProbe
{
    static readonly int[] ValidLengths = [40, 119, 48, 512];
    const int Warmup = 1;
    const int Iters = 3;

    public static int Run()
    {
        Console.WriteLine("=== Silk ModernBERT encoder profile (GPU, Silk, F32) ===");
        if (!TryResolveCheckpoint(out string modelPath, out string configPath, out string agentPath))
        {
            Console.WriteLine("Checkpoint absent.");
            return 0;
        }

        using var session = OpenSession(modelPath, configPath, agentPath);
        session.Encoder.ProfilingEnabled = true;
        Console.WriteLine($"Device: {session.DeviceName}");
        Console.WriteLine();

        foreach (int valid in ValidLengths)
        {
            int seqLen = Math.Max(valid, 64);
            int[] ids = BuildIds(seqLen, 50368, 1);
            // ensure validLength <= seqLen
            int vl = Math.Min(valid, seqLen);
            for (int i = 0; i < Warmup; i++)
                session.Forward(ids, vl, LayaQuestionType.Choice, [Math.Min(10, vl - 1)]);

            var samples = new double[Iters];
            Dictionary<string, long> last = new();
            for (int i = 0; i < Iters; i++)
            {
                var sw = Stopwatch.StartNew();
                session.Forward(ids, vl, LayaQuestionType.Choice, [Math.Min(10, vl - 1)]);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMilliseconds;
                last = new Dictionary<string, long>(session.Encoder.LastPhaseMs);
            }
            Array.Sort(samples);
            double median = samples[samples.Length / 2];
            Console.WriteLine($"seqLen={seqLen,-3} valid={vl,-3}  total median {median,7:F0} ms  (encode {session.LastEncodeMilliseconds} ms)");
            foreach (var kv in last.OrderBy(x => x.Key))
                Console.WriteLine($"     {kv.Key,-10} {kv.Value,6} ms");
            Console.WriteLine();
        }
        return 0;
    }

    static SilkLayaSession OpenSession(string modelPath, string configPath, string agentPath)
    {
        var tensors = SafeTensorsLoader.Read(modelPath);
        var config = ModernBertConfig.FromJson(File.ReadAllText(configPath));
        int headLayers = 2;
        if (File.Exists(agentPath))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(agentPath));
            if (doc.RootElement.TryGetProperty("head_layers", out var hl)) headLayers = hl.GetInt32();
        }
        return SilkLayaSession.Open(config, tensors, headLayers);
    }

    static bool TryResolveCheckpoint(out string modelPath, out string configPath, out string agentPath)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
        {
            string model = Path.Combine(dir.FullName, "samples", "data", "laya", "model.safetensors");
            string config = Path.Combine(dir.FullName, "samples", "data", "laya", "encoder", "config.json");
            string agent = Path.Combine(dir.FullName, "samples", "data", "laya", "rl_agent_config.json");
            if (File.Exists(model) && File.Exists(config))
            {
                modelPath = model; configPath = config; agentPath = agent; return true;
            }
        }
        modelPath = ""; configPath = ""; agentPath = "";
        return false;
    }

    static int[] BuildIds(int length, int vocab, int padId)
    {
        var rng = new Random(42);
        var ids = new int[length];
        for (int i = 0; i < length; i++) ids[i] = rng.Next(1, vocab);
        ids[^1] = padId;
        return ids;
    }
}
