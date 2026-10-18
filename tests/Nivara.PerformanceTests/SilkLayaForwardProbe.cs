using System.Text.Json;
using Nivara.AI;
using Nivara.Silk;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// <c>--silk-laya-forward</c>: dispatch guard only for now. A full profile needs the Laya checkpoint
/// and uses the same forward contract as `tests/Nivara.PerformanceTests/LayaForwardProbe`, but
/// through `SilkLayaSession` (the in-process TypeSafe -> Nivara route). When run without weights it
/// exits 0 after printing the missing-checkpoint condition. It is evidence for one kernel family at
/// a time only if all structural guards pass.
/// </summary>
internal static class SilkLayaForwardProbe
{
    static readonly int[] ValidLengths = [40, 119, 48, 512];
    const int Warmup = 1;
    const int Iters = 3;

    public static int Run()
    {
        Console.WriteLine("=== Silk Laya forward profile (GPU, Silk, F32) ===");
        if (!TryResolveCheckpoint(out string modelPath, out string configPath, out string agentPath))
        {
            Console.WriteLine("Checkpoint absent. Re-run from the repository root on a machine that has");
            Console.WriteLine("samples/data/laya/model.safetensors and samples/data/laya/encoder/config.json.");
            return 0;
        }

        using var session = OpenSession(modelPath, configPath, agentPath);
        Console.WriteLine($"Device: {session.DeviceName}");
        Console.WriteLine($"1 warmup + {Iters} timed passes per sequence length, median reported.");
        Console.WriteLine();

        long voids = 0;
        foreach (int valid in ValidLengths)
        {
            int[] ids = BuildIds(valid, 50368, 1);
            int[] markers = [Math.Min(10, valid - 1), Math.Min(20, valid - 1)];
            for (int i = 0; i < Warmup; i++)
            {
                var outp = session.Forward(ids, valid, LayaQuestionType.Choice, markers);
                if (outp.Logits.Length == 0) voids++;
            }
            var samples = new double[Iters];
            for (int i = 0; i < Iters; i++)
            {
                var sw = Stopwatch.StartNew();
                var outp = session.Forward(ids, valid, LayaQuestionType.Choice, markers);
                sw.Stop();
                samples[i] = sw.Elapsed.TotalMilliseconds;
                if (!float.IsFinite(outp.Logits.FirstOrDefault())) voids++;
            }
            Array.Sort(samples);
            double median = samples.Length % 2 == 1 ? samples[samples.Length / 2]
                : (samples[samples.Length / 2 - 1] + samples[samples.Length / 2]) / 2.0;
            Console.WriteLine($"  valid={valid,-3}  median {median,8:F0} ms   min {samples[0],8:F0}   max {samples[^1],8:F0}");
        }

        Console.WriteLine();
        Console.WriteLine(voids == 0 ? "Done." : $"*** {voids} structural void(s).");
        return voids == 0 ? 0 : 1;
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
