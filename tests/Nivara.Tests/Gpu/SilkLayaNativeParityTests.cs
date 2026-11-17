using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public class SilkLayaNativeParityTests
{
    static Dictionary<string, (Half[] Data, int[] Shape)>? tensorsHalf;
    static ModernBertConfig? config;
    static int headLayers = 2;

    static void RequireCheckpoint()
    {
        string modelDir = Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..", "samples", "data", "laya");
        string modelPath = Path.Combine(modelDir, "model.safetensors");
        string encoderConfigPath = Path.Combine(modelDir, "encoder", "config.json");
        string agentConfigPath = Path.Combine(modelDir, "rl_agent_config.json");
        string[] missing = new[] { modelPath, encoderConfigPath, agentConfigPath }
            .Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
            Assert.Ignore($"Laya checkpoint absent. Missing: {string.Join(", ", missing)}");
    }

    [Test]
    public void Forward_F16Storage_SilkMatchesCpuNative()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        RequireCheckpoint();

        string modelDir = Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..", "samples", "data", "laya");
        string modelPath = Path.Combine(modelDir, "model.safetensors");
        string encoderConfigPath = Path.Combine(modelDir, "encoder", "config.json");
        string agentConfigPath = Path.Combine(modelDir, "rl_agent_config.json");

        if (tensorsHalf == null)
        {
            tensorsHalf = SafeTensorsLoader.Read<Half>(modelPath);
            config = ModernBertConfig.FromJson(File.ReadAllText(encoderConfigPath));
            using var doc = JsonDocument.Parse(File.ReadAllText(agentConfigPath));
            headLayers = doc.RootElement.GetProperty("head_layers").GetInt32();
        }

        var cfg = config!;
        var hl = headLayers;

        // CPU native reference (F16 storage, F32 compute).
        var cpuEncoder = ModernBertEncoder<float, Half>.LoadWeights(tensorsHalf!, cfg, "encoder");
        var cpuHead = LayaDecisionHead<float, Half>.LoadWeights(tensorsHalf!, hl);

        int[] ids = [0, 1, 2, 3];
        int valid = 4;
        int[] markers = [0];

        var cpuHidden = cpuEncoder.Forward(ids, valid);
        var cpuOut = cpuHead.Forward(cpuHidden, LayaQuestionType.Choice, markers, valid);

        // Silk device (F16 storage).
        using var silkSession = SilkLayaSession.Open(cfg, tensorsHalf!, hl);
        var silkOut = silkSession.Forward(ids, valid, LayaQuestionType.Choice, markers);

        // Argmax agreement.
        Assert.That(ArgMax(silkOut.Logits), Is.EqualTo(ArgMax(cpuOut.Logits)));

        // Bound: max|diff| <= 1e-3*(1+|ref|).
        float maxAbs = 0f;
        double maxScaled = 0.0;
        for (int i = 0; i < cpuOut.Logits.Length; i++)
        {
            float diff = MathF.Abs(silkOut.Logits[i] - cpuOut.Logits[i]);
            if (diff > maxAbs) maxAbs = diff;
            double scaled = diff / (1.0 + Math.Abs(cpuOut.Logits[i]));
            if (scaled > maxScaled) maxScaled = scaled;
        }
        Assert.That(maxScaled, Is.LessThanOrEqualTo(1e-3),
            $"max scaled residual {maxScaled:E3} exceeds 1e-3 bound (maxAbs={maxAbs:E3})");
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
