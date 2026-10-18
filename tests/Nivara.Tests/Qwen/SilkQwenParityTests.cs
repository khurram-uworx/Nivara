using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
public class SilkQwenParityTests
{
    [Test]
    public void ForwardPrefill_SyntheticWeights_SilkMatchesCpuArgmaxAndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = SilkQwenTestWeights.Synthesize();
        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);

        int[] prompt = [1, 2, 3, 4];
        var cpuLogitsTensor = cpuModel.Forward(prompt);
        int seqLen = prompt.Length;
        var cpuLast = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
            cpuLast[i] = cpuLogitsTensor.Data[(seqLen - 1) * config.VocabSize + i];

        int cpuArgmax = ArgMax(cpuLast);

        using var session = SilkQwenSession.Open(
            SilkQwenTestWeights.ToSilkConfig(config, 16), tensors);
        var silkLast = session.ForwardPrefill(prompt);
        int silkArgmax = ArgMax(silkLast);

        Assert.That(silkArgmax, Is.EqualTo(cpuArgmax));

        float maxAbs = 0f;
        for (int i = 0; i < config.VocabSize; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuLast[i] - silkLast[i]));
        float scale = 1f + cpuLast.Max(v => MathF.Abs(v));
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3f * scale));
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
