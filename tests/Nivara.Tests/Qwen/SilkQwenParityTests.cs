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

    /// <summary>
    /// The teacher loop is prefill plus one cached step per token, and only the prefill half was
    /// pinned. A divergence here would silently change teacher labels, so this asserts the decoded
    /// step that produces every token after the first.
    /// </summary>
    [Test]
    public void ForwardCached_SyntheticWeights_SilkMatchesCpuArgmaxAndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = SilkQwenTestWeights.Synthesize();
        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);

        int[] prompt = [1, 2, 3, 4];
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        var cpuPrefill = cpuModel.ForwardPrefill(prompt, cache);
        var prefillLast = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
            prefillLast[i] = cpuPrefill.Data[i];
        int next = ArgMax(prefillLast);

        var cpuStep = cpuModel.ForwardCached(next, prompt.Length, cache);
        var cpuStepLast = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
            cpuStepLast[i] = cpuStep.Data[i];

        using var session = SilkQwenSession.Open(
            SilkQwenTestWeights.ToSilkConfig(config, 16), tensors);
        var silkPrefill = session.ForwardPrefill(prompt);
        Assert.That(ArgMax(silkPrefill), Is.EqualTo(next),
            "prefill argmax must agree before the cached step can be compared");

        var silkStep = session.ForwardCached(next, prompt.Length);
        Assert.That(ArgMax(silkStep), Is.EqualTo(ArgMax(cpuStepLast)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < config.VocabSize; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuStepLast[i] - silkStep[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(cpuStepLast[i]));
        }
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
