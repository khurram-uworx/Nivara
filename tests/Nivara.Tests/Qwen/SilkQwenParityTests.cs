using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
[Category("Gpu")]
public class SilkQwenParityTests
{
    [Test]
    public void ForwardPrefill_SyntheticWeights_SilkMatchesCpuArgmaxAndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = QwenTestWeights.Synthesize();
        var cpuModel = LlamaLoader.Load<float, float>(config, tensors);

        int[] prompt = [1, 2, 3, 4];
        var cpuLogitsTensor = cpuModel.Forward(prompt);
        int seqLen = prompt.Length;
        var cpuLast = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
            cpuLast[i] = cpuLogitsTensor.Data[(seqLen - 1) * config.VocabSize + i];

        int cpuArgmax = ArgMax(cpuLast);

        using var session = SilkQwenSession.Open(
            QwenTestWeights.ToSilkConfig(config, 16), tensors);
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

        var (tensors, config) = QwenTestWeights.Synthesize();
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
            QwenTestWeights.ToSilkConfig(config, 16), tensors);
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

    [Test]
    public void GreedyDecode_SyntheticWeights_CpuAndSilkMatchAndReportTiming()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = QwenTestWeights.Synthesize();
        int[] prompt = [1, 2, 3, 4];
        const int maxNewTokens = 4;

        var cpu = GreedyDecodeCpu(config, tensors, prompt, maxNewTokens);
        using var session = SilkQwenSession.Open(QwenTestWeights.ToSilkConfig(config, 16), tensors);
        var silk = GreedyDecodeSilk(session, prompt, maxNewTokens, config.EosTokenId);

        Assert.That(silk.GeneratedIds, Is.EqualTo(cpu.GeneratedIds));
        Assert.That(cpu.LastLogits.All(float.IsFinite), Is.True);
        Assert.That(silk.LastLogits.All(float.IsFinite), Is.True);

        double cpuDecodeTokS = cpu.DecodeMs > 0 ? cpu.GeneratedIds.Count * 1000.0 / cpu.DecodeMs : 0;
        double silkDecodeTokS = silk.DecodeMs > 0 ? silk.GeneratedIds.Count * 1000.0 / silk.DecodeMs : 0;
        TestContext.Progress.WriteLine(
            $"Qwen synthetic CPU/Silk: prompt={prompt.Length}, generated={cpu.GeneratedIds.Count}, " +
            $"cpu prefill={cpu.PrefillMs} ms, cpu decode={cpu.DecodeMs} ms ({cpuDecodeTokS:F2} tok/s), " +
            $"silk prefill={silk.PrefillMs} ms, silk decode={silk.DecodeMs} ms ({silkDecodeTokS:F2} tok/s)");
    }

    static (List<int> GeneratedIds, float[] LastLogits, long PrefillMs, long DecodeMs) GreedyDecodeCpu(
        LlamaConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int[] prompt,
        int maxNewTokens)
    {
        var model = LlamaLoader.Load<float, float>(config, tensors);
        int kvWidth = config.NumKeyValueHeads * (config.HiddenSize / config.NumAttentionHeads);
        using var cache = new LlamaKVCache<float>(config.NumHiddenLayers, kvWidth);

        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        var prefill = model.ForwardPrefill(prompt, cache);
        prefillSw.Stop();
        var last = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++) last[i] = prefill.Data[i];
        int next = ArgMax(last);

        var generated = new List<int>(maxNewTokens);
        int position = prompt.Length;
        var decodeSw = System.Diagnostics.Stopwatch.StartNew();
        for (int t = 0; t < maxNewTokens; t++)
        {
            generated.Add(next);
            if (next == config.EosTokenId)
                break;
            var step = model.ForwardCached(next, position++, cache);
            for (int i = 0; i < config.VocabSize; i++) last[i] = step.Data[i];
            next = ArgMax(last);
        }
        decodeSw.Stop();
        return (generated, last, prefillSw.ElapsedMilliseconds, decodeSw.ElapsedMilliseconds);
    }

    static (List<int> GeneratedIds, float[] LastLogits, long PrefillMs, long DecodeMs) GreedyDecodeSilk(
        SilkQwenSession session,
        int[] prompt,
        int maxNewTokens,
        int eosTokenId)
    {
        session.ResetCache();
        var prefillSw = System.Diagnostics.Stopwatch.StartNew();
        var last = session.ForwardPrefill(prompt);
        prefillSw.Stop();
        int next = ArgMax(last);

        var generated = new List<int>(maxNewTokens);
        int position = prompt.Length;
        var decodeSw = System.Diagnostics.Stopwatch.StartNew();
        for (int t = 0; t < maxNewTokens; t++)
        {
            generated.Add(next);
            if (next == eosTokenId)
                break;
            last = session.ForwardCached(next, position++);
            next = ArgMax(last);
        }
        decodeSw.Stop();
        return (generated, last, prefillSw.ElapsedMilliseconds, decodeSw.ElapsedMilliseconds);
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
