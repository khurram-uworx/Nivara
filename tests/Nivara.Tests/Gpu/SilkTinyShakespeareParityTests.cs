using Nivara.AutoDiff;
using Nivara.Silk;
using NivaraChat.Transformer;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// CPU (<see cref="BatchedTransformer{T}"/>) vs Silk/OpenCL parity for the
/// TinyShakespeare batched transformer (#1314). Mirrors the Qwen/SmolLM parity
/// test style: top-1 token equality plus logits closeness within 1e-3.
/// </summary>
[TestFixture]
[Category("Gpu")]
public class SilkTinyShakespeareParityTests
{
    const int Vocab = 32;
    const int NEmbd = 16;
    const int NLayer = 1;
    const int NHead = 2;
    const int Block = 8;

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    static float[] ToFloats(ReverseGradTensor<float> tensor)
    {
        tensor.Data.TryGetSpan(out var span);
        int vocab = tensor.Shape.Length > 0 ? tensor.Shape[^1] : span.Length;
        var result = new float[vocab];
        if (span.Length >= vocab)
        {
            for (int i = 0; i < vocab; i++) result[i] = span[i];
        }
        else
        {
            for (int i = 0; i < vocab; i++) result[i] = float.CreateChecked(tensor[i]);
        }
        return result;
    }

    static void AssertLogitsClose(float[] cpu, float[] silk)
    {
        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < cpu.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpu[i] - silk[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(cpu[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3f * scale));
    }

    [Test]
    public void ForwardPrefill_RandomWeights_SilkMatchesCpuTop1AndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        using var model = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        int[] prompt = [1, 2, 3, 4];

        var cpuTensor = model.ForwardPrefill(prompt);
        var cpuLogits = ToFloats(cpuTensor);
        int cpuNext = ArgMax(cpuLogits);

        var weights = model.ToSilkWeights();
        var config = new TinyShakespeareConfig
        {
            HiddenSize = NEmbd,
            NumHiddenLayers = NLayer,
            NumAttentionHeads = NHead,
            VocabSize = Vocab,
            MaxSeqLen = Block,
        };
        using var session = SilkTinyShakespeareSession.Open(config, weights);
        var silkLogits = session.ForwardPrefill(prompt);
        int silkNext = ArgMax(silkLogits);

        Assert.That(silkNext, Is.EqualTo(cpuNext));
        AssertLogitsClose(cpuLogits, silkLogits);
    }

    [Test]
    public void ForwardCached_RandomWeights_SilkMatchesCpuTop1AndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        using var model = new BatchedTransformer<float>(Vocab, NEmbd, NLayer, NHead, Block);
        int[] prompt = [1, 2, 3, 4];

        var cpuPrefillTensor = model.ForwardPrefill(prompt);
        var cpuPrefillLogits = ToFloats(cpuPrefillTensor);
        int cpuNext = ArgMax(cpuPrefillLogits);

        var weights = model.ToSilkWeights();
        var config = new TinyShakespeareConfig
        {
            HiddenSize = NEmbd,
            NumHiddenLayers = NLayer,
            NumAttentionHeads = NHead,
            VocabSize = Vocab,
            MaxSeqLen = Block,
        };
        using var session = SilkTinyShakespeareSession.Open(config, weights);
        var silkPrefillLogits = session.ForwardPrefill(prompt);
        int silkNext = ArgMax(silkPrefillLogits);

        Assert.That(silkNext, Is.EqualTo(cpuNext), "prefill top-1 must agree before cached steps");

        int position = prompt.Length;
        int cpuToken = cpuNext;
        int silkToken = silkNext;
        for (int step = 0; step < 3; step++)
        {
            var cpuStepTensor = model.ForwardCached(cpuToken, position);
            var cpuStepLogits = ToFloats(cpuStepTensor);
            int cpuStepNext = ArgMax(cpuStepLogits);

            var silkStepLogits = session.ForwardCached(silkToken, position);
            int silkStepNext = ArgMax(silkStepLogits);

            Assert.That(silkStepNext, Is.EqualTo(cpuStepNext), $"cached step {step} top-1");
            AssertLogitsClose(cpuStepLogits, silkStepLogits);

            cpuToken = cpuStepNext;
            silkToken = silkStepNext;
            position++;
        }
    }
}