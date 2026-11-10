using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

[TestFixture]
[Category("Gpu")]
public class SilkQwenBf16ParityTests
{
    [Test]
    public void ForwardPrefill_Bf16Raw_SilkMatchesF32UpcastArgmaxAndLogits()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = QwenTestWeights.Synthesize();

        var bf16Tensors = QwenTestWeights.ToBFloat16(tensors);

        int[] prompt = [1, 2, 3, 4];

        using var f32Session = SilkQwenSession.Open(
            QwenTestWeights.ToSilkConfig(config, 16), tensors);
        using var bf16Session = SilkQwenSession.Open(
            QwenTestWeights.ToSilkConfig(config, 16), bf16Tensors);

        var f32Logits = f32Session.ForwardPrefill(prompt);
        var bf16Logits = bf16Session.ForwardPrefill(prompt);

        Assert.That(ArgMax(bf16Logits), Is.EqualTo(ArgMax(f32Logits)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < f32Logits.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(f32Logits[i] - bf16Logits[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(f32Logits[i]));
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
