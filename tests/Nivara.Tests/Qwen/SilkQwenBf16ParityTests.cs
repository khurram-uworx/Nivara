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

    /// <summary>
    /// #1222 regression: the BF16 cached-decode path (issue #1219's crash site) must agree
    /// with the F32-upcast path on both the argmax and the logits, exercising
    /// <see cref="SilkQwenSession.ForwardCachedArgmax"/> at least once.
    /// </summary>
    [Test]
    public void ForwardCachedArgmax_Bf16Raw_SilkMatchesF32UpcastTokenStream()
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

        int f32Next = f32Session.ForwardPrefillArgmax(prompt);
        int bf16Next = bf16Session.ForwardPrefillArgmax(prompt);
        Assert.That(bf16Next, Is.EqualTo(f32Next),
            "prefill argmax must agree before the cached step can be compared");

        int position = prompt.Length;
        var f32Tokens = new List<int>();
        var bf16Tokens = new List<int>();
        for (int t = 0; t < 4; t++)
        {
            // Guard against runaway: break on stop id 0 (none defined for synthetic; cap at 4).
            int f32Tok = f32Session.ForwardCachedArgmax(f32Next, position);
            int bf16Tok = bf16Session.ForwardCachedArgmax(bf16Next, position);
            Assert.That(bf16Tok, Is.EqualTo(f32Tok),
                $"cached-decode argmax diverged at step {t}");
            f32Tokens.Add(f32Tok);
            bf16Tokens.Add(bf16Tok);
            f32Next = f32Tok;
            bf16Next = bf16Tok;
            position++;
        }

        // Logits parity on the first cached step (re-run from the prefill argmax).
        f32Session.ResetCache();
        bf16Session.ResetCache();
        _ = f32Session.ForwardPrefillArgmax(prompt);
        _ = bf16Session.ForwardPrefillArgmax(prompt);
        var f32Logits = f32Session.ForwardCached(f32Next, prompt.Length);
        var bf16Logits = bf16Session.ForwardCached(bf16Next, prompt.Length);

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
