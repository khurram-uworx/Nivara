using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.SmolLM;

/// <summary>
/// #1232 regression: the BF16 cached-decode path (issue #1219's crash site —
/// <c>CL_OUT_OF_RESOURCES</c> in <c>ArgMaxDevice</c> during
/// <see cref="SilkSmolLMSession.ForwardCachedArgmax"/>) must agree with the
/// F32-upcast path on both the argmax and the logits. This is the SmolLM-side
/// counterpart of <see cref="Nivara.Tests.Qwen.SilkQwenBf16ParityTests"/>.
/// </summary>
/// <remarks>
/// The crash happened because the tied LM head buffer (<c>tokEmbTransposed</c>)
/// is the one BF16 buffer allocated tightly (<c>AllocateBytes(len * sizeof(ushort))</c>);
/// the F32 <c>RmsNormGemv</c> kernel was being handed that buffer and read it as
/// <c>float*</c>, i.e. 2x past the end. Layer BF16 weights are allocated 2x
/// oversized (<c>Allocate(len)</c>), which is why the layer path stayed silent.
/// PR #1231 fixed both call sites by routing through the <c>RmsNormGemv</c>
/// dispatcher; this test pins that routing at the session level. The fixture
/// mirrors <see cref="Nivara.Tests.Qwen.SilkQwenBf16ParityTests"/> (same prompt/decode shape)
/// but keeps the real SmolLM-135M widths (hidden 576 / 9 heads / 3 kv heads /
/// inter 1536) so the non-whole-group gemv tail path is exercised.
/// </remarks>
[TestFixture]
[Category("Gpu")]
public class SilkSmolLMBf16ParityTests
{
    [Test]
    public void ForwardCachedArgmax_Bf16Raw_SilkMatchesF32UpcastTokenStream()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var (tensors, config) = SmolLMTestWeights.Synthesize();
        var bf16Tensors = SmolLMTestWeights.ToBFloat16(tensors);

        int[] prompt = [1, 2, 3, 4];

        using var f32Session = SilkSmolLMSession.Open(config, tensors);
        using var bf16Session = SilkSmolLMSession.Open(config, bf16Tensors);

        int f32Next = f32Session.ForwardPrefillArgmax(prompt);
        int bf16Next = bf16Session.ForwardPrefillArgmax(prompt);
        Assert.That(bf16Next, Is.EqualTo(f32Next),
            "prefill argmax must agree before the cached step can be compared");

        int position = prompt.Length;
        for (int t = 0; t < 4; t++)
        {
            int f32Tok = f32Session.ForwardCachedArgmax(f32Next, position);
            int bf16Tok = bf16Session.ForwardCachedArgmax(bf16Next, position);
            Assert.That(bf16Tok, Is.EqualTo(f32Tok),
                $"cached-decode argmax diverged at step {t}");
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
}
