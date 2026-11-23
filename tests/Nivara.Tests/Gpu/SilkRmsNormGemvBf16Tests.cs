using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity for the BF16 <c>rms_norm_gemv_bf16</c> kernel (issue #1466): output must match a
/// C# reference that applies the same BF16 widening (<c>BF16_TO_F32</c>) in the same ascending-k
/// accumulation order. Bit-identical, including tail widths and hasBias/no-bias.
/// </summary>
[Category("Gpu")]
public sealed class SilkRmsNormGemvBf16Tests
{
    const float Eps = 1e-5f;
    const int ACols = 40;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk rms_norm_gemv_bf16 verification."); throw; }
    }

    static ushort ToBf16(float f) => (ushort)(BitConverter.SingleToInt32Bits(f) >> 16);

    static float FromBf16(ushort b) => BitConverter.Int32BitsToSingle(((int)b) << 16);

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return v;
    }

    [Test]
    public void RmsNormGemvBf16_BiasFree_MatchesReference()
    {
        RunCase(hasBias: false, cols: 96);
        RunCase(hasBias: false, cols: 27);
    }

    [Test]
    public void RmsNormGemvBf16_Biased_MatchesReference()
    {
        RunCase(hasBias: true, cols: 96);
        RunCase(hasBias: true, cols: 25);
    }

    static void RunCase(bool hasBias, int cols)
    {
        using var silk = Device();
        var rng = new Random(unchecked((int)(hasBias ? 0xBF160002u : 0xBF160001u)));
        float[] x = Randoms(ACols, (uint)rng.Next());
        float[] gamma = Randoms(ACols, (uint)rng.Next());
        float[] wFloat = Randoms(ACols * cols, (uint)rng.Next());
        ushort[] w = new ushort[wFloat.Length];
        for (int i = 0; i < w.Length; i++) w[i] = ToBf16(wFloat[i]);
        float[] bias = hasBias ? Randoms(cols, (uint)rng.Next()) : [];

        // reference (C#)
        float ssq = 0f;
        for (int k = 0; k < ACols; k++) ssq += x[k] * x[k];
        float normScale = 1f / MathF.Sqrt(ssq / ACols + Eps);
        float[] expected = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            float acc = 0f;
            for (int k = 0; k < ACols; k++)
            {
                float av = x[k] * normScale * gamma[k];
                acc += av * FromBf16(w[k * cols + j]);
            }
            expected[j] = hasBias ? acc + bias[j] : acc;
        }

        // device
        nint sx = silk.Allocate(x.Length); silk.Upload(sx, x);
        nint sg = silk.Allocate(gamma.Length); silk.Upload(sg, gamma);
        nint sw = silk.Allocate(w.Length); silk.Upload(sw, w);
        nint sbias = hasBias ? silk.Allocate(bias.Length) : 0;
        if (hasBias) silk.Upload(sbias, bias);
        nint sout = silk.Allocate(cols);
        var slot = new SilkLauncher.GemvSlot(sw, hasBias ? sbias : 0, sout, cols);
        SilkLauncher.RmsNormGemvBf16(silk, sx, sg, [slot], hasBias, aRows: 1, aCols: ACols, eps: Eps);
        silk.Finish();
        float[] actual = silk.Download(sout, cols);

        for (int j = 0; j < cols; j++)
        {
            int want = BitConverter.SingleToInt32Bits(expected[j]);
            int got = BitConverter.SingleToInt32Bits(actual[j]);
            Assert.That(got, Is.EqualTo(want), $"cols={cols} hasBias={hasBias} element {j}: expected 0x{want:X8}, got 0x{got:X8}");
        }

        silk.Release(sx); silk.Release(sg); silk.Release(sw);
        if (hasBias) silk.Release(sbias);
        silk.Release(sout);
    }
}
