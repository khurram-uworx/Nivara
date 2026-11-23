using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity for the BF16 <c>gemm_gemv_add_bf16</c> kernel (issue #1466): output must match a
/// C# reference applying the same BF16 widening in ascending-k accumulation order. Bit-identical,
/// including tail widths.
/// </summary>
[Category("Gpu")]
public sealed class SilkGemmGemvAddBf16Tests
{
    const int ACols = 40;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk gemm_gemv_add_bf16 verification."); throw; }
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
    public void GemmGemvAddBf16_MatchesReference()
    {
        foreach (int bCols in new[] { 25, 26, 27, 96 })
        {
            using var silk = Device();
            float[] a = Randoms(ACols, 0x68100001u);
            float[] bFloat = Randoms(ACols * bCols, 0x68100002u);
            ushort[] b = new ushort[bFloat.Length];
            for (int i = 0; i < b.Length; i++) b[i] = ToBf16(bFloat[i]);
            float[] residual = Randoms(bCols, 0x68100003u);

            float[] expected = new float[bCols];
            for (int j = 0; j < bCols; j++)
            {
                float acc = 0f;
                for (int k = 0; k < ACols; k++) acc += a[k] * FromBf16(b[k * bCols + j]);
                expected[j] = acc + residual[j];
            }

            nint sa = silk.Allocate(a.Length); silk.Upload(sa, a);
            nint sb = silk.Allocate(b.Length); silk.Upload(sb, b);
            nint sres = silk.Allocate(residual.Length); silk.Upload(sres, residual);
            nint sout = silk.Allocate(bCols);
            SilkLauncher.GemmGemvAddBf16(silk, sa, sb, sout, sres, ACols, bCols);
            silk.Finish();
            float[] actual = silk.Download(sout, bCols);

            for (int j = 0; j < bCols; j++)
            {
                int want = BitConverter.SingleToInt32Bits(expected[j]);
                int got = BitConverter.SingleToInt32Bits(actual[j]);
                Assert.That(got, Is.EqualTo(want), $"bCols={bCols} element {j}: expected 0x{want:X8}, got 0x{got:X8}");
            }

            silk.Release(sa); silk.Release(sb); silk.Release(sres); silk.Release(sout);
        }
    }
}
