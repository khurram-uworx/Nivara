using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the decode-shaped GEMM (#681): <c>gemm_gemv_add</c> must reproduce
/// <c>gemm_row4_add</c> with <c>aRows == 1</c> bit for bit on identical inputs,
/// including tail widths. The contraction and K-order are the same; only the
/// launch geometry differs, so a bit difference means the decode shape changed
/// the accumulation.
/// </summary>
public sealed class SilkGemmGemvAddTests
{
    const int ACols = 40;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk gemm_gemv_add verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    [Test]
    public void GemmGemvAdd_MatchesGemmRow4Add_BitForBit()
    {
        int[] widths = [25, 26, 27, 96];
        using var silk = Device();
        foreach (int bCols in widths)
        {
            float[] a = Randoms(ACols, 0x68100001u);
            float[] b = Randoms(ACols * bCols, 0x68100002u);
            float[] residual = Randoms(bCols, 0x68100003u);

            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sres = silk.Allocate(residual.Length);
            nint scTile = silk.Allocate(bCols);
            nint scFlat = silk.Allocate(bCols);
            silk.Upload(sa, a);
            silk.Upload(sb, b);
            silk.Upload(sres, residual);

            SilkLauncher.GemmAdd(silk, sa, sb, scTile, sres, 1, ACols, bCols);
            SilkLauncher.GemmGemvAdd(silk, sa, sb, scFlat, sres, ACols, bCols);
            silk.Finish();

            float[] tile = silk.Download(scTile, bCols);
            float[] flat = silk.Download(scFlat, bCols);

            for (int i = 0; i < bCols; i++)
            {
                int want = BitConverter.SingleToInt32Bits(tile[i]);
                int got = BitConverter.SingleToInt32Bits(flat[i]);
                Assert.That(got, Is.EqualTo(want), $"bCols={bCols} element {i} differs: tile 0x{want:X8}, flat 0x{got:X8}");
            }

            silk.Release(sa); silk.Release(sb); silk.Release(sres);
            silk.Release(scTile); silk.Release(scFlat);
        }
    }
}