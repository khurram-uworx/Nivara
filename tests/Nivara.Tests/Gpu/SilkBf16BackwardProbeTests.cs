using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkBf16BackwardProbeTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    static ushort[] ToBf16(float[] f)
    {
        var r = new ushort[f.Length];
        for (int i = 0; i < f.Length; i++) r[i] = BitConverter.BFloat16ToUInt16Bits((BFloat16)f[i]);
        return r;
    }

    static float[] FromBf16(ushort[] u)
    {
        var r = new float[u.Length];
        for (int i = 0; i < u.Length; i++) r[i] = (float)BitConverter.UInt16BitsToBFloat16(u[i]);
        return r;
    }

    static void AssertBits(float[] actual, float[] expected, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), label);
        for (int i = 0; i < expected.Length; i++)
        {
            int want = BitConverter.SingleToInt32Bits(expected[i]);
            int got = BitConverter.SingleToInt32Bits(actual[i]);
            Assert.That(got, Is.EqualTo(want), $"{label}[{i}]: expected 0x{want:X8}, got 0x{got:X8}");
        }
    }

    [Test]
    public void GemmNtBf16_MatchesF32OnDecodedWeights()
    {
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (5, 2, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(m * k, 0x11330001);
            float[] b = Randoms(n * k, 0x11330002);
            var bf = ToBf16(b);
            var bdec = FromBf16(bf);

            nint sa = silk.Allocate(a.Length);
            nint sbf = silk.AllocateUshort(bf.Length);
            nint scBf = silk.Allocate(m * n);
            nint scF = silk.Allocate(m * n);
            nint sbF = silk.Allocate(bdec.Length);
            silk.Upload(sa, a);
            silk.Upload(sbf, bf);
            silk.Upload(sbF, bdec);

            SilkLauncher.GemmNtBf16(silk, sa, sbf, scBf, m, k, n);
            SilkLauncher.GemmNt(silk, sa, sbF, scF, m, k, n);
            silk.Finish();
            var gotBf = silk.Download(scBf, m * n);
            var gotF = silk.Download(scF, m * n);
            AssertBits(gotBf, gotF, $"gemm_row4_nt_bf16 m={m} k={k} n={n}");

            silk.Release(sa); silk.Release(sbf); silk.Release(scBf); silk.Release(scF); silk.Release(sbF);
        }
    }

    [Test]
    public void GemmTnBf16_MatchesF32OnDecodedWeights()
    {
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (2, 5, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(k * m, 0x11330011);
            float[] b = Randoms(k * n, 0x11330012);
            var bf = ToBf16(b);
            var bdec = FromBf16(bf);

            nint sa = silk.Allocate(a.Length);
            nint sbf = silk.AllocateUshort(bf.Length);
            nint scBf = silk.Allocate(m * n);
            nint scF = silk.Allocate(m * n);
            nint sbF = silk.Allocate(bdec.Length);
            silk.Upload(sa, a);
            silk.Upload(sbf, bf);
            silk.Upload(sbF, bdec);

            SilkLauncher.GemmTnBf16(silk, sa, sbf, scBf, m, k, n);
            SilkLauncher.GemmTn(silk, sa, sbF, scF, m, k, n);
            silk.Finish();
            var gotBf = silk.Download(scBf, m * n);
            var gotF = silk.Download(scF, m * n);
            AssertBits(gotBf, gotF, $"gemm_row4_tn_bf16 m={m} k={k} n={n}");

            silk.Release(sa); silk.Release(sbf); silk.Release(scBf); silk.Release(scF); silk.Release(sbF);
        }
    }

    [Test]
    public void GemmNtResidualBf16_MatchesF32OnDecodedWeights()
    {
        (int m, int k, int n)[] shapes = [(17, 33, 9), (16, 32, 64), (1, 1, 1), (5, 2, 2)];
        using var silk = Device();
        foreach (var (m, k, n) in shapes)
        {
            float[] a = Randoms(m * k, 0x113301D1);
            float[] b = Randoms(n * k, 0x113301D2);
            float[] residual = Randoms(m * n, 0x113301D3);
            var bf = ToBf16(b);
            var bdec = FromBf16(bf);

            nint sa = silk.Allocate(a.Length);
            nint sbf = silk.AllocateUshort(bf.Length);
            nint scBf = silk.Allocate(m * n);
            nint scF = silk.Allocate(m * n);
            nint sbF = silk.Allocate(bdec.Length);
            nint sr = silk.Allocate(residual.Length);
            silk.Upload(sa, a);
            silk.Upload(sbf, bf);
            silk.Upload(sbF, bdec);
            silk.Upload(sr, residual);

            SilkLauncher.GemmNtResidualBf16(silk, sa, sbf, scBf, sr, m, k, n);
            // F32 residual variant
            SilkLauncher.GemmNtResidual(silk, sa, sbF, scF, sr, m, k, n);
            silk.Finish();
            var gotBf = silk.Download(scBf, m * n);
            var gotF = silk.Download(scF, m * n);
            AssertBits(gotBf, gotF, $"gemm_row4_nt_residual_bf16 m={m} k={k} n={n}");

            silk.Release(sa); silk.Release(sbf); silk.Release(scBf); silk.Release(scF); silk.Release(sbF); silk.Release(sr);
        }
    }
}
