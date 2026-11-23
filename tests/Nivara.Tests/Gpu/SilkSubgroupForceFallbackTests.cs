using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Verifies the force-fallback hook (#1484): on a subgroup-capable device, constructing
/// <see cref="SilkRuntime"/> with subgroups disabled must report no support and make the
/// subgroup launchers fall back to the base kernels (bit-identical for gemm).
/// </summary>
[Category("Gpu")]
public sealed class SilkSubgroupForceFallbackTests
{
    const int ACols = 40;

    [Test]
    public void ForceDisabled_ReportsNoSupportAndKernelsAbsent()
    {
        using var silk = new SilkRuntime(enableSubgroups: false);
        Assert.That(silk.SupportsSubgroups, Is.False);
        Assert.That(silk.HasKernel(SilkKernels.GemmGemvAddSubgroup), Is.False);
        Assert.That(silk.HasKernel(SilkKernels.RmsNormGemvSubgroup), Is.False);
    }

    [Test]
    public void GemmGemvAddSubgroup_FallsBack_BitForBit()
    {
        using var silk = new SilkRuntime(enableSubgroups: false);
        var rng = new Random(0x14840001);
        float[] a = Randoms(ACols, rng);
        float[] b = Randoms(ACols * 96, rng);
        float[] residual = Randoms(96, rng);

        nint sa = silk.Allocate(a.Length);
        nint sb = silk.Allocate(b.Length);
        nint sres = silk.Allocate(residual.Length);
        nint sBase = silk.Allocate(96);
        nint sSub = silk.Allocate(96);
        silk.Upload(sa, a); silk.Upload(sb, b); silk.Upload(sres, residual);

        SilkLauncher.GemmGemvAdd(silk, sa, sb, sBase, sres, ACols, 96);
        SilkLauncher.GemmGemvAddSubgroup(silk, sa, sb, sSub, sres, ACols, 96);
        silk.Finish();

        float[] baseOut = silk.Download(sBase, 96);
        float[] subOut = silk.Download(sSub, 96);
        for (int i = 0; i < 96; i++)
        {
            Assert.That(BitConverter.SingleToInt32Bits(subOut[i]),
                Is.EqualTo(BitConverter.SingleToInt32Bits(baseOut[i])),
                $"element {i}");
        }

        silk.Release(sa); silk.Release(sb); silk.Release(sres);
        silk.Release(sBase); silk.Release(sSub);
    }

    static float[] Randoms(int count, Random rng)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }
}
