using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Capability-gated parity for the K-split subgroup decode GEMV (gemm_gemv_add) variant (#1486).
/// The K-split variant distributes the contraction over k across subgroup lanes and reduces
/// partial dot products via sub_group_reduce_add, which changes accumulation order (strided
/// partial sums), so results differ from the base kernel by a few ULP - a tolerance is used
/// with that justification. On devices without subgroup support the subgroup launcher falls
/// back to the base kernel and that path is bit-identical (see SilkSubgroupForceFallbackTests).
/// </summary>
[Category("Gpu")]
public sealed class SilkGemmGemvAddSubgroupTests
{
    const int ACols = 40;

    [Test]
    public void GemmGemvAddSubgroup_MatchesBaseKernel_WithinTolerance()
    {
        using var silk = Device();
        uint seed = 0x14720001u;
        foreach (int bCols in new[] { 25, 26, 27, 96 })
        {
            var rng = new Random((int)seed);
            float[] a = Randoms(ACols, rng);
            float[] b = Randoms(ACols * bCols, rng);
            float[] residual = Randoms(bCols, rng);

            nint sa = silk.Allocate(a.Length);
            nint sb = silk.Allocate(b.Length);
            nint sres = silk.Allocate(residual.Length);
            nint sBase = silk.Allocate(bCols);
            nint sSub = silk.Allocate(bCols);
            silk.Upload(sa, a);
            silk.Upload(sb, b);
            silk.Upload(sres, residual);

            SilkLauncher.GemmGemvAdd(silk, sa, sb, sBase, sres, ACols, bCols);
            SilkLauncher.GemmGemvAddSubgroup(silk, sa, sb, sSub, sres, ACols, bCols);
            silk.Finish();

            float[] baseOut = silk.Download(sBase, bCols);
            float[] subOut = silk.Download(sSub, bCols);

            for (int i = 0; i < bCols; i++)
            {
                float diff = MathF.Abs(baseOut[i] - subOut[i]);
                float tol = 1e-4f * MathF.Max(1f, MathF.Abs(baseOut[i]));
                if (diff > tol)
                    Assert.Fail($"bCols={bCols} element {i} differs by {diff} (tol {tol}): {baseOut[i]:R} vs {subOut[i]:R}");
            }

            silk.Release(sa); silk.Release(sb); silk.Release(sres);
            silk.Release(sBase); silk.Release(sSub);
        }
    }

    [Test]
    public void SupportsSubgroups_ConsistentWithKernelAvailability()
    {
        using var silk = Device();
        Assert.That(
            silk.HasKernel(Nivara.Silk.SilkKernels.GemmGemvAddSubgroup),
            Is.EqualTo(silk.SupportsSubgroups));
    }

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk subgroup verification."); throw; }
    }

    static float[] Randoms(int count, Random rng)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }
}
