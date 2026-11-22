using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the decode MLP decomposition (#1453 P1): replacing the single-WG fused
/// <c>mlp_block</c> kernel with multi-WG <c>rms_norm_gemv</c> (gate+up) +
/// <c>swiglu</c> + <c>gemm_gemv_add</c> (down+residual) must reproduce the
/// fused kernel bit for bit on identical inputs. Accumulation order is preserved
/// (ascending K, same norm, same elementwise ops).
/// </summary>
[Category("Gpu")]
public sealed class SilkMlpBlockDecompositionTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk MLP decomposition verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    [Test]
    public void MlpBlockDecomposition_MatchesFusedKernel_BitForBit()
    {
        RunCase(64, 128);
        RunCase(96, 256);
        RunCase(576, 1536); // SmolLM-135M shape
        RunCase(896, 4864); // Qwen2.5-0.5B shape
    }

    static void RunCase(int hidden, int intermediate)
    {
        const float eps = 1e-6f;
        using var silk = Device();

        float[] x = Randoms(hidden, 0x14530001u);
        float[] gamma = Randoms(hidden, 0x14530002u);
        float[] wGate = Randoms(hidden * intermediate, 0x14530003u);
        float[] wUp = Randoms(hidden * intermediate, 0x14530004u);
        float[] wDown = Randoms(intermediate * hidden, 0x14530005u);
        float[] residual = Randoms(hidden, 0x14530006u);

        nint sx = silk.Allocate(x.Length);
        nint sg = silk.Allocate(gamma.Length);
        nint swg = silk.Allocate(wGate.Length);
        nint swu = silk.Allocate(wUp.Length);
        nint swd = silk.Allocate(wDown.Length);
        nint sres = silk.Allocate(residual.Length);
        nint sfused = silk.Allocate(hidden);
        nint sdecomp = silk.Allocate(hidden);
        nint sgate = silk.Allocate(intermediate);
        nint sup = silk.Allocate(intermediate);
        nint smlp = silk.Allocate(intermediate);

        silk.Upload(sx, x);
        silk.Upload(sg, gamma);
        silk.Upload(swg, wGate);
        silk.Upload(swu, wUp);
        silk.Upload(swd, wDown);
        silk.Upload(sres, residual);

        SilkLauncher.MlpBlock(silk, sx, sg, swg, swu, swd, sfused, sres, hidden, intermediate, eps);

        SilkLauncher.RmsNormGemv(silk, sx, sg,
            [new(swg, 0, sgate, intermediate), new(swu, 0, sup, intermediate)],
            hasBias: false, aRows: 1, aCols: hidden, eps: eps);
        SilkLauncher.SwiGlu(silk, sgate, sup, smlp, 1, intermediate);
        SilkLauncher.GemmGemvAdd(silk, smlp, swd, sdecomp, sres, intermediate, hidden);
        silk.Finish();

        float[] fused = silk.Download(sfused, hidden);
        float[] decomp = silk.Download(sdecomp, hidden);

        for (int i = 0; i < hidden; i++)
        {
            int want = BitConverter.SingleToInt32Bits(fused[i]);
            int got = BitConverter.SingleToInt32Bits(decomp[i]);
            Assert.That(got, Is.EqualTo(want),
                $"hidden={hidden} inter={intermediate} element {i} differs: fused 0x{want:X8}, decomp 0x{got:X8}, fused={fused[i]:G9}, decomp={decomp[i]:G9}");
        }

        silk.Release(sx); silk.Release(sg); silk.Release(swg); silk.Release(swu); silk.Release(swd);
        silk.Release(sres); silk.Release(sfused); silk.Release(sdecomp);
        silk.Release(sgate); silk.Release(sup); silk.Release(smlp);
    }
}
