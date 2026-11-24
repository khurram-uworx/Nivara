using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins decode MLP Option B (#1470): fused <c>rms_norm_gemv_swiglu</c> (norm+gate+up+swiglu)
/// + <c>gemm_gemv_add</c> (down+residual) must reproduce the 3-dispatch Option A decomposition
/// bit for bit on identical F32 inputs. Accumulation order is preserved.
/// </summary>
[Category("Gpu")]
public sealed class SilkMlpBlockOptionBTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk MLP Option B verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    [Test]
    public void MlpBlockOptionB_MatchesOptionA_BitForBit_F32()
    {
        RunF32Case(64, 128);
        RunF32Case(96, 256);
        RunF32Case(576, 1536); // SmolLM-135M
        RunF32Case(896, 4864); // Qwen2.5-0.5B
    }

    [Test]
    public void MlpBlockOptionB_MatchesOptionA_BitForBit_Bf16()
    {
        RunBf16Case(64, 128);
        RunBf16Case(96, 256);
        RunBf16Case(576, 1536); // SmolLM-135M
        RunBf16Case(896, 4864); // Qwen2.5-0.5B
    }

    static void RunF32Case(int hidden, int intermediate)
    {
        const float eps = 1e-6f;
        using var silk = Device();

        float[] x = Randoms(hidden, 0x14700001u);
        float[] gamma = Randoms(hidden, 0x14700002u);
        float[] wGate = Randoms(hidden * intermediate, 0x14700003u);
        float[] wUp = Randoms(hidden * intermediate, 0x14700004u);
        float[] wDown = Randoms(intermediate * hidden, 0x14700005u);
        float[] residual = Randoms(hidden, 0x14700006u);

        nint sx = silk.Allocate(x.Length);
        nint sg = silk.Allocate(gamma.Length);
        nint swg = silk.Allocate(wGate.Length);
        nint swu = silk.Allocate(wUp.Length);
        nint swd = silk.Allocate(wDown.Length);
        nint sres = silk.Allocate(residual.Length);
        nint soptA = silk.Allocate(hidden);
        nint soptB = silk.Allocate(hidden);
        nint sgate = silk.Allocate(intermediate);
        nint sup = silk.Allocate(intermediate);
        nint smlp = silk.Allocate(intermediate);

        silk.Upload(sx, x);
        silk.Upload(sg, gamma);
        silk.Upload(swg, wGate);
        silk.Upload(swu, wUp);
        silk.Upload(swd, wDown);
        silk.Upload(sres, residual);

        // Option A: 3-dispatch decomposition.
        SilkLauncher.RmsNormGemv(silk, sx, sg,
            [new(swg, 0, sgate, intermediate), new(swu, 0, sup, intermediate)],
            hasBias: false, aRows: 1, aCols: hidden, eps: eps);
        SilkLauncher.SwiGlu(silk, sgate, sup, smlp, 1, intermediate);
        SilkLauncher.GemmGemvAdd(silk, smlp, swd, soptA, sres, intermediate, hidden);

        // Option B: 2-dispatch fused.
        SilkLauncher.RmsNormGemvSwiGlu(silk, sx, sg, swg, swu, smlp, hidden, intermediate, eps);
        SilkLauncher.GemmGemvAdd(silk, smlp, swd, soptB, sres, intermediate, hidden);
        silk.Finish();

        float[] optA = silk.Download(soptA, hidden);
        float[] optB = silk.Download(soptB, hidden);
        for (int i = 0; i < hidden; i++)
        {
            int want = BitConverter.SingleToInt32Bits(optA[i]);
            int got = BitConverter.SingleToInt32Bits(optB[i]);
            Assert.That(got, Is.EqualTo(want),
                $"F32 hidden={hidden} inter={intermediate} element {i} differs: A 0x{want:X8}, B 0x{got:X8}, A={optA[i]:G9}, B={optB[i]:G9}");
        }

        silk.Release(sx); silk.Release(sg); silk.Release(swg); silk.Release(swu); silk.Release(swd);
        silk.Release(sres); silk.Release(soptA); silk.Release(soptB);
        silk.Release(sgate); silk.Release(sup); silk.Release(smlp);
    }

    static void RunBf16Case(int hidden, int intermediate)
    {
        const float eps = 1e-6f;
        using var silk = Device();

        float[] x = Randoms(hidden, 0x14710001u);
        float[] gamma = Randoms(hidden, 0x14710002u);
        ushort[] wGateBf16 = ToBf16(Randoms(hidden * intermediate, 0x14710003u));
        ushort[] wUpBf16 = ToBf16(Randoms(hidden * intermediate, 0x14710004u));
        ushort[] wDownBf16 = ToBf16(Randoms(intermediate * hidden, 0x14710005u));
        float[] residual = Randoms(hidden, 0x14710006u);

        nint sx = silk.Allocate(x.Length);
        nint sg = silk.Allocate(gamma.Length);
        nint swg = silk.Allocate(wGateBf16.Length);
        nint swu = silk.Allocate(wUpBf16.Length);
        nint swd = silk.Allocate(wDownBf16.Length);
        nint sres = silk.Allocate(residual.Length);
        nint soptA = silk.Allocate(hidden);
        nint soptB = silk.Allocate(hidden);
        nint sgate = silk.Allocate(intermediate);
        nint sup = silk.Allocate(intermediate);
        nint smlp = silk.Allocate(intermediate);

        silk.Upload(sx, x);
        silk.Upload(sg, gamma);
        silk.Upload(swg, wGateBf16);
        silk.Upload(swu, wUpBf16);
        silk.Upload(swd, wDownBf16);
        silk.Upload(sres, residual);

        SilkLauncher.RmsNormGemvBf16(silk, sx, sg,
            [new(swg, 0, sgate, intermediate), new(swu, 0, sup, intermediate)],
            hasBias: false, aRows: 1, aCols: hidden, eps: eps);
        SilkLauncher.SwiGlu(silk, sgate, sup, smlp, 1, intermediate);
        SilkLauncher.GemmGemvAddBf16(silk, smlp, swd, soptA, sres, intermediate, hidden);

        SilkLauncher.RmsNormGemvSwiGluBf16(silk, sx, sg, swg, swu, smlp, hidden, intermediate, eps);
        SilkLauncher.GemmGemvAddBf16(silk, smlp, swd, soptB, sres, intermediate, hidden);
        silk.Finish();

        float[] optA = silk.Download(soptA, hidden);
        float[] optB = silk.Download(soptB, hidden);
        for (int i = 0; i < hidden; i++)
        {
            int want = BitConverter.SingleToInt32Bits(optA[i]);
            int got = BitConverter.SingleToInt32Bits(optB[i]);
            Assert.That(got, Is.EqualTo(want),
                $"BF16 hidden={hidden} inter={intermediate} element {i} differs: A 0x{want:X8}, B 0x{got:X8}, A={optA[i]:G9}, B={optB[i]:G9}");
        }

        silk.Release(sx); silk.Release(sg); silk.Release(swg); silk.Release(swu); silk.Release(swd);
        silk.Release(sres); silk.Release(soptA); silk.Release(soptB);
        silk.Release(sgate); silk.Release(sup); silk.Release(smlp);
    }

    static ushort[] ToBf16(float[] values)
    {
        var result = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            result[i] = (ushort)(bits >> 16);
        }
        return result;
    }
}
