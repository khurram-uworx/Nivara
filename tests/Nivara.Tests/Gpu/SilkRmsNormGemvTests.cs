using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the fused cached-decode projection kernel (#670): for every slot, <c>rms_norm_gemv</c> must
/// reproduce the single-projection kernel it replaced bit for bit, not within a tolerance.
/// </summary>
/// <remarks>
/// <para>
/// The assertion is exact because the claim is exact. <c>rms_norm_gemv</c> is
/// <c>rms_norm_lm_head</c>'s accumulation text with a per-slot weight/output selection and an
/// optional bias in the store, so a change of K-order, of the norm reduction or of the contraction
/// shows up as a different bit pattern. A tolerance would hide precisely the defect this fixture
/// exists to catch — and the documented byte-identity against ILGPU rests on the same argument.
/// </para>
/// <para>
/// On a real model the fused kernel's bounds are barely exercised: Qwen's widths (896 / 128 / 4864 /
/// 151936) and SmolLM's 1536 are whole numbers of 128-column groups, and SmolLM's 576 and 192 are
/// whole numbers of 4-column lanes, so every live lane owns a full <c>BLOCK</c> span and only a
/// SmolLM group is short. The widths here leave the edges deliberately: <c>aCols</c> is not a
/// multiple of <c>GemmTileSize</c>, some slots end mid-group — which is where the
/// <c>col &gt;= bCols</c> early return fires — and three of them end mid-*lane*, which is where the
/// per-lane <c>avail</c> bound fires.
/// </para>
/// </remarks>
public sealed class SilkRmsNormGemvTests
{
    const float Eps = 1e-5f;

    /// <summary>Not a multiple of <c>GemmTileSize</c> (16), so the K-loop tail guards run.</summary>
    const int ACols = 40;

    /// <summary>One full group plus a partial one (two groups at <c>GemmTileCols</c> 64).</summary>
    const int WideCols = 96;

    /// <summary>A single partial group.</summary>
    const int NarrowCols = 24;

    [Test]
    public void RmsNormGemv_ThreeBiasedSlots_MatchesEachSingleProjection()
    {
        using var silk = Device();
        var rng = new Random(670);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots =
        [
            new(Randoms(ACols * 96, rng), Randoms(96, rng), 96),
            new(Randoms(ACols * 24, rng), Randoms(24, rng), 24),
            new(Randoms(ACols * 40, rng), Randoms(40, rng), 40),
        ];

        float[][] fused = RunFused(silk, x, gamma, slots, hasBias: true);
        float[][] singles = RunSingleProjections(silk, x, gamma, slots, hasBias: true);

        for (int s = 0; s < slots.Length; s++)
            AssertBitIdentical(singles[s], fused[s], $"biased slot {s} (cols {slots[s].Cols})");
    }

    [Test]
    public void RmsNormGemv_TwoBiasFreeSlots_MatchesEachSingleProjectionKernel()
    {
        using var silk = Device();
        var rng = new Random(671);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots =
        [
            new(Randoms(ACols * 96, rng), [], 96),
            new(Randoms(ACols * 96, rng), [], 96),
        ];

        float[][] fused = RunFused(silk, x, gamma, slots, hasBias: false);
        float[][] reference = RunSingleProjections(silk, x, gamma, slots, hasBias: false);

        for (int s = 0; s < slots.Length; s++)
            AssertBitIdentical(reference[s], fused[s], $"bias-free slot {s}");
    }

    [Test]
    public void RmsNormGemv_OneBiasFreeSlot_MatchesSingleProjectionKernel()
    {
        using var silk = Device();
        var rng = new Random(672);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots = [new(Randoms(ACols * 200, rng), [], 200)];

        float[][] fused = RunFused(silk, x, gamma, slots, hasBias: false);
        float[][] reference = RunSingleProjections(silk, x, gamma, slots, hasBias: false);

        AssertBitIdentical(reference[0], fused[0], "single slot (head shape)");
    }

    [Test]
    public void RmsNormGemv_WidthsThatStraddleALane_MatchesSingleProjectionKernel()
    {
        using var silk = Device();
        var rng = new Random(673);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots =
        [
            new(Randoms(ACols * 25, rng), Randoms(25, rng), 25),
            new(Randoms(ACols * 26, rng), Randoms(26, rng), 26),
            new(Randoms(ACols * 27, rng), Randoms(27, rng), 27),
        ];

        float[][] fused = RunFused(silk, x, gamma, slots, hasBias: true);
        float[][] singles = RunSingleProjections(silk, x, gamma, slots, hasBias: true);

        for (int s = 0; s < slots.Length; s++)
            AssertBitIdentical(singles[s], fused[s], $"biased straddling slot {s} (cols {slots[s].Cols})");
    }

    [Test]
    public void RmsNormGemv_BiasFreeStraddlingWidth_MatchesSingleProjectionKernel()
    {
        using var silk = Device();
        var rng = new Random(674);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots = [new(Randoms(ACols * 27, rng), [], 27)];

        float[][] fused = RunFused(silk, x, gamma, slots, hasBias: false);
        float[][] reference = RunSingleProjections(silk, x, gamma, slots, hasBias: false);

        AssertBitIdentical(reference[0], fused[0], "bias-free straddling slot (cols 27)");
    }

    static SilkRuntime Device()
    {
        try
        {
            return new SilkRuntime();
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping Silk rms_norm_gemv verification.");
            throw;
        }
    }

    static float[][] RunFused(SilkRuntime silk, float[] x, float[] gamma, SlotData[] slots, bool hasBias)
    {
        var owned = new List<nint>();
        var outHandles = new nint[slots.Length];
        var result = new float[slots.Length][];
        try
        {
            nint sx = Upload(silk, owned, x);
            nint sg = Upload(silk, owned, gamma);
            var kernelSlots = new SilkLauncher.GemvSlot[slots.Length];
            for (int i = 0; i < slots.Length; i++)
            {
                nint weight = Upload(silk, owned, slots[i].Weight);
                nint bias = hasBias ? Upload(silk, owned, slots[i].Bias) : 0;
                outHandles[i] = Allocate(silk, owned, slots[i].Cols);
                kernelSlots[i] = new SilkLauncher.GemvSlot(weight, bias, outHandles[i], slots[i].Cols);
            }

            SilkLauncher.RmsNormGemv(
                silk, sx, sg, kernelSlots, hasBias, aRows: 1, aCols: x.Length, eps: Eps);
            silk.Finish();

            for (int i = 0; i < slots.Length; i++)
                result[i] = silk.Download(outHandles[i], slots[i].Cols);
        }
        finally
        {
            foreach (nint handle in owned) silk.Release(handle);
        }
        return result;
    }

    static float[][] RunSingleProjections(
        SilkRuntime silk, float[] x, float[] gamma, SlotData[] slots, bool hasBias)
    {
        var owned = new List<nint>();
        var outHandles = new nint[slots.Length];
        var result = new float[slots.Length][];
        try
        {
            nint sx = Upload(silk, owned, x);
            nint sg = Upload(silk, owned, gamma);
            for (int i = 0; i < slots.Length; i++)
            {
                nint weight = Upload(silk, owned, slots[i].Weight);
                outHandles[i] = Allocate(silk, owned, slots[i].Cols);
                SilkLauncher.RmsNormLmHead(
                    silk, sx, sg, weight, outHandles[i],
                    aRows: 1, aCols: x.Length, bCols: slots[i].Cols, eps: Eps);
            }
            silk.Finish();

            for (int i = 0; i < slots.Length; i++)
            {
                result[i] = silk.Download(outHandles[i], slots[i].Cols);
                if (hasBias)
                    for (int j = 0; j < result[i].Length; j++)
                        result[i][j] += slots[i].Bias[j];
            }
        }
        finally
        {
            foreach (nint handle in owned) silk.Release(handle);
        }
        return result;
    }

    static nint Upload(SilkRuntime silk, List<nint> owned, float[] data)
    {
        nint buffer = silk.Allocate(data.Length);
        owned.Add(buffer);
        silk.Upload(buffer, data);
        return buffer;
    }

    static nint Allocate(SilkRuntime silk, List<nint> owned, int count)
    {
        nint buffer = silk.Allocate(count);
        owned.Add(buffer);
        return buffer;
    }

    static float[] Randoms(int count, Random rng)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    static void AssertBitIdentical(float[] expected, float[] actual, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label}: length changed");
        for (int i = 0; i < expected.Length; i++)
        {
            int want = BitConverter.SingleToInt32Bits(expected[i]);
            int got = BitConverter.SingleToInt32Bits(actual[i]);
            if (want != got)
                Assert.Fail(
                    $"{label}: element {i} differs. expected {expected[i]:R} (0x{want:X8}), got {actual[i]:R} (0x{got:X8})");
        }
    }

    sealed record SlotData(float[] Weight, float[] Bias, int Cols);
}
