using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Capability-gated parity for the K-split subgroup decode GEMV variant (#1486).
/// The K-split variant splits both the RMS-norm sum of squares and the per-output-column
/// contraction over k across subgroup lanes and reduces via sub_group_reduce_add, which
/// changes accumulation order (strided partial sums), so results differ from the base
/// kernel by a few ULP - a tolerance is used with that justification. On devices
/// without subgroup support the subgroup launcher falls back to the base kernel and the
/// comparison is trivially exact.
/// </summary>
[Category("Gpu")]
public sealed class SilkRmsNormGemvSubgroupTests
{
    const float Eps = 1e-5f;
    const int ACols = 40;

    [Test]
    public void RmsNormGemvSubgroup_MatchesBaseKernel_WithinTolerance()
    {
        using var silk = Device();
        var rng = new Random(1472);
        float[] x = Randoms(ACols, rng);
        float[] gamma = Randoms(ACols, rng);
        SlotData[] slots =
        [
            new(Randoms(ACols * 96, rng), Randoms(96, rng), 96),
            new(Randoms(ACols * 24, rng), Randoms(24, rng), 24),
        ];

        float[][] baseOut = Run(silk, x, gamma, slots, hasBias: true, subgroup: false);
        float[][] subOut = Run(silk, x, gamma, slots, hasBias: true, subgroup: true);

        for (int s = 0; s < slots.Length; s++)
            AssertWithinTolerance(baseOut[s], subOut[s], $"biased slot {s}");
    }

    [Test]
    public void SupportsSubgroups_IsReported()
    {
        using var silk = Device();
        // Smoke: property is readable and consistent with kernel availability.
        Assert.That(
            silk.HasKernel(Nivara.Silk.SilkKernels.RmsNormGemvSubgroup),
            Is.EqualTo(silk.SupportsSubgroups));
    }

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk subgroup verification."); throw; }
    }

    static float[][] Run(SilkRuntime silk, float[] x, float[] gamma, SlotData[] slots, bool hasBias, bool subgroup)
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

            if (subgroup)
                SilkLauncher.RmsNormGemvSubgroup(silk, sx, sg, kernelSlots, hasBias, aRows: 1, aCols: x.Length, eps: Eps);
            else
                SilkLauncher.RmsNormGemv(silk, sx, sg, kernelSlots, hasBias, aRows: 1, aCols: x.Length, eps: Eps);
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
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    static void AssertWithinTolerance(float[] expected, float[] actual, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label}: length changed");
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = 1e-4f * MathF.Max(1f, MathF.Abs(expected[i]));
            if (diff > tol)
                Assert.Fail($"{label}: element {i} differs by {diff} (tol {tol}): {expected[i]:R} vs {actual[i]:R}");
        }
    }

    sealed record SlotData(float[] Weight, float[] Bias, int Cols);
}
