using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkBackwardBf16ActKernelTests
{
    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk backward BF16 kernel verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    static ushort ToBf16(float f) => (ushort)(BitConverter.SingleToInt32Bits(f) >> 16);

    static void AssertWithinBound(float[] got, float[] reference, string label)
    {
        Assert.That(got.Length, Is.EqualTo(reference.Length), label);
        for (int i = 0; i < reference.Length; i++)
        {
            float bound = 1e-2f * (1f + Math.Abs(reference[i]));
            Assert.That(Math.Abs(got[i] - reference[i]), Is.LessThanOrEqualTo(bound),
                $"{label}[{i}]: got {got[i]}, ref {reference[i]}, bound {bound}");
        }
    }

    [Test]
    public void GeluBackwardBf16Act_MatchesF32Reference_WithinBound()
    {
        const int count = 1000;
        using var silk = Device();
        float[] d = Randoms(count, 0x11490001);
        float[] act = Randoms(count, 0x11490002);

        nint sd = silk.Allocate(d.Length);
        nint sa = silk.Allocate(act.Length);
        nint sy = silk.Allocate(count);
        silk.Upload(sd, d);
        silk.Upload(sa, act);
        SilkLauncher.GeluBackward(silk, sd, sa, sy, count);
        silk.Finish();
        float[] reference = silk.Download(sy, count);

        var d16 = d.Select(ToBf16).ToArray();
        var a16 = act.Select(ToBf16).ToArray();
        nint sd16 = silk.AllocateUshort(count);
        nint sa16 = silk.AllocateUshort(count);
        nint sy16 = silk.AllocateUshort(count);
        silk.Upload(sd16, d16.AsSpan());
        silk.Upload(sa16, a16.AsSpan());
        SilkLauncher.GeluBackwardBf16Act(silk, sd16, sa16, sy16, count);
        silk.Finish();
        var got16 = silk.DownloadUshort(sy16, count);
        float[] got = got16.Select(u => SilkRuntime.WidenBf16(u)).ToArray();

        AssertWithinBound(got, reference, "gelu_backward_bf16_act");

        silk.Release(sd); silk.Release(sa); silk.Release(sy);
        silk.Release(sd16); silk.Release(sa16); silk.Release(sy16);
    }

    [Test]
    public void ColSumBf16Act_MatchesF32Reference_WithinBound()
    {
        (int rows, int cols)[] shapes = [(7, 5), (33, 2), (16, 8), (1, 1)];
        using var silk = Device();
        foreach (var (rows, cols) in shapes)
        {
            float[] x = Randoms(rows * cols, 0x11490011);
            const float scale = 0.25f;

            nint sx = silk.Allocate(x.Length);
            nint sy = silk.Allocate(cols);
            silk.Upload(sx, x);
            SilkLauncher.ColSum(silk, sx, sy, rows, cols, scale);
            silk.Finish();
            float[] reference = silk.Download(sy, cols);

            var x16 = x.Select(ToBf16).ToArray();
            nint sx16 = silk.AllocateUshort(x.Length);
            nint sy16 = silk.Allocate(cols);
            silk.Upload(sx16, x16.AsSpan());
            SilkLauncher.ColSumBf16Act(silk, sx16, sy16, rows, cols, scale);
            silk.Finish();
            float[] got = silk.Download(sy16, cols);

            AssertWithinBound(got, reference, $"col_sum_bf16_act rows={rows} cols={cols}");

            silk.Release(sx); silk.Release(sy);
            silk.Release(sx16); silk.Release(sy16);
        }
    }

    [Test]
    public void LayerNormBackwardBf16Act_MatchesF32Reference_WithinBound()
    {
        const int rows = 4, cols = 8;
        using var silk = Device();
        float[] d = Randoms(rows * cols, 0x11490021);
        float[] x = Randoms(rows * cols, 0x11490022);
        float[] gamma = Randoms(cols, 0x11490023);
        const float eps = 1e-5f;

        nint sd = silk.Allocate(d.Length);
        nint sx = silk.Allocate(x.Length);
        nint sg = silk.Allocate(gamma.Length);
        nint sdx = silk.Allocate(rows * cols);
        nint sdga = silk.Allocate(rows * cols);
        nint sdba = silk.Allocate(rows * cols);
        silk.Upload(sd, d);
        silk.Upload(sx, x);
        silk.Upload(sg, gamma);
        SilkLauncher.LayerNormBackward(silk, sd, sx, sg, sdx, sdga, sdba, rows, cols, eps);
        silk.Finish();
        float[] refDx = silk.Download(sdx, rows * cols);
        float[] refDga = silk.Download(sdga, rows * cols);
        float[] refDba = silk.Download(sdba, rows * cols);

        var d16 = d.Select(ToBf16).ToArray();
        var x16 = x.Select(ToBf16).ToArray();
        nint sd16 = silk.AllocateUshort(d.Length);
        nint sx16 = silk.AllocateUshort(x.Length);
        nint sdx16 = silk.AllocateUshort(rows * cols);
        nint sdga16 = silk.Allocate(rows * cols);
        nint sdba16 = silk.Allocate(rows * cols);
        silk.Upload(sd16, d16.AsSpan());
        silk.Upload(sx16, x16.AsSpan());
        silk.Upload(sg, gamma);
        SilkLauncher.LayerNormBackwardBf16Act(silk, sd16, sx16, sg, sdx16, sdga16, sdba16, rows, cols, eps);
        silk.Finish();
        var gotDx16 = silk.DownloadUshort(sdx16, rows * cols);
        float[] gotDx = gotDx16.Select(u => SilkRuntime.WidenBf16(u)).ToArray();
        float[] gotDga = silk.Download(sdga16, rows * cols);
        float[] gotDba = silk.Download(sdba16, rows * cols);

        AssertWithinBound(gotDx, refDx, "layer_norm_backward_bf16_act dx");
        AssertWithinBound(gotDga, refDga, "layer_norm_backward_bf16_act dgammaAcc");
        AssertWithinBound(gotDba, refDba, "layer_norm_backward_bf16_act dbetaAcc");

        silk.Release(sd); silk.Release(sx); silk.Release(sg); silk.Release(sdx); silk.Release(sdga); silk.Release(sdba);
        silk.Release(sd16); silk.Release(sx16); silk.Release(sdx16); silk.Release(sdga16); silk.Release(sdba16);
    }
}
