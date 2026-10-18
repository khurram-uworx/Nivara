using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the fused decode attention + O-projection (#691): bit-identity against the
/// two-kernel path (<c>decode_gqa_attention</c> + <c>gemm_gemv_add</c>) on identical
/// inputs, including tail widths and short kvLen. Uses a small head count so the fused
/// path is runnable on devices with a 256-slot work-group limit.
/// </summary>
public sealed class SilkDecodeOprojFusedTests
{
    const int NumHeads = 4;
    const int NumKvHeads = 2;
    const int HeadDim = 8;
    const int QDim = NumHeads * HeadDim; // 32

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk fused O-proj verification."); throw; }
    }

    static float[] Randoms(int count, uint seed)
    {
        var rng = new Random((int)seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return values;
    }

    [Test]
    public void Fused_MatchesTwoKernelPath_BitForBit()
    {
        using var silk = Device();
        if (!silk.CanFuseDecodeOproj(NumHeads, HeadDim))
            Assert.Ignore("Device cannot run the fused O-proj kernel; skipping.");

        float scale = 1f / MathF.Sqrt(HeadDim);
        int[] kvLens = [1, 7, 32];
        int[] widths = [25, 26, 27, 64];

        foreach (int kvLen in kvLens)
            foreach (int hiddenDim in widths)
            {
                int kvDim = NumKvHeads * HeadDim;
                float[] q = Randoms(QDim, 0x69100001u);
                float[] kc = Randoms(kvLen * kvDim, 0x69100002u);
                float[] vc = Randoms(kvLen * kvDim, 0x69100003u);
                float[] wo = Randoms(QDim * hiddenDim, 0x69100004u);
                float[] res = Randoms(hiddenDim, 0x69100005u);

                nint sq = silk.Allocate(q.Length);
                nint skc = silk.Allocate(kc.Length);
                nint svc = silk.Allocate(vc.Length);
                nint swo = silk.Allocate(wo.Length);
                nint sres = silk.Allocate(res.Length);
                nint sctx = silk.Allocate(QDim);
                nint sfused = silk.Allocate(hiddenDim);
                nint stwo = silk.Allocate(hiddenDim);

                silk.Upload(sq, q);
                silk.Upload(skc, kc);
                silk.Upload(svc, vc);
                silk.Upload(swo, wo);
                silk.Upload(sres, res);

                SilkLauncher.DecodeGqaAttentionOprojFused(silk, sq, skc, svc, swo, sres, sfused,
                    kvLen, NumHeads, NumKvHeads, HeadDim, hiddenDim, scale);
                SilkLauncher.DecodeGqaAttention(silk, sq, skc, svc, sctx, kvLen, NumHeads, NumKvHeads, HeadDim, scale);
                SilkLauncher.GemmGemvAdd(silk, sctx, swo, stwo, sres, QDim, hiddenDim);
                silk.Finish();

                float[] fused = silk.Download(sfused, hiddenDim);
                float[] two = silk.Download(stwo, hiddenDim);

                for (int i = 0; i < hiddenDim; i++)
                {
                    int a = BitConverter.SingleToInt32Bits(fused[i]);
                    int b = BitConverter.SingleToInt32Bits(two[i]);
                    Assert.That(a, Is.EqualTo(b), $"kvLen={kvLen} hiddenDim={hiddenDim} element {i}: fused 0x{a:X8}, two 0x{b:X8}");
                }

                silk.Release(sq); silk.Release(skc); silk.Release(svc); silk.Release(swo);
                silk.Release(sres); silk.Release(sctx); silk.Release(sfused); silk.Release(stwo);
            }
    }
}
