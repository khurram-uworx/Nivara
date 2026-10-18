using Nivara.AutoDiff.Operations;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins <c>decode_gqa_attention</c> against the CPU <c>AttentionKernels.DecodeAttention</c>
/// reference across small and large <c>kvLen</c>, including the reduction path that splits
/// <c>kvLen</c> across work-group lanes.
/// </summary>
public sealed class SilkDecodeAttentionParityTests
{
    const int NumHeads = 14;
    const int NumKvHeads = 2;
    const int HeadDim = 64;

    static SilkRuntime Device()
    {
        try
        {
            return new SilkRuntime();
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping Silk decode attention verification.");
            throw;
        }
    }

    [Test]
    public void DecodeGqaAttention_MatchesCpu_AcrossKvLens()
    {
        using var silk = Device();
        float scale = 1f / MathF.Sqrt(HeadDim);
        int kvWidth = NumKvHeads * HeadDim;
        var rng = new Random(42);

        foreach (int kvLen in new[] { 1, 2, 64, 512, 1024, 4096 })
        {
            var q = new float[NumHeads * HeadDim];
            var kCache = new float[kvLen * kvWidth];
            var vCache = new float[kvLen * kvWidth];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
            for (int i = 0; i < kCache.Length; i++) kCache[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
            for (int i = 0; i < vCache.Length; i++) vCache[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;

            var cpuOut = new float[NumHeads * HeadDim];
            AttentionKernels<float>.DecodeAttention(q, kCache, vCache, cpuOut, kvLen, NumHeads, NumKvHeads, HeadDim, scale);

            nint qBuf = silk.Allocate(q.Length);
            nint kBuf = silk.Allocate(kCache.Length);
            nint vBuf = silk.Allocate(vCache.Length);
            nint outBuf = silk.Allocate(cpuOut.Length);
            silk.Upload(qBuf, q);
            silk.Upload(kBuf, kCache);
            silk.Upload(vBuf, vCache);

            SilkLauncher.DecodeGqaAttention(silk, qBuf, kBuf, vBuf, outBuf, kvLen, NumHeads, NumKvHeads, HeadDim, scale);
            silk.Finish();
            var silkOut = silk.Download(outBuf, cpuOut.Length);

            silk.Release(qBuf);
            silk.Release(kBuf);
            silk.Release(vBuf);
            silk.Release(outBuf);

            float maxAbs = 0f;
            float maxCpu = 0f;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuOut[i] - silkOut[i]));
                maxCpu = MathF.Max(maxCpu, MathF.Abs(cpuOut[i]));
            }
            Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3f * (1f + maxCpu)),
                $"kvLen={kvLen}: maxAbs={maxAbs}, scale={1f + maxCpu}");
        }
    }
}
