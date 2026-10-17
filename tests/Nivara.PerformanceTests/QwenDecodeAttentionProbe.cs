using Nivara.Silk;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

static class QwenDecodeAttentionProbe
{
    const int NumHeads = 14;
    const int NumKvHeads = 2;
    const int HeadDim = 64;
    const int QDim = NumHeads * HeadDim;
    const int KvDim = NumKvHeads * HeadDim;
    static readonly float Scale = 1f / MathF.Sqrt(HeadDim);

    public static int Run()
    {
        using var runtime = new SilkRuntime();
        try { runtime.EnsureAttentionLocalMemory(HeadDim); }
        catch (Exception e) { Console.WriteLine("EnsureAttentionLocalMemory failed: " + e.Message); }

        int maxKv = 4096;
        var q = new float[QDim];
        var kCache = new float[maxKv * KvDim];
        var vCache = new float[maxKv * KvDim];
        var rng = new Random(42);
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        for (int i = 0; i < kCache.Length; i++) kCache[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        for (int i = 0; i < vCache.Length; i++) vCache[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;

        nint qBuf = runtime.Allocate(q.Length);
        nint kBuf = runtime.Allocate(kCache.Length);
        nint vBuf = runtime.Allocate(vCache.Length);
        runtime.Upload(qBuf, q);
        runtime.Upload(kBuf, kCache);
        runtime.Upload(vBuf, vCache);

        foreach (int kvLen in new[] { 512, 1024, 4096 })
        {
            nint outBuf = runtime.Allocate(QDim);

            // warmup
            for (int i = 0; i < 3; i++)
            {
                SilkLauncher.DecodeGqaAttention(runtime, qBuf, kBuf, vBuf, outBuf, kvLen, NumHeads, NumKvHeads, HeadDim, Scale);
                runtime.Finish();
            }

            var times = new List<double>();
            for (int i = 0; i < 7; i++)
            {
                var sw = Stopwatch.StartNew();
                SilkLauncher.DecodeGqaAttention(runtime, qBuf, kBuf, vBuf, outBuf, kvLen, NumHeads, NumKvHeads, HeadDim, Scale);
                runtime.Finish();
                sw.Stop();
                times.Add(sw.Elapsed.TotalMilliseconds);
            }
            times.Sort();
            Console.WriteLine($"kvLen={kvLen}: median {times[times.Count / 2]:F3} ms");

            runtime.Release(outBuf);
        }

        runtime.Release(qBuf);
        runtime.Release(kBuf);
        runtime.Release(vBuf);
        return 0;
    }
}

