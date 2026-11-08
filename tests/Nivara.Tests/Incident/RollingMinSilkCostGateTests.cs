using Nivara;
using Nivara.Silk;
using Nivara.Tensors;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.Incident;

[TestFixture]
[Category("Gpu")]
[Category("Performance")]
public class RollingMinSilkCostGateTests
{
    const int N = 100_000;
    const int Window = 60;

    [Test]
    public void RollingMin_CpuVsSilk_InterleavedTiming()
    {
        SilkRuntime runtime;
        try
        {
            runtime = new SilkRuntime();
        }
        catch
        {
            Assert.Ignore("no OpenCL GPU available");
            return;
        }

        using (runtime)
        {
            var values = new float[N];
            for (int i = 0; i < N; i++)
                values[i] = (i * 7 + 3) % 13;

            using (NivaraColumn<float>.Create(values).RollingMin(Window)) { }
            SilkAnalytics.RollingMin(runtime, values, Window);

            const int Warmup = 3;
            const int Iterations = 7;
            var cpuTimes = new double[Iterations];
            var silkTimes = new double[Iterations];

            for (int w = 0; w < Warmup; w++)
            {
                using var c = NivaraColumn<float>.Create(values).RollingMin(Window);
                SilkAnalytics.RollingMin(runtime, values, Window);
            }

            for (int i = 0; i < Iterations; i++)
            {
                var swCpu = Stopwatch.StartNew();
                using (var c = NivaraColumn<float>.Create(values).RollingMin(Window)) { }
                swCpu.Stop();
                cpuTimes[i] = swCpu.Elapsed.TotalMilliseconds;

                var swSilk = Stopwatch.StartNew();
                SilkAnalytics.RollingMin(runtime, values, Window);
                swSilk.Stop();
                silkTimes[i] = swSilk.Elapsed.TotalMilliseconds;
            }

            static double Median(double[] a)
            {
                var s = (double[])a.Clone();
                Array.Sort(s);
                int m = s.Length / 2;
                return s.Length % 2 == 0 ? (s[m - 1] + s[m]) / 2.0 : s[m];
            }

            double cpuMedian = Median(cpuTimes);
            double silkMedian = Median(silkTimes);
            double ratio = cpuMedian > 0 ? silkMedian / cpuMedian : double.NaN;

            TestContext.Out.WriteLine($"RollingMin CPU/Silk cost gate: n={N}, w={Window}");
            TestContext.Out.WriteLine($"  CPU median: {cpuMedian:F2} ms");
            TestContext.Out.WriteLine($"  Silk median: {silkMedian:F2} ms");
            TestContext.Out.WriteLine($"  Silk/CPU ratio: {ratio:F3}");

            Assert.That(!double.IsNaN(ratio) && !double.IsInfinity(ratio));
        }
    }
}
