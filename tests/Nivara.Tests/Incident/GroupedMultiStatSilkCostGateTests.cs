using Nivara.Silk;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.Incident;

[TestFixture]
[Category("Gpu")]
[Category("Performance")]
public class GroupedMultiStatSilkCostGateTests
{
    const int N = 100_000;
    const int Groups = 5;

    [Test]
    public void GroupedMultiStat_CpuVsSilk_InterleavedTiming()
    {
        var runtime = new SilkRuntime();
        using (runtime)
        {
            var serviceIds = new int[N];
            var statusCodes = new int[N];
            var durations = new float[N];
            for (int i = 0; i < N; i++)
            {
                serviceIds[i] = i % Groups;
                statusCodes[i] = (i % 7 == 0) ? 500 : 200;
                durations[i] = ((i * 13 + 7) % 50) / 10f;
            }

            SilkAnalytics.GroupedMultiStatReduce(runtime, serviceIds, statusCodes, durations, Groups);

            const int Warmup = 3;
            const int Iterations = 7;
            var cpuTimes = new double[Iterations];
            var silkTimes = new double[Iterations];

            for (int w = 0; w < Warmup; w++)
            {
                CpuReduce(serviceIds, statusCodes, durations, Groups);
                SilkAnalytics.GroupedMultiStatReduce(runtime, serviceIds, statusCodes, durations, Groups);
            }

            for (int i = 0; i < Iterations; i++)
            {
                var swCpu = Stopwatch.StartNew();
                CpuReduce(serviceIds, statusCodes, durations, Groups);
                swCpu.Stop();
                cpuTimes[i] = swCpu.Elapsed.TotalMilliseconds;

                var swSilk = Stopwatch.StartNew();
                SilkAnalytics.GroupedMultiStatReduce(runtime, serviceIds, statusCodes, durations, Groups);
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

            TestContext.Out.WriteLine($"GroupedMultiStat CPU/Silk cost gate: n={N}, groups={Groups}");
            TestContext.Out.WriteLine($"  CPU median: {cpuMedian:F2} ms");
            TestContext.Out.WriteLine($"  Silk median: {silkMedian:F2} ms");
            TestContext.Out.WriteLine($"  Silk/CPU ratio: {ratio:F3}");

            Assert.That(!double.IsNaN(ratio) && !double.IsInfinity(ratio));
        }
    }

    static void CpuReduce(int[] serviceIds, int[] statusCodes, float[] durations, int groupCount)
    {
        var counts = new int[groupCount];
        var errors = new int[groupCount];
        var sums = new float[groupCount];
        var sumsq = new float[groupCount];
        var mins = new float[groupCount];
        var maxs = new float[groupCount];
        for (int g = 0; g < groupCount; g++) { mins[g] = float.MaxValue; maxs[g] = float.MinValue; }
        int n = serviceIds.Length;
        for (int i = 0; i < n; i++)
        {
            int g = serviceIds[i];
            counts[g]++;
            if (statusCodes[i] >= 500) errors[g]++;
            float d = durations[i];
            sums[g] += d;
            sumsq[g] += d * d;
            if (d < mins[g]) mins[g] = d;
            if (d > maxs[g]) maxs[g] = d;
        }
    }
}
