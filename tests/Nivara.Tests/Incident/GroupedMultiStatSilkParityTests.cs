using Nivara;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
[Category("Gpu")]
public class GroupedMultiStatSilkParityTests
{
    [Test]
    public void SilkGroupedMultiStat_MatchesCpu()
    {
        var runtime = new SilkRuntime();
        using (runtime)
        {
            const int n = 1_000;
            const int groupCount = 5;
            var serviceIds = new int[n];
            var statusCodes = new int[n];
            var durations = new float[n];
            for (int i = 0; i < n; i++)
            {
                serviceIds[i] = i % groupCount;
                statusCodes[i] = (i % 7 == 0) ? 500 : 200;
                durations[i] = ((i * 13 + 7) % 50) / 10f;
            }

            // CPU reference
            var cpuCounts = new int[groupCount];
            var cpuErrors = new int[groupCount];
            var cpuSums = new float[groupCount];
            var cpuSumsq = new float[groupCount];
            var cpuMins = new float[groupCount];
            var cpuMaxs = new float[groupCount];
            var has = new bool[groupCount];
            for (int i = 0; i < groupCount; i++) { cpuMins[i] = float.MaxValue; cpuMaxs[i] = float.MinValue; }
            for (int i = 0; i < n; i++)
            {
                int g = serviceIds[i];
                cpuCounts[g]++;
                if (statusCodes[i] >= 500) cpuErrors[g]++;
                float d = durations[i];
                cpuSums[g] += d;
                cpuSumsq[g] += d * d;
                if (!has[g]) { has[g] = true; cpuMins[g] = cpuMaxs[g] = d; }
                else { cpuMins[g] = Math.Min(cpuMins[g], d); cpuMaxs[g] = Math.Max(cpuMaxs[g], d); }
            }
            for (int g = 0; g < groupCount; g++)
                if (!has[g]) { cpuMins[g] = 0; cpuMaxs[g] = 0; }

            var silk = SilkAnalytics.GroupedMultiStatReduce(runtime, serviceIds, statusCodes, durations, groupCount);

            for (int g = 0; g < groupCount; g++)
            {
                Assert.That(silk.Counts[g], Is.EqualTo(cpuCounts[g]));
                Assert.That(silk.Errors[g], Is.EqualTo(cpuErrors[g]));
                Assert.That(silk.Sums[g], Is.EqualTo(cpuSums[g]).Within(0.05));
                Assert.That(silk.Sumsq[g], Is.EqualTo(cpuSumsq[g]).Within(0.5));
                Assert.That(silk.Mins[g], Is.EqualTo(cpuMins[g]).Within(1e-6));
                Assert.That(silk.Maxs[g], Is.EqualTo(cpuMaxs[g]).Within(1e-6));
            }
        }
    }
}
