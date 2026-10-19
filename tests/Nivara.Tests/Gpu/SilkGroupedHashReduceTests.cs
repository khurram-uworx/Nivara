using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[TestFixture]
public class SilkGroupedHashReduceTests
{
    [Test]
    public void GroupedHashReduce_MatchesCpuReference()
    {
        var serviceIds = new[] { 0, 1, 0, 2, 1, 0, 2, 2, 1, 0 };
        var statusCodes = new[] { 200, 500, 200, 404, 500, 200, 200, 500, 200, 500 };
        var durations = new[] { 1.5f, 2.0f, 1.0f, 3.0f, 2.5f, 1.2f, 0.8f, 4.0f, 1.1f, 2.2f };
        const int groupCount = 3;

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
            var (counts, errors, sums) = SilkAnalytics.GroupedHashReduce(
                runtime, serviceIds, statusCodes, durations, groupCount);

            // CPU reference
            var refCounts = new int[groupCount];
            var refErrors = new int[groupCount];
            var refSums = new float[groupCount];
            for (int i = 0; i < serviceIds.Length; i++)
            {
                int g = serviceIds[i];
                refCounts[g]++;
                if (statusCodes[i] >= 500) refErrors[g]++;
                refSums[g] += durations[i];
            }

            Assert.That(counts, Is.EqualTo(refCounts));
            Assert.That(errors, Is.EqualTo(refErrors));
            for (int g = 0; g < groupCount; g++)
                Assert.That(sums[g], Is.EqualTo(refSums[g]).Within(1e-5f));
        }
    }
}
