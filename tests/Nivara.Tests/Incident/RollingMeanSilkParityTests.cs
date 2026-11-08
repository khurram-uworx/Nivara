using Nivara;
using Nivara.Silk;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
[Category("Gpu")]
public class RollingMeanSilkParityTests
{
    [Test]
    public void SilkRollingMean_MatchesCpu_IntWindow()
    {
        var runtime = new SilkRuntime();
        using (runtime)
        {
            const int n = 200;
            const int window = 60;
            var values = new int[n];
            for (int i = 0; i < n; i++)
                values[i] = (i * 7 + 3) % 13;

            var cpu = NivaraColumn<int>.Create(values).RollingMean(window);
            var silk = SilkAnalytics.RollingMean(runtime, values.Select(v => (float)v).ToArray(), window);

            Assert.That(cpu.Length, Is.EqualTo(n));
            Assert.That(silk.Means.Length, Is.EqualTo(n));
            Assert.That(silk.Valid.Length, Is.EqualTo(n));
            Assert.That(silk.Valid.Count(x => x), Is.EqualTo(n - window + 1));

            double maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                if (!silk.Valid[i]) continue;
                double cpuVal = (double)cpu.GetValue(i)!;
                double diff = System.Math.Abs(cpuVal - silk.Means[i]);
                if (diff > maxDiff) maxDiff = diff;
                Assert.That(diff, Is.LessThanOrEqualTo(1e-3),
                    $"max diff exceeded at row {i}: {diff}");
            }
            Assert.That(maxDiff, Is.LessThanOrEqualTo(1e-3));
        }
    }
}
