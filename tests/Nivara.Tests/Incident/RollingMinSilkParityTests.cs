using Nivara;
using Nivara.Silk;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
[Category("Gpu")]
public class RollingMinSilkParityTests
{
    [Test]
    public void SilkRollingMin_MatchesCpu_FloatWindow()
    {
        var runtime = new SilkRuntime();
        using (runtime)
        {
            const int n = 200;
            const int window = 60;
            var values = new float[n];
            for (int i = 0; i < n; i++)
                values[i] = (i * 7 + 3) % 13;

            var cpu = NivaraColumn<float>.Create(values).RollingMin(window);
            var silk = SilkAnalytics.RollingMin(runtime, values, window);

            Assert.That(cpu.Length, Is.EqualTo(n));
            Assert.That(silk.Values.Length, Is.EqualTo(n));
            Assert.That(silk.Valid.Length, Is.EqualTo(n));
            Assert.That(silk.Valid.Count(x => x), Is.EqualTo(n - window + 1));

            for (int i = 0; i < n; i++)
            {
                if (!silk.Valid[i]) continue;
                float cpuVal = (float)cpu.GetValue(i)!;
                Assert.That(silk.Values[i], Is.EqualTo(cpuVal));
            }
        }
    }
}
