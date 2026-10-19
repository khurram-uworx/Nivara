using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class GpuProfitabilityTests
{
    [Test]
    public void Decide_ReturnsNotRequested_WhenNeitherFlag()
    {
        var (offload, reason, _, _) = GpuProfitability.Decide(false, false, 100, 50);
        Assert.That(offload, Is.False);
        Assert.That(reason, Is.EqualTo("gpu not requested"));
    }

    [Test]
    public void Decide_ReturnsForced_WhenGpuForce()
    {
        var (offload, reason, _, _) = GpuProfitability.Decide(false, true, 100, 200);
        Assert.That(offload, Is.True);
        Assert.That(reason, Does.StartWith("forced"));
    }

    [Test]
    public void Decide_ReturnsNotProfitable_WhenEstimatedGpuTooSlow()
    {
        var (offload, reason, _, _) = GpuProfitability.Decide(true, false, 100, 200);
        Assert.That(offload, Is.False);
        Assert.That(reason, Does.StartWith("not profitable"));
    }

    [Test]
    public void Decide_ReturnsProfitable_WhenEstimatedGpuFaster()
    {
        var (offload, reason, _, _) = GpuProfitability.Decide(true, false, 100, 50);
        Assert.That(offload, Is.True);
        Assert.That(reason, Does.StartWith("profitable"));
    }

    [Test]
    public void EstimateCpuMs_ScalesSampleToFullRowCount()
    {
        double ms = GpuProfitability.EstimateCpuMs(10, 1000, 10000);
        Assert.That(ms, Is.EqualTo(100).Within(1e-9));
    }

    [Test]
    public void CalibrateCpuMs_ReturnsZero_WhenEmpty()
    {
        var ms = GpuProfitability.CalibrateCpuMs(Array.Empty<int>(), Array.Empty<int>(), Array.Empty<float>(), 0, 0);
        Assert.That(ms, Is.EqualTo(0));
    }
}
