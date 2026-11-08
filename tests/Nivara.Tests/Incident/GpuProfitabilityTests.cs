using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class SilkProfitabilityTests
{
    [Test]
    public void Decide_ReturnsNotRequested_WhenNeitherFlag()
    {
        var (offload, reason, _, _) = SilkProfitability.Decide(false, false, 100, 50);
        Assert.That(offload, Is.False);
        Assert.That(reason, Is.EqualTo("gpu not requested"));
    }

    [Test]
    public void Decide_ReturnsForced_WhenGpuForce()
    {
        var (offload, reason, _, _) = SilkProfitability.Decide(false, true, 100, 200);
        Assert.That(offload, Is.True);
        Assert.That(reason, Does.StartWith("forced"));
    }

    [Test]
    public void Decide_ReturnsNotProfitable_WhenEstimatedGpuTooSlow()
    {
        var (offload, reason, _, _) = SilkProfitability.Decide(true, false, 100, 200);
        Assert.That(offload, Is.False);
        Assert.That(reason, Does.StartWith("not profitable"));
    }

    [Test]
    public void Decide_ReturnsProfitable_WhenEstimatedGpuFaster()
    {
        var (offload, reason, _, _) = SilkProfitability.Decide(true, false, 100, 50);
        Assert.That(offload, Is.True);
        Assert.That(reason, Does.StartWith("profitable"));
    }

    [Test]
    public void Decide_WhenEstimatedGpuEqualsCpu_IsNotProfitable()
    {
        var (offload, reason, _, _) = SilkProfitability.Decide(true, false, 100, 100);
        Assert.That(offload, Is.False);
        Assert.That(reason, Does.StartWith("not profitable"));
    }

    [Test]
    public void EstimateCpuMs_ScalesSampleToFullRowCount()
    {
        double ms = SilkProfitability.EstimateCpuMs(10, 1000, 10000);
        Assert.That(ms, Is.EqualTo(100).Within(1e-9));
    }

    [Test]
    public void EstimateGpuMs_ChargesFixedTransferAndPerRowCosts()
    {
        // 1 ms fixed + 12_000 bytes / 5_000_000 bytes-per-ms + 1_000 rows * 0.0003 ms/row.
        double ms = SilkProfitability.EstimateGpuMs(1000, 12_000);
        Assert.That(ms, Is.EqualTo(1.0 + 0.0024 + 0.3).Within(1e-9));
    }

    [Test]
    public void CalibrateCpuMs_ReturnsZero_WhenEmpty()
    {
        var ms = SilkProfitability.CalibrateCpuMs(Array.Empty<int>(), Array.Empty<int>(), Array.Empty<float>(), 0, 0);
        Assert.That(ms, Is.EqualTo(0));
    }

    [Test]
    public void CalibrateCpuMs_RejectsMismatchedArrays()
    {
        Assert.That(
            () => SilkProfitability.CalibrateCpuMs(new[] { 0 }, Array.Empty<int>(), new[] { 1f }, 1, 1),
            Throws.ArgumentException);
    }

    [Test]
    public void EstimateCpuMs_IncreasesWithRowCount()
    {
        double small = SilkProfitability.EstimateCpuMs(10, 1000, 1000);
        double large = SilkProfitability.EstimateCpuMs(10, 1000, 10000);
        Assert.That(large, Is.GreaterThan(small));
    }

    [Test]
    public void EstimateGpuMs_IncreasesWithTransferBytes()
    {
        double small = SilkProfitability.EstimateGpuMs(1000, 1_000);
        double large = SilkProfitability.EstimateGpuMs(1000, 100_000);
        Assert.That(large, Is.GreaterThan(small));
    }

    [Test]
    public void Decide_FlipsToProfitable_AsCpuEstimateGrows()
    {
        var (notProfitable, _, _, _) = SilkProfitability.Decide(true, false, 10, 100);
        var (profitable, _, _, _) = SilkProfitability.Decide(true, false, 1000, 100);
        Assert.That(notProfitable, Is.False);
        Assert.That(profitable, Is.True);
    }
}
