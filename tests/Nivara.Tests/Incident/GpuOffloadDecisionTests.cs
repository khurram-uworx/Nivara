using Nivara.Samples.Incident;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class GpuOffloadDecisionTests
{
    const int TotalRecords = 5_000;

    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-dec-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", TotalRecords);
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void ForcedRun_RecordsOffloadedDecisionWithMeasuredFields()
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
            using var frame = Analysis.AnalyzeGroupedAggregation(tempDir, Scenarios.A, useGpu: true, gpuForce: true);
            var decision = Analysis.LastGpuOffloadDecision;
            Assert.That(decision, Is.Not.Null);
            Assert.That(decision!.Offloaded, Is.True);
            Assert.That(decision.CpuMs, Is.GreaterThanOrEqualTo(0));
            Assert.That(decision.GpuMs, Is.GreaterThanOrEqualTo(0));
            Assert.That(decision.TransferBytes, Is.GreaterThan(0));
            Assert.That(decision.Kernel, Is.Not.Null.And.Not.Empty);
        }
    }

    [Test]
    public void GatedRun_RecordsAReasonString()
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
            using var frame = Analysis.AnalyzeGroupedAggregation(tempDir, Scenarios.A, useGpu: true, gpuForce: false);
            var decision = Analysis.LastGpuOffloadDecision;
            Assert.That(decision, Is.Not.Null);
            Assert.That(decision!.Reason, Does.Contain("profitable").Or.Contain("not profitable")
                .Or.Contain("gpu not requested").Or.Contain("forced"));
        }
    }
}
