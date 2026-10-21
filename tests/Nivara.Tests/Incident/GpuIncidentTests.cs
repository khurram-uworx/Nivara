using Nivara.Samples.Incident;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class GpuIncidentTests
{
    const int TotalRecords = 10_000;

    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-gpu-{Guid.NewGuid():N}");
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
    public void GroupedAggregation_GpuPath_MatchesCpuReference()
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
            var scenario = Scenarios.A;

            using var cpu = Analysis.AnalyzeGroupedAggregation(tempDir, scenario, useGpu: false);
            using var gpu = Analysis.AnalyzeGroupedAggregation(tempDir, scenario, useGpu: true, gpuForce: true);

            var decision = Analysis.LastGpuOffloadDecision;
            Assert.That(decision, Is.Not.Null);
            Assert.That(decision!.Offloaded, Is.True,
                $"a GPU is present but the forced GPU path fell back to CPU: {decision.Reason}");

            Assert.That(cpu.RowCount, Is.EqualTo(gpu.RowCount));

            var cpuByService = IndexByService(cpu);
            var gpuByService = IndexByService(gpu);

            foreach (var (service, cpuTotal) in cpuByService)
            {
                Assert.That(gpuByService, Does.ContainKey(service));
                Assert.That(gpuByService[service].Total, Is.EqualTo(cpuTotal.Total));
                Assert.That(gpuByService[service].Avg, Is.EqualTo(cpuTotal.Avg).Within(1e-4));
            }
        }
    }

    [Test]
    public void GroupedAggregation_GpuPath_MatchesCpuReference_At1K()
    {
        RunParityAt(1_000);
    }

    [Test]
    public void GroupedAggregation_GpuPath_MatchesCpuReference_At10K()
    {
        RunParityAt(10_000);
    }

    [Test]
    public void GroupedAggregation_GpuPath_MatchesCpuReference_At100K()
    {
        RunParityAt(100_000);
    }

    static void RunParityAt(int recordCount)
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"inc-gpu-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            DatasetGenerator.GenerateFromRecordCount(tempDir, "A", recordCount);

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
                var scenario = Scenarios.A;
                using var cpu = Analysis.AnalyzeGroupedAggregation(tempDir, scenario, useGpu: false);
                using var gpu = Analysis.AnalyzeGroupedAggregation(tempDir, scenario, useGpu: true, gpuForce: true);

                var decision = Analysis.LastGpuOffloadDecision;
                Assert.That(decision, Is.Not.Null);
                Assert.That(decision!.Offloaded, Is.True,
                    $"a GPU is present but the forced GPU path fell back to CPU: {decision.Reason}");

                Assert.That(cpu.RowCount, Is.EqualTo(gpu.RowCount));

                var cpuByService = IndexByService(cpu);
                var gpuByService = IndexByService(gpu);

                foreach (var (service, cpuTotal) in cpuByService)
                {
                    Assert.That(gpuByService, Does.ContainKey(service));
                    Assert.That(gpuByService[service].Total, Is.EqualTo(cpuTotal.Total));
                    Assert.That(gpuByService[service].Avg, Is.EqualTo(cpuTotal.Avg).Within(1e-3));
                }
            }
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    static Dictionary<string, (int Total, double Avg)> IndexByService(NivaraFrame frame)
    {
        var serviceCol = frame.GetColumn<string>("Service");
        var totalCol = frame.GetColumn<int>("TotalRequests");
        var avgCol = frame.GetColumn<double>("AvgDurationMs");
        var result = new Dictionary<string, (int Total, double Avg)>();
        for (int i = 0; i < frame.RowCount; i++)
        {
            result[(string)serviceCol.GetValue(i)!] = ((int)totalCol.GetValue(i)!, (double)avgCol.GetValue(i)!);
        }
        return result;
    }
}
