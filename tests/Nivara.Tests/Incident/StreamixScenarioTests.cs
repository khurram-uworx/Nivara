using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class StreamixScenarioTests
{
    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "StreamixScenarioTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", 10_000);
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public async Task FaultTolerantStreaming_ProcessesAllIncidentRows()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunFaultTolerantStreaming(
            tempDir, scenario, chunkSize: 1000);

        Assert.That(summary.TotalRows, Is.GreaterThan(0));
        Assert.That(summary.ChunksProcessed, Is.GreaterThan(0));
    }

    [Test]
    public async Task WindowedAnalytics_CollectsWindowResults()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunWindowedAnalytics(
            tempDir, scenario, chunkSize: 1000);

        Assert.That(summary.TotalRows, Is.GreaterThan(0));
        Assert.That(summary.Windows, Is.Not.Empty);
        Assert.That(summary.Windows[0].WindowStart, Is.Not.EqualTo(default(DateTimeOffset)));
        Assert.That(summary.Windows[0].AverageDurationMs, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public async Task OnlineAutoDiffLearning_CompletesTrainingBatches()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunOnlineAutoDiffLearning(
            tempDir, scenario, batchSize: 512, epochs: 1);

        Assert.That(summary.TrainingBatches, Is.GreaterThan(0));
        Assert.That(summary.FinalLoss, Is.Not.NaN);
    }
}
