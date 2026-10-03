using Nivara.Expressions;
using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class StreamixScenarioTests
{
    const int TotalRecords = 10_000;
    const int ChunkSize = 1000;

    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "StreamixScenarioTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", TotalRecords);
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
            tempDir, scenario, chunkSize: ChunkSize);

        Assert.Multiple(() =>
        {
            Assert.That(summary.TotalRows, Is.GreaterThan(0));
            Assert.That(summary.ChunksProcessed, Is.GreaterThan(0));
        });
    }

    [Test]
    public async Task FaultTolerantStreaming_LosesNoRowsAcrossTheStreamixBridge()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunFaultTolerantStreaming(
            tempDir, scenario, chunkSize: ChunkSize);

        // Counted a second way, off the same source and the same incident-window filter but
        // without the Streamix bridge. A bridge that silently dropped or re-delivered a chunk
        // would still satisfy TotalRows > 0; only a like-for-like comparison catches that.
        int expected;
        using (var query = Ingestion.LoadParquet(Path.Combine(tempDir, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(scenario.IncidentStart.Ticks))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(scenario.IncidentEnd.Ticks)))
        using (var frame = query.Collect())
        {
            expected = frame.RowCount;
        }

        Assert.That(expected, Is.GreaterThan(0), "the incident window should contain rows to stream");
        Assert.That(summary.TotalRows, Is.EqualTo(expected),
            "streamed rows should match the rows in the incident window -- a difference means the "
            + "Streamix bridge dropped or duplicated a chunk");
    }

    [Test]
    public async Task WindowedAnalytics_CollectsWindowResults()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunWindowedAnalytics(
            tempDir, scenario, chunkSize: ChunkSize);

        Assert.That(summary.TotalRows, Is.GreaterThan(0));
        Assert.That(summary.Windows, Is.Not.Empty);
        Assert.That(summary.Windows[0].WindowStart, Is.Not.EqualTo(default(DateTimeOffset)));
        Assert.That(summary.Windows[0].AverageDurationMs, Is.GreaterThanOrEqualTo(0));
        Assert.That(summary.Windows[0].RollingAvgDurationMs, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public async Task WindowedAnalytics_WindowsPartitionTheIncidentRowsExactly()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunWindowedAnalytics(
            tempDir, scenario, chunkSize: ChunkSize);

        // Exact, not a tolerance band: totalRows is accumulated per window, so a partition that
        // double-counted or lost a chunk would break the identity. WindowResult.Empty contributes
        // 0 to both sides, so a window that arrived with no rows does not skew it.
        Assert.Multiple(() =>
        {
            Assert.That(summary.Windows.Sum(w => w.RowCount), Is.EqualTo(summary.TotalRows),
                "every counted row should belong to exactly one window");

            foreach (var window in summary.Windows)
            {
                var where = $"window [{window.WindowStart:O} .. {window.WindowEnd:O}]";

                Assert.That(window.RowCount, Is.GreaterThanOrEqualTo(0), $"{where}: negative row count");

                Assert.That(window.ErrorRate, Is.InRange(0.0, 1.0),
                    $"{where}: ErrorRate {window.ErrorRate} is outside [0, 1]");
                Assert.That(double.IsFinite(window.AverageDurationMs), Is.True,
                    $"{where}: AverageDurationMs {window.AverageDurationMs} is not finite");
                Assert.That(double.IsFinite(window.RollingAvgDurationMs), Is.True,
                    $"{where}: RollingAvgDurationMs {window.RollingAvgDurationMs} is not finite");

                if (window.RowCount == 0)
                    continue;

                Assert.That(window.WindowEnd, Is.GreaterThanOrEqualTo(window.WindowStart),
                    $"{where}: WindowEnd precedes WindowStart");
                Assert.That(window.WindowStart, Is.GreaterThanOrEqualTo(scenario.IncidentStart),
                    $"{where}: WindowStart precedes the incident window the query filters to");
                Assert.That(window.WindowEnd, Is.LessThanOrEqualTo(scenario.IncidentEnd),
                    $"{where}: WindowEnd follows the incident window the query filters to");
            }
        });
    }

    [Test]
    public void WindowResult_Empty_IsAZeroRowSentinel()
    {
        var empty = StreamixScenarios.WindowResult.Empty;

        // WindowResult.Empty is the value RunWindowedAnalytics returns for a window that arrives
        // with no rows, which is why the partition identity above tolerates a zero row count.
        Assert.Multiple(() =>
        {
            Assert.That(empty.RowCount, Is.EqualTo(0));
            Assert.That(empty.WindowStart, Is.EqualTo(default(DateTimeOffset)));
            Assert.That(empty.WindowEnd, Is.EqualTo(default(DateTimeOffset)));
            Assert.That(empty.AverageDurationMs, Is.EqualTo(0.0));
            Assert.That(empty.RollingAvgDurationMs, Is.EqualTo(0.0));
            Assert.That(empty.ErrorRate, Is.EqualTo(0.0));
        });
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

    [Test]
    public async Task OnlineAutoDiffLearning_EndsAtAFiniteNonNegativeLoss()
    {
        var scenario = Scenarios.Get("A");
        var summary = await StreamixScenarios.RunOnlineAutoDiffLearning(
            tempDir, scenario, batchSize: 512, epochs: 1);

        Assert.Multiple(() =>
        {
            Assert.That(summary.TrainingBatches, Is.GreaterThan(0));
            Assert.That(summary.Model, Is.Not.Null);

            // Is.Not.NaN passes for +/-Infinity, which would sail through the assertion above even
            // though it means training diverged. MSE cannot be negative, so both bounds are real.
            Assert.That(float.IsFinite(summary.FinalLoss), Is.True,
                $"FinalLoss diverged to {summary.FinalLoss}");
            Assert.That(summary.FinalLoss, Is.GreaterThanOrEqualTo(0f),
                $"FinalLoss {summary.FinalLoss} is negative, but MSELoss cannot be");
        });
    }
}
