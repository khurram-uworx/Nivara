using Nivara.IO;
using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class AnalysisTests
{
    const int TotalRecords = 10_000;

    string tempDir = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-analysis-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        foreach (var sid in new[] { "A", "B", "C", "D" })
        {
            var dir = Path.Combine(tempDir, sid);
            Directory.CreateDirectory(dir);
            DatasetGenerator.GenerateFromRecordCount(dir, sid, TotalRecords);
        }
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("C")]
    [TestCase("D")]
    public void DegradationOrdering_NonEmptyResults(string sid)
    {
        var dir = Path.Combine(tempDir, sid);
        var scenario = Scenarios.Get(sid);
        using var qf = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame = qf.Collect();
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("C")]
    [TestCase("D")]
    public void DeploymentCorrelation_NonEmptyResults(string sid)
    {
        var dir = Path.Combine(tempDir, sid);
        var scenario = Scenarios.Get(sid);
        using var frame = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [Test]
    public void ScenarioC_SaturationOrdering_ShowsAffectedServices()
    {
        var dir = Path.Combine(tempDir, "C");
        var scenario = Scenarios.Get("C");
        using var frame = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        Assert.That(frame.RowCount, Is.GreaterThan(0));

        var serviceCol = frame.GetColumn<string>("Service");
        var services = new HashSet<string>();
        for (int i = 0; i < frame.RowCount; i++)
            services.Add((string)serviceCol.GetValue(i)!);

        Assert.That(services, Does.Contain("gateway"));
        Assert.That(services, Does.Contain("orders"));
        Assert.That(services, Does.Contain("payments"));
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("D")]
    public void SaturationOrdering_NonEmptyResults(string sid)
    {
        var dir = Path.Combine(tempDir, sid);
        var scenario = Scenarios.Get(sid);
        using var frame = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("C")]
    [TestCase("D")]
    public void RegionalPartitioning_NonEmptyResults(string sid)
    {
        var dir = Path.Combine(tempDir, sid);
        var scenario = Scenarios.Get(sid);
        using var frame = Analysis.AnalyzeRegionalPartitioning(dir, scenario);
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [TestCase("A")]
    [TestCase("B")]
    [TestCase("C")]
    [TestCase("D")]
    public void GroupedAggregation_NonEmptyResults(string sid)
    {
        var dir = Path.Combine(tempDir, sid);
        var scenario = Scenarios.Get(sid);
        using var frame = Analysis.AnalyzeGroupedAggregation(dir, scenario);
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [Test]
    public void DegradationOrdering_Determinism()
    {
        var dir = Path.Combine(tempDir, "A");
        var scenario = Scenarios.Get("A");

        using var qf1 = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame1 = qf1.Collect();
        using var qf2 = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame2 = qf2.Collect();

        Assert.That(frame2.RowCount, Is.EqualTo(frame1.RowCount));
    }

    [Test]
    public void Diagnostics_ReturnsValidData()
    {
        var dir = Path.Combine(tempDir, "A");
        var scenario = Scenarios.Get("A");
        using var qf = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame = qf.Collect();
        var diag = qf.GetExecutionDiagnostics();
        Assert.That(diag, Is.Not.Null);
        Assert.That(diag!.RowsRead, Is.GreaterThan(0));
        Assert.That(diag.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void ScenarioA_DegradationOrdering_OrdersFirst()
    {
        var dir = Path.Combine(tempDir, "A");
        var scenario = Scenarios.Get("A");
        using var qf = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame = qf.Collect();

        var serviceCol = frame.GetColumn<string>("Service");
        var services = new HashSet<string>();
        for (int i = 0; i < frame.RowCount; i++)
            services.Add((string)serviceCol.GetValue(i)!);

        Assert.That(services, Does.Contain("orders"));
        Assert.That(services, Does.Contain("checkout"));
        Assert.That(services, Does.Contain("payments"));
        Assert.That(services, Does.Contain("gateway"));
    }

    [Test]
    public void ScenarioB_DeploymentCorrelation_DeployAtMinute17()
    {
        var dir = Path.Combine(tempDir, "B");
        var scenario = Scenarios.Get("B");

        var deployFrame = NivaraParquetReader.ReadParquet(Path.Combine(dir, "deployments.parquet"));
        var deployTsCol = deployFrame.GetColumn<long>("Timestamp");
        var deploySvcCol = deployFrame.GetColumn<string>("Service");

        long firstDeployTs = (long)deployTsCol.GetValue(0)!;
        string firstDeploySvc = (string)deploySvcCol.GetValue(0)!;

        Assert.That(firstDeploySvc, Is.EqualTo("orders"));
        Assert.That(firstDeployTs, Is.EqualTo(scenario.Events[0].Timestamp.Ticks));

        deployFrame.Dispose();
    }

    [Test]
    public void ScenarioD_RegionalPartitioning_ApSouth1Present()
    {
        var dir = Path.Combine(tempDir, "D");
        var scenario = Scenarios.Get("D");
        using var frame = Analysis.AnalyzeRegionalPartitioning(dir, scenario);

        var regionCol = frame.GetColumn<string>("Region");
        var regions = new HashSet<string>();
        for (int i = 0; i < frame.RowCount; i++)
            regions.Add((string)regionCol.GetValue(i)!);

        Assert.That(regions, Does.Contain("ap-south-1"));
    }

    [Test]
    public void ParquetCsvConvergence_SameAnalysisSameResults()
    {
        var dir = Path.Combine(tempDir, "A");

        using var pqQf = Ingestion.LoadParquet(Path.Combine(dir, "requests.parquet"));
        using var pqFrame = pqQf.Collect();

        using var csvQf = Ingestion.LoadCsv(Path.Combine(dir, "requests.csv"));
        using var csvFrame = csvQf.Collect();

        Assert.That(csvFrame.RowCount, Is.EqualTo(pqFrame.RowCount));
    }

    [Test]
    public void ReplayConvergence_MaterializeThenReAnalyze_SameResults()
    {
        var dir = Path.Combine(tempDir, "A");
        var scenario = Scenarios.Get("A");

        using var qf = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame = qf.Collect();

        var serviceCol = frame.GetColumn<string>("Service");
        var services = new HashSet<string>();
        for (int i = 0; i < frame.RowCount; i++)
            services.Add((string)serviceCol.GetValue(i)!);

        using var qf2 = Analysis.AnalyzeDegradationOrdering(dir, scenario);
        using var frame2 = qf2.Collect();
        Assert.That(frame2.RowCount, Is.EqualTo(frame.RowCount));

        var serviceCol2 = frame2.GetColumn<string>("Service");
        var services2 = new HashSet<string>();
        for (int i = 0; i < frame2.RowCount; i++)
            services2.Add((string)serviceCol2.GetValue(i)!);

        Assert.That(services2, Is.EquivalentTo(services));
    }

    [Test]
    public void DeploymentCorrelation_B_HasDeploymentColumns()
    {
        var dir = Path.Combine(tempDir, "B");
        var scenario = Scenarios.Get("B");
        using var frame = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);

        Assert.That(frame.ColumnNames, Does.Contain("DeploymentVersion"));
        Assert.That(frame.ColumnNames, Does.Contain("TimeSinceDeploySec"));
        Assert.That(frame.ColumnNames, Does.Contain("ErrorCategory"));

        var versionCol = frame.GetColumn<string>("DeploymentVersion");
        int nonEmptyVersions = 0;
        for (int i = 0; i < frame.RowCount; i++)
        {
            var v = (string)versionCol.GetValue(i)!;
            if (!string.IsNullOrEmpty(v)) nonEmptyVersions++;
        }
        Assert.That(nonEmptyVersions, Is.GreaterThan(0));
    }

    [Test]
    public void DeploymentCorrelation_B_ErrorCategoriesAreValid()
    {
        var dir = Path.Combine(tempDir, "B");
        var scenario = Scenarios.Get("B");
        using var frame = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);

        var categoryCol = frame.GetColumn<string>("ErrorCategory");
        var categories = new HashSet<string>();
        for (int i = 0; i < frame.RowCount; i++)
            categories.Add((string)categoryCol.GetValue(i)!);

        Assert.That(categories, Does.Contain("server_error"));
        Assert.That(categories, Does.Contain("success"));
    }

    [Test]
    public void DeploymentCorrelation_Determinism()
    {
        var dir = Path.Combine(tempDir, "B");
        var scenario = Scenarios.Get("B");
        using var frame1 = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);
        using var frame2 = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);
        Assert.That(frame2.RowCount, Is.EqualTo(frame1.RowCount));
    }

    [Test]
    public void SaturationOrdering_C_HasQuantileAndPeakColumns()
    {
        var dir = Path.Combine(tempDir, "C");
        var scenario = Scenarios.Get("C");
        using var frame = Analysis.AnalyzeSaturationOrdering(dir, scenario);

        Assert.That(frame.ColumnNames, Does.Contain("PeakQueueDepth"));
        Assert.That(frame.ColumnNames, Does.Contain("P50QueueDepth"));
        Assert.That(frame.ColumnNames, Does.Contain("P95QueueDepth"));
        Assert.That(frame.ColumnNames, Does.Contain("P99QueueDepth"));
        Assert.That(frame.ColumnNames, Does.Contain("StdDevQueueDepth"));

        var peakCol = frame.GetColumn<int>("PeakQueueDepth");
        var p50Col = frame.GetColumn<double>("P50QueueDepth");
        var p95Col = frame.GetColumn<double>("P95QueueDepth");
        var p99Col = frame.GetColumn<double>("P99QueueDepth");

        for (int i = 0; i < frame.RowCount; i++)
        {
            var peak = (int)peakCol.GetValue(i)!;
            var p50 = (double)p50Col.GetValue(i)!;
            var p95 = (double)p95Col.GetValue(i)!;
            var p99 = (double)p99Col.GetValue(i)!;
            Assert.That(peak, Is.GreaterThanOrEqualTo(0),
                $"PeakQueueDepth ({peak}) should be >= 0 at row {i}");
            Assert.That(p95, Is.GreaterThanOrEqualTo(p50),
                $"P95 ({p95}) should be >= P50 ({p50}) at row {i}");
            Assert.That(p99, Is.GreaterThanOrEqualTo(p95),
                $"P99 ({p99}) should be >= P95 ({p95}) at row {i}");
        }
    }

    [Test]
    public void SaturationOrdering_Determinism()
    {
        var dir = Path.Combine(tempDir, "C");
        var scenario = Scenarios.Get("C");
        using var frame1 = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        using var frame2 = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        Assert.That(frame2.RowCount, Is.EqualTo(frame1.RowCount));
    }

    [Test]
    public void RegionalPartitioning_D_HasErrorRateRankAndPercentRank()
    {
        var dir = Path.Combine(tempDir, "D");
        var scenario = Scenarios.Get("D");
        using var frame = Analysis.AnalyzeRegionalPartitioning(dir, scenario);

        Assert.That(frame.ColumnNames, Does.Contain("ErrorRate"));
        Assert.That(frame.ColumnNames, Does.Contain("ErrorRank"));
        Assert.That(frame.ColumnNames, Does.Contain("MaxDurationPercentRank"));
        Assert.That(frame.ColumnNames, Does.Contain("P50Duration"));
        Assert.That(frame.ColumnNames, Does.Contain("P95Duration"));

        var errorRateCol = frame.GetColumn<double>("ErrorRate");
        var rankCol = frame.GetColumn<long>("ErrorRank");
        var pctRankCol = frame.GetColumn<double>("MaxDurationPercentRank");

        for (int i = 0; i < frame.RowCount; i++)
        {
            var rate = (double)errorRateCol.GetValue(i)!;
            Assert.That(rate, Is.GreaterThanOrEqualTo(0.0).And.LessThanOrEqualTo(1.0),
                $"ErrorRate out of range at row {i}");
            var rank = (long)rankCol.GetValue(i)!;
            Assert.That(rank, Is.GreaterThan(0),
                $"ErrorRank should be positive at row {i}");
            var pctRank = (double)pctRankCol.GetValue(i)!;
            Assert.That(pctRank, Is.GreaterThanOrEqualTo(0.0).And.LessThanOrEqualTo(1.0),
                $"MaxDurationPercentRank out of range at row {i}");
        }
    }

    [Test]
    public void RegionalPartitioning_Determinism()
    {
        var dir = Path.Combine(tempDir, "D");
        var scenario = Scenarios.Get("D");
        using var frame1 = Analysis.AnalyzeRegionalPartitioning(dir, scenario);
        using var frame2 = Analysis.AnalyzeRegionalPartitioning(dir, scenario);
        Assert.That(frame2.RowCount, Is.EqualTo(frame1.RowCount));
    }
}
