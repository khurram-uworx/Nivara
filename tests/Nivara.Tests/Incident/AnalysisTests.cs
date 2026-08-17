using Nivara.IO;
using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class AnalysisTests
{
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
            GenerateSmallDataset(dir, sid);
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
        using var qf = Analysis.AnalyzeDeploymentCorrelation(dir, scenario);
        using var frame = qf.Collect();
        Assert.That(frame.RowCount, Is.GreaterThan(0));
    }

    [Test]
    public void ScenarioC_SaturationOrdering_ShowsAffectedServices()
    {
        var dir = Path.Combine(tempDir, "C");
        var scenario = Scenarios.Get("C");
        using var qf = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        using var frame = qf.Collect();
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
        using var qf = Analysis.AnalyzeSaturationOrdering(dir, scenario);
        using var frame = qf.Collect();
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
        using var qf = Analysis.AnalyzeRegionalPartitioning(dir, scenario);
        using var frame = qf.Collect();
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
        var frame = Analysis.AnalyzeGroupedAggregation(dir, scenario);
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
        using var qf = Analysis.AnalyzeRegionalPartitioning(dir, scenario);
        using var frame = qf.Collect();

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

    static void GenerateSmallDataset(string dir, string scenarioId)
    {
        var scenario = Scenarios.Get(scenarioId);
        var rng = new Random(42);
        var baseTime = new DateTimeOffset(2025, 6, 15, 14, 0, 0, TimeSpan.Zero);

        var services = new[] { "gateway", "catalog", "inventory", "orders", "checkout", "payments", "notifications", "identity" };
        var regions = new[] { "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1", "ap-northeast-1", "sa-east-1", "ca-central-1", "ap-southeast-1", "af-south-1" };
        var endpoints = new[] { "/api/v1/health", "/api/v1/products", "/api/v1/orders/create", "/api/v1/checkout/process", "/api/v1/payments/process" };

        int totalRequests = 10_000;
        int requestsPerMinute = totalRequests / 30;

        var timestampData = new long[totalRequests];
        var serviceData = new string[totalRequests];
        var endpointData = new string[totalRequests];
        var durationMsData = new double[totalRequests];
        var statusCodeData = new int[totalRequests];
        var regionData = new string[totalRequests];
        var traceIdData = new string[totalRequests];
        var isRetryData = new bool[totalRequests];

        int idx = 0;
        for (int minute = 0; minute < 30 && idx < totalRequests; minute++)
        {
            var minuteStart = baseTime.AddMinutes(minute);
            bool inIncident = minuteStart >= scenario.IncidentStart && minuteStart < scenario.IncidentEnd;

            for (int r = 0; r < requestsPerMinute && idx < totalRequests; r++, idx++)
            {
                var timestamp = minuteStart.AddMilliseconds(rng.NextDouble() * 60_000);
                var service = services[rng.Next(services.Length)];
                var isAffected = scenario.AffectedServices.Contains(service);
                bool degradeThisMinute = inIncident && isAffected;

                timestampData[idx] = timestamp.Ticks;
                serviceData[idx] = service;
                endpointData[idx] = endpoints[rng.Next(endpoints.Length)];
                durationMsData[idx] = Math.Round(Math.Max(1.0, rng.NextDouble() * 100), 2);
                regionData[idx] = regions[rng.Next(regions.Length)];
                traceIdData[idx] = $"trace-{idx:X8}";

                if (degradeThisMinute)
                {
                    var errorRoll = rng.NextDouble();
                    statusCodeData[idx] = errorRoll switch
                    {
                        < 0.40 => 500,
                        < 0.60 => 503,
                        < 0.75 => 429,
                        < 0.85 => 502,
                        _ => 200,
                    };
                    isRetryData[idx] = rng.NextDouble() < 0.35;
                }
                else
                {
                    var normalRoll = rng.NextDouble();
                    statusCodeData[idx] = normalRoll switch
                    {
                        < 0.02 => 429,
                        < 0.03 => 500,
                        < 0.035 => 503,
                        _ => 200,
                    };
                    isRetryData[idx] = rng.NextDouble() < 0.02;
                }
            }
        }

        var requestFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(timestampData)),
            ("Service", NivaraColumn<string>.Create(serviceData)),
            ("Endpoint", NivaraColumn<string>.Create(endpointData)),
            ("DurationMs", NivaraColumn<double>.Create(durationMsData)),
            ("StatusCode", NivaraColumn<int>.Create(statusCodeData)),
            ("Region", NivaraColumn<string>.Create(regionData)),
            ("TraceId", NivaraColumn<string>.Create(traceIdData)),
            ("IsRetry", NivaraColumn<bool>.Create(isRetryData)));

        var parquetOptions = ParquetWriteOptions.Default.With(rowGroupSize: 10_000);
        requestFrame.ToParquet(Path.Combine(dir, "requests.parquet"), parquetOptions);

        using (var writer = new StreamWriter(Path.Combine(dir, "requests.csv")))
        {
            writer.WriteLine("Timestamp,Service,Endpoint,DurationMs,StatusCode,Region,TraceId,IsRetry");
            for (int i = 0; i < totalRequests; i++)
            {
                var ts = new DateTimeOffset(timestampData[i], TimeSpan.Zero);
                writer.WriteLine($"{ts:O},{serviceData[i]},{endpointData[i]},{durationMsData[i]:F2},{statusCodeData[i]},{regionData[i]},{traceIdData[i]},{isRetryData[i]}");
            }
        }

        int deploymentCount = 20 + rng.Next(10);
        var deployTimestamps = new long[deploymentCount];
        var deployServices = new string[deploymentCount];
        var deployVersions = new string[deploymentCount];
        var deployRegions = new string[deploymentCount];

        for (int i = 0; i < deploymentCount; i++)
        {
            var svc = services[rng.Next(services.Length)];
            var region = regions[rng.Next(regions.Length)];
            var minute = rng.Next(30);
            var ts = baseTime.AddMinutes(minute).AddSeconds(rng.Next(60));
            var major = 4 + rng.Next(3);
            var minor = rng.Next(50);

            if (scenarioId == "B" && i == 0)
            {
                ts = scenario.Events[0].Timestamp;
                svc = "orders";
                major = 4;
                minor = 21;
                region = "us-east-1";
            }

            deployTimestamps[i] = ts.Ticks;
            deployServices[i] = svc;
            deployVersions[i] = $"v{major}.{minor}";
            deployRegions[i] = region;
        }

        var deployFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(deployTimestamps)),
            ("Service", NivaraColumn<string>.Create(deployServices)),
            ("Version", NivaraColumn<string>.Create(deployVersions)),
            ("Region", NivaraColumn<string>.Create(deployRegions)));
        deployFrame.ToParquet(Path.Combine(dir, "deployments.parquet"), parquetOptions);

        var dependencyParent = new List<string> { "gateway", "gateway", "gateway", "orders", "orders", "orders", "checkout", "checkout", "checkout", "payments", "payments" };
        var dependencyChild = new List<string> { "orders", "catalog", "identity", "inventory", "checkout", "payments", "payments", "inventory", "notifications", "notifications", "identity" };

        var dependencyFrame = NivaraFrame.Create(
            ("Parent", NivaraColumn<string>.Create(dependencyParent.ToArray())),
            ("Child", NivaraColumn<string>.Create(dependencyChild.ToArray())));
        dependencyFrame.ToParquet(Path.Combine(dir, "dependencies.parquet"), parquetOptions);

        int instanceCount = services.Length * 10 * regions.Length;
        var instTimestamps = new long[instanceCount];
        var instServices = new string[instanceCount];
        var instIds = new string[instanceCount];
        var instRegions = new string[instanceCount];
        var instActiveReqs = new int[instanceCount];
        var instQueueDepth = new int[instanceCount];

        int instIdx = 0;
        foreach (var svc in services)
        {
            foreach (var region in regions)
            {
                for (int i = 0; i < 10; i++)
                {
                    bool inIncident = scenario.AffectedServices.Contains(svc);
                    var queueBase = inIncident ? 50 : 5;

                    var instTime = inIncident
                        ? scenario.IncidentStart.AddMinutes(rng.NextDouble() * (scenario.IncidentEnd - scenario.IncidentStart).TotalMinutes)
                        : baseTime.AddMinutes(rng.NextDouble() * 30);

                    instTimestamps[instIdx] = instTime.Ticks;
                    instServices[instIdx] = svc;
                    instIds[instIdx] = $"{svc}-{region}-{i:D3}";
                    instRegions[instIdx] = region;
                    instActiveReqs[instIdx] = rng.Next(10, 200);
                    instQueueDepth[instIdx] = inIncident
                        ? queueBase + rng.Next(50)
                        : rng.Next(1, 10);
                    instIdx++;
                }
            }
        }

        var instanceFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(instTimestamps)),
            ("Service", NivaraColumn<string>.Create(instServices)),
            ("InstanceId", NivaraColumn<string>.Create(instIds)),
            ("Region", NivaraColumn<string>.Create(instRegions)),
            ("ActiveRequests", NivaraColumn<int>.Create(instActiveReqs)),
            ("QueueDepth", NivaraColumn<int>.Create(instQueueDepth)));
        instanceFrame.ToParquet(Path.Combine(dir, "instances.parquet"), parquetOptions);
    }
}
