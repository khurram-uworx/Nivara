using Nivara.IO;
using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class IngestionTests
{
    string tempDir = null!;
    const int TotalRows = 10_000;
    const int RowGroupSize = 100;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-ingestion-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        GenerateSmallDataset(tempDir, "A");
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void LoadParquet_ReturnsCorrectRowCount()
    {
        using var qf = Ingestion.LoadParquet(Path.Combine(tempDir, "requests.parquet"));
        using var frame = qf.Collect();
        Assert.That(frame.RowCount, Is.EqualTo(TotalRows));
    }

    [Test]
    public void LoadCsv_ReturnsIdenticalData()
    {
        using var pqQf = Ingestion.LoadParquet(Path.Combine(tempDir, "requests.parquet"));
        using var pqFrame = pqQf.Collect();

        using var csvQf = Ingestion.LoadCsv(Path.Combine(tempDir, "requests.csv"));
        using var csvFrame = csvQf.Collect();

        Assert.That(csvFrame.RowCount, Is.EqualTo(pqFrame.RowCount));

        var pqServiceCol = pqFrame.GetColumn<string>("Service");
        var csvServiceCol = csvFrame.GetColumn<string>("Service");
        var pqStatusCol = pqFrame.GetColumn<int>("StatusCode");
        var csvStatusCol = csvFrame.GetColumn<int>("StatusCode");

        for (int i = 0; i < pqFrame.RowCount; i++)
        {
            Assert.That(csvServiceCol.GetValue(i), Is.EqualTo(pqServiceCol.GetValue(i)),
                $"Service mismatch at row {i}");
            Assert.That(csvStatusCol.GetValue(i), Is.EqualTo(pqStatusCol.GetValue(i)),
                $"StatusCode mismatch at row {i}");
        }
    }

    [Test]
    public async Task StreamChunks_YieldsExpectedChunkCount()
    {
        int chunkCount = 0;
        await foreach (var chunk in Ingestion.StreamChunks(
            Path.Combine(tempDir, "requests.parquet"), RowGroupSize))
        {
            chunkCount++;
            chunk.Dispose();
        }

        Assert.That(chunkCount, Is.EqualTo(TotalRows / RowGroupSize));
    }

    [Test]
    public async Task StreamChunks_DisposesResources()
    {
        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in Ingestion.StreamChunks(
            Path.Combine(tempDir, "requests.parquet"), RowGroupSize))
        {
            chunks.Add(chunk);
            if (chunks.Count >= 3) break;
        }

        foreach (var chunk in chunks)
            chunk.Dispose();

        Assert.That(chunks.Count, Is.EqualTo(3));
    }

    [Test]
    public async Task StreamChunks_CancellationStopsStream()
    {
        using var cts = new CancellationTokenSource();
        int chunkCount = 0;
        const int cancelAfter = 3;

        try
        {
            await foreach (var chunk in Ingestion.StreamChunks(
                Path.Combine(tempDir, "requests.parquet"), RowGroupSize, cts.Token))
            {
                chunkCount++;
                chunk.Dispose();
                if (chunkCount >= cancelAfter)
                    cts.Cancel();
            }
        }
        catch (OperationCanceledException)
        {
        }

        Assert.That(chunkCount, Is.EqualTo(cancelAfter));
    }

    static void GenerateSmallDataset(string dir, string scenarioId)
    {
        var scenario = Scenarios.Get(scenarioId);
        var rng = new Random(42);
        var baseTime = new DateTimeOffset(2025, 6, 15, 14, 0, 0, TimeSpan.Zero);

        var services = new[] { "gateway", "catalog", "inventory", "orders", "checkout", "payments", "notifications", "identity" };
        var regions = new[] { "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1" };
        var endpoints = new[] { "/api/v1/health", "/api/v1/products", "/api/v1/orders/create", "/api/v1/checkout/process", "/api/v1/payments/process" };

        int requestsPerMinute = TotalRows / 30;

        var timestampData = new long[TotalRows];
        var serviceData = new string[TotalRows];
        var endpointData = new string[TotalRows];
        var durationMsData = new double[TotalRows];
        var statusCodeData = new int[TotalRows];
        var regionData = new string[TotalRows];
        var traceIdData = new string[TotalRows];
        var isRetryData = new bool[TotalRows];

        int idx = 0;
        for (int minute = 0; minute < 30 && idx < TotalRows; minute++)
        {
            var minuteStart = baseTime.AddMinutes(minute);
            bool inIncident = minuteStart >= scenario.IncidentStart && minuteStart < scenario.IncidentEnd;

            for (int r = 0; r < requestsPerMinute && idx < TotalRows; r++, idx++)
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

        var parquetOptions = ParquetWriteOptions.Default.With(rowGroupSize: RowGroupSize);
        requestFrame.ToParquet(Path.Combine(dir, "requests.parquet"), parquetOptions);

        using (var writer = new StreamWriter(Path.Combine(dir, "requests.csv")))
        {
            writer.WriteLine("Timestamp,Service,Endpoint,DurationMs,StatusCode,Region,TraceId,IsRetry");
            for (int i = 0; i < TotalRows; i++)
            {
                var ts = new DateTimeOffset(timestampData[i], TimeSpan.Zero);
                writer.WriteLine($"{ts:O},{serviceData[i]},{endpointData[i]},{durationMsData[i]:F2},{statusCodeData[i]},{regionData[i]},{traceIdData[i]},{isRetryData[i]}");
            }
        }

        int deploymentCount = 10;
        var deployTimestamps = new long[deploymentCount];
        var deployServices = new string[deploymentCount];
        var deployVersions = new string[deploymentCount];
        var deployRegions = new string[deploymentCount];

        for (int i = 0; i < deploymentCount; i++)
        {
            deployTimestamps[i] = baseTime.AddMinutes(rng.Next(30)).Ticks;
            deployServices[i] = services[rng.Next(services.Length)];
            deployVersions[i] = $"v{4 + rng.Next(3)}.{rng.Next(50)}";
            deployRegions[i] = regions[rng.Next(regions.Length)];
        }

        var deployFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(deployTimestamps)),
            ("Service", NivaraColumn<string>.Create(deployServices)),
            ("Version", NivaraColumn<string>.Create(deployVersions)),
            ("Region", NivaraColumn<string>.Create(deployRegions)));
        deployFrame.ToParquet(Path.Combine(dir, "deployments.parquet"), parquetOptions);

        int instanceCount = services.Length * 5 * regions.Length;
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
                for (int i = 0; i < 5; i++)
                {
                    bool inIncident = scenario.AffectedServices.Contains(svc);
                    instTimestamps[instIdx] = (inIncident
                        ? scenario.IncidentStart.AddMinutes(rng.NextDouble() * (scenario.IncidentEnd - scenario.IncidentStart).TotalMinutes)
                        : baseTime.AddMinutes(rng.NextDouble() * 30)).Ticks;
                    instServices[instIdx] = svc;
                    instIds[instIdx] = $"{svc}-{region}-{i:D3}";
                    instRegions[instIdx] = region;
                    instActiveReqs[instIdx] = rng.Next(10, 200);
                    instQueueDepth[instIdx] = inIncident ? 50 + rng.Next(50) : rng.Next(1, 10);
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
