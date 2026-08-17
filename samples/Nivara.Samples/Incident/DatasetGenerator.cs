using Nivara.IO;

namespace Nivara.Samples.Incident;

public static class DatasetGenerator
{
    static readonly string[] Services = ["gateway", "catalog", "inventory", "orders", "checkout", "payments", "notifications", "identity"];
    static readonly string[] Regions = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-south-1", "ap-northeast-1", "sa-east-1", "ca-central-1", "ap-southeast-1", "af-south-1"];

    static readonly Dictionary<string, (string[] Endpoints, double BaseMeanMs, double BaseStdMs)> ServiceProfiles = new()
    {
        ["gateway"] = (["/api/v1/health", "/api/v1/status", "/api/v1/routes"], 8, 2),
        ["catalog"] = (["/api/v1/products", "/api/v1/search", "/api/v1/categories"], 25, 8),
        ["inventory"] = (["/api/v1/stock", "/api/v1/reserve", "/api/v1/adjust"], 30, 10),
        ["orders"] = (["/api/v1/orders/create", "/api/v1/orders/status", "/api/v1/orders/cancel"], 50, 15),
        ["checkout"] = (["/api/v1/checkout/process", "/api/v1/checkout/validate", "/api/v1/cart"], 80, 20),
        ["payments"] = (["/api/v1/payments/process", "/api/v1/payments/refund", "/api/v1/payments/verify"], 120, 30),
        ["notifications"] = (["/api/v1/notify/email", "/api/v1/notify/sms", "/api/v1/notify/push"], 40, 12),
        ["identity"] = (["/api/v1/auth/login", "/api/v1/auth/token", "/api/v1/auth/validate"], 20, 6),
    };

    static readonly Dictionary<string, List<(string Child, double Weight)>> DependencyGraph = new()
    {
        ["gateway"] = [("orders", 0.4), ("catalog", 0.3), ("identity", 0.3)],
        ["orders"] = [("inventory", 0.3), ("checkout", 0.4), ("payments", 0.3)],
        ["checkout"] = [("payments", 0.5), ("inventory", 0.3), ("notifications", 0.2)],
        ["payments"] = [("notifications", 0.6), ("identity", 0.4)],
        ["catalog"] = [],
        ["inventory"] = [],
        ["notifications"] = [],
        ["identity"] = [],
    };

    public static void Generate(string datasetPath, string scenarioId, int scale)
        => GenerateFromRecordCount(datasetPath, scenarioId, 10_000_000L * scale);

    public static void GenerateFromRecordCount(string datasetPath, string scenarioId, long totalRecords)
    {
        var scenario = Scenarios.Get(scenarioId);
        Directory.CreateDirectory(datasetPath);

        var rng = new Random(42);
        var baseTime = new DateTimeOffset(2025, 6, 15, 14, 0, 0, TimeSpan.Zero);
        var incidentStart = scenario.IncidentStart;
        var incidentEnd = scenario.IncidentEnd;
        var durationMinutes = 30.0;

        int totalRequests = (int)totalRecords;
        int requestsPerMinute = (int)(totalRequests / durationMinutes);

        var affectedRegion = scenarioId == "D" ? "ap-south-1" : null;

        var servicesPerInstance = new Dictionary<string, int>();
        foreach (var svc in Services)
            servicesPerInstance[svc] = 10 + rng.Next(5);

        var timestampData = new long[totalRequests];
        var serviceData = new string[totalRequests];
        var endpointData = new string[totalRequests];
        var durationMsData = new double[totalRequests];
        var statusCodeData = new int[totalRequests];
        var regionData = new string[totalRequests];
        var traceIdData = new string[totalRequests];
        var isRetryData = new bool[totalRequests];

        int requestIdx = 0;
        for (int minute = 0; minute < (int)durationMinutes; minute++)
        {
            var minuteStart = baseTime.AddMinutes(minute);
            var minuteEnd = baseTime.AddMinutes(minute + 1);
            bool inIncident = minuteStart >= incidentStart && minuteStart < incidentEnd;

            for (int r = 0; r < requestsPerMinute && requestIdx < totalRequests; r++, requestIdx++)
            {
                var timestamp = minuteStart.AddMilliseconds(rng.NextDouble() * 60_000);
                var service = Services[rng.Next(Services.Length)];
                var (endpoints, baseMean, baseStd) = ServiceProfiles[service];
                var endpoint = endpoints[rng.Next(endpoints.Length)];
                var region = Regions[rng.Next(Regions.Length)];

                double meanMs = baseMean;
                double stdMs = baseStd;
                bool isAffected = scenario.AffectedServices.Contains(service);
                bool isAffectedRegion = affectedRegion != null && region == affectedRegion;
                bool degradeThisMinute = inIncident && (isAffected || isAffectedRegion);

                if (degradeThisMinute)
                {
                    var eventForService = scenario.Events
                        .Where(e => e.Service == service)
                        .OrderBy(e => e.Timestamp)
                        .FirstOrDefault();
                    if (eventForService != null)
                    {
                        var minutesSinceEvent = (timestamp - eventForService.Timestamp).TotalMinutes;
                        var rampFactor = Math.Min(1.0, Math.Max(0, minutesSinceEvent / 3.0));
                        meanMs *= 1.0 + (eventForService.Magnitude - 1.0) * rampFactor;
                        stdMs *= 1.0 + (eventForService.Magnitude - 1.0) * 0.5 * rampFactor;
                    }
                }

                var durationMs = Math.Max(1.0, BoxMuller(rng, meanMs, stdMs));

                int statusCode;
                if (degradeThisMinute)
                {
                    var errorRoll = rng.NextDouble();
                    statusCode = errorRoll switch
                    {
                        < 0.40 => 500,
                        < 0.60 => 503,
                        < 0.75 => 429,
                        < 0.85 => 502,
                        _ => 200,
                    };
                }
                else
                {
                    var normalRoll = rng.NextDouble();
                    statusCode = normalRoll switch
                    {
                        < 0.02 => 429,
                        < 0.03 => 500,
                        < 0.035 => 503,
                        _ => 200,
                    };
                }

                bool isRetry = !degradeThisMinute ? rng.NextDouble() < 0.02
                    : rng.NextDouble() < 0.35;

                timestampData[requestIdx] = timestamp.Ticks;
                serviceData[requestIdx] = service;
                endpointData[requestIdx] = endpoint;
                durationMsData[requestIdx] = Math.Round(durationMs, 2);
                statusCodeData[requestIdx] = statusCode;
                regionData[requestIdx] = region;
                traceIdData[requestIdx] = $"trace-{requestIdx:X8}";
                isRetryData[requestIdx] = isRetry;
            }
        }

        int deploymentCount = 20 + rng.Next(10);
        var deployTimestamps = new long[deploymentCount];
        var deployServices = new string[deploymentCount];
        var deployVersions = new string[deploymentCount];
        var deployRegions = new string[deploymentCount];

        for (int i = 0; i < deploymentCount; i++)
        {
            var svc = Services[rng.Next(Services.Length)];
            var region = Regions[rng.Next(Regions.Length)];
            var minute = rng.Next((int)durationMinutes);
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

        var dependencyParent = new List<string>();
        var dependencyChild = new List<string>();
        foreach (var (parent, children) in DependencyGraph)
        {
            foreach (var (child, _) in children)
            {
                dependencyParent.Add(parent);
                dependencyChild.Add(child);
            }
        }

        int instanceCount = 0;
        foreach (var svc in Services)
            instanceCount += servicesPerInstance[svc] * Regions.Length;

        var instTimestamps = new long[instanceCount];
        var instServices = new string[instanceCount];
        var instIds = new string[instanceCount];
        var instRegions = new string[instanceCount];
        var instActiveReqs = new int[instanceCount];
        var instQueueDepth = new int[instanceCount];

        int instIdx = 0;
        foreach (var svc in Services)
        {
            int perRegion = servicesPerInstance[svc];
            foreach (var region in Regions)
            {
                for (int i = 0; i < perRegion; i++)
                {
                    bool inIncident = baseTime.AddMinutes(10) >= incidentStart;
                    var queueBase = inIncident && scenario.AffectedServices.Contains(svc) ? 50 : 5;

                    instTimestamps[instIdx] = baseTime.Ticks;
                    instServices[instIdx] = svc;
                    instIds[instIdx] = $"{svc}-{region}-{i:D3}";
                    instRegions[instIdx] = region;
                    instActiveReqs[instIdx] = rng.Next(10, 200);
                    instQueueDepth[instIdx] = (int)BoxMuller(rng, queueBase, queueBase * 0.3);
                    instIdx++;
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
        requestFrame.ToParquet(Path.Combine(datasetPath, "requests.parquet"), parquetOptions);

        WriteCsv(Path.Combine(datasetPath, "requests.csv"),
            timestampData, serviceData, endpointData, durationMsData,
            statusCodeData, regionData, traceIdData, isRetryData, totalRequests);

        var deployFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(deployTimestamps)),
            ("Service", NivaraColumn<string>.Create(deployServices)),
            ("Version", NivaraColumn<string>.Create(deployVersions)),
            ("Region", NivaraColumn<string>.Create(deployRegions)));
        deployFrame.ToParquet(Path.Combine(datasetPath, "deployments.parquet"), parquetOptions);

        var dependencyFrame = NivaraFrame.Create(
            ("Parent", NivaraColumn<string>.Create(dependencyParent.ToArray())),
            ("Child", NivaraColumn<string>.Create(dependencyChild.ToArray())));
        dependencyFrame.ToParquet(Path.Combine(datasetPath, "dependencies.parquet"), parquetOptions);

        var instanceFrame = NivaraFrame.Create(
            ("Timestamp", NivaraColumn<long>.Create(instTimestamps)),
            ("Service", NivaraColumn<string>.Create(instServices)),
            ("InstanceId", NivaraColumn<string>.Create(instIds)),
            ("Region", NivaraColumn<string>.Create(instRegions)),
            ("ActiveRequests", NivaraColumn<int>.Create(instActiveReqs)),
            ("QueueDepth", NivaraColumn<int>.Create(instQueueDepth)));
        instanceFrame.ToParquet(Path.Combine(datasetPath, "instances.parquet"), parquetOptions);
    }

    static double BoxMuller(Random rng, double mean, double stddev)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stddev * normal;
    }

    static void WriteCsv(string path,
        long[] timestamps, string[] services, string[] endpoints,
        double[] durationMs, int[] statusCodes, string[] regions,
        string[] traceIds, bool[] isRetries, int count)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("Timestamp,Service,Endpoint,DurationMs,StatusCode,Region,TraceId,IsRetry");
        for (int i = 0; i < count; i++)
        {
            var ts = new DateTimeOffset(timestamps[i], TimeSpan.Zero);
            writer.WriteLine($"{ts:O},{services[i]},{endpoints[i]},{durationMs[i]:F2},{statusCodes[i]},{regions[i]},{traceIds[i]},{isRetries[i]}");
        }
    }
}
