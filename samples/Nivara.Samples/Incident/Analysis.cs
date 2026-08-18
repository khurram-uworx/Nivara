using Nivara.Expressions;
using Nivara.IO;
using Nivara.Linq;
using Nivara.Operations;
using Nivara.Query;

namespace Nivara.Samples.Incident;

public sealed class RequestRow
{
    public long Timestamp { get; set; }
    public string Service { get; set; } = "";
    public string Endpoint { get; set; } = "";
    public double DurationMs { get; set; }
    public int StatusCode { get; set; }
    public string Region { get; set; } = "";
    public string TraceId { get; set; } = "";
    public bool IsRetry { get; set; }
    public double DurationPercentRank { get; set; }
}

public sealed class DeploymentRow
{
    public long Timestamp { get; set; }
    public string Service { get; set; } = "";
    public string Version { get; set; } = "";
    public string Region { get; set; } = "";
}

public sealed class InstanceRow
{
    public long Timestamp { get; set; }
    public string Service { get; set; } = "";
    public string InstanceId { get; set; } = "";
    public string Region { get; set; } = "";
    public int ActiveRequests { get; set; }
    public int QueueDepth { get; set; }
    public int PeakQueueDepth { get; set; }
}

public static class Analysis
{
    public static QueryFrame AnalyzeDegradationOrdering(string datasetPath, IncidentScenario scenario)
    {
        var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        return frame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .RollingMean("StatusCode", "ErrorRate", 60)
            .Shift("ErrorRate", "PrevErrorRate", 1)
            .Select(
                ColumnExpressions.Col("Service"),
                ColumnExpressions.Col("Endpoint"),
                ColumnExpressions.Col("Timestamp"),
                ColumnExpressions.Col("StatusCode"),
                ColumnExpressions.Col("ErrorRate"),
                ColumnExpressions.Col("PrevErrorRate"));
    }

    public static NivaraFrame AnalyzeDeploymentCorrelation(string datasetPath, IncidentScenario scenario)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        using var requestsQf = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        using var requests = requestsQf
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .Collect();

        using var deployments = NivaraParquetReader.ReadParquet(Path.Combine(datasetPath, "deployments.parquet"));

        var requestTsCol = requests.GetColumn<long>("Timestamp");
        var requestSvcCol = requests.GetColumn<string>("Service");
        var requestEndpointCol = requests.GetColumn<string>("Endpoint");
        var requestStatusCol = requests.GetColumn<int>("StatusCode");
        var requestRegionCol = requests.GetColumn<string>("Region");

        var deployTsCol = deployments.GetColumn<long>("Timestamp");
        var deploySvcCol = deployments.GetColumn<string>("Service");
        var deployVerCol = deployments.GetColumn<string>("Version");

        var deployLookup = new Dictionary<string, (long Timestamp, string Version)>();
        for (int d = 0; d < deployments.RowCount; d++)
        {
            var svc = (string)deploySvcCol.GetValue(d)!;
            var ts = (long)deployTsCol.GetValue(d)!;
            if (!deployLookup.TryGetValue(svc, out var existing) || ts > existing.Timestamp)
                deployLookup[svc] = (ts, (string)deployVerCol.GetValue(d)!);
        }

        int rowCount = requests.RowCount;
        var serviceOut = new string[rowCount];
        var endpointOut = new string[rowCount];
        var timestampOut = new long[rowCount];
        var statusCodeOut = new int[rowCount];
        var regionOut = new string[rowCount];
        var deployVersionOut = new string[rowCount];
        var timeSinceDeployOut = new double[rowCount];
        var errorCategoryOut = new string[rowCount];

        for (int i = 0; i < rowCount; i++)
        {
            var svc = (string)requestSvcCol.GetValue(i)!;
            serviceOut[i] = svc;
            endpointOut[i] = (string)requestEndpointCol.GetValue(i)!;
            timestampOut[i] = (long)requestTsCol.GetValue(i)!;
            statusCodeOut[i] = (int)requestStatusCol.GetValue(i)!;
            regionOut[i] = (string)requestRegionCol.GetValue(i)!;

            if (deployLookup.TryGetValue(svc, out var deploy) && deploy.Timestamp <= timestampOut[i])
            {
                deployVersionOut[i] = deploy.Version;
                timeSinceDeployOut[i] = TimeSpan.FromTicks(timestampOut[i] - deploy.Timestamp).TotalSeconds;
            }
            else
            {
                deployVersionOut[i] = "";
                timeSinceDeployOut[i] = -1;
            }

            errorCategoryOut[i] = statusCodeOut[i] switch
            {
                >= 500 => "server_error",
                >= 400 => "client_error",
                _ => "success"
            };
        }

        deployments.Dispose();

        return NivaraFrame.Create(
            ("Service", NivaraColumn<string>.Create(serviceOut)),
            ("Endpoint", NivaraColumn<string>.Create(endpointOut)),
            ("Timestamp", NivaraColumn<long>.Create(timestampOut)),
            ("StatusCode", NivaraColumn<int>.Create(statusCodeOut)),
            ("Region", NivaraColumn<string>.Create(regionOut)),
            ("DeploymentVersion", NivaraColumn<string>.Create(deployVersionOut)),
            ("TimeSinceDeploySec", NivaraColumn<double>.Create(timeSinceDeployOut)),
            ("ErrorCategory", NivaraColumn<string>.Create(errorCategoryOut)));
    }

    public static NivaraFrame AnalyzeSaturationOrdering(string datasetPath, IncidentScenario scenario)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        using var instanceQf = Ingestion.LoadParquet(Path.Combine(datasetPath, "instances.parquet"));
        using var instances = instanceQf
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .RollingMax("QueueDepth", "PeakQueueDepth", 10, new WindowSpec().PartitionBy("Service"))
            .Collect();

        var typedQuery = instances.Query<InstanceRow>();
        var grouped = typedQuery
            .GroupBy(r => r.Service)
            .Select(g => new
            {
                Service = g.Key,
                InstanceCount = g.Count(),
                PeakQueueDepth = g.Max(r => r.PeakQueueDepth),
                P50QueueDepth = g.Quantile(r => r.QueueDepth, 0.50),
                P95QueueDepth = g.Quantile(r => r.QueueDepth, 0.95),
                P99QueueDepth = g.Quantile(r => r.QueueDepth, 0.99),
                StdDevQueueDepth = g.StdDev(r => r.QueueDepth),
            });

        return grouped.Collect();
    }

    public static NivaraFrame AnalyzeRegionalPartitioning(string datasetPath, IncidentScenario scenario)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        using var requestsQf = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        using var requests = requestsQf
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .PercentRank("DurationPercentRank",
                [new SortKey("DurationMs", SortDirection.Ascending)],
                "Region")
            .Collect();

        var typedQuery = requests.Query<RequestRow>();
        var grouped = typedQuery
            .GroupBy(r => r.Region)
            .Select(g => new
            {
                Region = g.Key,
                TotalRequests = g.Count(),
                ErrorCount = g.Sum(r => r.StatusCode >= 500 ? 1 : 0),
                ErrorRate = g.Average(r => r.StatusCode >= 500 ? 1.0 : 0.0),
                P50Duration = g.Quantile(r => r.DurationMs, 0.50),
                P95Duration = g.Quantile(r => r.DurationMs, 0.95),
                MaxDurationPercentRank = g.Max(r => r.DurationPercentRank),
            });

        var result = grouped.Collect();

        var regionCol = result.GetColumn<string>("Region");
        var errorRateCol = result.GetColumn<double>("ErrorRate");
        int rowCount = result.RowCount;

        var ranks = new long[rowCount];
        var sortedIndices = Enumerable.Range(0, rowCount)
            .OrderByDescending(i => (double)errorRateCol.GetValue(i)!)
            .ThenBy(i => (string)regionCol.GetValue(i)!)
            .ToArray();
        for (int rank = 0; rank < rowCount; rank++)
            ranks[sortedIndices[rank]] = rank + 1;

        var columns = new (string Name, IColumn Column)[result.ColumnNames.Count + 1];
        for (int c = 0; c < result.ColumnNames.Count; c++)
            columns[c] = (result.ColumnNames[c], result.GetColumn(result.ColumnNames[c]));
        columns[result.ColumnNames.Count] = ("ErrorRank", NivaraColumn<long>.Create(ranks));
        return NivaraFrame.Create(columns);
    }

    public static NivaraFrame AnalyzeGroupedAggregation(string datasetPath, IncidentScenario scenario)
    {
        var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        var result = frame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Collect();

        var serviceCol = result.GetColumn<string>("Service");
        var statusCol = result.GetColumn<int>("StatusCode");
        var durationCol = result.GetColumn<double>("DurationMs");

        var services = new Dictionary<string, (int Errors, int Total, double TotalDuration)>();
        for (int i = 0; i < result.RowCount; i++)
        {
            var svc = (string)serviceCol.GetValue(i)!;
            if (!services.TryGetValue(svc, out var acc))
                acc = (0, 0, 0);
            acc.Total++;
            acc.TotalDuration += (double)durationCol.GetValue(i)!;
            if ((int)statusCol.GetValue(i)! >= 500) acc.Errors++;
            services[svc] = acc;
        }

        var svcNames = services.Keys.OrderBy(k => k).ToArray();
        var errorRates = svcNames.Select(s => services[s].Total > 0 ? (double)services[s].Errors / services[s].Total : 0).ToArray();
        var avgDurations = svcNames.Select(s => services[s].Total > 0 ? services[s].TotalDuration / services[s].Total : 0).ToArray();

        return NivaraFrame.Create(
            ("Service", NivaraColumn<string>.Create(svcNames)),
            ("ErrorRate", NivaraColumn<double>.Create(errorRates)),
            ("AvgDurationMs", NivaraColumn<double>.Create(avgDurations)),
            ("TotalRequests", NivaraColumn<int>.Create(svcNames.Select(s => services[s].Total).ToArray())));
    }

    public static NivaraFrame AnalyzeGroupedAggregationWithTypedLinq(string datasetPath, IncidentScenario scenario)
    {
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        var collected = frame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Collect();

        var typedQuery = collected.Query<RequestRow>();

        var grouped = typedQuery
            .GroupBy(r => r.Service)
            .Select(g => new
            {
                Service = g.Key,
                TotalRequests = g.Count(),
                ErrorCount = g.Sum(r => r.StatusCode >= 500 ? 1 : 0),
                AvgDuration = g.Average(r => r.DurationMs),
            });

        return grouped.Collect();
    }
}
