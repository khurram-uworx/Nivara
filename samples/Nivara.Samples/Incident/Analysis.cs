using Nivara.Expressions;
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

    public static QueryFrame AnalyzeDeploymentCorrelation(string datasetPath, IncidentScenario scenario)
    {
        var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        return frame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .Select(
                ColumnExpressions.Col("Service"),
                ColumnExpressions.Col("Endpoint"),
                ColumnExpressions.Col("Timestamp"),
                ColumnExpressions.Col("StatusCode"),
                ColumnExpressions.Col("Region"));
    }

    public static QueryFrame AnalyzeSaturationOrdering(string datasetPath, IncidentScenario scenario)
    {
        var instanceFrame = Ingestion.LoadParquet(Path.Combine(datasetPath, "instances.parquet"));
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        return instanceFrame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .RollingMax("QueueDepth", "MaxQueueDepth", 60)
            .Select(
                ColumnExpressions.Col("Service"),
                ColumnExpressions.Col("InstanceId"),
                ColumnExpressions.Col("Region"),
                ColumnExpressions.Col("QueueDepth"),
                ColumnExpressions.Col("MaxQueueDepth"));
    }

    public static QueryFrame AnalyzeRegionalPartitioning(string datasetPath, IncidentScenario scenario)
    {
        var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet"));
        var incidentStart = scenario.IncidentStart.Ticks;
        var incidentEnd = scenario.IncidentEnd.Ticks;

        return frame
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(incidentStart))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(incidentEnd))
            .Sort("Timestamp", SortDirection.Ascending)
            .PercentRank(
                "ErrorRank",
                [new SortKey("StatusCode", SortDirection.Descending)],
                "Region")
            .Select(
                ColumnExpressions.Col("Region"),
                ColumnExpressions.Col("Service"),
                ColumnExpressions.Col("StatusCode"),
                ColumnExpressions.Col("DurationMs"),
                ColumnExpressions.Col("ErrorRank"));
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
