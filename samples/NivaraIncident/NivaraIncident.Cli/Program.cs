using Nivara;
using Nivara.Samples.Incident;
using System.Diagnostics;

var mode = args.Length > 0 ? args[0] : "";
string datasetPath = "./data/incident-lab";
string scenarioId = "A";
int scale = 1;
long records = 0;
int chunkSize = 100_000;
bool stream = false;
bool benchmark = false;

for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--dataset" && i + 1 < args.Length) datasetPath = args[++i];
    else if (args[i] == "--scenario" && i + 1 < args.Length) scenarioId = args[++i];
    else if (args[i] == "--scale" && i + 1 < args.Length) scale = int.Parse(args[++i]);
    else if (args[i] == "--records" && i + 1 < args.Length) records = long.Parse(args[++i]);
    else if (args[i] == "--chunk-size" && i + 1 < args.Length) chunkSize = int.Parse(args[++i]);
    else if (args[i] == "--stream") stream = true;
    else if (args[i] == "--benchmark") benchmark = true;
}

var scenario = Scenarios.Get(scenarioId);

switch (mode)
{
    case "generate":
        Generate(datasetPath, scenarioId, scale, records);
        break;
    case "analyze":
        await Analyze(datasetPath, scenario, stream, chunkSize, benchmark);
        break;
    case "replay":
        await Replay(datasetPath, scenario, chunkSize);
        break;
    default:
        PrintUsage();
        break;
}

void Generate(string dsPath, string sid, int sc, long rec)
{
    if (rec > 0)
        Console.WriteLine($"Generating dataset for scenario {sid} ({rec:N0} records)...");
    else
        Console.WriteLine($"Generating dataset for scenario {sid} (scale {sc})...");

    var sw = Stopwatch.StartNew();
    if (rec > 0)
        DatasetGenerator.GenerateFromRecordCount(dsPath, sid, rec);
    else
        DatasetGenerator.Generate(dsPath, sid, sc);
    sw.Stop();
    Console.WriteLine($"Done in {sw.Elapsed.TotalSeconds:F1}s");
    Console.WriteLine($"Output: {dsPath}");
}

async Task Analyze(string dsPath, IncidentScenario sc, bool doStream, int cs, bool doBenchmark)
{
    Console.WriteLine($"Scenario: {sc.Id} — {sc.Name}");
    Console.WriteLine($"Dataset: {dsPath}");
    Console.WriteLine();

    if (doBenchmark)
    {
        Console.WriteLine("=== Benchmark: Nivara ===");
        Console.WriteLine();

        var sw1 = Stopwatch.StartNew();
        var degradation = Analysis.AnalyzeDegradationOrdering(dsPath, sc).Collect();
        sw1.Stop();
        Console.WriteLine($"  Degradation Ordering:   {sw1.ElapsedMilliseconds} ms  ({degradation.RowCount:N0} rows)");

        var sw2 = Stopwatch.StartNew();
        var deployment = Analysis.AnalyzeDeploymentCorrelation(dsPath, sc);
        sw2.Stop();
        Console.WriteLine($"  Deployment Correlation: {sw2.ElapsedMilliseconds} ms  ({deployment.RowCount:N0} rows)");

        var sw3 = Stopwatch.StartNew();
        var saturation = Analysis.AnalyzeSaturationOrdering(dsPath, sc);
        sw3.Stop();
        Console.WriteLine($"  Saturation Ordering:    {sw3.ElapsedMilliseconds} ms  ({saturation.RowCount:N0} rows)");

        var sw4 = Stopwatch.StartNew();
        var regional = Analysis.AnalyzeRegionalPartitioning(dsPath, sc);
        sw4.Stop();
        Console.WriteLine($"  Regional Partitioning:  {sw4.ElapsedMilliseconds} ms  ({regional.RowCount:N0} rows)");

        var sw5 = Stopwatch.StartNew();
        var grouped = Analysis.AnalyzeGroupedAggregation(dsPath, sc);
        sw5.Stop();
        Console.WriteLine($"  Grouped Aggregation:    {sw5.ElapsedMilliseconds} ms  ({grouped.RowCount:N0} rows)");

        var totalMs = sw1.ElapsedMilliseconds + sw2.ElapsedMilliseconds + sw3.ElapsedMilliseconds + sw4.ElapsedMilliseconds + sw5.ElapsedMilliseconds;
        Console.WriteLine();
        Console.WriteLine($"  Total:                  {totalMs} ms ({totalMs / 1000.0:F2}s)");

        deployment.Dispose();
        saturation.Dispose();
        regional.Dispose();
        grouped.Dispose();
        return;
    }

    var sw = Stopwatch.StartNew();

    Console.WriteLine("=== Degradation Ordering ===");
    var degradationQf = Analysis.AnalyzeDegradationOrdering(dsPath, sc);
    if (doStream)
    {
        int rowCount = 0;
        await foreach (var chunk in degradationQf.AsStream(cs))
        {
            rowCount += chunk.RowCount;
            PrintFrameSummary(chunk);
        }
        Console.WriteLine($"  Total rows streamed: {rowCount}");
    }
    else
    {
        PrintFrameSummary(degradationQf.Collect());
    }

    Console.WriteLine();
    Console.WriteLine("=== Deployment Correlation ===");
    PrintFrameSummary(Analysis.AnalyzeDeploymentCorrelation(dsPath, sc));

    Console.WriteLine();
    Console.WriteLine("=== Saturation Ordering ===");
    PrintFrameSummary(Analysis.AnalyzeSaturationOrdering(dsPath, sc));

    Console.WriteLine();
    Console.WriteLine("=== Regional Partitioning ===");
    PrintFrameSummary(Analysis.AnalyzeRegionalPartitioning(dsPath, sc));

    Console.WriteLine();
    Console.WriteLine("=== Grouped Aggregation ===");
    PrintFrameSummary(Analysis.AnalyzeGroupedAggregation(dsPath, sc));

    Console.WriteLine();
    sw.Stop();
    Console.WriteLine($"Total analysis time: {sw.Elapsed.TotalSeconds:F1}s");
}

async Task Replay(string dsPath, IncidentScenario sc, int cs)
{
    Console.WriteLine($"Replaying scenario {sc.Id} — {sc.Name}");
    Console.WriteLine($"Chunk size: {cs}");

    var sw = Stopwatch.StartNew();
    int chunkCount = 0;
    long totalRows = 0;

    await foreach (var chunk in Ingestion.StreamChunks(Path.Combine(dsPath, "requests.parquet"), cs))
    {
        chunkCount++;
        totalRows += chunk.RowCount;

        var firstTs = chunk.GetColumn<long>("Timestamp").GetValue(0);
        var lastTs = chunk.GetColumn<long>("Timestamp").GetValue(chunk.RowCount - 1);
        Console.WriteLine($"  Chunk {chunkCount}: {chunk.RowCount} rows [{firstTs}..{lastTs}]");
    }

    sw.Stop();
    Console.WriteLine();
    Console.WriteLine($"Replayed {totalRows:N0} rows in {chunkCount} chunks ({sw.Elapsed.TotalSeconds:F1}s)");
}

void PrintFrameSummary(NivaraFrame frame)
{
    Console.WriteLine($"  Columns: {frame.ColumnNames.Count}  Rows: {frame.RowCount}");
    foreach (var name in frame.ColumnNames)
    {
        var col = frame.GetColumn(name);
        Console.WriteLine($"    {name}: {col.ElementType.Name} ({col.Length} values)");
    }
}

void PrintUsage()
{
    Console.WriteLine("Nivara Incident Lab");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  NivaraIncident.Cli generate  --dataset <path> --scenario <A|B|C|D> [--scale <N>] [--records <N>]");
    Console.WriteLine("  NivaraIncident.Cli analyze   --dataset <path> --scenario <A|B|C|D> [--stream] [--chunk-size <N>] [--benchmark]");
    Console.WriteLine("  NivaraIncident.Cli replay    --dataset <path> --scenario <A|B|C|D> --chunk-size <N>");
}
