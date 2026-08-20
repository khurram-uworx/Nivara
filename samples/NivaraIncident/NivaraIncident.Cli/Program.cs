using Nivara;
using Nivara.Diagnostics;
using Nivara.Expressions;
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
int iterations = 5;

for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--dataset" && i + 1 < args.Length) datasetPath = args[++i];
    else if (args[i] == "--scenario" && i + 1 < args.Length) scenarioId = args[++i];
    else if (args[i] == "--scale" && i + 1 < args.Length) scale = int.Parse(args[++i]);
    else if (args[i] == "--records" && i + 1 < args.Length) records = long.Parse(args[++i]);
    else if (args[i] == "--chunk-size" && i + 1 < args.Length) chunkSize = int.Parse(args[++i]);
    else if (args[i] == "--iterations" && i + 1 < args.Length) iterations = int.Parse(args[++i]);
    else if (args[i] == "--stream") stream = true;
    else if (args[i] == "--benchmark") benchmark = true;
}

var scenarioIds = scenarioId.Equals("all", StringComparison.OrdinalIgnoreCase)
    ? new[] { "A", "B", "C", "D" }
    : new[] { scenarioId };

switch (mode)
{
    case "generate":
        foreach (var sid in scenarioIds)
            Generate(datasetPath, sid, scale, records);
        break;
    case "analyze":
        foreach (var sid in scenarioIds)
            await Analyze(datasetPath, Scenarios.Get(sid), stream, chunkSize, benchmark, iterations);
        break;
    case "bench-stream":
        foreach (var sid in scenarioIds)
            BenchStream(datasetPath, Scenarios.Get(sid));
        break;
    case "bench-kernels":
        foreach (var sid in scenarioIds)
            BenchKernels(datasetPath, Scenarios.Get(sid));
        break;
    case "replay":
        foreach (var sid in scenarioIds)
            await Replay(datasetPath, Scenarios.Get(sid), chunkSize);
        break;
    case "streamix":
        foreach (var sid in scenarioIds)
            await RunStreamixScenarios(datasetPath, Scenarios.Get(sid), chunkSize);
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

async Task Analyze(string dsPath, IncidentScenario sc, bool doStream, int cs, bool doBenchmark, int iters)
{
    Console.WriteLine($"Scenario: {sc.Id} — {sc.Name}");
    Console.WriteLine($"Dataset: {dsPath}");
    Console.WriteLine();

    if (doBenchmark)
    {
        Console.WriteLine($"=== Benchmark: Nivara ({iters} iterations, median) ===");
        Console.WriteLine();

        int warmup = 1;
        int totalRuns = warmup + iters;

        // Analysis A: returns QueryFrame — full diagnostics available
        var degradationTimes = new double[iters];
        long degradationRows = 0;
        for (int run = 0; run < totalRuns; run++)
        {
            GC.Collect(2, GCCollectionMode.Forced, true);
            var memBefore = GC.GetTotalMemory(true);
            var iterSw = Stopwatch.StartNew();
            using var qf = Analysis.AnalyzeDegradationOrdering(dsPath, sc);
            using var frame = qf.Collect();
            iterSw.Stop();
            var memAfter = GC.GetTotalMemory(false);
            degradationRows = frame.RowCount;

            if (run >= warmup)
            {
                degradationTimes[run - warmup] = iterSw.Elapsed.TotalMilliseconds;
                var diag = qf.GetExecutionDiagnostics();
                if (run == warmup)
                {
                    Console.WriteLine($"  Degradation Ordering:");
                    Console.WriteLine($"    Rows: {frame.RowCount:N0} returned");
                    if (diag != null)
                        Console.WriteLine($"    Diagnostics: {diag.RowsRead:N0} read, {diag.MaterializedColumns} cols, {diag.TotalExecutionTime.TotalMilliseconds:F1}ms exec, {diag.MemoryAllocated / 1024.0 / 1024.0:F2} MB allocated");
                    Console.WriteLine($"    GC peak: {(memAfter - memBefore) / 1024.0 / 1024.0:F2} MB");
                }
            }
        }
        Console.WriteLine($"    Median: {Median(degradationTimes):F1} ms  ({degradationRows:N0} rows)");
        Console.WriteLine();

        // Analyses B-E: return NivaraFrame — report elapsed/rows only
        RunBenchmarkIteration("Deployment Correlation", iters, warmup,
            () => Analysis.AnalyzeDeploymentCorrelation(dsPath, sc));

        RunBenchmarkIteration("Saturation Ordering", iters, warmup,
            () => Analysis.AnalyzeSaturationOrdering(dsPath, sc));

        RunBenchmarkIteration("Regional Partitioning", iters, warmup,
            () => Analysis.AnalyzeRegionalPartitioning(dsPath, sc));

        RunBenchmarkIteration("Grouped Aggregation", iters, warmup,
            () => Analysis.AnalyzeGroupedAggregation(dsPath, sc));

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

void BenchStream(string dsPath, IncidentScenario sc)
{
    Console.WriteLine($"=== Streaming vs Eager Benchmark: {sc.Id} — {sc.Name} ===");
    Console.WriteLine($"Dataset: {dsPath}");
    Console.WriteLine();

    var chunkSizes = new[] { 10_000, 50_000, 100_000, 500_000 };

    // Window-heavy analysis (Sort + RollingMean + Shift + RollingMax)
    Console.WriteLine("  Degradation Ordering (window ops: Sort + RollingMean + Shift):");
    Console.WriteLine("  ChunkSize    Chunks    Elapsed      PeakMem       Rows");

    foreach (var cs in chunkSizes)
    {
        GC.Collect(2, GCCollectionMode.Forced, true);
        long peakMem = 0;
        int chunkCount = 0;
        long totalRows = 0;

        var sw = Stopwatch.StartNew();
        var qf = Analysis.AnalyzeDegradationOrdering(dsPath, sc);
        var stream = qf.AsStream(cs);

        // Synchronously enumerate to measure
        var enumerator = stream.GetAsyncEnumerator();
        try
        {
            while (true)
            {
                bool moved;
                try { moved = enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult(); }
                catch { break; }
                if (!moved) break;
                chunkCount++;
                totalRows += enumerator.Current.RowCount;
                long mem = GC.GetTotalMemory(false);
                if (mem > peakMem) peakMem = mem;
            }
        }
        finally { enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
        sw.Stop();

        string fallback = chunkCount <= 1 ? " (FALLBACK)" : "";
        Console.WriteLine("  " + cs.ToString().PadRight(12) + chunkCount.ToString().PadLeft(8) + sw.ElapsedMilliseconds.ToString().PadLeft(8) + " ms " + (peakMem / 1024.0 / 1024.0).ToString("F2").PadLeft(10) + " MB " + totalRows.ToString("N0").PadLeft(10) + fallback);
        qf.Dispose();
    }

    // Filter-only prefix (no window ops) for contrast
    Console.WriteLine();
    Console.WriteLine("  Filter-only (no window ops):");
    Console.WriteLine("  ChunkSize    Chunks    Elapsed      PeakMem       Rows");

    var scenario = sc;
    foreach (var cs in chunkSizes)
    {
        GC.Collect(2, GCCollectionMode.Forced, true);
        long peakMem = 0;
        int chunkCount = 0;
        long totalRows = 0;

        var sw = Stopwatch.StartNew();
        var qf = Ingestion.LoadParquet(Path.Combine(dsPath, "requests.parquet"))
            .Filter(ColumnExpressions.Col("Timestamp") >= ColumnExpressions.Lit(scenario.IncidentStart.Ticks))
            .Filter(ColumnExpressions.Col("Timestamp") <= ColumnExpressions.Lit(scenario.IncidentEnd.Ticks));
        var stream = qf.AsStream(cs);

        var enumerator = stream.GetAsyncEnumerator();
        try
        {
            while (true)
            {
                bool moved;
                try { moved = enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult(); }
                catch { break; }
                if (!moved) break;
                chunkCount++;
                totalRows += enumerator.Current.RowCount;
                long mem = GC.GetTotalMemory(false);
                if (mem > peakMem) peakMem = mem;
            }
        }
        finally { enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
        sw.Stop();

        Console.WriteLine("  " + cs.ToString().PadRight(12) + chunkCount.ToString().PadLeft(8) + sw.ElapsedMilliseconds.ToString().PadLeft(8) + " ms " + (peakMem / 1024.0 / 1024.0).ToString("F2").PadLeft(10) + " MB " + totalRows.ToString("N0").PadLeft(10));
        qf.Dispose();
    }

    Console.WriteLine();
}

void BenchKernels(string dsPath, IncidentScenario sc)
{
    Console.WriteLine($"=== Kernel Vectorization Report: {sc.Id} — {sc.Name} ===");
    Console.WriteLine($"Dataset: {dsPath}");
    Console.WriteLine();

    DiagnosticsTracker.IsEnabled = true;

    // Analysis A: Degradation Ordering (QueryFrame pipeline — full kernel path)
    DiagnosticsTracker.ClearRecordedOperations();
    GC.Collect(2, GCCollectionMode.Forced, true);
    var sw = Stopwatch.StartNew();
    using (var qf = Analysis.AnalyzeDegradationOrdering(dsPath, sc))
    using (var frame = qf.Collect()) { }
    sw.Stop();
    var summaryA = DiagnosticsTracker.GetSummary();
    Console.WriteLine($"  Degradation Ordering ({sw.ElapsedMilliseconds}ms):");
    Console.WriteLine($"    Operations: {summaryA.TotalOperations} total, {summaryA.VectorizedOperations} vectorized, {summaryA.ScalarOperations} scalar");
    Console.WriteLine($"    Vectorization rate: {summaryA.VectorizationRate:F1}%");
    PrintTopOps(summaryA);

    // Analysis C: Saturation Ordering (Filter + Sort + RollingMax + Collect + typed LINQ GroupBy)
    DiagnosticsTracker.ClearRecordedOperations();
    GC.Collect(2, GCCollectionMode.Forced, true);
    sw.Restart();
    using (var satFrame = Analysis.AnalyzeSaturationOrdering(dsPath, sc)) { }
    sw.Stop();
    var summaryC = DiagnosticsTracker.GetSummary();
    Console.WriteLine($"  Saturation Ordering ({sw.ElapsedMilliseconds}ms):");
    Console.WriteLine($"    Operations: {summaryC.TotalOperations} total, {summaryC.VectorizedOperations} vectorized, {summaryC.ScalarOperations} scalar");
    Console.WriteLine($"    Vectorization rate: {summaryC.VectorizationRate:F1}%");
    PrintTopOps(summaryC);

    // Analysis D: Regional Partitioning (Filter + PercentRank + Collect + typed LINQ GroupBy)
    DiagnosticsTracker.ClearRecordedOperations();
    GC.Collect(2, GCCollectionMode.Forced, true);
    sw.Restart();
    using (var regFrame = Analysis.AnalyzeRegionalPartitioning(dsPath, sc)) { }
    sw.Stop();
    var summaryD = DiagnosticsTracker.GetSummary();
    Console.WriteLine($"  Regional Partitioning ({sw.ElapsedMilliseconds}ms):");
    Console.WriteLine($"    Operations: {summaryD.TotalOperations} total, {summaryD.VectorizedOperations} vectorized, {summaryD.ScalarOperations} scalar");
    Console.WriteLine($"    Vectorization rate: {summaryD.VectorizationRate:F1}%");
    PrintTopOps(summaryD);

    DiagnosticsTracker.IsEnabled = false;
    DiagnosticsTracker.ClearRecordedOperations();

    Console.WriteLine();
    Console.WriteLine("  Note: rank/quantile scalar fallback is expected. Hand-rolled C# loops");
    Console.WriteLine("  (Analyses B, E) run outside Nivara kernels and are not measured here.");
}

void PrintTopOps(OperationSummary summary)
{
    if (summary.KernelTypes.Count > 0)
    {
        Console.WriteLine("    Kernel types:");
        foreach (var kvp in summary.KernelTypes.OrderByDescending(kv => kv.Value).Take(5))
            Console.WriteLine($"      {kvp.Key}: {kvp.Value}");
    }
    if (summary.OperationTypes.Count > 0)
    {
        Console.WriteLine("    Operation types:");
        foreach (var kvp in summary.OperationTypes.OrderByDescending(kv => kv.Value).Take(5))
            Console.WriteLine($"      {kvp.Key}: {kvp.Value}");
    }
    Console.WriteLine();
}

void RunBenchmarkIteration(string name, int iters, int warmup, Func<NivaraFrame> run)
{
    var times = new double[iters];
    long rowCount = 0;
    int totalRuns = warmup + iters;
    for (int i = 0; i < totalRuns; i++)
    {
        GC.Collect(2, GCCollectionMode.Forced, true);
        var sw = Stopwatch.StartNew();
        using var frame = run();
        sw.Stop();
        rowCount = frame.RowCount;
        if (i >= warmup)
            times[i - warmup] = sw.Elapsed.TotalMilliseconds;
    }
    Console.WriteLine("  " + name.PadRight(24) + Median(times).ToString("F1").PadLeft(8) + " ms  (" + rowCount.ToString("N0") + " rows)");
}

static double Median(double[] values)
{
    var sorted = (double[])values.Clone();
    Array.Sort(sorted);
    int mid = sorted.Length / 2;
    return sorted.Length % 2 == 0
        ? (sorted[mid - 1] + sorted[mid]) / 2.0
        : sorted[mid];
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

async Task RunStreamixScenarios(string dsPath, IncidentScenario sc, int cs)
{
    Console.WriteLine($"=== Streamix Scenarios: {sc.Id} — {sc.Name} ===");
    Console.WriteLine($"Dataset: {dsPath}");
    Console.WriteLine();

    Console.WriteLine("--- Scenario 1: Fault-Tolerant Streaming (Retry + Checkpoint) ---");
    var sw1 = Stopwatch.StartNew();
    var summary1 = await StreamixScenarios.RunFaultTolerantStreaming(dsPath, sc, cs);
    sw1.Stop();
    Console.WriteLine($"  Rows processed: {summary1.TotalRows:N0}  ({summary1.ChunksProcessed} chunks)");
    Console.WriteLine($"  Time: {sw1.Elapsed.TotalSeconds:F1}s");
    Console.WriteLine();

    Console.WriteLine("--- Scenario 2: Event-Time Windowed Analytics ---");
    var sw2 = Stopwatch.StartNew();
    var summary2 = await StreamixScenarios.RunWindowedAnalytics(dsPath, sc, cs);
    sw2.Stop();
    Console.WriteLine($"  Total rows: {summary2.TotalRows:N0}  ({summary2.Windows.Count} windows)");
    Console.WriteLine($"  Time: {sw2.Elapsed.TotalSeconds:F1}s");
    if (summary2.Windows.Count > 0)
    {
        var avgDuration = summary2.Windows.Average(w => w.AverageDurationMs);
        var avgRolling = summary2.Windows.Average(w => w.RollingAvgDurationMs);
        var avgErrorRate = summary2.Windows.Average(w => w.ErrorRate);
        Console.WriteLine($"  Avg duration: {avgDuration:F1}ms  Rolling avg: {avgRolling:F1}ms  Avg error rate: {avgErrorRate:P1}");
    }
    Console.WriteLine();

    Console.WriteLine("--- Scenario 3: Online AutoDiff Learning ---");
    var sw3 = Stopwatch.StartNew();
    var summary3 = await StreamixScenarios.RunOnlineAutoDiffLearning(dsPath, sc, batchSize: 128, epochs: 3);
    sw3.Stop();
    Console.WriteLine($"  Training batches: {summary3.TrainingBatches:N0}  Final loss: {summary3.FinalLoss:F4}");
    Console.WriteLine($"  Time: {sw3.Elapsed.TotalSeconds:F1}s");
    Console.WriteLine();
}

void PrintUsage()
{
    Console.WriteLine("Nivara Incident Lab");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  NivaraIncident.Cli generate      --dataset <path> --scenario <A|B|C|D|all> [--scale <N>] [--records <N>]");
    Console.WriteLine("  NivaraIncident.Cli analyze       --dataset <path> --scenario <A|B|C|D|all> [--stream] [--chunk-size <N>] [--benchmark] [--iterations <N>]");
    Console.WriteLine("  NivaraIncident.Cli bench-stream  --dataset <path> --scenario <A|B|C|D|all>");
    Console.WriteLine("  NivaraIncident.Cli bench-kernels --dataset <path> --scenario <A|B|C|D|all>");
    Console.WriteLine("  NivaraIncident.Cli replay        --dataset <path> --scenario <A|B|C|D|all> --chunk-size <N>");
    Console.WriteLine("  NivaraIncident.Cli streamix      --dataset <path> --scenario <A|B|C|D|all> [--chunk-size <N>]");
}
