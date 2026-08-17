using Nivara.Samples.Incident;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

static class IncidentLabBenchmark
{
    static readonly string[] ScenarioIds = ["A", "B", "C", "D"];

    public static void Run(string[] args)
    {
        var scale = args.Length > 0 && int.TryParse(args[0], out var s) ? s : 1;
        var tempDir = Path.Combine(Path.GetTempPath(), $"incident-perf-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        Console.WriteLine("Incident Lab Benchmark");
        Console.WriteLine($"  Scale: {scale}x");
        Console.WriteLine($"  Temp:  {tempDir}");
        Console.WriteLine();

        var genSw = Stopwatch.StartNew();
        foreach (var sid in ScenarioIds)
        {
            var dir = Path.Combine(tempDir, sid);
            Directory.CreateDirectory(dir);
            DatasetGenerator.Generate(dir, sid, scale);
        }
        genSw.Stop();
        Console.WriteLine($"  Generate (4 scenarios):  {genSw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine();

        foreach (var sid in ScenarioIds)
        {
            var dir = Path.Combine(tempDir, sid);
            var scenario = Scenarios.Get(sid);

            Console.WriteLine($"  Scenario {sid} — {scenario.Name}");

            var degradationSw = Stopwatch.StartNew();
            using (var qf = Analysis.AnalyzeDegradationOrdering(dir, scenario))
            {
                var frame = qf.Collect();
                Console.WriteLine($"    Degradation ordering:  {degradationSw.Elapsed.TotalMilliseconds:F0}ms  ({frame.RowCount} rows)");
            }

            var deploySw = Stopwatch.StartNew();
            using (var qf = Analysis.AnalyzeDeploymentCorrelation(dir, scenario))
            {
                var frame = qf.Collect();
                Console.WriteLine($"    Deployment correlation:{deploySw.Elapsed.TotalMilliseconds:F0}ms  ({frame.RowCount} rows)");
            }

            var saturationSw = Stopwatch.StartNew();
            using (var qf = Analysis.AnalyzeSaturationOrdering(dir, scenario))
            {
                var frame = qf.Collect();
                Console.WriteLine($"    Saturation ordering:   {saturationSw.Elapsed.TotalMilliseconds:F0}ms  ({frame.RowCount} rows)");
            }

            var regionalSw = Stopwatch.StartNew();
            using (var qf = Analysis.AnalyzeRegionalPartitioning(dir, scenario))
            {
                var frame = qf.Collect();
                Console.WriteLine($"    Regional partitioning: {regionalSw.Elapsed.TotalMilliseconds:F0}ms  ({frame.RowCount} rows)");
            }

            var groupedSw = Stopwatch.StartNew();
            var grouped = Analysis.AnalyzeGroupedAggregation(dir, scenario);
            Console.WriteLine($"    Grouped aggregation:   {groupedSw.Elapsed.TotalMilliseconds:F0}ms  ({grouped.RowCount} groups)");
            Console.WriteLine();
        }

        Directory.Delete(tempDir, true);
    }
}
