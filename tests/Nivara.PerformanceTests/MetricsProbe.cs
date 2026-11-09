using Nivara.Metrics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// <c>--metrics</c>: throughput and allocation picture for the frame-scoped metrics
/// operations in <c>Nivara.Metrics</c>. Human-run only.
/// </summary>
static class MetricsProbe
{
    static readonly int[] RowCounts = [10_000, 100_000, 1_000_000];
    static readonly int[] ClassCounts = [10, 100, 1_000];
    const int CurveRows = 100_000;

    public static int Run()
    {
        Console.WriteLine("=== Nivara metrics throughput (human-run) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        if (IsDebugBuild())
        {
            Console.WriteLine("WARNING: Debug build — numbers are not representative. Run with -c Release.");
        }
        Console.WriteLine();
        Console.WriteLine($"{"operation",-28} {"n",10} {"k",6} {"ms/call",10} {"rows/s",14} {"bytes/call",12}");

        foreach (int n in RowCounts)
        {
            var regressionPredicted = NivaraColumn<double>.Create(RegressionValues(n));
            var regressionActual = NivaraColumn<double>.Create(RegressionValues(n, 1.0));
            foreach (int k in ClassCounts)
            {
                var classesPredicted = NivaraColumn<int>.Create(ClassValues(n, k));
                var classesActual = NivaraColumn<int>.Create(ClassValues(n, k));

                var reg = Measure(() => _ = NivaraMetrics.RegressionSummaryFrame(regressionPredicted, regressionActual));
                var confusion = Measure(() => _ = NivaraMetrics.ConfusionMatrixFrame(classesPredicted, classesActual));
                var perClass = Measure(() => _ = NivaraMetrics.PerClassMetricsFrame(classesPredicted, classesActual));

                PrintRow("RegressionSummaryFrame", n, k, reg);
                PrintRow("ConfusionMatrixFrame", n, k, confusion);
                PrintRow("PerClassMetricsFrame", n, k, perClass);
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Class-count curve (n = {CurveRows:N0})");
        Console.WriteLine($"{"k",6} {"k²",12} {"ms/call",10} {"bytes/call",12} {"bytes/k²",12}");

        foreach (int k in ClassCounts)
        {
            var p = NivaraColumn<int>.Create(ClassValues(CurveRows, k));
            var a = NivaraColumn<int>.Create(ClassValues(CurveRows, Math.Max(1, k - 1)));
            var m = Measure(() => _ = NivaraMetrics.ConfusionMatrixFrame(p, a));
            Console.WriteLine(
                $"{k,6} {(long)k * k,12} {m.MedianMs,10:F3} {m.Bytes,12} {(m.Bytes / (double)((long)k * k)):12:F2}");
        }

        Console.WriteLine();
        Console.WriteLine("The confusion-matrix frame grows as O(k²) (one row and one column per class).");
        return 0;
    }

    static void PrintRow(string operation, int n, int k, (double MedianMs, long Bytes) measurement)
    {
        double rowsPerSecond = measurement.MedianMs > 0 ? n / (measurement.MedianMs / 1000.0) : double.PositiveInfinity;
        Console.WriteLine(
            $"{operation,-28} {n,10} {k,6} {measurement.MedianMs,10:F3} {rowsPerSecond,14:F0} {measurement.Bytes,12}");
    }

    static double[] RegressionValues(int n, double offset = 0.0)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = i + offset;
        return values;
    }

    static int[] ClassValues(int n, int k)
    {
        if (k <= 0)
            k = 1;
        var values = new int[n];
        for (int i = 0; i < n; i++)
            values[i] = i % k;
        return values;
    }

    static (double MedianMs, long Bytes) Measure(Action operation, int warmups = 3, int samples = 5)
    {
        for (int i = 0; i < warmups; i++)
            operation();

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var timings = new double[samples];
        long bestBytes = long.MaxValue;
        for (int i = 0; i < samples; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long beforeBytes = GC.GetAllocatedBytesForCurrentThread();
            var stopwatch = Stopwatch.StartNew();
            operation();
            stopwatch.Stop();
            long afterBytes = GC.GetAllocatedBytesForCurrentThread();
            timings[i] = stopwatch.Elapsed.TotalMilliseconds;
            bestBytes = Math.Min(bestBytes, afterBytes - beforeBytes);
        }

        Array.Sort(timings);
        return (timings[samples / 2], bestBytes);
    }

#if DEBUG
    static bool IsDebugBuild() => true;
#else
    static bool IsDebugBuild() => false;
#endif
}