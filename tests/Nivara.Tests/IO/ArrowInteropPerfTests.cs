using Nivara.IO;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.IO;

[TestFixture]
[Category("Performance")]
public class ArrowInteropPerfTests
{
    [Test]
    public void ToArrowTable_CopyPath_IntColumn_Throughput()
    {
        RunDirectionalBenchmark<int>("ToArrowTable", 100_000);
    }

    [Test]
    public void FromArrowTable_CopyPath_IntColumn_Throughput()
    {
        RunDirectionalBenchmark<int>("FromArrowTable", 100_000);
    }

    [Test]
    public void ToArrowTable_CopyPath_DoubleColumn_Throughput()
    {
        RunDirectionalBenchmark<double>("ToArrowTable", 100_000);
    }

    [Test]
    public void FromArrowTable_CopyPath_DoubleColumn_Throughput()
    {
        RunDirectionalBenchmark<double>("FromArrowTable", 100_000);
    }

    [Test]
    public void ToArrowTable_CopyPath_IntColumn_Throughput_1M()
    {
        RunDirectionalBenchmark<int>("ToArrowTable", 1_000_000);
    }

    [Test]
    public void FromArrowTable_CopyPath_IntColumn_Throughput_1M()
    {
        RunDirectionalBenchmark<int>("FromArrowTable", 1_000_000);
    }

    [Test]
    public void ToArrowTable_CopyPath_DoubleColumn_Throughput_1M()
    {
        RunDirectionalBenchmark<double>("ToArrowTable", 1_000_000);
    }

    [Test]
    public void FromArrowTable_CopyPath_DoubleColumn_Throughput_1M()
    {
        RunDirectionalBenchmark<double>("FromArrowTable", 1_000_000);
    }

    [Test]
    public void ToArrowTable_ZeroCopyVsCopy_IntColumn_100K()
    {
        RunInterleavedExportComparison<int>(100_000);
    }

    [Test]
    public void ToArrowTable_ZeroCopyVsCopy_IntColumn_1M()
    {
        RunInterleavedExportComparison<int>(1_000_000);
    }

    [Test]
    public void ToArrowTable_ZeroCopyVsCopy_DoubleColumn_100K()
    {
        RunInterleavedExportComparison<double>(100_000);
    }

    [Test]
    public void ToArrowTable_ZeroCopyVsCopy_DoubleColumn_1M()
    {
        RunInterleavedExportComparison<double>(1_000_000);
    }

    static void RunInterleavedExportComparison<T>(int size)
    {
        TimingGuards.RequireOptimizedBuildForTiming();

        var rng = new Random(42);
        var data = new T[size];
        for (int i = 0; i < size; i++)
            data[i] = typeof(T) == typeof(int) ? (T)(object)(i % 100_000) : (T)(object)(rng.NextDouble());

        var frame = NivaraFrame.Create(("Values", NivaraColumn<T>.Create(data)));
        var copyOptions = new ArrowConversionOptions();
        var zeroCopyOptions = new ArrowConversionOptions { UseZeroCopy = true };

        Action copy = () => ArrowInterop.ToArrowTable(frame, copyOptions);
        Action zeroCopy = () => ArrowInterop.ToArrowTable(frame, zeroCopyOptions);

        copy();
        zeroCopy();

        const int rounds = 11;
        var copyMs = new double[rounds];
        var zeroCopyMs = new double[rounds];
        var stopwatch = new Stopwatch();

        for (int round = 0; round < rounds; round++)
        {
            // Alternate the order so load drift cannot systematically favour one route.
            if ((round & 1) == 0)
            {
                copyMs[round] = Time(copy);
                zeroCopyMs[round] = Time(zeroCopy);
            }
            else
            {
                zeroCopyMs[round] = Time(zeroCopy);
                copyMs[round] = Time(copy);
            }
        }

        var medianCopy = Median(copyMs);
        var medianZeroCopy = Median(zeroCopyMs);

        // No timing assertion. The payoff claim is structural - the zero-copy route has no
        // element loop - and this probe only corroborates it for a human running Release.
        TestContext.Out.WriteLine(
            $"ToArrowTable <{typeof(T).Name}> ({size} elements): copy {medianCopy:F3}ms, " +
            $"zero-copy {medianZeroCopy:F3}ms, ratio {medianZeroCopy / medianCopy:F4} " +
            $"({rounds} interleaved rounds, median of each route)");

        double Time(Action action)
        {
            stopwatch.Restart();
            action();
            stopwatch.Stop();
            return stopwatch.Elapsed.TotalMilliseconds;
        }
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        System.Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return (sorted.Length & 1) == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2.0;
    }

    static void RunDirectionalBenchmark<T>(string direction, int size)
    {
        var rng = new Random(42);
        var data = new T[size];
        for (int i = 0; i < size; i++)
            data[i] = typeof(T) == typeof(int) ? (T)(object)(i % 100_000) : (T)(object)(rng.NextDouble());

        var frame = NivaraFrame.Create(("Values", NivaraColumn<T>.Create(data)));
        var table = ArrowInterop.ToArrowTable(frame);

        Action action = direction switch
        {
            "ToArrowTable" => () => ArrowInterop.ToArrowTable(frame),
            "FromArrowTable" => () => ArrowInterop.FromArrowTable(table),
            _ => throw new ArgumentException($"Unknown direction '{direction}'")
        };

        var time = MeasureBestOfFiveMs(action);

        double perSec = size / (time / 1000.0);
        TestContext.Out.WriteLine(
            $"{direction} <{typeof(T).Name}> ({size} elements): {time:F2}ms ({perSec:F0} el/s) " +
            $"- copy path baseline for ARROW-ROADMAP Phase D zero-copy payoff");
    }

    static double MeasureBestOfFiveMs(Action action)
    {
        var best = double.MaxValue;
        var sw = new Stopwatch();
        for (int i = 0; i < 5; i++)
        {
            sw.Restart();
            action();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }
}
