using Nivara.IO;
using NUnit.Framework;
using System.Diagnostics;

namespace Nivara.Tests.IO;

[TestFixture]
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
