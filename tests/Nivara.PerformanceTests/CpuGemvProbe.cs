using Nivara.AutoDiff.Operations;
using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand GEMV comparison for issue #1119 part 2. M=1 was excluded from
/// <c>--cpu-gemm</c>. This mode does not register a provider and does not change
/// <c>MultiplyCore</c>. A production <c>IGemvProvider</c> waits on a human-run
/// median ratio of at least 2× (managed / sgemv at 1 OpenBLAS thread).
/// </summary>
internal static class CpuGemvProbe
{
    const int Warmups = 1;
    const int Rounds = 20;
    const double GateMaxAbs = 1e-3;
    const int MissingDllExit = 2;

    static readonly (string Name, int K, int N)[] Shapes =
    [
        ("hidden-384", 384, 384),
        ("hidden-768", 768, 768),
        ("hidden-1024", 1024, 1024),
        ("hidden-4096", 4096, 4096),
        ("k768-n30522", 768, 30522),
        ("k1024-n30522", 1024, 30522),
        ("k4096-n30522", 4096, 30522),
    ];

    static readonly (string Name, int Threads)[] BlasLegs =
    [
        ("sgemv-1t", 1),
        ("sgemv-4t", 4),
        ("sgemv-8t", 8),
    ];

    public static int Run()
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("CPU GEMV probe: managed single-row MatMulTransposedB vs cblas_sgemv");
        Console.WriteLine("Bind bar: managed median / sgemv-1t median >= 2. This process does not bind.");
        Console.WriteLine();

        if (!OpenBlasInterop.TryInitialize(out string detail))
        {
            Console.Error.WriteLine("OpenBLAS is not available: " + detail);
            return MissingDllExit;
        }

        Console.WriteLine($"  {detail}");
        Console.WriteLine($"  corename: {OpenBlasInterop.CoreName}");
        Console.WriteLine($"  Gate: maxAbs(leg - double dot) <= {GateMaxAbs}");
        Console.WriteLine($"  Timing: {Warmups} untimed warmup, then {Rounds} interleaved rounds; median and min.");
        Console.WriteLine();
        Console.WriteLine($"{"shape",-16} {"K",-6} {"N",-7} {"leg",-12} {"maxAbs",-10} {"gate",-6} {"median ms",-10} {"min ms",-8} {"GMAC/s",-8} {"vs managed",-12}");

        int failures = 0;
        foreach (var shape in Shapes)
            failures += Measure(shape.Name, shape.K, shape.N);

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? "Gate: every leg passed."
            : $"Gate: {failures} leg(s) failed.");
        return failures == 0 ? 0 : 1;
    }

    static int Measure(string name, int k, int n)
    {
        var random = new Random(7);
        var a = new float[k];
        var b = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)random.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)random.NextDouble();
        var truth = DoubleDot(a, b, k, n);

        var legs = new Action<float[]>[1 + BlasLegs.Length];
        var labels = new string[legs.Length];
        legs[0] = y => GradKernels.MatMulTransposedB(a, b, y, 1, k, n);
        labels[0] = "managed";
        for (int i = 0; i < BlasLegs.Length; i++)
        {
            int threads = BlasLegs[i].Threads;
            legs[i + 1] = y => Sgemv(a, b, y, k, n, threads);
            labels[i + 1] = BlasLegs[i].Name;
        }

        var buffers = new float[legs.Length][];
        var abs = new double[legs.Length];
        int failures = 0;
        for (int i = 0; i < legs.Length; i++)
        {
            buffers[i] = new float[n];
            legs[i](buffers[i]);
            abs[i] = MaxAbs(buffers[i], truth);
            if (abs[i] > GateMaxAbs)
                failures++;
        }

        var samples = new List<double>[legs.Length];
        for (int i = 0; i < legs.Length; i++)
            samples[i] = new List<double>(Rounds);

        for (int i = 0; i < legs.Length; i++)
            legs[i](buffers[i]);

        for (int round = 0; round < Rounds; round++)
        {
            int first = round % legs.Length;
            for (int step = 0; step < legs.Length; step++)
            {
                int leg = (first + step) % legs.Length;
                long start = Stopwatch.GetTimestamp();
                legs[leg](buffers[leg]);
                samples[leg].Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }
        }

        double managedMs = Median(samples[0]);
        long macs = (long)k * n;
        for (int i = 0; i < legs.Length; i++)
        {
            double median = Median(samples[i]);
            double min = Min(samples[i]);
            double gmacs = macs / (median * 1e6);
            string versus = i == 0 ? "-" : $"{managedMs / median:F2}x";
            string gate = abs[i] > GateMaxAbs ? "FAIL" : "PASS";
            Console.WriteLine($"{name,-16} {k,-6} {n,-7} {labels[i],-12} {abs[i],-10:E2} {gate,-6} {median,-10:F3} {min,-8:F3} {gmacs,-8:F1} {versus,-12}");
        }

        return failures;
    }

    static void Sgemv(float[] a, float[] b, float[] y, int k, int n, int threads)
    {
        OpenBlasInterop.SetNumThreads(threads);
        OpenBlasInterop.Sgemv(
            OpenBlasInterop.RowMajor, OpenBlasInterop.NoTrans,
            n, k, 1f, ref b[0], k, ref a[0], 1, 0f, ref y[0], 1);
    }

    static double[] DoubleDot(float[] a, float[] b, int k, int n)
    {
        var truth = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            int row = j * k;
            for (int t = 0; t < k; t++)
                sum += a[t] * (double)b[row + t];
            truth[j] = sum;
        }

        return truth;
    }

    static double MaxAbs(float[] got, double[] truth)
    {
        double max = 0;
        for (int i = 0; i < truth.Length; i++)
        {
            double delta = Math.Abs(got[i] - truth[i]);
            if (delta > max)
                max = delta;
        }

        return max;
    }

    static double Median(List<double> samples)
    {
        var ordered = new List<double>(samples);
        ordered.Sort();
        int mid = ordered.Count / 2;
        return ordered.Count % 2 == 1
            ? ordered[mid]
            : (ordered[mid - 1] + ordered[mid]) / 2;
    }

    static double Min(List<double> samples)
    {
        double min = samples[0];
        for (int i = 1; i < samples.Count; i++)
            if (samples[i] < min)
                min = samples[i];
        return min;
    }
}
