using Nivara.AutoDiff.Operations;
using Nivara.Extensions.Blas;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.SimdProbe;

/// <summary>
/// A/B probe: managed <c>GradKernels.MatMulTransposedB</c> (the in-tree path) vs the
/// OpenBLAS-backed path, for both GEMM (<c>aRows &gt; 1</c>) and GEMV (<c>aRows == 1</c>).
///
/// <para>The BLAS leg registers the provider with <c>threadCount = ProcessorCount</c>
/// and <c>openBlasThreads = 1</c> — the hybrid integration shape (Nivara owns the row
/// parallelism; OpenBLAS is called single-threaded per chunk) recommended by the
/// BLAS-MKL probe — so the comparison reflects the candidate wiring rather than a
/// single-threaded OpenBLAS call.</para>
///
/// <para>Methodology follows the SimdProbe/transpose rules: correctness gate first
/// (<c>maxAbs(leg - double truth) &lt;= 1e-3</c>), untimed warmup of both legs, then
/// interleaved A/B rounds alternating which leg is measured first, reporting medians.</para>
/// </summary>
internal static class BlasProbe
{
    const int Rounds = 20;
    const int Warmups = 5;
    const double GateMaxAbs = 1e-3;
    const int MissingDllExit = 2;

    static readonly (string Name, int M, int K, int N)[] GemmShapes =
    [
        ("square-256", 256, 256, 256),
        ("square-512", 512, 512, 512),
        ("tall-1024x64", 1024, 1024, 64),
        ("batched-8x768", 8, 768, 768),
    ];

    static readonly (string Name, int K, int N)[] GemvShapes =
    [
        ("hidden-768", 768, 768),
        ("hidden-4096", 4096, 4096),
        ("vocab-768x30522", 768, 30522),
    ];

    public static int Run()
    {
        Console.WriteLine("=== BLAS A/B probe: managed vs OpenBLAS (GEMM + GEMV) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Interleaved: {Rounds} rounds/shape, {Warmups} warmups/leg, gate maxAbs <= {GateMaxAbs}.");
        Console.WriteLine();

        if (!OpenBlasInterop.TryInitialize(out string detail))
        {
            Console.Error.WriteLine("OpenBLAS is not available: " + detail);
            return MissingDllExit;
        }
        Console.WriteLine($"  {detail}");
        Console.WriteLine($"  corename: {OpenBlasInterop.CoreName}");
        Console.WriteLine();

        int failures = 0;
        Console.WriteLine("--- GEMM (aRows > 1) ---");
        foreach (var (name, m, k, n) in GemmShapes)
            failures += MeasureGemm(name, m, k, n);

        Console.WriteLine();
        Console.WriteLine("--- GEMV (aRows == 1) ---");
        foreach (var (name, k, n) in GemvShapes)
            failures += MeasureGemv(name, k, n);

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? "Gate: every leg passed."
            : $"Gate: {failures} leg(s) failed.");
        return failures == 0 ? 0 : 1;
    }

    static int MeasureGemm(string name, int m, int k, int n)
    {
        var rng = new Random(7);
        var a = new float[m * k];
        var bt = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < bt.Length; i++) bt[i] = (float)rng.NextDouble();

        var truth = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                int aBase = i * k;
                int bBase = j * k;
                for (int t = 0; t < k; t++)
                    sum += a[aBase + t] * (double)bt[bBase + t];
                truth[i * n + j] = sum;
            }

        var managedBuf = new float[m * n];
        var blasBuf = new float[m * n];

        Action managed = () =>
        {
            NivaraBlas.UseManaged();
            GradKernels.MatMulTransposedB(a, bt, managedBuf, m, k, n);
        };
        Action blas = () =>
        {
            NivaraBlas.UseBlas(threadCount: Environment.ProcessorCount, openBlasThreads: 1);
            GradKernels.MatMulTransposedB(a, bt, blasBuf, m, k, n);
        };

        // Correctness (untimed).
        managed();
        blas();
        double absManaged = MaxAbs(managedBuf, truth);
        double absBlas = MaxAbs(blasBuf, truth);
        int failures = 0;
        if (absManaged > GateMaxAbs) failures++;
        if (absBlas > GateMaxAbs) failures++;

        // Warm up both legs.
        for (int w = 0; w < Warmups; w++)
        {
            managed();
            blas();
        }

        var managedSamples = new List<double>(Rounds);
        var blasSamples = new List<double>(Rounds);

        for (int round = 0; round < Rounds; round++)
        {
            if (round % 2 == 0)
            {
                managedSamples.Add(Time(managed));
                blasSamples.Add(Time(blas));
            }
            else
            {
                blasSamples.Add(Time(blas));
                managedSamples.Add(Time(managed));
            }
        }

        double managedMs = Median(managedSamples);
        double blasMs = Median(blasSamples);
        double ratio = blasMs > 0 ? managedMs / blasMs : double.NaN;
        long macs = (long)m * k * n;

        Console.WriteLine($"{name,-18} M={m,-5} K={k,-5} N={n,-5}  " +
                          $"managed {managedMs,8:F3} ms  blas {blasMs,8:F3} ms  " +
                          $"ratio {ratio,6:F2}x  macs {macs / 1e6,7:F1} M  " +
                          $"maxAbs m={absManaged:E1} b={absBlas:E1}  " +
                          $"{(failures == 0 ? "PASS" : "FAIL")}");
        return failures;
    }

    static int MeasureGemv(string name, int k, int n)
    {
        var rng = new Random(7);
        var a = new float[k];
        var b = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        var truth = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            int row = j * k;
            for (int t = 0; t < k; t++)
                sum += a[t] * (double)b[row + t];
            truth[j] = sum;
        }

        var managedBuf = new float[n];
        var blasBuf = new float[n];

        Action managed = () =>
        {
            NivaraBlas.UseManaged();
            GradKernels.MatMulTransposedB(a, b, managedBuf, 1, k, n);
        };
        Action blas = () =>
        {
            NivaraBlas.UseBlas(threadCount: Environment.ProcessorCount, openBlasThreads: 1);
            GradKernels.MatMulTransposedB(a, b, blasBuf, 1, k, n);
        };

        managed();
        blas();
        double absManaged = MaxAbs(managedBuf, truth);
        double absBlas = MaxAbs(blasBuf, truth);
        int failures = 0;
        if (absManaged > GateMaxAbs) failures++;
        if (absBlas > GateMaxAbs) failures++;

        for (int w = 0; w < Warmups; w++)
        {
            managed();
            blas();
        }

        var managedSamples = new List<double>(Rounds);
        var blasSamples = new List<double>(Rounds);

        for (int round = 0; round < Rounds; round++)
        {
            if (round % 2 == 0)
            {
                managedSamples.Add(Time(managed));
                blasSamples.Add(Time(blas));
            }
            else
            {
                blasSamples.Add(Time(blas));
                managedSamples.Add(Time(managed));
            }
        }

        double managedMs = Median(managedSamples);
        double blasMs = Median(blasSamples);
        double ratio = blasMs > 0 ? managedMs / blasMs : double.NaN;
        long macs = (long)k * n;

        Console.WriteLine($"{name,-18} K={k,-5} N={n,-6}  " +
                          $"managed {managedMs,8:F3} ms  blas {blasMs,8:F3} ms  " +
                          $"ratio {ratio,6:F2}x  macs {macs / 1e6,7:F1} M  " +
                          $"maxAbs m={absManaged:E1} b={absBlas:E1}  " +
                          $"{(failures == 0 ? "PASS" : "FAIL")}");
        return failures;
    }

    static double Time(Action action)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    static double Median(List<double> samples)
    {
        var ordered = new List<double>(samples);
        ordered.Sort();
        int mid = ordered.Count / 2;
        return ordered.Count % 2 == 1
            ? ordered[mid]
            : (ordered[mid - 1] + ordered[mid]) / 2.0;
    }

    static double MaxAbs(float[] got, double[] truth)
    {
        double max = 0;
        for (int i = 0; i < truth.Length; i++)
        {
            double delta = Math.Abs(got[i] - truth[i]);
            if (delta > max) max = delta;
        }
        return max;
    }
}
