using System.Diagnostics;

namespace Nivara.PerformanceTests;

/// <summary>
/// Native Silk-only GEMM baseline/comparison harness (no ILGPU leg). Runs the Silk
/// <c>gemm</c> kernel over the standard GEMM shapes and compares its output and timing
/// against an ascending-K CPU reference (mul+add), so measurement and perf tuning work
/// on Silk-only machines where the ILGPU runtime is unavailable.
/// </summary>
internal static class SilkGemmProbe
{
    const int Warmups = 1;
    const int TimedRounds = 5;
    const double GateMaxAbs = 1e-3;

    public static int Run(string[] args)
    {
        GemmBenchmark.PrintPowerState();
        Console.WriteLine("Silk-only GEMM baseline: Silk `gemm` vs ascending-K CPU reference (mul+add)");
        Console.WriteLine($"  Gate  : maxAbs(silk - cpu) <= {GateMaxAbs:E0}");
        Console.WriteLine($"  Timing: {Warmups} warmup + best-of-{TimedRounds} launches");
        Console.WriteLine();

        Nivara.Silk.SilkRuntime silk;
        try
        {
            silk = new Nivara.Silk.SilkRuntime();
        }
        catch (Exception ex)
        {
            Console.WriteLine("UNBUILT: --silk-gemm requires the Silk (OpenCL) runtime; none was available.");
            Console.WriteLine($"  {ex.GetType().Name}: {ex.Message}");
            return 10;
        }

        using (silk)
        {
            Console.WriteLine($"{"shape",-20} {"M",-5} {"K",-5} {"N",-6} {"byteDiffs",-10} {"maxAbs",-10} {"gate",-6} {"cpu ms",-9} {"silk ms",-9} {"x",-6}");
            Console.WriteLine(new string('-', 96));

            int failures = 0;
            foreach (var shape in GemmBenchmark.s_shapes)
            {
                int m = shape.Arows, k = shape.Acols, n = shape.Bcols;

                uint seed = 0x9E3779B9u;
                var a = new float[m * k];
                var w = new float[n * k];
                for (int i = 0; i < a.Length; i++) a[i] = GemmBenchmark.NextUniformF(ref seed);
                for (int i = 0; i < w.Length; i++) w[i] = GemmBenchmark.NextUniformF(ref seed);

                // w is already [N x K] row-major, so Bt == w.
                var bt = w;

                var cpuOut = new float[m * n];
                Action runCpu = () =>
                {
                    for (int i = 0; i < m; i++)
                    {
                        int aBase = i * k;
                        int outBase = i * n;
                        for (int j = 0; j < n; j++)
                        {
                            float acc = 0.0f;
                            int bBase = j * k;
                            for (int kk = 0; kk < k; kk++)
                                acc += a[aBase + kk] * bt[bBase + kk];
                            cpuOut[outBase + j] = acc;
                        }
                    }
                };

                int elements = m * n;
                nint sa = silk.Allocate(m * k);
                nint sb = silk.Allocate(n * k);
                nint sc = silk.Allocate(elements);
                silk.Upload(sa, a);
                silk.Upload(sb, bt);

                Action runSilk = () =>
                {
                    silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 0, sa);
                    silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 1, sb);
                    silk.SetBuffer(Nivara.Silk.SilkKernels.Gemm, 2, sc);
                    silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 3, (uint)m);
                    silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 4, (uint)k);
                    silk.SetUInt32(Nivara.Silk.SilkKernels.Gemm, 5, (uint)n);
                    silk.Run(
                        Nivara.Silk.SilkKernels.Gemm,
                        [(nuint)Nivara.Silk.SilkKernels.GemmGlobalRows(m), (nuint)Nivara.Silk.SilkKernels.GemmGlobalCols(n)],
                        [(nuint)Nivara.Silk.SilkKernels.GemmTileSize, (nuint)Nivara.Silk.SilkKernels.GemmTileSize]);
                    silk.Finish();
                };

                runCpu();
                runSilk();
                float[] silkOut = silk.Download(sc, elements);

                int byteDiffs = 0;
                double maxAbs = 0.0;
                for (int i = 0; i < elements; i++)
                {
                    if (BitConverter.SingleToInt32Bits(silkOut[i]) != BitConverter.SingleToInt32Bits(cpuOut[i]))
                        byteDiffs++;
                    maxAbs = Math.Max(maxAbs, Math.Abs(silkOut[i] - cpuOut[i]));
                }

                double cpuUs = GemmBenchmark.TimeBest(runCpu, Warmups, TimedRounds);
                double silkUs = GemmBenchmark.TimeBest(runSilk, Warmups, TimedRounds);
                double cpuMs = cpuUs / 1000.0;
                double silkMs = silkUs / 1000.0;

                bool failed = maxAbs > GateMaxAbs;
                if (failed) failures++;
                double ratio = cpuMs > 0 ? silkMs / cpuMs : 0.0;
                Console.WriteLine($"{shape.Name,-20} {m,-5} {k,-5} {n,-6} {byteDiffs,-10} {maxAbs,-10:E2} {(failed ? "FAIL" : "PASS"),-6} {cpuMs,-9:F2} {silkMs,-9:F2} {ratio,-6:F2}");

                silk.Release(sa);
                silk.Release(sb);
                silk.Release(sc);
            }

            Console.WriteLine();
            Console.WriteLine(failures == 0
                ? "Gate PASS — all Silk GEMM cells within tolerance of the CPU reference."
                : $"Gate FAIL — {failures} cell(s) exceed {GateMaxAbs} vs CPU reference.");

            return failures;
        }
    }
}
