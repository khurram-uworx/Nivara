using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Nivara.SimdProbe;

/// <summary>
/// Probe: can .NET 11 hardware intrinsics accelerate BFloat16 / Half compute via
/// widen-compute-narrow, where the BCL TensorPrimitives path runs scalar loops?
/// </summary>
internal class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== BFloat16 / Half SIMD Probe ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        Console.WriteLine($"Vector128.IsHardwareAccelerated: {Vector128.IsHardwareAccelerated}");
        Console.WriteLine();

        string mode = args.Length > 0 ? args[0] : "all";
        return mode switch
        {
            "cpu" => CpuIdProbe.Run(),
            "support" => SupportReport.Run(),
            "correctness" => Correctness.RunAll(),
            "benchmark" => Benchmark.RunAll(),
            "scalar" => ScalarKernelProbe.Run(),
            "all" => CpuIdProbe.Run() + SupportReport.Run() + Correctness.RunAll() + Benchmark.RunAll(),
            _ => SupportReport.Run() + Correctness.RunAll() + Benchmark.RunAll()
        };
    }
}
