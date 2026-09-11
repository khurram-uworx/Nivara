using System.Runtime.InteropServices;
using Nivara.GpuProbe.LevelZero;

namespace Nivara.GpuProbe;

/// <summary>
/// Probe: can .NET access Intel GPU compute via Level Zero? Pure P/Invoke against
/// the inbox ze_loader.dll — no packages. Enumerates drivers/devices (Arc 140T iGPU,
/// possibly the NPU) and then runs hand-authored SPIR-V kernels for real.
/// </summary>
internal class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Intel Level Zero GPU Probe ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}  OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine();

        string mode = args.Length > 0 ? args[0] : "l0";
        return mode switch
        {
            "l0" => L0Probe.Run() + L0Run.Run(),
            "list" => L0Probe.Run(),
            "run" => L0Run.Run(),
            "spv" => SpvDump.Run(),
            "ocl" => OclProbe.Run(),
            "all" => L0Probe.Run() + L0Run.Run(),
            _ => L0Probe.Run() + L0Run.Run()
        };
    }
}