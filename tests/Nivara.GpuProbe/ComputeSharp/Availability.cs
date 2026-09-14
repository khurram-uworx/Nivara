#if WINDOWS
using System.Runtime.Versioning;
using ComputeSharp;

namespace Nivara.GpuProbe.ComputeSharp;

/// <summary>
/// <c>computesharp</c> mode diagnostics: the default D3D12 device ComputeSharp will run on
/// and its hardware-acceleration status (a WARP software device means the leg's gate row is
/// UNBUILT — never a CPU run). Prints the failure when no device is reachable.
/// </summary>
/// <remarks>Compiled only on Windows hosts (<c>WINDOWS</c> symbol). See the csproj gate.</remarks>
[SupportedOSPlatform("windows6.2")] // D3D12 — Windows-only, like the hand-rolled DX12 leg
internal static class Availability
{
    public static int Run()
    {
        Console.WriteLine();
        Console.WriteLine("--- ComputeSharp availability ---");
        try
        {
            using GraphicsDevice device = GraphicsDevice.GetDefault();
            Console.WriteLine($"  device: {device.Name}");
            Console.WriteLine($"  hardware accelerated: {device.IsHardwareAccelerated}");
            if (!device.IsHardwareAccelerated)
            {
                Console.WriteLine("  WARP (software) device only — ComputeSharp leg will report UNBUILT, not run on CPU");
                return 1;
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  unavailable: {ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }
}
#endif