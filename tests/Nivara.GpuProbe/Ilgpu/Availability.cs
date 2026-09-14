using ILGPU;
using ILGPU.Runtime.OpenCL;

namespace Nivara.GpuProbe.Ilgpu;

/// <summary>
/// <c>ilgpu</c> mode diagnostics: the ILGPU version, every OpenCL device the driver
/// exposes, and the accelerator the leg will run on. Prints the failure + hint when the
/// OpenCL backend is unreachable.
/// </summary>
internal static class Availability
{
    public static int Run()
    {
        Console.WriteLine();
        Console.WriteLine("--- ILGPU availability ---");
        try
        {
            using var context = Context.Create(builder => builder.OpenCL().Optimize(OptimizationLevel.O2));
            int count = 0;
            foreach (CLDevice device in context.GetCLDevices())
            {
                count++;
                Console.WriteLine($"  CL device: {device.Name} | {device.VendorName} | {device.DeviceType}");
            }
            if (count == 0)
            {
                Console.WriteLine("  no OpenCL devices (in-box OpenCL.dll + Intel graphics driver required)");
                return 1;
            }
            CLDevice? gpu = IlgpuLeg.SelectGPU(context);
            if (gpu is null)
            {
                Console.WriteLine("  no OpenCL GPU device (Intel) — ILGPU leg will not run");
                return 1;
            }
            using CLAccelerator accelerator = gpu.CreateCLAccelerator(context);
            Console.WriteLine($"  accelerator: {accelerator.Name} | {accelerator.DeviceType}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  unavailable: {ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }
}