using System.Runtime.InteropServices;
using Nivara.GpuProbe.LevelZero;

namespace Nivara.GpuProbe;

/// <summary>
/// Probe: can .NET access Intel GPU compute? Pure P/Invoke — no packages. The Level
/// Zero leg enumerates drivers/devices (Arc 140T iGPU, possibly the NPU) and runs
/// hand-authored SPIR-V kernels; the DX12 leg runs a hand-rolled compute pipeline;
/// the OpenVINO leg loads the pip-installed openvino_c.dll and drives IR models on
/// GPU; the ILGPU leg (phase 4a, issue #431) JIT-compiles C# kernels to OpenCL C.
/// Each GPU backend runs the same SmolLM-shaped kernel fixtures, gated against
/// the production Nivara CPU kernels (`kernels` mode).
/// </summary>
internal class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Intel GPU Compute Probe (Level Zero / DX12 / OpenVINO / SYCL) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}  OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine();

        string mode = args.Length > 0 ? args[0] : "l0";
        int OVRun()
            => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(),
                ("OV (bf16 IR)", OpenVino.OpenVinoLeg.RunBf16),
                ("OV (f32 + hint)", OpenVino.OpenVinoLeg.RunF32));
        return mode switch
        {
            "l0" => L0Probe.Run() + L0Run.Run(),
            "list" => L0Probe.Run(),
            "run" => L0Run.Run(),
            "spv" => SpvDump.Run(),
            "ocl" => OclProbe.Run(),
            "dx12" => D3d12Check.Run() + D3d12.D3d12Compute.Run(),
            "sycl" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "SYCL (oneAPI)", Sycl.SyclLeg.RunLeg),
            "ov" => OpenVino.Availability.Run() + OVRun(),
            "ilgpu" => Ilgpu.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg),
            "kernels" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(),
                ("SYCL (oneAPI)", Sycl.SyclLeg.RunLeg),
                ("DX12 (hand-rolled)", D3d12.D3d12Compute.RunLeg),
                ("OV (bf16 IR)", OpenVino.OpenVinoLeg.RunBf16),
                ("OV (f32 + hint)", OpenVino.OpenVinoLeg.RunF32),
                ("ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg)),
            "all" => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run() + OpenVino.Availability.Run() + OVRun(),
            _ => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run() + OpenVino.Availability.Run() + OVRun()
        };
    }
}