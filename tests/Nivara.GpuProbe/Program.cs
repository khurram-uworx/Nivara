using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Nivara.GpuProbe.LevelZero;

namespace Nivara.GpuProbe;

/// <summary>
/// Probe: can .NET access Intel GPU compute? Pure P/Invoke — no packages — with two
/// deliberate exceptions: the phase-4a ILGPU leg (issue #431) and the phase-4b
/// ComputeSharp leg (issue #432), the probe's only NuGet references, because there the
/// package *is* the toolchain (pure-managed C#→OpenCL / C#→HLSL→DXIL JIT runtimes).
/// The Level
/// Zero leg enumerates drivers/devices (Arc 140T iGPU, possibly the NPU) and runs
/// hand-authored SPIR-V kernels; the DX12 leg runs a hand-rolled compute pipeline;
/// the OpenVINO leg loads the pip-installed openvino_c.dll and drives IR models on
/// GPU; the ILGPU leg JIT-compiles C# kernels to OpenCL C; the ComputeSharp leg
/// source-generates HLSL from C# kernels and runs them through D3D12.
/// Each GPU backend runs the same SmolLM-shaped kernel fixtures, gated against
/// the production Nivara CPU kernels (`kernels` mode).
/// </summary>
[SupportedOSPlatform("windows6.2")] // whole probe is Windows GPU plumbing (L0/DX12/OpenCL/D3D12)
internal class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Intel GPU Compute Probe (Level Zero / DX12 / OpenVINO / SYCL / ILGPU / ComputeSharp) ===");
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
#if WINDOWS
            "computesharp" => ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg),
#endif
            "kernels" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(),
                ("SYCL (oneAPI)", Sycl.SyclLeg.RunLeg),
                ("DX12 (hand-rolled)", D3d12.D3d12Compute.RunLeg),
                ("OV (bf16 IR)", OpenVino.OpenVinoLeg.RunBf16),
                ("OV (f32 + hint)", OpenVino.OpenVinoLeg.RunF32),
                ("ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg)
#if WINDOWS
                , ("ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg)
#endif
                ),
            "all" => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run() + OpenVino.Availability.Run() + OVRun()
#if WINDOWS
                + ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg)
#endif
                ,
            _ => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run() + OpenVino.Availability.Run() + OVRun()
#if WINDOWS
                + ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg)
#endif
        };
    }
}