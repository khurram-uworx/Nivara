using Nivara.GpuProbe.LevelZero;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

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
/// source-generates HLSL from C# kernels and runs them through D3D12; the
/// Silk.NET OpenCL leg compiles hand-authored OpenCL C in-process through
/// Silk.NET's managed bindings to the in-box ICD (no external toolchain).
/// Each GPU backend runs the same SmolLM-shaped kernel fixtures, gated against
/// the production Nivara CPU kernels (`kernels` mode). A `--rounds N` argument
/// (default 5) sets how many times the gate sweeps its legs.
/// </summary>
[SupportedOSPlatform("windows6.2")] // whole probe is Windows GPU plumbing (L0/DX12/OpenCL/D3D12)
internal class Program
{
    /// <summary>Exit code for a command-line usage error — distinct from a gate run's
    /// exit code, which is the count of unexpected failed kernels.</summary>
    private const int UsageExitCode = 64;

    static int Main(string[] args)
    {
        Console.WriteLine("=== Intel GPU Compute Probe (Level Zero / DX12 / OpenVINO / SYCL / ILGPU / ComputeSharp) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}  OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine();

        int rounds = Kernels.KernelGate.DefaultRounds;
        string? modeArg = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--rounds")
            {
                if (i + 1 >= args.Length || !int.TryParse(args[i + 1], out int parsed) || parsed < 1)
                {
                    string got = i + 1 < args.Length ? args[i + 1] : "<missing>";
                    Console.Error.WriteLine($"usage: --rounds expects a positive integer (got '{got}')");
                    return UsageExitCode;
                }
                rounds = parsed;
                i++;
            }
            else
            {
                modeArg ??= args[i];
            }
        }
        string mode = modeArg ?? "l0";
        int OVRun()
            => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), rounds,
                ("OV (bf16 IR)", OpenVino.OpenVinoLeg.RunBf16, Kernels.KernelDiagnosis.OvBf16Silu()),
                ("OV (f32 + hint)", OpenVino.OpenVinoLeg.RunF32, (Kernels.KernelDiagnosis?)null));
        return mode switch
        {
            "l0" => L0Probe.Run() + L0Run.Run(),
            "list" => L0Probe.Run(),
            "run" => L0Run.Run(),
            "spv" => SpvDump.Run(),
            "ocl" => OclProbe.Run(),
            "silk" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "Silk.NET OpenCL", OpenCl.SilkLeg.RunLeg, rounds),
            "dx12" => D3d12Check.Run() + D3d12.D3d12Compute.Run(rounds),
            "sycl" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "SYCL (oneAPI)", Sycl.SyclLeg.RunLeg, rounds),
            "ov" => OpenVino.Availability.Run() + OVRun(),
            "ilgpu" => Ilgpu.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg, rounds),
#if WINDOWS
            "computesharp" => ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg, rounds),
#endif
            "kernels" => Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), rounds,
                ("SYCL (oneAPI)", Sycl.SyclLeg.RunLeg, (Kernels.KernelDiagnosis?)null),
                ("DX12 (hand-rolled)", D3d12.D3d12Compute.RunLeg, null),
                ("OV (bf16 IR)", OpenVino.OpenVinoLeg.RunBf16, Kernels.KernelDiagnosis.OvBf16Silu()),
                ("OV (f32 + hint)", OpenVino.OpenVinoLeg.RunF32, null),
                ("ILGPU (OpenCL)", Ilgpu.IlgpuLeg.RunLeg, null),
                ("Silk.NET OpenCL", OpenCl.SilkLeg.RunLeg, null)
#if WINDOWS
                , ("ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg, null)
#endif
                ),
            "all" => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run(rounds) + OpenVino.Availability.Run() + OVRun()
#if WINDOWS
                + ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg, rounds)
#endif
                ,
            _ => L0Probe.Run() + L0Run.Run() + D3d12Check.Run() + D3d12.D3d12Compute.Run(rounds) + OpenVino.Availability.Run() + OVRun()
#if WINDOWS
                + ComputeSharp.Availability.Run() + Kernels.KernelGate.Run(Kernels.KernelFixtures.Generate(), "ComputeSharp (DXIL)", ComputeSharp.ComputeSharpLeg.RunLeg, rounds)
#endif
        };
    }
}
