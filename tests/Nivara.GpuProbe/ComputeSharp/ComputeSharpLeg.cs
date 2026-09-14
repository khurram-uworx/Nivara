#if WINDOWS
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Versioning;
using ComputeSharp;
using Nivara.GpuProbe.D3d12;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.ComputeSharp;

/// <summary>
/// The ComputeSharp leg of the kernel probe (issue #432, phase 4b): the probe's second
/// NuGet-based GPU backend and its first managed-DX12 path. ComputeSharp is a pure-managed
/// C# kernel JIT (MIT) that source-generates HLSL from ordinary C# structs
/// (<see cref="ComputeSharpKernels"/>), compiles it to DXIL in-process via the DXC binaries
/// bundled by <c>ComputeSharp.Dxc</c> (no external compiler or toolchain), and dispatches it
/// on the Arc 140T iGPU through D3D12 — the managed story the probe's hand-rolled
/// <c>D3d12Compute</c> leg predicted. BF16 stays the wire format (packed 2-per-uint via
/// <c>D3d12.GemvKernels.PackBf16</c>, widened in-shader by <c>Hlsl.AsFloat</c> — byte-identical
/// transport to the D3D12 and ILGPU legs).
///
/// Setup cost is split from steady state like the other legs: the first dispatch through each
/// kernel pays the DXC HLSL→DXIL compile + descriptor pipeline creation (timed as the <c>dxc</c>
/// split), and per-kernel times are 1 warmup + best-of-25 synchronized dispatches (the OpenVINO
/// methodology, matching <see cref="Ilgpu.IlgpuLeg"/>). Every <c>GraphicsDevice.For</c> call is a
/// full submit+wait round-trip (the compute context runs the command list and waits for
/// completion), so the steady-state numbers are comparable to the ILGPU
/// <c>dispatch + stream.Synchronize()</c> idiom.
///
/// The device assert makes a silent CPU fallback impossible: <see cref="GraphicsDevice.GetDefault"/>
/// walks adapters with WARP last, and we additionally reject a non-hardware device, so the leg
/// returns <c>null</c> (the KernelGate UNBUILT row) instead of running on the software rasterizer.
/// </summary>
/// <remarks>Compiled only on Windows hosts (<c>WINDOWS</c> symbol): ComputeSharp's source
/// generator needs the Windows SDK host. See the csproj gate.</remarks>
[SupportedOSPlatform("windows6.2")] // D3D12 — Windows-only, like the hand-rolled DX12 leg
internal static class ComputeSharpLeg
{
    private const int TimingPasses = 25;

    public static LegResults? RunLeg(KernelFixtures fixtures)
    {
        try
        {
            return Run(fixtures);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ComputeSharp: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static LegResults? Run(KernelFixtures fixtures)
    {
        Console.WriteLine("--- ComputeSharp leg (C# → HLSL source-gen → DXIL via bundled DXC on D3D12) ---");

        using GraphicsDevice device = GraphicsDevice.GetDefault();
        Console.WriteLine($"  device: {device.Name}");
        if (!device.IsHardwareAccelerated)
        {
            Console.WriteLine("  ComputeSharp: default device is WARP (software rasterizer) — refusing CPU fallback, leg unavailable");
            return null;
        }

        var sw = new Stopwatch();

        // Persistent buffers, allocated once and reused across all timed iterations
        // (the D3D12 one-shot-transient lesson — see docs/DX12.md). The shaders read
        // packed-BF16 input as uint (2 elements per 4-byte element) and write f32.
        using ReadOnlyBuffer<uint> dot16Input = device.AllocateReadOnlyBuffer(PackConcat(fixtures.Dot16A, fixtures.Dot16B));
        using ReadOnlyBuffer<uint> siluInput = device.AllocateReadOnlyBuffer(GemvKernels.PackBf16(fixtures.SiluX));
        using ReadOnlyBuffer<uint> gemvInput = device.AllocateReadOnlyBuffer(PackConcat(fixtures.GemvW, fixtures.GemvX));
        using ReadWriteBuffer<float> dot16Output = device.AllocateReadWriteBuffer<float>(1);
        using ReadWriteBuffer<float> siluOutput = device.AllocateReadWriteBuffer<float>(KernelFixtures.HiddenSize);
        using ReadWriteBuffer<float> gemvOutput = device.AllocateReadWriteBuffer<float>(KernelFixtures.IntermediateSize);

        var (dot16, dot16DxcUs, dot16Us) = RunKernel(
            () => device.For(1, new ComputeSharpKernels.Dot16Shader(dot16Input, dot16Output, KernelFixtures.Dot16Length)),
            () => dot16Output.ToArray(),
            sw);
        Console.WriteLine($"  [dot16] dxc {dot16DxcUs,8:F0} µs | steady {dot16Us,8:F1} µs");

        var (silu, siluDxcUs, siluUs) = RunKernel(
            () => device.For(KernelFixtures.HiddenSize, new ComputeSharpKernels.SiluShader(siluInput, siluOutput, KernelFixtures.HiddenSize)),
            () => siluOutput.ToArray(),
            sw);
        Console.WriteLine($"  [silu]  dxc {siluDxcUs,8:F0} µs | steady {siluUs,8:F1} µs");

        var (gemv, gemvDxcUs, gemvUs) = RunKernel(
            () => device.For(KernelFixtures.IntermediateSize, new ComputeSharpKernels.GemvShader(
                gemvInput, gemvOutput, KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize,
                KernelFixtures.IntermediateSize * KernelFixtures.HiddenSize)),
            () => gemvOutput.ToArray(),
            sw);
        Console.WriteLine($"  [gemv]  dxc {gemvDxcUs,8:F0} µs | steady {gemvUs,8:F1} µs");

        return new LegResults(dot16[0], silu, gemv, dot16Us, siluUs, gemvUs);
    }

    /// <summary>First dispatch (DXC HLSL→DXIL compile + descriptor pipeline) timed, one warmup,
    /// then best-of-N steady state.</summary>
    private static (float[] Results, double DxcUs, double BestUs) RunKernel(Action dispatch, Func<float[]> read, Stopwatch sw)
    {
        sw.Restart();
        dispatch();
        double dxcUs = sw.Elapsed.TotalMicroseconds;

        dispatch(); // warmup — steady-state scheduling

        double bestUs = double.PositiveInfinity;
        for (int i = 0; i < TimingPasses; i++)
        {
            sw.Restart();
            dispatch();
            sw.Stop();
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMicroseconds);
        }
        return (read(), dxcUs, bestUs);
    }

    /// <summary>Concatenates BF16 fixtures, then packs 2-per-uint so element parity is
    /// preserved across the boundary (the kernels widen by absolute element index).</summary>
    private static uint[] PackConcat(params BFloat16[][] arrays)
    {
        int total = 0;
        foreach (var array in arrays)
            total += array.Length;
        var flat = new BFloat16[total];
        int offset = 0;
        foreach (var array in arrays)
        {
            array.CopyTo(flat, offset);
            offset += array.Length;
        }
        return GemvKernels.PackBf16(flat);
    }
}
#endif