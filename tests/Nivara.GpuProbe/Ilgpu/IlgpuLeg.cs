using System.Diagnostics;
using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Nivara.GpuProbe.D3d12;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.Ilgpu;

/// <summary>
/// The ILGPU leg of the kernel probe (issue #431, phase 4a): a pure-managed C#-kernel
/// runtime that JIT-compiles the three production SmolLM kernels to OpenCL C and runs
/// them on the Arc 140T iGPU through the in-box Windows <c>OpenCL.dll</c> ICD + Intel
/// graphics driver. BF16 stays the wire format (packed 2-per-uint, widened in-shader —
/// byte-identical transport to the D3D12 leg). Setup cost is split from steady state:
/// context/accelerator creation and kernel JIT + first dispatch are timed separately, and
/// per-kernel times are 1 warmup + best-of-25 synchronized dispatches (the OpenVINO
/// methodology, matching <see cref="OpenVino.OpenVinoLeg"/>).
///
/// The explicit OpenCL-only context + GPU-device assert make a silent CPU fallback
/// impossible: if no OpenCL GPU device is reachable the leg returns <c>null</c> (the
/// KernelGate UNBUILT row) instead of running anywhere else.
/// </summary>
internal static class IlgpuLeg
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
            Console.WriteLine($"  ILGPU: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static LegResults? Run(KernelFixtures fixtures)
    {
        using var context = Context.Create(builder => builder.OpenCL().Optimize(OptimizationLevel.O2));
        CLDevice? device = SelectGPU(context);
        if (device is null)
        {
            Console.WriteLine("  ILGPU: no OpenCL GPU device found (in-box OpenCL.dll + Intel driver must expose the Arc iGPU)");
            return null;
        }

        var sw = new Stopwatch();

        sw.Restart();
        using CLAccelerator accelerator = device.CreateCLAccelerator(context);
        accelerator.Synchronize();
        double acceleratorUs = sw.Elapsed.TotalMicroseconds;
        Console.WriteLine($"  device: {accelerator.Name} ({accelerator.VendorName}) | type {accelerator.DeviceType} | acc setup {acceleratorUs,8:F0} µs");
        if (accelerator.DeviceType != CLDeviceType.CL_DEVICE_TYPE_GPU)
            throw new InvalidOperationException($"OpenCL accelerator {accelerator.Name} is not a GPU (got {accelerator.DeviceType}) — no CPU fallback");

        using AcceleratorStream stream = accelerator.CreateStream();

        // Persistent buffers, allocated once and reused across all timed iterations
        // (the D3D12 one-shot-transient lesson — see docs/DX12.md).
        using MemoryBuffer1D<uint, Stride1D.Dense> dot16Input = accelerator.Allocate1D<uint>(KernelFixtures.Dot16Length);
        using MemoryBuffer1D<uint, Stride1D.Dense> siluInput = accelerator.Allocate1D<uint>(KernelFixtures.HiddenSize / 2);
        using MemoryBuffer1D<uint, Stride1D.Dense> gemvInput = accelerator.Allocate1D<uint>(
            (KernelFixtures.IntermediateSize * KernelFixtures.HiddenSize + KernelFixtures.HiddenSize) / 2);
        using MemoryBuffer1D<float, Stride1D.Dense> dot16Output = accelerator.Allocate1D<float>(1);
        using MemoryBuffer1D<float, Stride1D.Dense> siluOutput = accelerator.Allocate1D<float>(KernelFixtures.HiddenSize);
        using MemoryBuffer1D<float, Stride1D.Dense> gemvOutput = accelerator.Allocate1D<float>(KernelFixtures.IntermediateSize);

        dot16Input.CopyFromCPU(PackConcat(fixtures.Dot16A, fixtures.Dot16B));
        siluInput.CopyFromCPU(GemvKernels.PackBf16(fixtures.SiluX));
        gemvInput.CopyFromCPU(PackConcat(fixtures.GemvW, fixtures.GemvX));

        // Kernel JIT compile happens at load (OpenCL program build → IGC); the first
        // dispatch below additionally pays any driver-side lazy compile. Both are the
        // "setup" half of the split; the timed passes are steady state only.
        var dot16Kernel = accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<uint>, ArrayView<float>, int>(IlgpuKernels.Dot16Kernel);
        var siluKernel = accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<uint>, ArrayView<float>, int>(IlgpuKernels.SiluKernel);
        var gemvKernel = accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<uint>, ArrayView<float>, int, int, int>(IlgpuKernels.GemvKernel);

        var (dot16, dot16JitUs, dot16Us) = RunKernel(
            () => { dot16Kernel(stream, 1, dot16Input.View, dot16Output.View, KernelFixtures.Dot16Length); stream.Synchronize(); },
            () => dot16Output.AsContiguous().GetAsArray(),
            sw);
        Console.WriteLine($"  [dot16] jit {dot16JitUs,8:F0} µs | steady {dot16Us,8:F1} µs");

        var (silu, siluJitUs, siluUs) = RunKernel(
            () => { siluKernel(stream, KernelFixtures.HiddenSize, siluInput.View, siluOutput.View, KernelFixtures.HiddenSize); stream.Synchronize(); },
            () => siluOutput.AsContiguous().GetAsArray(),
            sw);
        Console.WriteLine($"  [silu]  jit {siluJitUs,8:F0} µs | steady {siluUs,8:F1} µs");

        var (gemv, gemvJitUs, gemvUs) = RunKernel(
            () =>
            {
                gemvKernel(stream, KernelFixtures.IntermediateSize, gemvInput.View, gemvOutput.View,
                    KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize,
                    KernelFixtures.IntermediateSize * KernelFixtures.HiddenSize);
                stream.Synchronize();
            },
            () => gemvOutput.AsContiguous().GetAsArray(),
            sw);
        Console.WriteLine($"  [gemv]  jit {gemvJitUs,8:F0} µs | steady {gemvUs,8:F1} µs");

        return new LegResults(dot16[0], silu, gemv, dot16Us, siluUs, gemvUs);
    }

    /// <summary>First dispatch (driver JIT) timed, one warmup, then best-of-N steady state.</summary>
    private static (float[] Results, double JitUs, double BestUs) RunKernel(Action dispatch, Func<float[]> read, Stopwatch sw)
    {
        sw.Restart();
        dispatch();
        double jitUs = sw.Elapsed.TotalMicroseconds;

        dispatch(); // warmup — steady-state scheduling

        double bestUs = double.PositiveInfinity;
        for (int i = 0; i < TimingPasses; i++)
        {
            sw.Restart();
            dispatch();
            sw.Stop();
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMicroseconds);
        }
        return (read(), jitUs, bestUs);
    }

    /// <summary>Picks the first OpenCL GPU device from an Intel vendor — never the CPU.</summary>
    internal static CLDevice? SelectGPU(Context context)
    {
        foreach (CLDevice device in context.GetCLDevices())
        {
            if (device.DeviceType == CLDeviceType.CL_DEVICE_TYPE_GPU &&
                (device.Name?.Contains("Intel", StringComparison.OrdinalIgnoreCase) ?? false))
                return device;
        }
        return null;
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