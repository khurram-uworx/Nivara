using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;

namespace Nivara.Samples.Gpu;

/// <summary>
/// OpenCL-only ILGPU runtime for the DistilBERT GPU scenario (docs/BERT-GPU.md):
/// context + accelerator + explicit stream lifecycle with a hard GPU-device assert —
/// a silent CPU fallback is impossible (same contract as the probe's IlgpuLeg,
/// docs/ILGPU.md). Context is built OpenCL-only at OptimizationLevel.O2; XMath from
/// ILGPU.Algorithms is available to kernels without an explicit EnableAlgorithms call
/// (probe-verified on ILGPU 1.5.3).
/// </summary>
public sealed class IlgpuRuntime : IDisposable
{
    private readonly Context context;
    private readonly CLAccelerator accelerator;
    private readonly AcceleratorStream stream;

    public CLAccelerator Accelerator => accelerator;
    public AcceleratorStream Stream => stream;
    public string DeviceName => $"{accelerator.Name} ({accelerator.VendorName})";

    public IlgpuRuntime()
    {
        context = Context.Create(builder => builder.OpenCL().Optimize(OptimizationLevel.O2));
        CLDevice? device = SelectGpu(context);
        if (device is null)
        {
            throw new InvalidOperationException(
                "--gpu requires an OpenCL GPU device (the in-box OpenCL.dll + Intel driver exposing the Arc iGPU); none was found.");
        }

        accelerator = device.CreateCLAccelerator(context);
        if (accelerator.DeviceType != CLDeviceType.CL_DEVICE_TYPE_GPU)
        {
            throw new InvalidOperationException(
                $"OpenCL accelerator {accelerator.Name} is not a GPU (got {accelerator.DeviceType}); --gpu has no CPU fallback.");
        }

        stream = accelerator.CreateStream();
    }

    /// <summary>Picks the first OpenCL GPU device from an Intel vendor — never the CPU.</summary>
    public static CLDevice? SelectGpu(Context context)
    {
        foreach (CLDevice device in context.GetCLDevices())
        {
            if (device.DeviceType == CLDeviceType.CL_DEVICE_TYPE_GPU &&
                (device.Name?.Contains("Intel", StringComparison.OrdinalIgnoreCase) ?? false))
            {
                return device;
            }
        }

        return null;
    }

    public MemoryBuffer1D<float, Stride1D.Dense> Allocate1D(int length) => accelerator.Allocate1D<float>(length);

    public void Synchronize() => stream.Synchronize();

    public void Dispose()
    {
        stream.Dispose();
        accelerator.Dispose();
        context.Dispose();
    }
}