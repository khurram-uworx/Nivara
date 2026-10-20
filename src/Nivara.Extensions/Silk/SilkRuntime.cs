using Silk.NET.OpenCL;
using System.Text;

namespace Nivara.Silk;

/// <summary>
/// Host-side OpenCL plumbing for <see cref="SilkKernels"/>: an Intel/AMD GPU is selected (and
/// re-asserted as a GPU — there is no CPU fallback), the OpenCL C source is compiled in-process
/// by the driver with its build log decoded on failure, and buffers/dispatches are wrapped in
/// helpers that check every status where it is produced.
///
/// <para>
/// The API usage mirrors the verified Silk.NET 2.23.0 calls in
/// <c>tests/Nivara.GpuProbe/OpenCl/SilkLeg.cs</c> (docs/SILK.md §4). The one deliberate
/// difference is status handling: the probe ORs the <c>clSetKernelArg</c> results into one slot,
/// which loses <em>which</em> argument failed; here each call is checked individually and the
/// failure names the kernel and argument index. That failure mode is worth the extra lines — a
/// mismatched argument count makes the kernel read uninitialized scalars and answers fast and
/// wrong rather than erroring.
/// </para>
///
/// <para>
/// Public because <see cref="SilkBertEncoderRunner"/> takes one, the same way
/// <c>BertEncoderGpuRunner</c> takes an <c>IlgpuRuntime</c>: the caller owns the
/// device lifetime and the runner only borrows it.
/// </para>
/// </summary>
public sealed unsafe class SilkRuntime : IDisposable
{
    readonly CL cl;
    nint device;
    nint context;
    nint queue;
    nint program;
    readonly Dictionary<string, nint> kernels = new();
    bool disposed;

    public string DeviceName { get; }
    public string Vendor { get; }
    public ulong MaxWorkGroupSize { get; }
    public ulong LocalMemorySize { get; }

    /// <summary>
    /// Builds the context, the in-process program and every kernel in
    /// <see cref="SilkKernels.KernelNames"/>. Throws (rather than falling back to the CPU) when
    /// no OpenCL GPU is reachable, so a run on a machine without a driver fails
    /// with the reason instead of quietly measuring the host.
    /// </summary>
    public SilkRuntime()
    {
        cl = CL.GetApi();
        try
        {
            nint dev = SelectGpu(cl, out string name, out string vendor, out ulong maxWorkGroupSize, out ulong localMemorySize);
            if (dev == 0)
                throw new InvalidOperationException(
                    "Silk requires an OpenCL GPU device (the in-box OpenCL.dll + Intel/AMD driver exposing an iGPU); none was found.");

            // The enumeration asked for GPUs; a driver that answers with something else must fail
            // loudly rather than silently turn this into a CPU measurement.
            if (!IsGpu(cl, dev))
                throw new InvalidOperationException(
                    $"Silk found {name}, which reports a non-GPU device type — refusing to run (no CPU fallback).");

            device = dev;
            DeviceName = name;
            Vendor = vendor;
            MaxWorkGroupSize = maxWorkGroupSize;
            LocalMemorySize = localMemorySize;

            int err = 0;
            context = cl.CreateContext(null, 1, &dev, null, null, out err);
            if (err != 0 || context == 0)
                throw new InvalidOperationException($"Silk clCreateContext failed ({err}).");

            queue = cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out err);
            if (err != 0 || queue == 0)
                throw new InvalidOperationException($"Silk clCreateCommandQueue failed ({err}).");

            program = BuildProgram();
            foreach (string kernelName in SilkKernels.KernelNames)
            {
                nint kernel = cl.CreateKernel(program, kernelName, out err);
                if (err != 0 || kernel == 0)
                    throw new InvalidOperationException($"Silk clCreateKernel('{kernelName}') failed ({err}).");
                kernels[kernelName] = kernel;
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public string Describe()
        => $"{DeviceName} ({Vendor}) | max WG {MaxWorkGroupSize} | local {LocalMemorySize} B";

    /// <summary>
    /// Rejects a head dimension the fused attention tile cannot hold (it mirrors
    /// <c>GpuBuffers.ValidateAttentionLocalMemory</c> on the ILGPU path): the kernel's
    /// accumulator tile is a compile-time constant, so a larger <c>headDim</c> would read past it.
    /// </summary>
    public void EnsureAttentionLocalMemory(int headDim)
    {
        if (headDim > SilkKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"Silk attention supports headDim up to {SilkKernels.MaxHeadDim}; got {headDim}.");

        ulong needed = (ulong)SilkKernels.MaxHeadDim * SilkKernels.AttentionGroupSize * sizeof(float);
        if (LocalMemorySize < needed)
            throw new InvalidOperationException(
                $"Silk attention needs {needed} B of local memory per work-group; {DeviceName} reports {LocalMemorySize} B.");
    }

    /// <summary>
    /// Returns true when the device can run the fused decode attention + O-projection kernel:
    /// enough work-items per group for all heads and enough local memory for the context tile.
    /// </summary>
    public bool CanFuseDecodeOproj(int numHeads, int headDim)
    {
        if (numHeads <= 0 || headDim <= 0) return false;
        if (MaxWorkGroupSize < (ulong)numHeads * (ulong)SilkKernels.AttentionGroupSize) return false;
        try
        {
            EnsureFusedDecodeOprojLocalMemory(numHeads, headDim);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Rejects a configuration the fused O-projection tile cannot hold (mirrors
    /// <see cref="EnsureAttentionLocalMemory"/>): local memory must hold the per-head
    /// context plus one head's online-softmax partials.
    /// </summary>
    public void EnsureFusedDecodeOprojLocalMemory(int numHeads, int headDim)
    {
        if (headDim > SilkKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"Silk fused O-proj supports headDim up to {SilkKernels.MaxHeadDim}; got {headDim}.");

        if (numHeads > 64)
            throw new InvalidOperationException(
                $"Silk fused O-proj supports up to 64 heads; got {numHeads}.");

        ulong needed = ((ulong)numHeads * (ulong)headDim
                      + (ulong)SilkKernels.AttentionGroupSize
                      + (ulong)SilkKernels.AttentionGroupSize
                      + (ulong)SilkKernels.AttentionGroupSize * (ulong)SilkKernels.MaxHeadDim)
                     * sizeof(float);
        if (LocalMemorySize < needed)
            throw new InvalidOperationException(
                $"Silk fused O-proj needs {needed} B of local memory per work-group; {DeviceName} reports {LocalMemorySize} B.");
    }

    /// <summary>
    /// Grows <paramref name="buffer"/> when it is smaller than <paramref name="floatCount"/>,
    /// allocating it on first use. Never shrinks. The same shape as
    /// <c>GpuBuffers.Ensure</c>:
    /// a runner declares workspace as <see cref="SilkBuffer.Empty"/> and the first forward sizes it.
    /// </summary>
    public SilkBuffer Ensure(SilkBuffer buffer, int floatCount)
    {
        if (floatCount < 0)
            throw new ArgumentOutOfRangeException(nameof(floatCount), floatCount, "floatCount cannot be negative.");
        if (buffer.Handle != 0 && buffer.Capacity >= floatCount)
            return buffer;
        Release(buffer.Handle);
        return floatCount == 0 ? SilkBuffer.Empty : new SilkBuffer(Allocate(floatCount), floatCount);
    }

    public nint Allocate(int floatCount) => AllocateBytes((ulong)floatCount * sizeof(float));

    public nint AllocateBytes(ulong bytes)
    {
        nint buffer = cl.CreateBuffer(context, MemFlags.ReadWrite, (nuint)bytes, null, out int err);
        if (err != 0 || buffer == 0)
            throw new InvalidOperationException($"Silk clCreateBuffer({bytes} B) failed ({err}).");
        return buffer;
    }

    /// <summary>
    /// Creates a read-only sub-buffer view of <paramref name="elementCount"/> floats starting at
    /// <paramref name="elementOffset"/> into <paramref name="parent"/> — the OpenCL spelling of the
    /// ILGPU <c>SubView</c> the reference runner hands to the fused attention kernel, whose q/k/v
    /// are the three contiguous blocks of the packed qkv result. The view shares the parent's
    /// storage, so the parent must outlive it; the caller releases it via <see cref="Release"/>.
    /// </summary>
    public nint CreateSubview(nint parent, int elementOffset, int elementCount)
    {
        var region = new BufferRegion
        {
            Origin = (nuint)((ulong)elementOffset * sizeof(float)),
            Size = (nuint)((ulong)elementCount * sizeof(float))
        };

        int err = 0;
        nint view = cl.CreateSubBuffer(parent, MemFlags.ReadOnly, BufferCreateType.Region, ref region, out err);
        if (err != 0 || view == 0)
            throw new InvalidOperationException(
                $"Silk clCreateSubBuffer(offset {elementOffset}, {elementCount} floats) failed ({err}).");
        return view;
    }

    /// <summary>Blocking write of exactly <paramref name="data"/>.Length elements.</summary>
    public void Upload(nint buffer, ReadOnlySpan<float> data)
    {
        fixed (float* p = data)
            WriteRaw(buffer, p, (ulong)data.Length * sizeof(float));
    }

    /// <summary>Blocking write of exactly <paramref name="data"/>.Length elements.</summary>
    public void Upload(nint buffer, ReadOnlySpan<ushort> data)
    {
        fixed (ushort* p = data)
            WriteRaw(buffer, p, (ulong)data.Length * sizeof(ushort));
    }

    /// <summary>Blocking write of exactly <paramref name="data"/>.Length elements.</summary>
    public void Upload(nint buffer, ReadOnlySpan<int> data)
    {
        fixed (int* p = data)
            WriteRaw(buffer, p, (ulong)data.Length * sizeof(int));
    }

    /// <summary>Blocking read of <paramref name="count"/> floats.</summary>
    public float[] Download(nint buffer, int count)
    {
        var values = new float[count];
        fixed (float* p = values)
        {
            int err = cl.EnqueueReadBuffer(queue, buffer, true, 0, (nuint)(count * sizeof(float)), p, 0, null, null);
            if (err != 0)
                throw new InvalidOperationException($"Silk readback of {count} floats failed ({err}).");
        }
        return values;
    }

    /// <summary>Blocking read of <paramref name="count"/> ints.</summary>
    public int[] DownloadInt(nint buffer, int count)
    {
        var values = new int[count];
        fixed (int* p = values)
        {
            int err = cl.EnqueueReadBuffer(queue, buffer, true, 0, (nuint)(count * sizeof(int)), p, 0, null, null);
            if (err != 0)
                throw new InvalidOperationException($"Silk readback of {count} ints failed ({err}).");
        }
        return values;
    }

    void WriteRaw(nint buffer, void* p, ulong bytes)
    {
        int err = cl.EnqueueWriteBuffer(queue, buffer, true, 0, (nuint)bytes, p, 0, null, null);
        if (err != 0)
            throw new InvalidOperationException($"Silk upload of {bytes} B failed ({err}).");
    }

    /// <summary>Blocking device-to-device copy of <paramref name="count"/> floats from
    /// <paramref name="src"/> at <paramref name="srcOffset"/> to <paramref name="dst"/> at
    /// <paramref name="dstOffset"/>.</summary>
    public void CopyBuffer(nint src, nint dst, int srcOffset, int dstOffset, int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (count == 0) return;
        int err = cl.EnqueueCopyBuffer(
            queue, src, dst,
            (nuint)((ulong)srcOffset * sizeof(float)),
            (nuint)((ulong)dstOffset * sizeof(float)),
            (nuint)((ulong)count * sizeof(float)),
            0, null, null);
        if (err != 0)
            throw new InvalidOperationException($"Silk clEnqueueCopyBuffer({count} floats) failed ({err}).");
    }

    public void SetBuffer(string kernelName, uint index, nint buffer)
    {
        nint local = buffer;
        int err = cl.SetKernelArg(Kernel(kernelName), index, (nuint)sizeof(nint), &local);
        FailArg(kernelName, index, err);
    }

    public void SetUInt32(string kernelName, uint index, uint value)
    {
        int err = cl.SetKernelArg(Kernel(kernelName), index, sizeof(uint), &value);
        FailArg(kernelName, index, err);
    }

    public void SetInt32(string kernelName, uint index, int value)
    {
        int err = cl.SetKernelArg(Kernel(kernelName), index, sizeof(int), &value);
        FailArg(kernelName, index, err);
    }

    public void SetFloat32(string kernelName, uint index, float value)
    {
        int err = cl.SetKernelArg(Kernel(kernelName), index, sizeof(float), &value);
        FailArg(kernelName, index, err);
    }

    void FailArg(string kernelName, uint index, int err)
    {
        if (err != 0)
            throw new InvalidOperationException($"Silk clSetKernelArg('{kernelName}', arg {index}) failed ({err}).");
    }

    /// <summary>
    /// Enqueues <paramref name="kernelName"/> over the given global/local work sizes (non-blocking).
    /// The global size must already be a whole multiple of the local size —
    /// the <see cref="SilkKernels"/> geometry helpers guarantee that — because OpenCL rejects a
    /// non-multiple and the guard would otherwise be dead in the kernel.
    /// </summary>
    public void Run(string kernelName, ReadOnlySpan<nuint> global, ReadOnlySpan<nuint> local)
    {
        if (global.Length != local.Length)
            throw new ArgumentException("global and local work sizes must have the same rank.", nameof(local));

        nuint offset = 0;
        fixed (nuint* g = global)
        fixed (nuint* l = local)
        {
            int err = cl.EnqueueNdrangeKernel(
                queue, Kernel(kernelName), (uint)global.Length, &offset, g, l, 0, null, null);
            if (err != 0)
                throw new InvalidOperationException($"Silk dispatch of '{kernelName}' failed ({err}).");
        }
    }

    /// <summary>Blocks until every queued command (kernels, copies) on the device queue has
    /// completed, and checks its completion status. Launches and copies are now non-blocking, so
    /// callers needing strict synchronization (timing, host reads of non-downloaded data) must
    /// call this (or use a blocking <see cref="Download"/> / upload).</summary>
    public void Finish()
    {
        int status = cl.Finish(queue);
        if (status != 0)
            throw new InvalidOperationException($"Silk clFinish failed ({status}).");
    }

    /// <summary>A device buffer plus the float capacity it was allocated for. <see cref="Handle"/> 0 is empty.</summary>
    public readonly record struct SilkBuffer(nint Handle, int Capacity)
    {
        public static SilkBuffer Empty => default;
    }

    public void Release(nint buffer)
    {
        if (buffer != 0)
            cl.ReleaseMemObject(buffer);
    }

    nint Kernel(string name)
        => kernels.TryGetValue(name, out nint kernel)
            ? kernel
            : throw new InvalidOperationException($"Silk has no kernel named '{name}'.");

    /// <summary>Compiles <see cref="SilkKernels.Source"/> in-process; on failure the driver's
    /// build log is decoded (the single most useful diagnostics for Intel/AMD drivers).</summary>
    nint BuildProgram()
    {
        int err = 0;
        nint built = cl.CreateProgramWithSource(context, 1, [SilkKernels.Source], null, out err);
        if (err != 0 || built == 0)
            throw new InvalidOperationException($"Silk clCreateProgramWithSource failed ({err}).");

        // num_devices = 0 means "every device in the context" — the one device asked for.
        if (cl.BuildProgram(built, 0, null, (string?)null, null, null) == 0)
            return built;

        string log = BuildLog(built);
        cl.ReleaseProgram(built);
        throw new InvalidOperationException("Silk clBuildProgram failed:" + Environment.NewLine + log);
    }

    string BuildLog(nint built)
    {
        nuint size = 0;
        if (cl.GetProgramBuildInfo(built, device, ProgramBuildInfo.BuildLog, 0, null, &size) != 0 || size == 0)
            return "    (no build log available)";
        byte[] buffer = new byte[(int)size];
        fixed (byte* p = buffer)
            cl.GetProgramBuildInfo(built, device, ProgramBuildInfo.BuildLog, (nuint)buffer.Length, p, null);
        int end = Array.IndexOf(buffer, (byte)0);
        return "    " + Encoding.UTF8.GetString(buffer, 0, end < 0 ? buffer.Length : end).Trim();
    }

    public void Dispose()
    {
        if (disposed)
            return;
        disposed = true;

        foreach (nint kernel in kernels.Values)
            if (kernel != 0)
                cl.ReleaseKernel(kernel);
        kernels.Clear();
        if (program != 0)
        {
            cl.ReleaseProgram(program);
            program = 0;
        }
        if (queue != 0)
        {
            cl.ReleaseCommandQueue(queue);
            queue = 0;
        }
        if (context != 0)
        {
            cl.ReleaseContext(context);
            context = 0;
        }
        device = 0;
    }

    static unsafe nint SelectGpu(CL cl, out string name, out string vendor, out ulong maxWorkGroupSize, out ulong localMemorySize)
    {
        name = vendor = "";
        maxWorkGroupSize = 0;
        localMemorySize = 0;

        uint platformCount = 0;
        if (cl.GetPlatformIDs(0, null, &platformCount) != 0 || platformCount == 0)
            return 0;

        var platforms = new nint[platformCount];
        fixed (nint* pp = platforms)
            if (cl.GetPlatformIDs(platformCount, pp, null) != 0)
                return 0;

        nint fallback = 0;
        foreach (nint platform in platforms)
        {
            uint deviceCount = 0;
            if (cl.GetDeviceIDs(platform, DeviceType.Gpu, 0, null, &deviceCount) != 0 || deviceCount == 0)
                continue;

            var devices = new nint[deviceCount];
            fixed (nint* dp = devices)
                if (cl.GetDeviceIDs(platform, DeviceType.Gpu, deviceCount, dp, null) != 0)
                    continue;

            foreach (nint candidate in devices)
            {
                string candidateVendor = DeviceInfoString(cl, candidate, DeviceInfo.Vendor);
                bool preferred = candidateVendor.Contains("Intel", StringComparison.OrdinalIgnoreCase)
                              || candidateVendor.Contains("AMD", StringComparison.OrdinalIgnoreCase);
                if (preferred || fallback == 0)
                {
                    fallback = candidate;
                    name = DeviceInfoString(cl, candidate, DeviceInfo.Name);
                    vendor = candidateVendor;
                    ulong wg = 0;
                    cl.GetDeviceInfo(candidate, DeviceInfo.MaxWorkGroupSize, sizeof(ulong), &wg, null);
                    maxWorkGroupSize = wg;
                    ulong lm = 0;
                    cl.GetDeviceInfo(candidate, DeviceInfo.LocalMemSize, sizeof(ulong), &lm, null);
                    localMemorySize = lm;
                }
                if (preferred)
                    break;
            }

            if (fallback != 0)
                return fallback;
        }
        return fallback;
    }

    static unsafe bool IsGpu(CL cl, nint device)
    {
        DeviceType type = DeviceType.None;
        cl.GetDeviceInfo(device, DeviceInfo.Type, sizeof(DeviceType), &type, null);
        return (type & DeviceType.Gpu) != 0;
    }

    /// <summary>Two-call OpenCL info pattern; trims at the first NUL rather than assuming
    /// <c>size - 1</c>, since some drivers omit the terminator (docs/SILK.md §4).</summary>
    static unsafe string DeviceInfoString(CL cl, nint device, DeviceInfo info)
    {
        nuint size = 0;
        if (cl.GetDeviceInfo(device, info, 0, null, &size) != 0 || size == 0)
            return "";
        byte[] buffer = new byte[(int)size];
        fixed (byte* p = buffer)
            cl.GetDeviceInfo(device, info, (nuint)buffer.Length, p, null);
        int end = Array.IndexOf(buffer, (byte)0);
        return Encoding.UTF8.GetString(buffer, 0, end < 0 ? buffer.Length : end);
    }
}
