using Silk.NET.OpenCL;
using System.Text;

namespace Nivara.Samples.Gpu;

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
/// </summary>
internal sealed unsafe class SilkRuntime : IDisposable
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
    /// no OpenCL GPU is reachable, so a <c>--silk</c> run on a machine without a driver fails
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
                    "--silk requires an OpenCL GPU device (the in-box OpenCL.dll + Intel/AMD driver exposing an iGPU); none was found.");

            // The enumeration asked for GPUs; a driver that answers with something else must fail
            // loudly rather than silently turn this into a CPU measurement.
            if (!IsGpu(cl, dev))
                throw new InvalidOperationException(
                    $"--silk found {name}, which reports a non-GPU device type — refusing to run (no CPU fallback).");

            device = dev;
            DeviceName = name;
            Vendor = vendor;
            MaxWorkGroupSize = maxWorkGroupSize;
            LocalMemorySize = localMemorySize;

            int err = 0;
            context = cl.CreateContext(null, 1, &dev, null, null, out err);
            if (err != 0 || context == 0)
                throw new InvalidOperationException($"--silk clCreateContext failed ({err}).");

            queue = cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out err);
            if (err != 0 || queue == 0)
                throw new InvalidOperationException($"--silk clCreateCommandQueue failed ({err}).");

            program = BuildProgram();
            foreach (string kernelName in SilkKernels.KernelNames)
            {
                nint kernel = cl.CreateKernel(program, kernelName, out err);
                if (err != 0 || kernel == 0)
                    throw new InvalidOperationException($"--silk clCreateKernel('{kernelName}') failed ({err}).");
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
    /// <see cref="GpuBuffers.ValidateAttentionLocalMemory"/> on the ILGPU path): the kernel's
    /// accumulator tile is a compile-time constant, so a larger <c>headDim</c> would read past it.
    /// </summary>
    public void EnsureAttentionLocalMemory(int headDim)
    {
        if (headDim > SilkKernels.MaxHeadDim)
            throw new InvalidOperationException(
                $"--silk attention supports headDim up to {SilkKernels.MaxHeadDim}; got {headDim}.");

        ulong needed = (ulong)SilkKernels.MaxHeadDim * SilkKernels.AttentionGroupSize * sizeof(float);
        if (LocalMemorySize < needed)
            throw new InvalidOperationException(
                $"--silk attention needs {needed} B of local memory per work-group; {DeviceName} reports {LocalMemorySize} B.");
    }

    public nint Allocate(int floatCount) => AllocateBytes((ulong)floatCount * sizeof(float));

    public nint AllocateBytes(ulong bytes)
    {
        nint buffer = cl.CreateBuffer(context, MemFlags.ReadWrite, (nuint)bytes, null, out int err);
        if (err != 0 || buffer == 0)
            throw new InvalidOperationException($"--silk clCreateBuffer({bytes} B) failed ({err}).");
        return buffer;
    }

    /// <summary>Blocking write of exactly <paramref name="data"/>.Length elements.</summary>
    public void Upload(nint buffer, ReadOnlySpan<float> data)
    {
        fixed (float* p = data)
            WriteRaw(buffer, p, (ulong)data.Length * sizeof(float));
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
                throw new InvalidOperationException($"--silk readback of {count} floats failed ({err}).");
        }
        return values;
    }

    void WriteRaw(nint buffer, void* p, ulong bytes)
    {
        int err = cl.EnqueueWriteBuffer(queue, buffer, true, 0, (nuint)bytes, p, 0, null, null);
        if (err != 0)
            throw new InvalidOperationException($"--silk upload of {bytes} B failed ({err}).");
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
            throw new InvalidOperationException($"--silk clSetKernelArg('{kernelName}', arg {index}) failed ({err}).");
    }

    /// <summary>
    /// Enqueues <paramref name="kernelName"/> over the given global/local work sizes and blocks
    /// until it completes. The global size must already be a whole multiple of the local size —
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
                throw new InvalidOperationException($"--silk dispatch of '{kernelName}' failed ({err}).");
        }

        int finish = cl.Finish(queue);
        if (finish != 0)
            throw new InvalidOperationException($"--silk clFinish after '{kernelName}' failed ({finish}).");
    }

    public void Release(nint buffer)
    {
        if (buffer != 0)
            cl.ReleaseMemObject(buffer);
    }

    nint Kernel(string name)
        => kernels.TryGetValue(name, out nint kernel)
            ? kernel
            : throw new InvalidOperationException($"--silk has no kernel named '{name}'.");

    /// <summary>Compiles <see cref="SilkKernels.Source"/> in-process; on failure the driver's
    /// build log is decoded (the single most useful diagnostics for Intel/AMD drivers).</summary>
    nint BuildProgram()
    {
        int err = 0;
        nint built = cl.CreateProgramWithSource(context, 1, [SilkKernels.Source], null, out err);
        if (err != 0 || built == 0)
            throw new InvalidOperationException($"--silk clCreateProgramWithSource failed ({err}).");

        // num_devices = 0 means "every device in the context" — the one device asked for.
        if (cl.BuildProgram(built, 0, null, (string?)null, null, null) == 0)
            return built;

        string log = BuildLog(built);
        cl.ReleaseProgram(built);
        throw new InvalidOperationException("--silk clBuildProgram failed:" + Environment.NewLine + log);
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
