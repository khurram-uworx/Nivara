using Nivara.GpuProbe.D3d12;
using Nivara.GpuProbe.Kernels;
using Silk.NET.OpenCL;
using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace Nivara.GpuProbe.OpenCl;

/// <summary>
/// The Silk.NET OpenCL leg of the kernel probe: hand-authored OpenCL C compiled
/// in-process by the driver (<c>clCreateProgramWithSource</c> + <c>clBuildProgram</c>) —
/// no external toolchain, no subprocess — and dispatched on an Intel/AMD iGPU through
/// Silk.NET's managed bindings to the OpenCL C API. This is the "true OpenCL" baseline
/// the abstraction-hiding legs (ILGPU, SYCL, ComputeSharp) are compared against.
///
/// BF16 stays the wire format: inputs are packed 2-per-uint by
/// <see cref="GemvKernels.PackBf16"/> — the same byte-identical transport the DX12,
/// ComputeSharp and ILGPU legs upload — and widened to f32 in-shader by the same
/// element-parity <c>as_float</c> trick, so a widening difference cannot masquerade as
/// a kernel difference. Build options are deliberately empty (no
/// <c>-cl-fast-relaxed-math</c>): the gate compares against the precise CPU SiLU, and
/// relaxing the math would turn a precision regression into a timing "win".
///
/// Setup cost is split from steady state the same way as <see cref="Ilgpu.IlgpuLeg"/>:
/// program build and each kernel's first dispatch are reported separately, and the
/// timed passes are 1 warmup + best-of-<see cref="TimingPasses"/> synchronized
/// dispatches. Device selection is GPU-only with an explicit type assertion, so a
/// silent CPU fallback is impossible — with no GPU the leg returns <c>null</c>
/// (the KernelGate UNBUILT row) instead of running anywhere else.
/// </summary>
internal static class SilkLeg
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
            Console.WriteLine($"  Silk.NET OpenCL: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static LegResults? Run(KernelFixtures fixtures)
    {
        Console.WriteLine("--- Silk.NET OpenCL leg (managed bindings, in-process OpenCL C compile) ---");

        CL cl = CL.GetApi();
        nint device = SelectGpu(cl, out string deviceName, out string vendor, out ulong maxWorkGroupSize);
        if (device == 0)
        {
            Console.WriteLine("  Silk.NET OpenCL: no OpenCL GPU device reachable (in-box OpenCL ICD + Intel/AMD driver must expose an iGPU)");
            return null;
        }
        Console.WriteLine($"  device: {deviceName} ({vendor}) | max WG {maxWorkGroupSize} | OpenCL {DeviceVersion(cl, device)}");
        PrintExtensions(cl, device);

        // Explicitly re-assert GPU-ness: the enumeration above asked for GPUs, and a
        // driver that answers with something else must fail loudly rather than silently
        // turn this leg into a CPU measurement.
        if (!IsGpu(cl, device))
        {
            Console.WriteLine($"  Silk.NET OpenCL: {deviceName} reports a non-GPU device type — refusing to run");
            return null;
        }

        int local = (int)Math.Clamp((long)GemvKernels.ThreadsPerGroup, 1, (long)Math.Max(1, maxWorkGroupSize));

        var stopwatch = new Stopwatch();
        unsafe
        {
            int err = 0;
            nint context = cl.CreateContext(null, 1, &device, null, null, out err);
            if (err != 0 || context == 0)
            {
                Console.WriteLine($"  [FAIL] clCreateContext: {err}");
                return null;
            }

            nint queue = cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out err);
            if (err != 0 || queue == 0)
            {
                Console.WriteLine($"  [FAIL] clCreateCommandQueue: {err}");
                cl.ReleaseContext(context);
                return null;
            }

            nint program = 0;
            try
            {
                uint dot16Length = KernelFixtures.Dot16Length;
                uint hidden = KernelFixtures.HiddenSize;
                uint rows = KernelFixtures.IntermediateSize;
                uint cols = KernelFixtures.HiddenSize;

                stopwatch.Restart();
                program = BuildProgram(cl, context, device, KernelSource());
                double buildUs = stopwatch.Elapsed.TotalMicroseconds;
                if (program == 0)
                {
                    Console.WriteLine("  [FAIL] program build (build log above)");
                    return null;
                }
                Console.WriteLine($"  program: OpenCL C built in-process | build {buildUs,8:F0} µs");

                // Persistent buffers, allocated once and reused across every timed
                // iteration (the D3D12 one-shot-transient lesson — see DX12.md). They are
                // created *inside* the try so that a partial failure still releases the
                // buffers that were already created.
                uint[] dot16Input = PackConcat(fixtures.Dot16A, fixtures.Dot16B);
                uint[] siluInput = GemvKernels.PackBf16(fixtures.SiluX);
                uint[] gemvInput = PackConcat(fixtures.GemvW, fixtures.GemvX);

                nint dot16In = 0, siluIn = 0, gemvIn = 0, dot16Out = 0, siluOut = 0, gemvOut = 0;
                try
                {
                    // Each status is checked where it is produced: these all write to the
                    // same `err` slot, so a single check at the end would let the last
                    // call overwrite an earlier failure and the leg would proceed on a
                    // half-created set of buffers.
                    dot16In = Upload(cl, context, queue, dot16Input, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] dot16 input upload: {err}"); return null; }
                    siluIn = Upload(cl, context, queue, siluInput, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] silu input upload: {err}"); return null; }
                    gemvIn = Upload(cl, context, queue, gemvInput, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] gemv input upload: {err}"); return null; }

                    dot16Out = Output(cl, context, 1, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] dot16 output buffer: {err}"); return null; }
                    siluOut = Output(cl, context, KernelFixtures.HiddenSize, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] silu output buffer: {err}"); return null; }
                    gemvOut = Output(cl, context, KernelFixtures.IntermediateSize, out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] gemv output buffer: {err}"); return null; }

                    nint dot16Kernel = CreateKernel(cl, program, "dot16", out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] clCreateKernel(dot16): {err}"); return null; }
                    nint siluKernel = CreateKernel(cl, program, "silu", out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] clCreateKernel(silu): {err}"); return null; }
                    nint gemvKernel = CreateKernel(cl, program, "gemv", out err);
                    if (err != 0) { Console.WriteLine($"  [FAIL] clCreateKernel(gemv): {err}"); return null; }

                    try
                    {
                        var dot16 = Measure(stopwatch, "dot16",
                            () => Dispatch(cl, queue, dot16Kernel, dot16In, dot16Out, 1u, 1, [dot16Length]),
                            () => Download(cl, queue, dot16Out, 1, "dot16"), out double dot16FirstUs, out double dot16Us);
                        if (dot16 is null) return null;
                        Console.WriteLine($"  [dot16] first {dot16FirstUs,8:F0} µs | steady {dot16Us,8:F1} µs");

                        var silu = Measure(stopwatch, "silu",
                            () => Dispatch(cl, queue, siluKernel, siluIn, siluOut, hidden, local, [hidden]),
                            () => Download(cl, queue, siluOut, KernelFixtures.HiddenSize, "silu"), out double siluFirstUs, out double siluUs);
                        if (silu is null) return null;
                        Console.WriteLine($"  [silu]  first {siluFirstUs,8:F0} µs | steady {siluUs,8:F1} µs");

                        var gemv = Measure(stopwatch, "gemv",
                            () => Dispatch(cl, queue, gemvKernel, gemvIn, gemvOut, rows, local, [rows, cols]),
                            () => Download(cl, queue, gemvOut, KernelFixtures.IntermediateSize, "gemv"), out double gemvFirstUs, out double gemvUs);
                        if (gemv is null) return null;
                        Console.WriteLine($"  [gemv]  first {gemvFirstUs,8:F0} µs | steady {gemvUs,8:F1} µs");

                        return new LegResults(dot16[0], silu, gemv, dot16Us, siluUs, gemvUs);
                    }
                    finally
                    {
                        if (dot16Kernel != 0) cl.ReleaseKernel(dot16Kernel);
                        if (siluKernel != 0) cl.ReleaseKernel(siluKernel);
                        if (gemvKernel != 0) cl.ReleaseKernel(gemvKernel);
                    }
                }
                finally
                {
                    // Handles can legitimately be 0 here: a failed allocation or a failed
                    // clCreateKernel returns before all of them were created.
                    if (dot16In != 0) cl.ReleaseMemObject(dot16In);
                    if (siluIn != 0) cl.ReleaseMemObject(siluIn);
                    if (gemvIn != 0) cl.ReleaseMemObject(gemvIn);
                    if (dot16Out != 0) cl.ReleaseMemObject(dot16Out);
                    if (siluOut != 0) cl.ReleaseMemObject(siluOut);
                    if (gemvOut != 0) cl.ReleaseMemObject(gemvOut);
                }
            }
            finally
            {
                if (program != 0) cl.ReleaseProgram(program);
                cl.ReleaseCommandQueue(queue);
                cl.ReleaseContext(context);
            }
        }
    }

    /// <summary>
    /// The three production SmolLM kernels in one OpenCL C translation unit. Element
    /// layout mirrors <see cref="GemvKernels"/> exactly: dot16 reads
    /// <c>[a(16) | b(16)]</c>, silu reads <c>[x(n)]</c>, gemv reads <c>[w(rows*cols) | x(cols)]</c>.
    /// Every kernel guards <c>id &gt;= n</c> because the global size is rounded up to a
    /// whole number of work-groups (OpenCL requires global_work_size to be a multiple of
    /// local_work_size). Shapes are runtime kernel arguments, not baked into the
    /// source, so the same translation unit serves any fixture size.</summary>
    private static string KernelSource()
        => """
__kernel void dot16(__global const uint* gIn, __global float* gOut, const uint k)
{
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i)
        acc += Widen(gIn[i >> 1], i) * Widen(gIn[(k + i) >> 1], k + i);
    gOut[0] = acc;
}

__kernel void silu(__global const uint* gIn, __global float* gOut, const uint n)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    float x = Widen(gIn[i >> 1], i);
    gOut[i] = x / (1.0f + exp(-x));
}

__kernel void gemv(__global const uint* gIn, __global float* gOut, const uint rows, const uint cols)
{
    uint row = get_global_id(0);
    if (row >= rows) return;
    float acc = 0.0f;
    uint weightElems = rows * cols;
    for (uint k = 0; k < cols; ++k)
    {
        uint w = row * cols + k;
        acc += Widen(gIn[w >> 1], w) * Widen(gIn[(weightElems + k) >> 1], weightElems + k);
    }
    gOut[row] = acc;
}
""";

    /// <summary>BF16 → f32 widen for the packed-2-per-uint transport: element 2k lives in
    /// the HIGH half of the uint, element 2k+1 in the LOW half, so a widening change can
    /// never be mistaken for a kernel difference.</summary>
    private const string Widen = "float Widen(uint packed, uint element)\n" +
        "{\n" +
        "    return (element & 1u) == 0u ? as_float(packed & 0xFFFF0000u) : as_float((packed & 0xFFFFu) << 16);\n" +
        "}\n\n";

    /// <summary>Compiles the OpenCL C source in-process; prints the driver build log on
    /// failure (the single most useful diagnostic for Intel/AMD driver problems).</summary>
    private static unsafe nint BuildProgram(CL cl, nint context, nint device, string body)
    {
        string source = Widen + body;
        int err = 0;
        nint program = cl.CreateProgramWithSource(context, 1, [source], null, out err);
        if (err != 0 || program == 0)
        {
            Console.WriteLine($"  [FAIL] clCreateProgramWithSource: {err}");
            return 0;
        }

        // num_devices = 0: build for every device in the context (the spec-defined
        // "all devices" form), which is exactly the one device we asked for.
        if (cl.BuildProgram(program, 0, null, (string?)null, null, null) == 0)
            return program;

        Console.WriteLine("  [FAIL] clBuildProgram:");
        Console.WriteLine(BuildLog(cl, program, device));
        cl.ReleaseProgram(program);
        return 0;
    }

    private static unsafe string BuildLog(CL cl, nint program, nint device)
    {
        nuint size = 0;
        if (cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0, null, &size) != 0 || size == 0)
            return "    (no build log available)";
        byte[] buffer = new byte[(int)size];
        fixed (byte* p = buffer)
            cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, (nuint)buffer.Length, p, null);
        int end = Array.IndexOf(buffer, (byte)0);
        return "    " + Encoding.UTF8.GetString(buffer, 0, end < 0 ? buffer.Length : end).Trim();
    }

    /// <summary>First dispatch timed, one warmup, then best-of-N steady state. A non-zero
    /// status from any dispatch aborts the measurement and yields <c>null</c> — the leg
    /// reports UNBUILT rather than gating a value the driver never produced.</summary>
    private static float[]? Measure(Stopwatch sw, string name, Func<int> dispatch,
        Func<float[]?> read, out double firstUs, out double bestUs)
    {
        bestUs = 0;
        sw.Restart();
        int err = dispatch();
        firstUs = sw.Elapsed.TotalMicroseconds;
        if (err != 0)
        {
            Console.WriteLine($"  [FAIL] {name} dispatch: {err}");
            return null;
        }

        err = dispatch(); // warmup — steady-state scheduling
        if (err != 0)
        {
            Console.WriteLine($"  [FAIL] {name} warmup: {err}");
            return null;
        }

        bestUs = double.PositiveInfinity;
        for (int i = 0; i < TimingPasses; i++)
        {
            sw.Restart();
            err = dispatch();
            sw.Stop();
            if (err != 0)
            {
                Console.WriteLine($"  [FAIL] {name} timed pass {i}: {err}");
                return null;
            }
            bestUs = Math.Min(bestUs, sw.Elapsed.TotalMicroseconds);
        }
        return read();
    }

    private static unsafe nint CreateKernel(CL cl, nint program, string name, out int err)
        => cl.CreateKernel(program, name, out err);

    /// <summary>Uploads one packed-uint buffer and returns the mem object. The handle is
    /// returned even when the write fails, so the caller's finally still releases it.</summary>
    private static unsafe nint Upload(CL cl, nint context, nint queue, uint[] data, out int err)
    {
        nint buffer = cl.CreateBuffer(context, MemFlags.ReadOnly, (nuint)(data.Length * sizeof(uint)), null, out err);
        if (err != 0) return 0;
        fixed (uint* p = data)
            err = cl.EnqueueWriteBuffer(queue, buffer, true, 0, (nuint)(data.Length * sizeof(uint)), p, 0, null, null);
        return buffer;
    }

    private static unsafe nint Output(CL cl, nint context, int count, out int err)
        => cl.CreateBuffer(context, MemFlags.ReadWrite, (nuint)(count * sizeof(float)), null, out err);

    /// <summary>Blocking readback. Returns <c>null</c> on failure rather than throwing:
    /// <see cref="Kernels.KernelGate"/> does not guard its legs, so an escaping exception
    /// would abort the whole multi-leg run instead of marking this leg UNBUILT.</summary>
    private static unsafe float[]? Download(CL cl, nint queue, nint buffer, int count, string name)
    {
        var values = new float[count];
        int err;
        fixed (float* p = values)
            err = cl.EnqueueReadBuffer(queue, buffer, true, 0, (nuint)(count * sizeof(float)), p, 0, null, null);
        if (err != 0)
        {
            Console.WriteLine($"  [FAIL] {name} readback: {err}");
            return null;
        }
        return values;
    }

    /// <summary>Sets the packed-input/output buffer arguments (0, 1), then the scalar
    /// arguments (2..) in order, then enqueues a synchronized 1-D NDRange and waits.
    /// Every OpenCL status is checked: a mismatched argument count fails
    /// <c>clSetKernelArg</c>, and letting that pass would silently measure a kernel
    /// reading uninitialized scalars — a wrong answer reported as a fast one.</summary>
    private static unsafe int Dispatch(CL cl, nint queue, nint kernel,
        nint input, nint output, uint globalCount, int localSize, uint[] scalars)
    {
        int err = cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), &input);
        err |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint), &output);
        for (int i = 0; i < scalars.Length; i++)
        {
            uint value = scalars[i];
            err |= cl.SetKernelArg(kernel, (uint)i + 2, sizeof(uint), &value);
        }
        if (err != 0)
            return err;

        nuint global = ((globalCount + (uint)localSize - 1) / (uint)localSize) * (uint)localSize;
        nuint offset = 0;
        nuint local = (nuint)localSize;
        err = cl.EnqueueNdrangeKernel(queue, kernel, 1, &offset, &global, &local, 0, null, null);
        if (err != 0)
            return err;
        return cl.Finish(queue);
    }

    /// <summary>Picks an Intel or AMD OpenCL GPU, else the first GPU on any platform.
    /// Every platform and GPU device is logged as it is considered (marking the one
    /// selected) — this is the leg's discovery output, so a machine with no Intel/AMD
    /// iGPU still shows what OpenCL actually exposed.</summary>
    private static unsafe nint SelectGpu(CL cl, out string name, out string vendor, out ulong maxWorkGroupSize)
    {
        name = vendor = "";
        maxWorkGroupSize = 0;
        uint platformCount = 0;
        if (cl.GetPlatformIDs(0, null, &platformCount) != 0 || platformCount == 0)
        {
            Console.WriteLine("  [FAIL] no OpenCL platforms");
            return 0;
        }

        var platforms = new nint[platformCount];
        fixed (nint* pp = platforms)
            if (cl.GetPlatformIDs(platformCount, pp, null) != 0)
                return 0;

        nint fallback = 0;
        string? owner = null; // platform that supplied the current `name`/`fallback`
        foreach (nint platform in platforms)
        {
            string platformName = InfoString(cl, 0, PlatformInfo.Name, platform);
            uint deviceCount = 0;
            if (cl.GetDeviceIDs(platform, DeviceType.Gpu, 0, null, &deviceCount) != 0 || deviceCount == 0)
            {
                Console.WriteLine($"  platform: {platformName} | no GPU devices");
                continue;
            }

            var devices = new nint[deviceCount];
            fixed (nint* dp = devices)
                if (cl.GetDeviceIDs(platform, DeviceType.Gpu, deviceCount, dp, null) != 0)
                    continue;

            bool platformPreferred = false;
            foreach (nint device in devices)
            {
                string deviceVendor = InfoString(cl, device, DeviceInfo.Vendor);
                bool preferred = deviceVendor.Contains("Intel", StringComparison.OrdinalIgnoreCase)
                              || deviceVendor.Contains("AMD", StringComparison.OrdinalIgnoreCase);
                if (preferred || fallback == 0)
                {
                    fallback = device;
                    owner = platformName;
                    name = InfoString(cl, device, DeviceInfo.Name);
                    vendor = deviceVendor;
                    ulong wg = 0;
                    cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, sizeof(ulong), &wg, null);
                    maxWorkGroupSize = wg;
                }
                platformPreferred |= preferred;
                if (preferred)
                    break;
            }

            // Mark the pick only on the platform that actually owns it — a later
            // non-preferred platform must not inherit the earlier platform's name.
            string label = platformPreferred || owner == platformName ? $"* {name}" : "(not selected)";
            Console.WriteLine($"  platform: {platformName} | GPU devices {deviceCount} | best: {label}");

            if (platformPreferred)
                return fallback;
        }
        return fallback;
    }

    /// <summary>Reports <c>CL_DEVICE_EXTENSIONS</c>, marking BF16 entries with <c>*</c>.
    /// This leg's kernels take BF16 as a wire format and widen to f32 in-shader (see
    /// <see cref="Widen"/>), so whether the driver offers a native BF16 type decides whether
    /// that widening is avoidable. The full list is printed because a BF16 extension can be
    /// spelled either way — this device ships <c>cl_intel_bfloat16_conversions</c>, which a
    /// substring search for "bf16" alone misses entirely and reports as absent.</summary>
    private static unsafe void PrintExtensions(CL cl, nint device)
    {
        string extensions = InfoString(cl, device, DeviceInfo.Extensions);
        var list = extensions.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (list.Length == 0)
        {
            Console.WriteLine("  extensions: (none reported)");
            return;
        }

        int bf16Count = 0;
        foreach (string e in list)
            if (IsBf16Extension(e))
                bf16Count++;

        Console.WriteLine($"  extensions: {list.Length} advertised, {bf16Count} BF16 (* marks them)");
        // Full list, wrapped: a native BF16 could in principle be advertised under a token
        // spelling neither "bf16" nor "bfloat", so counting matches alone cannot rule one out.
        const int Width = 96;
        string line = "   ";
        foreach (string e in list)
        {
            string token = IsBf16Extension(e) ? "*" + e : e;
            if (line.Length + token.Length + 1 > Width)
            {
                Console.WriteLine(line);
                line = "   ";
            }
            line += token + " ";
        }
        if (line.Length > 3)
            Console.WriteLine(line);
    }

    /// <summary>Matches both spellings: Intel ships <c>cl_intel_bfloat16_conversions</c>, so
    /// a "bf16"-only match reports this device as having no BF16 support when it does.</summary>
    private static bool IsBf16Extension(string extension)
        => extension.Contains("bf16", StringComparison.OrdinalIgnoreCase)
        || extension.Contains("bfloat", StringComparison.OrdinalIgnoreCase);

    private static unsafe bool IsGpu(CL cl, nint device)
    {
        DeviceType type = DeviceType.None;
        cl.GetDeviceInfo(device, DeviceInfo.Type, sizeof(DeviceType), &type, null);
        return (type & DeviceType.Gpu) != 0;
    }

    /// <summary>Two-call OpenCL info pattern: query the byte size, allocate, query into it.
    /// The returned strings are raw NUL-terminated bytes — trim at the first NUL rather
    /// than assuming <c>size - 1</c>, since some drivers omit the terminator.</summary>
    private static unsafe string InfoString(CL cl, nint device, DeviceInfo info)
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

    private static unsafe string InfoString(CL cl, int _, PlatformInfo info, nint platform)
    {
        nuint size = 0;
        if (cl.GetPlatformInfo(platform, info, 0, null, &size) != 0 || size == 0)
            return "";
        byte[] buffer = new byte[(int)size];
        fixed (byte* p = buffer)
            cl.GetPlatformInfo(platform, info, (nuint)buffer.Length, p, null);
        int end = Array.IndexOf(buffer, (byte)0);
        return Encoding.UTF8.GetString(buffer, 0, end < 0 ? buffer.Length : end);
    }

    private static unsafe string DeviceVersion(CL cl, nint device) => InfoString(cl, device, DeviceInfo.Version);

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