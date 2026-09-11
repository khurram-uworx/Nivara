using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// OpenCL diagnostic probe. The Intel graphics driver ships the OpenCL ICD in the
/// driver store (igdrcl64.dll) even though it is not registered as an ICD on this
/// machine. We load it directly to (a) run a real OpenCL C kernel on the iGPU
/// (ground-truth compute proof + timing on the same IGC stack) and (b) feed the
/// hand-authored SPIR-V through clCreateProgramWithIL, whose build log states the
/// exact reason the Level Zero driver rejected the modules.
/// </summary>
internal static class OclProbe
{
    private const uint CL_DEVICE_TYPE_GPU = 1u << 2;
    private const ulong CL_MEM_READ_WRITE = 1u << 1;
    private const uint CL_PROGRAM_BUILD_STATUS = 0x1181;
    private const uint CL_PROGRAM_BUILD_LOG = 0x1183;

    public delegate int ClGetPlatformIDs(uint numEntries, IntPtr platforms, out uint numPlatforms);
    public delegate int ClGetDeviceIDs(IntPtr platform, ulong deviceType, uint numEntries, IntPtr devices, out uint numDevices);
    public delegate IntPtr ClCreateContext(IntPtr properties, uint numDevices, IntPtr devices, IntPtr pfnNotify, IntPtr userData, out int errcodeRet);
    public delegate IntPtr ClCreateCommandQueue(IntPtr context, IntPtr device, ulong properties, out int errcodeRet);
    public delegate IntPtr ClCreateBuffer(IntPtr context, ulong flags, nuint size, IntPtr hostPtr, out int errcodeRet);
    public delegate int ClEnqueueWriteBuffer(IntPtr queue, IntPtr buffer, uint blockingWrite, nuint offset, nuint size, IntPtr ptr, uint numEventsInWaitList, IntPtr eventWaitList, out IntPtr evt);
    public delegate int ClEnqueueReadBuffer(IntPtr queue, IntPtr buffer, uint blockingRead, nuint offset, nuint size, IntPtr ptr, uint numEventsInWaitList, IntPtr eventWaitList, out IntPtr evt);
    public delegate IntPtr ClCreateProgramWithSource(IntPtr context, uint count, IntPtr strings, IntPtr lengths, out int errcodeRet);
    public delegate IntPtr ClCreateProgramWithIL(IntPtr context, IntPtr il, nuint length, out int errcodeRet);
    public delegate int ClBuildProgram(IntPtr program, uint numDevices, IntPtr deviceList, [MarshalAs(UnmanagedType.LPUTF8Str)] string? options, IntPtr pfnNotify, IntPtr userData);
    public delegate int ClGetProgramBuildInfo(IntPtr program, IntPtr device, uint paramName, nuint paramValueSize, IntPtr paramValue, out nuint paramValueSizeRet);
    public delegate IntPtr ClCreateKernel(IntPtr program, [MarshalAs(UnmanagedType.LPUTF8Str)] string kernelName, out int errcodeRet);
    public delegate int ClSetKernelArg(IntPtr kernel, uint argIndex, nuint argSize, IntPtr argValue);
    public delegate int ClEnqueueNDRangeKernel(IntPtr queue, IntPtr kernel, uint workDim, IntPtr globalWorkOffset, IntPtr globalWorkSize, IntPtr localWorkSize, uint numEventsInWaitList, IntPtr eventWaitList, out IntPtr evt);
    public delegate int ClFinish(IntPtr queue);
    public delegate int ClReleaseMemObject(IntPtr mem);
    public delegate int ClReleaseKernel(IntPtr kernel);
    public delegate int ClReleaseProgram(IntPtr program);
    public delegate IntPtr ClGetExtensionFunctionAddressForPlatform(IntPtr platform, [MarshalAs(UnmanagedType.LPUTF8Str)] string funcName);

    private static int failures;

    public static int Run()
    {
        Console.WriteLine("--- OpenCL diagnostic (ocl) ---");
        try
        {
            string? path = FindIgdrcl();
            if (path == null)
            {
                Console.WriteLine("  [FAIL] igdrcl64.dll not found in the driver store.");
                failures++;
                return failures;
            }

            using var icd = new OclLoader(path);
            Console.WriteLine($"  loaded OpenCL ICD (driver store): {Path.GetFileName(Path.GetDirectoryName(path))}");

            var clGetPlatformIDs = icd.Api<ClGetPlatformIDs>("clGetPlatformIDs");
            uint platformCount = 0;
            Check(clGetPlatformIDs(0, IntPtr.Zero, out platformCount), "clGetPlatformIDs (count)");
            if (platformCount == 0)
            {
                Console.WriteLine("  [FAIL] no OpenCL platforms.");
                failures++;
                return failures;
            }

            IntPtr platformSlot = Marshal.AllocHGlobal(IntPtr.Size);
            IntPtr platform;
            try
            {
                Check(clGetPlatformIDs(1, platformSlot, out _), "clGetPlatformIDs (list)");
                platform = Marshal.ReadIntPtr(platformSlot);
            }
            finally
            {
                Marshal.FreeHGlobal(platformSlot);
            }

            if (platform == IntPtr.Zero)
            {
                Console.WriteLine("  [FAIL] clGetPlatformIDs returned no handle.");
                failures++;
                return failures;
            }

            var clGetDeviceIDs = icd.Api<ClGetDeviceIDs>("clGetDeviceIDs");
            uint deviceCount = 0;
            Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, IntPtr.Zero, out deviceCount), "clGetDeviceIDs");
            if (deviceCount == 0)
            {
                Console.WriteLine("  [FAIL] no OpenCL GPU device.");
                failures++;
                return failures;
            }

            IntPtr deviceSlot = Marshal.AllocHGlobal((int)deviceCount * IntPtr.Size);
            IntPtr device;
            try
            {
                Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, deviceSlot, out _), "clGetDeviceIDs(list)");
                device = Marshal.ReadIntPtr(deviceSlot);
            }
            finally
            {
                Marshal.FreeHGlobal(deviceSlot);
            }

            var clCreateContext = icd.Api<ClCreateContext>("clCreateContext");
            IntPtr deviceSlot2 = Marshal.AllocHGlobal(IntPtr.Size);
            IntPtr context;
            try
            {
                Marshal.WriteIntPtr(deviceSlot2, device);
                context = clCreateContext(IntPtr.Zero, 1, deviceSlot2, IntPtr.Zero, IntPtr.Zero, out int ctxErr);
                Check(ctxErr, "clCreateContext");
            }
            finally
            {
                Marshal.FreeHGlobal(deviceSlot2);
            }

            if (context == IntPtr.Zero)
            {
                Console.WriteLine("  [FAIL] clCreateContext failed.");
                failures++;
                return failures;
            }

            var clCreateCommandQueue = icd.Api<ClCreateCommandQueue>("clCreateCommandQueue");
            IntPtr queue = clCreateCommandQueue(context, device, 0, out int cqErr);
            Check(cqErr, "clCreateCommandQueue");
            if (queue == IntPtr.Zero)
            {
                Console.WriteLine("  [FAIL] clCreateCommandQueue failed.");
                failures++;
                return failures;
            }

            try
            {
                Console.WriteLine();
                Console.WriteLine("  [test 3] OpenCL C kernel on the iGPU: 1M-element parallel add");
                RunSourceKernel(icd, context, device, queue);

                Console.WriteLine();
                Console.WriteLine("  [test 4] SPIR-V via clCreateProgramWithIL (diagnostic): add_parallel @ SPIR-V 1.0");
                RunIlDiagnostic(icd, context, device, SpvKernels.AddParallel(SpvKernels.Version10));

                Console.WriteLine();
                Console.WriteLine("  [test 5] SPIR-V via clCreateProgramWithIL (diagnostic): add_loop @ SPIR-V 1.0");
                RunIlDiagnostic(icd, context, device, SpvKernels.AddLoop(SpvKernels.Version10));
            }
            finally
            {
                icd.Api<ClReleaseMemObject>("clReleaseMemObject")(queue);
                icd.Api<ClReleaseMemObject>("clReleaseMemObject")(context);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [FAIL] OpenCL probe crashed: {ex.Message}");
            failures++;
        }

        Console.WriteLine();
        Console.WriteLine($"--- OpenCL diagnostic done: {failures} failure(s) ---");
        return failures;
    }

    private static void RunSourceKernel(OclLoader icd, IntPtr context, IntPtr device, IntPtr queue)
    {
        const int n = 1_000_000;

        IntPtr program = IntPtr.Zero, kernel = IntPtr.Zero;
        IntPtr aBuf = IntPtr.Zero, bBuf = IntPtr.Zero, cBuf = IntPtr.Zero, host = IntPtr.Zero;
        try
        {
            const string source = """
                __kernel void add_parallel(__global const float* a, __global const float* b, __global float* c) {
                    int i = get_global_id(0);
                    c[i] = a[i] + b[i];
                }
                """;

            IntPtr srcPtr = Marshal.StringToCoTaskMemUTF8(source);
            IntPtr srcSlot = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                Marshal.WriteIntPtr(srcSlot, srcPtr);
                program = icd.Api<ClCreateProgramWithSource>("clCreateProgramWithSource")(
                    context, 1, srcSlot, IntPtr.Zero, out int srcErr);
                Check(srcErr, "clCreateProgramWithSource");
            }
            finally
            {
                Marshal.FreeHGlobal(srcSlot);
                Marshal.FreeCoTaskMem(srcPtr);
            }

            if (program == IntPtr.Zero)
                return;

            var clBuildProgram = icd.Api<ClBuildProgram>("clBuildProgram");
            int buildErr = clBuildProgram(program, 1, device, null, IntPtr.Zero, IntPtr.Zero);
            if (buildErr != 0)
            {
                Console.WriteLine($"    [FAIL] clBuildProgram: {ClError(buildErr)}");
                Console.WriteLine("    build log:\n" + Indent(BuildLog(icd, program, device)));
                failures++;
                return;
            }

            var clCreateKernel = icd.Api<ClCreateKernel>("clCreateKernel");
            kernel = clCreateKernel(program, "add_parallel", out int kernErr);
            Check(kernErr, "clCreateKernel");
            if (kernel == IntPtr.Zero)
                return;

            var clCreateBuffer = icd.Api<ClCreateBuffer>("clCreateBuffer");
            nuint bytes = (nuint)n * 4;
            aBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, IntPtr.Zero, out int aErr);
            bBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, IntPtr.Zero, out int bErr);
            cBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, IntPtr.Zero, out int cErr);
            if (!Check(aErr, "clCreateBuffer a") || !Check(bErr, "clCreateBuffer b") || !Check(cErr, "clCreateBuffer c"))
                return;

            if (!FillAndWriteBuffers(icd, context, queue, aBuf, bBuf, n))
                return;

            var clSetKernelArg = icd.Api<ClSetKernelArg>("clSetKernelArg");
            IntPtr argSlot = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                var args = new[] { aBuf, bBuf, cBuf };
                for (int i = 0; i < args.Length; i++)
                {
                    Marshal.WriteIntPtr(argSlot, args[i]);
                    if (!Check(clSetKernelArg(kernel, (uint)i, (nuint)IntPtr.Size, argSlot), $"clSetKernelArg[{i}]"))
                        return;
                }
            }
            finally
            {
                Marshal.FreeHGlobal(argSlot);
            }

            var clEnqueueNDRangeKernel = icd.Api<ClEnqueueNDRangeKernel>("clEnqueueNDRangeKernel");
            var clFinish = icd.Api<ClFinish>("clFinish");

            IntPtr globalPtr = Marshal.AllocHGlobal(UIntPtr.Size);
            IntPtr localPtr = Marshal.AllocHGlobal(UIntPtr.Size);
            try
            {
                Marshal.WriteIntPtr(globalPtr, new IntPtr((long)n));
                Marshal.WriteIntPtr(localPtr, new IntPtr(256));

                int err = clEnqueueNDRangeKernel(queue, kernel, 1, IntPtr.Zero, globalPtr, localPtr, 0, IntPtr.Zero, out _);
                if (!Check(err, "clEnqueueNDRangeKernel"))
                    return;
                Check(clFinish(queue), "clFinish");

                // Verify.
                var clEnqueueReadBuffer = icd.Api<ClEnqueueReadBuffer>("clEnqueueReadBuffer");
                float[] c = new float[n];
                host = Marshal.AllocHGlobal((int)bytes);
                Check(clEnqueueReadBuffer(queue, cBuf, 1, 0, bytes, host, 0, IntPtr.Zero, out _), "clEnqueueReadBuffer");
                Check(clFinish(queue), "clFinish");
                Marshal.Copy(host, c, 0, n);

                float[] a = new float[n];
                float[] b = new float[n];
                for (int i = 0; i < n; i++)
                {
                    a[i] = i * 0.25f;
                    b[i] = i * 0.125f;
                }

                int mismatches = 0;
                for (int i = 0; i < n; i++)
                {
                    if (c[i] != a[i] + b[i])
                        mismatches++;
                }

                if (mismatches == 0)
                    Console.WriteLine($"    result: PASS ({n} elements verified)");
                else
                {
                    Console.WriteLine($"    result: FAIL ({mismatches}/{n} mismatches)");
                    failures++;
                }

                // Timing: re-enqueue + finish.
                var times = new long[8];
                int runs = 0;
                for (int i = 0; i < times.Length; i++)
                {
                    var sw = Stopwatch.StartNew();
                    int e = clEnqueueNDRangeKernel(queue, kernel, 1, IntPtr.Zero, globalPtr, localPtr, 0, IntPtr.Zero, out _);
                    if (e == 0) e = clFinish(queue);
                    sw.Stop();
                    if (e != 0)
                    {
                        Check(e, "timing run");
                        break;
                    }

                    times[runs++] = sw.Elapsed.Ticks * 100; // ns
                }

                var valid = times.Take(runs).Where(t => t > 0).OrderBy(t => t).ToArray();
                if (valid.Length > 0)
                {
                    long minNs = valid[0];
                    double medianNs = valid.Length % 2 == 0
                        ? (valid[valid.Length / 2 - 1] + valid[valid.Length / 2]) / 2.0
                        : valid[valid.Length / 2];
                    double addPerSec = n / (minNs / 1e9);
                    Console.WriteLine($"    timing ({runs} launches): min {minNs / 1000.0:F1} µs, median {medianNs / 1000.0:F1} µs" +
                                      $" -> {addPerSec:N0} FADD/s ({n} elems, 256 lwi)");
                }
            }
            finally
            {
                Marshal.FreeHGlobal(globalPtr);
                Marshal.FreeHGlobal(localPtr);
            }
        }
        finally
        {
            if (host != IntPtr.Zero)
                Marshal.FreeHGlobal(host);
            var release = icd.Api<ClReleaseMemObject>("clReleaseMemObject");
            if (cBuf != IntPtr.Zero) release(cBuf);
            if (bBuf != IntPtr.Zero) release(bBuf);
            if (aBuf != IntPtr.Zero) release(aBuf);
            if (kernel != IntPtr.Zero) icd.Api<ClReleaseKernel>("clReleaseKernel")(kernel);
            if (program != IntPtr.Zero) icd.Api<ClReleaseProgram>("clReleaseProgram")(program);
        }
    }

    private static bool FillAndWriteBuffers(OclLoader icd, IntPtr context, IntPtr queue, IntPtr aBuf, IntPtr bBuf, int n)
    {
        nuint bytes = (nuint)n * 4;
        IntPtr host = Marshal.AllocHGlobal((int)bytes);
        try
        {
            float[] a = new float[n];
            float[] b = new float[n];
            for (int i = 0; i < n; i++)
            {
                a[i] = i * 0.25f;
                b[i] = i * 0.125f;
            }

            var clEnqueueWriteBuffer = icd.Api<ClEnqueueWriteBuffer>("clEnqueueWriteBuffer");
            var clFinish = icd.Api<ClFinish>("clFinish");
            bool ok = true;
            Marshal.Copy(a, 0, host, n);
            ok &= Check(clEnqueueWriteBuffer(queue, aBuf, 1, 0, bytes, host, 0, IntPtr.Zero, out _), "clEnqueueWriteBuffer a");
            Marshal.Copy(b, 0, host, n);
            ok &= Check(clEnqueueWriteBuffer(queue, bBuf, 1, 0, bytes, host, 0, IntPtr.Zero, out _), "clEnqueueWriteBuffer b");
            ok &= Check(clFinish(queue), "clFinish");
            return ok;
        }
        finally
        {
            Marshal.FreeHGlobal(host);
        }
    }

    private static void RunIlDiagnostic(OclLoader icd, IntPtr context, IntPtr device, uint[] spirv)
    {
        IntPtr program = IntPtr.Zero;
        try
        {
            byte[] il = SpvKernels.ToBytes(spirv);
            IntPtr ilPtr = Marshal.AllocHGlobal(il.Length);
            try
            {
                Marshal.Copy(il, 0, ilPtr, il.Length);
                program = icd.Api<ClCreateProgramWithIL>("clCreateProgramWithIL")(
                    context, ilPtr, (nuint)il.Length, out int ilErr);
                Check(ilErr, "clCreateProgramWithIL");
            }
            finally
            {
                Marshal.FreeHGlobal(ilPtr);
            }

            if (program == IntPtr.Zero)
                return;

            int buildErr = icd.Api<ClBuildProgram>("clBuildProgram")(program, 1, device, null, IntPtr.Zero, IntPtr.Zero);
            if (buildErr == 0)
                Console.WriteLine("    clBuildProgram: OK (module compiled)");
            else
            {
                Console.WriteLine($"    clBuildProgram: {ClError(buildErr)}");
                Console.WriteLine("    build log:\n" + Indent(BuildLog(icd, program, device)));
                failures++;
            }
        }
        finally
        {
            if (program != IntPtr.Zero)
                icd.Api<ClReleaseProgram>("clReleaseProgram")(program);
        }
    }

    private static string BuildLog(OclLoader icd, IntPtr program, IntPtr device)
    {
        var info = icd.Api<ClGetProgramBuildInfo>("clGetProgramBuildInfo");
        nuint size = 0;
        info(program, device, CL_PROGRAM_BUILD_LOG, 0, IntPtr.Zero, out size);
        if (size == 0)
            return "(empty log)";

        IntPtr buf = Marshal.AllocHGlobal((int)size + 4);
        try
        {
            info(program, device, CL_PROGRAM_BUILD_LOG, (nuint)((int)size + 4), buf, out _);
            byte[] bytes = new byte[(int)size];
            Marshal.Copy(buf, bytes, 0, bytes.Length);
            return Encoding.UTF8.GetString(bytes).TrimEnd('\0').Trim();
        }
        finally
        {
            Marshal.FreeHGlobal(buf);
        }
    }

    private static string Indent(string text) => string.Join("\n", text.Split('\n').Select(line => "      " + line));

    private static bool Check(int result, string what)
    {
        if (result != 0)
        {
            Console.WriteLine($"  [FAIL] {what}: {ClError(result)}");
            failures++;
            return false;
        }

        return true;
    }

    private static string ClError(int e) => e switch
    {
        0 => "CL_SUCCESS",
        -1 => "CL_DEVICE_NOT_FOUND",
        -2 => "CL_DEVICE_NOT_AVAILABLE",
        -3 => "CL_COMPILER_NOT_AVAILABLE",
        -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        -5 => "CL_OUT_OF_RESOURCES",
        -6 => "CL_OUT_OF_HOST_MEMORY",
        -11 => "CL_BUILD_PROGRAM_FAILURE",
        -12 => "CL_MAP_FAILURE",
        -15 => "CL_COMPILE_PROGRAM_FAILURE",
        -16 => "CL_LINKER_NOT_AVAILABLE",
        -17 => "CL_LINK_PROGRAM_FAILURE",
        -30 => "CL_INVALID_VALUE",
        -31 => "CL_INVALID_DEVICE_TYPE",
        -32 => "CL_INVALID_PLATFORM",
        -33 => "CL_INVALID_DEVICE",
        -34 => "CL_INVALID_CONTEXT",
        -36 => "CL_INVALID_COMMAND_QUEUE",
        -38 => "CL_INVALID_MEM_OBJECT",
        -42 => "CL_INVALID_BINARY",
        -43 => "CL_INVALID_BUILD_OPTIONS",
        -44 => "CL_INVALID_PROGRAM",
        -45 => "CL_INVALID_PROGRAM_EXECUTABLE",
        -46 => "CL_INVALID_KERNEL_NAME",
        -47 => "CL_INVALID_KERNEL_DEFINITION",
        -48 => "CL_INVALID_KERNEL",
        -49 => "CL_INVALID_ARG_INDEX",
        -50 => "CL_INVALID_ARG_VALUE",
        -51 => "CL_INVALID_ARG_SIZE",
        -52 => "CL_INVALID_KERNEL_ARGS",
        -53 => "CL_INVALID_WORK_DIMENSION",
        -54 => "CL_INVALID_WORK_GROUP_SIZE",
        -55 => "CL_INVALID_WORK_ITEM_SIZE",
        -56 => "CL_INVALID_GLOBAL_OFFSET",
        -59 => "CL_INVALID_OPERATION",
        _ => $"CL error {e}"
    };

    private static string? FindIgdrcl()
    {
        string store = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "System32", "DriverStore", "FileRepository");
        var dirs = Directory.GetDirectories(store, "iigd_dch.inf_amd64_*")
            .OrderByDescending(d => Directory.GetLastWriteTime(d));
        foreach (string dir in dirs)
        {
            string candidate = Path.Combine(dir, "igdrcl64.dll");
            if (File.Exists(candidate))
                return candidate;
        }

        return null;
    }

    private sealed class OclLoader : IDisposable
    {
        private readonly IntPtr handle;

        public OclLoader(string path)
        {
            handle = LoadLibraryEx(path, IntPtr.Zero, 0);
            if (handle == IntPtr.Zero)
                throw new InvalidOperationException($"igdrcl64.dll failed to load (Win32 error 0x{Marshal.GetLastWin32Error():X8})");
        }

        public T Api<T>(string name) where T : Delegate
        {
            IntPtr fn = GetProcAddress(handle, name);
            if (fn == IntPtr.Zero)
                throw new EntryPointNotFoundException($"{name} is not exported by igdrcl64.dll");
            return Marshal.GetDelegateForFunctionPointer<T>(fn);
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
                FreeLibrary(handle);
        }
    }

    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
}