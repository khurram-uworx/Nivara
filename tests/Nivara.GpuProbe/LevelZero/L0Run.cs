using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// Real Level Zero execution probe: creates a context + compute queue, allocates
/// shared memory, compiles hand-authored SPIR-V (fp32 vector add) with the driver's
/// built-in IGC compiler, and verifies results. No SPIR-V toolchain required.
/// </summary>
internal static class L0Run
{
    public delegate uint ZeInit(uint flags);
    public delegate uint ZeDriverGet(ref uint count, IntPtr drivers);
    public delegate uint ZeDeviceGet(IntPtr driver, ref uint count, IntPtr devices);
    public delegate uint ZeDeviceGetProperties(IntPtr device, IntPtr props);
    public delegate uint ZeDeviceGetModuleProperties(IntPtr device, IntPtr props);
    public delegate uint ZeDeviceGetCommandQueueGroupProperties(IntPtr device, ref uint count, IntPtr props);
    public delegate uint ZeContextCreate(IntPtr driver, IntPtr desc, out IntPtr context);
    public delegate uint ZeCommandQueueCreate(IntPtr context, IntPtr device, IntPtr desc, out IntPtr queue);
    public delegate uint ZeCommandQueueExecuteCommandLists(IntPtr queue, uint numLists, IntPtr lists, IntPtr fence);
    public delegate uint ZeCommandQueueSynchronize(IntPtr queue, ulong timeout);
    public delegate uint ZeCommandQueueDestroy(IntPtr queue);
    public delegate uint ZeContextDestroy(IntPtr context);
    public delegate uint ZeCommandListCreate(IntPtr context, IntPtr device, IntPtr desc, out IntPtr list);
    public delegate uint ZeCommandListAppendLaunchKernel(IntPtr list, IntPtr kernel, IntPtr groupCount, IntPtr signalEvent, uint numWaitEvents, IntPtr waitEvents);
    public delegate uint ZeCommandListClose(IntPtr list);
    public delegate uint ZeCommandListDestroy(IntPtr list);
    public delegate uint ZeModuleCreate(IntPtr context, IntPtr device, IntPtr desc, out IntPtr module, out IntPtr buildLog);
    public delegate uint ZeModuleBuildLogGetString(IntPtr buildLog, ref nuint size, IntPtr pBuildLog);
    public delegate uint ZeModuleBuildLogDestroy(IntPtr buildLog);
    public delegate uint ZeModuleDestroy(IntPtr module);
    public delegate uint ZeKernelCreate(IntPtr module, IntPtr desc, out IntPtr kernel);
    public delegate uint ZeKernelDestroy(IntPtr kernel);
    public delegate uint ZeKernelSetArgumentValue(IntPtr kernel, uint argIndex, nuint argSize, IntPtr pArgValue);
    public delegate uint ZeMemAllocShared(IntPtr context, IntPtr deviceDesc, IntPtr hostDesc, nuint size, nuint alignment, IntPtr device, out IntPtr ptr);
    public delegate uint ZeMemFree(IntPtr context, IntPtr ptr);

    private delegate bool AllocSharedFn(nuint size, out IntPtr ptr);

    private static int failures;
    private static int diagnostics;

    public static int Run()
    {
        Console.WriteLine("--- Level Zero execution (run) ---");
        try
        {
            using var loader = new L0Loader();
            var zeInit = loader.GetProc<ZeInit>("zeInit");
            if (Check(zeInit(0), "zeInit"))
                Execute(loader);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [FAIL] Level Zero execution probe crashed: {ex.Message}");
            failures++;
        }

        Console.WriteLine();
        Console.WriteLine($"--- execution probe done: {failures} failure(s), {diagnostics} expected driver-bug diagnostic(s) ---");
        return failures;
    }

    private static void Execute(L0Loader loader)
    {
        var zeDriverGet = loader.GetProc<ZeDriverGet>("zeDriverGet");
        var zeDeviceGet = loader.GetProc<ZeDeviceGet>("zeDeviceGet");
        var zeDeviceGetProperties = loader.GetProc<ZeDeviceGetProperties>("zeDeviceGetProperties");
        var zeDeviceGetModuleProps = loader.GetProc<ZeDeviceGetModuleProperties>("zeDeviceGetModuleProperties");
        var zeDeviceGetCqGroups = loader.GetProc<ZeDeviceGetCommandQueueGroupProperties>("zeDeviceGetCommandQueueGroupProperties");
        var zeContextCreate = loader.GetProc<ZeContextCreate>("zeContextCreate");
        var zeCommandQueueCreate = loader.GetProc<ZeCommandQueueCreate>("zeCommandQueueCreate");
        var zeCommandQueueExecuteCommandLists = loader.GetProc<ZeCommandQueueExecuteCommandLists>("zeCommandQueueExecuteCommandLists");
        var zeCommandQueueSynchronize = loader.GetProc<ZeCommandQueueSynchronize>("zeCommandQueueSynchronize");
        var zeCommandQueueDestroy = loader.GetProc<ZeCommandQueueDestroy>("zeCommandQueueDestroy");
        var zeMemAllocShared = loader.GetProc<ZeMemAllocShared>("zeMemAllocShared");
        var zeMemFree = loader.GetProc<ZeMemFree>("zeMemFree");
        var zeContextDestroy = loader.GetProc<ZeContextDestroy>("zeContextDestroy");

        // Pick a GPU device (prefer integrated, skip subdevices) from the first driver.
        uint driverCount = 0;
        Check(zeDriverGet(ref driverCount, IntPtr.Zero), "zeDriverGet (count)");
        if (driverCount == 0)
        {
            Console.WriteLine("  [FAIL] no Level Zero drivers to execute on.");
            failures++;
            return;
        }

        IntPtr driverBlock = Marshal.AllocHGlobal((int)driverCount * IntPtr.Size);
        IntPtr driver;
        IntPtr device = IntPtr.Zero;
        string deviceName = "";
        try
        {
            Check(zeDriverGet(ref driverCount, driverBlock), "zeDriverGet (list)");
            driver = Marshal.ReadIntPtr(driverBlock);

            uint deviceCount = 0;
            Check(zeDeviceGet(driver, ref deviceCount, IntPtr.Zero), "zeDeviceGet (count)");
            IntPtr deviceBlock = Marshal.AllocHGlobal((int)deviceCount * IntPtr.Size);
            try
            {
                Check(zeDeviceGet(driver, ref deviceCount, deviceBlock), "zeDeviceGet (list)");
                for (int i = 0; i < deviceCount && device == IntPtr.Zero; i++)
                {
                    IntPtr candidate = Marshal.ReadIntPtr(deviceBlock, i * IntPtr.Size);
                    var dp = ZeDeviceProperties.New();
                    IntPtr dpPtr = L0Loader.PtrToStructure(ref dp);
                    try
                    {
                        if (Check(zeDeviceGetProperties(candidate, dpPtr), "zeDeviceGetProperties"))
                        {
                            var p = L0Loader.StructureFromPtr<ZeDeviceProperties>(dpPtr);
                            if (p.type == L0Constants.DEVICE_TYPE_GPU && (p.flags & L0Constants.PROP_SUBDEVICE) == 0)
                            {
                                device = candidate;
                                deviceName = L0Loader.AnsiString(p.name);
                            }
                        }
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(dpPtr);
                    }
                }
            }
            finally
            {
                Marshal.FreeHGlobal(deviceBlock);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(driverBlock);
        }

        if (device == IntPtr.Zero)
        {
            Console.WriteLine("  [FAIL] no GPU device found to execute on.");
            failures++;
            return;
        }

        Console.WriteLine($"  executing on: {deviceName}");

        var ctxDesc = new ZeContextDesc { stype = L0Constants.ST_CONTEXT_DESC, flags = 0 };
        IntPtr ctxDescPtr = L0Loader.PtrToStructure(ref ctxDesc);
        IntPtr context;
        try
        {
            if (!Check(zeContextCreate(driver, ctxDescPtr, out context), "zeContextCreate"))
                return;
        }
        finally
        {
            Marshal.FreeHGlobal(ctxDescPtr);
        }

        // Find the compute-capable command queue group.
        uint groupCount = 0;
        uint computeOrdinal = 0;
        IntPtr groupsPtr = IntPtr.Zero;
        try
        {
            Check(zeDeviceGetCqGroups(device, ref groupCount, IntPtr.Zero), "zeDeviceGetCommandQueueGroupProperties (count)");
            if (groupCount > 0)
            {
                groupsPtr = Marshal.AllocHGlobal((int)groupCount * Marshal.SizeOf<ZeCommandQueueGroupProperties>());
                if (Check(zeDeviceGetCqGroups(device, ref groupCount, groupsPtr), "zeDeviceGetCommandQueueGroupProperties (list)"))
                {
                    for (uint i = 0; i < groupCount; i++)
                    {
                        var g = L0Loader.StructureFromPtr<ZeCommandQueueGroupProperties>(
                            groupsPtr + (int)i * Marshal.SizeOf<ZeCommandQueueGroupProperties>());
                        if ((g.flags & L0Constants.CQ_GROUP_COMPUTE) != 0)
                        {
                            computeOrdinal = i;
                            break;
                        }
                    }

                    Console.WriteLine($"  compute queue group ordinal: {computeOrdinal}");
                }
            }
        }
        finally
        {
            if (groupsPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(groupsPtr);
        }

        var cqDesc = new ZeCommandQueueDesc
        {
            stype = L0Constants.ST_COMMAND_QUEUE_DESC,
            ordinal = computeOrdinal,
            index = 0,
            flags = 0,
            mode = 0,
            priority = 0
        };
        IntPtr cqDescPtr = L0Loader.PtrToStructure(ref cqDesc);
        IntPtr queue;
        try
        {
            if (!Check(zeCommandQueueCreate(context, device, cqDescPtr, out queue), "zeCommandQueueCreate"))
                return;
        }
        finally
        {
            Marshal.FreeHGlobal(cqDescPtr);
        }

        // SPIR-V versions the driver accepts. Only use the driver-reported maximum
        // (this driver reports 1.0); attempting higher versions hangs zeModuleCreate.
        var versions = new List<uint>();
        var mp = ZeDeviceModuleProperties.New();
        IntPtr mpPtr = L0Loader.PtrToStructure(ref mp);
        try
        {
            if (Check(zeDeviceGetModuleProps(device, mpPtr), "zeDeviceGetModuleProperties"))
            {
                var p = L0Loader.StructureFromPtr<ZeDeviceModuleProperties>(mpPtr);
                Console.WriteLine($"  SPIR-V supported (device): {L0Loader.SpirvVersion(p.spirvVersionSupported)}");
                if (p.spirvVersionSupported != 0)
                    versions.Add(p.spirvVersionSupported);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(mpPtr);
        }

        // Test 0: minimal kernel (module-only build) — bisects the IGC crash: if this
        // bare module also fails, the problem is the prologue, not the body constructs.
        Console.WriteLine();
        Console.WriteLine("  [test 0] minimal kernel (empty body): module-only build");
        var zeModuleCreateTest0 = loader.GetProc<ZeModuleCreate>("zeModuleCreate");
        bool minimalBuilt = false;
        foreach (uint version in versions)
        {
            if (TryCreateModule(loader, zeModuleCreateTest0, context, device, SpvKernels.Minimal(version), out IntPtr m0))
            {
                Console.WriteLine($"    module built OK (SPIR-V {L0Loader.SpirvVersion(version)})");
                loader.GetProc<ZeModuleDestroy>("zeModuleDestroy")(m0);
                minimalBuilt = true;
                break;
            }
        }

        if (!minimalBuilt)
        {
            Console.WriteLine("    [FAIL] minimal kernel module rejected by the driver");
            failures++;
        }

        // Bisection variants — each adds one layer beyond the minimal kernel so the
        // exact construct that crashes IGC is identifiable from a single run.
        foreach ((string name, uint[] module) in SpvKernels.BisectVariants(versions.Count > 0 ? versions[0] : 0x00010000))
        {
            Console.WriteLine();
            Console.WriteLine($"  [bisect] {name}");
            bool built = false;
            if (TryCreateModule(loader, zeModuleCreateTest0, context, device, module, out IntPtr mVariant))
            {
                Console.WriteLine("    module built OK");
                loader.GetProc<ZeModuleDestroy>("zeModuleDestroy")(mVariant);
                built = true;
            }

            if (!built)
            {
                // Expected where the variant exercises a known-broken IGC construct
                // (access chains, Private variables, Generic-pointers args) — these
                // are diagnostics, not probe failures.
                diagnostics++;
            }
        }

        AllocSharedFn allocShared = (nuint size, out IntPtr ptr) =>
        {
            var devDesc = new ZeDeviceMemAllocDesc { stype = L0Constants.ST_DEVICE_MEM_ALLOC_DESC, ordinal = 0 };
            var hostDesc = new ZeHostMemAllocDesc { stype = L0Constants.ST_HOST_MEM_ALLOC_DESC };
            IntPtr devDescPtr = L0Loader.PtrToStructure(ref devDesc);
            IntPtr hostDescPtr = L0Loader.PtrToStructure(ref hostDesc);
            try
            {
                uint r = zeMemAllocShared(context, devDescPtr, hostDescPtr, size, 64, device, out ptr);
                return Check(r, $"zeMemAllocShared ({size} bytes)");
            }
            finally
            {
                Marshal.FreeHGlobal(devDescPtr);
                Marshal.FreeHGlobal(hostDescPtr);
            }
        };

        try
        {
            Console.WriteLine();
            Console.WriteLine("  [test 1] add_parallel: atomic sum of lane ids -> counter (0+..+255 = 32640)");
            Console.WriteLine("    diagnostic sweep: simd8/simd16 local sizes exhibit an IGC bug (upper-half atomics are");
            Console.WriteLine("    dropped); only 1-lane workgroups compile simd1 atomics and land all 256 lanes. The");
            Console.WriteLine("    (localSize=1, groups=256) launch is the pass gate below.");
            foreach (var (localSize, groups) in new[] { (256u, 1u), (128u, 2u), (32u, 8u), (8u, 32u), (1u, 256u) })
            {
                uint workItems = localSize * groups;
                uint expected = workItems * (workItems - 1) / 2;
                bool isGate = localSize == 1 && groups == 256;
                Console.WriteLine();
                Console.WriteLine($"    localSize = {localSize}, groups = {groups} ({workItems} work items), expected counter = {expected}{(isGate ? "  [gate]" : "")}");
                TestKernel(
                    loader, context, device, queue, computeOrdinal, versions, allocShared,
                    v => SpvKernels.AddParallel(v, localSize), "add_parallel", argCount: 1, n: 1,
                    fill: (a, b, c) => Marshal.WriteInt32(a, 0, 0),
                    verify: (a, b, c) =>
                    {
                        int counter = Marshal.ReadInt32(a);
                        bool pass = counter == (int)expected;
                        Console.WriteLine(pass
                            ? $"      result: PASS ({workItems} lanes atomicAdded their ids -> {counter})"
                            : $"      result: FAIL (counter = {counter}, expected {expected})");
                        if (isGate)
                        {
                            if (!pass)
                                failures++;
                        }
                        else if (!pass)
                            diagnostics++;
                        return pass;
                    },
                    zeCommandQueueExecuteCommandLists, zeCommandQueueSynchronize, zeMemFree,
                    timing: false, opsPerLaunch: workItems, groupCountX: groups);
            }

            Console.WriteLine();
            Console.WriteLine("  [test 2] add_loop: 1 work item, 1,000,000 serial fp32 adds (OpPhi loop)");
            TestKernel(
                loader, context, device, queue, computeOrdinal, versions, allocShared,
                SpvKernels.AddLoop, "add_loop", argCount: 3, n: 1_000_000,
                fill: (a, b, c) =>
                {
                    float[] data = new float[1_000_000];
                    data[0] = 1.0f; // the kernel accumulates *a per iteration
                    Marshal.Copy(data, 0, a, data.Length);
                },
                verify: (a, b, c) =>
                {
                    float[] outData = new float[1_000_000];
                    Marshal.Copy(c, outData, 0, outData.Length);
                    bool pass = outData[0] == 1_000_000.0f;
                    Console.WriteLine(pass
                        ? "    result: PASS (c[0] = 1,000,000.0 after 1M serial adds of *a = 1.0)"
                        : $"    result: FAIL (c[0] = {outData[0]}, expected 1,000,000.0)");
                    if (!pass)
                        failures++;
                    return pass;
                },
                zeCommandQueueExecuteCommandLists, zeCommandQueueSynchronize, zeMemFree,
                timing: true, opsPerLaunch: 1_000_000);

            // BF16 widening on device, BFloat16-to-BFloat16. The host holds weights as
            // System.Numerics.BFloat16 — the raw 16-bit pattern IS the BFloat16 memory
            // layout (SafeTensorsLoader keeps BF16 weights that way) — and writes the
            // BFloat16 itself into the SAME shared buffer the kernel reads. On an iGPU
            // (unified DRAM) there is no copy in/out: zeMemAllocShared is used
            // everywhere and one allocation is visible to both the CPU and the GPU.
            // BFloat16 is the high half of the f32 the GPU emits (BF16 -> F32 is
            // lossless), so the result is loaded straight back into BFloat16 — no
            // ushort, no ToSingle, no host bit arithmetic anywhere in the round trip.
            // "No widening" holds for the input path (BF16 end to end); the kernel's
            // f32 accumulator is the hardware's native BF16 compute model (Xe2 DPAS
            // also accumulates BF16 products in f32 — f32 accumulation is required
            // for precision, never a host/API widening we introduced).
            float[] bf16Patterns = [1.0f, -2.0f, 1.5f, 123.5f];

            Console.WriteLine();
            Console.WriteLine("  [test 3] bf16_native: BF16->F32 via OpConvertBF16ToFINTEL (SPV_INTEL_bfloat16_conversion, capability 6115)");
            bool bf16NativeBuilt = false;
            foreach (uint version in versions)
            {
                if (TryCreateModule(loader, zeModuleCreateTest0, context, device, SpvKernels.Bf16Native(version), out IntPtr bf16Module))
                {
                    Console.WriteLine($"    module built OK (SPIR-V {L0Loader.SpirvVersion(version)}) — IGC honors the ZE_extension_bfloat16_conversions contract");
                    loader.GetProc<ZeModuleDestroy>("zeModuleDestroy")(bf16Module);
                    bf16NativeBuilt = true;
                    break;
                }
            }

            if (!bf16NativeBuilt)
            {
                Console.WriteLine("    [diagnostic] native BF16 module rejected: the driver advertises ZE_extension_bfloat16_conversions but IGC");
                Console.WriteLine("    refuses SPV_INTEL_bfloat16_conversion modules — BF16 must use the safe-subset widen path (test 4).");
                diagnostics++;
            }

            if (bf16NativeBuilt)
            {
                foreach (float value in bf16Patterns)
                {
                    var bf16 = BFloat16.CreateChecked(value);
                    TestKernel(
                        loader, context, device, queue, computeOrdinal, versions, allocShared,
                        v => SpvKernels.Bf16Native(v), "bf16_native", argCount: 2, n: 1,
                        fill: (a, b, c) => Marshal.StructureToPtr(bf16, a, false),
                        verify: (a, b, c) =>
                        {
                            var stored = Marshal.PtrToStructure<BFloat16>(IntPtr.Add(b, 2));
                            bool pass = stored == bf16;
                            Console.WriteLine(pass
                                ? $"      result: PASS (BF16 {value} widened natively, reloaded as BF16 {stored})"
                                : $"      result: FAIL (reloaded {stored}, expected {bf16})");
                            if (!pass)
                                failures++;
                            return pass;
                        },
                        zeCommandQueueExecuteCommandLists, zeCommandQueueSynchronize, zeMemFree,
                        timing: false, opsPerLaunch: 1);
                }
            }

            Console.WriteLine();
            Console.WriteLine("  [test 4] bf16_emul: BF16->F32 widen via OpUConvert + shift16 (safe subset, no extension)");
            foreach (float value in bf16Patterns)
            {
                var bf16 = BFloat16.CreateChecked(value);
                TestKernel(
                    loader, context, device, queue, computeOrdinal, versions, allocShared,
                    v => SpvKernels.Bf16EmulWiden(v), "bf16_emul", argCount: 2, n: 1,
                    fill: (a, b, c) => Marshal.StructureToPtr(bf16, a, false),
                    verify: (a, b, c) =>
                    {
                        var stored = Marshal.PtrToStructure<BFloat16>(IntPtr.Add(b, 2));
                        bool pass = stored == bf16;
                        Console.WriteLine(pass
                            ? $"      result: PASS (BF16 {value} widened by shift, reloaded as BF16 {stored})"
                            : $"      result: FAIL (reloaded {stored}, expected {bf16})");
                        if (!pass)
                            failures++;
                        return pass;
                    },
                    zeCommandQueueExecuteCommandLists, zeCommandQueueSynchronize, zeMemFree,
                    timing: false, opsPerLaunch: 1);
            }

            if (bf16NativeBuilt)
            {
                Console.WriteLine();
                Console.WriteLine("  [test 5] bf16_native_acc: 1M iterations of acc += widen(BF16 *a) (convert in an OpPhi loop)");
                TestKernel(
                    loader, context, device, queue, computeOrdinal, versions, allocShared,
                    v => SpvKernels.Bf16NativeAccumulate(v), "bf16_native_acc", argCount: 2, n: 1,
                    fill: (a, b, c) => Marshal.StructureToPtr(BFloat16.CreateChecked(1.0f), a, false),
                    verify: (a, b, c) =>
                    {
                        float stored = BitConverter.Int32BitsToSingle(Marshal.ReadInt32(b));
                        bool pass = stored == 1_000_000.0f;
                        Console.WriteLine(pass
                            ? "      result: PASS (c[0] = 1,000,000.0 after 1M native BF16 widens + fp32 adds of BF16 1.0)"
                            : $"      result: FAIL (c[0] = {stored}, expected 1,000,000.0)");
                        if (!pass)
                            failures++;
                        return pass;
                    },
                    zeCommandQueueExecuteCommandLists, zeCommandQueueSynchronize, zeMemFree,
                    timing: true, opsPerLaunch: 1_000_000);
            }
        }
        finally
        {
            zeCommandQueueDestroy(queue);
            zeContextDestroy(context);
        }
    }

    private static void TestKernel(
        L0Loader loader,
        IntPtr context,
        IntPtr device,
        IntPtr queue,
        uint computeOrdinal,
        List<uint> spirvVersions,
        AllocSharedFn allocShared,
        Func<uint, uint[]> buildSpv,
        string kernelName,
        int argCount,
        int n,
        Action<IntPtr, IntPtr, IntPtr> fill,
        Func<IntPtr, IntPtr, IntPtr, bool> verify,
        ZeCommandQueueExecuteCommandLists zeCqExecute,
        ZeCommandQueueSynchronize zeCqSync,
        ZeMemFree zeMemFree,
        bool timing,
        long opsPerLaunch,
        uint groupCountX = 1)
    {
        IntPtr module = IntPtr.Zero;
        IntPtr kernel = IntPtr.Zero;
        IntPtr aBuf = IntPtr.Zero;
        IntPtr bBuf = IntPtr.Zero;
        IntPtr cBuf = IntPtr.Zero;
        IntPtr list = IntPtr.Zero;
        IntPtr gcPtr = IntPtr.Zero;

        try
        {
            var zeCommandListCreate = loader.GetProc<ZeCommandListCreate>("zeCommandListCreate");
            var clDesc = new ZeCommandListDesc
            {
                stype = L0Constants.ST_COMMAND_LIST_DESC,
                commandQueueGroupOrdinal = computeOrdinal,
                flags = 0
            };
            IntPtr clDescPtr = L0Loader.PtrToStructure(ref clDesc);
            try
            {
                if (!Check(zeCommandListCreate(context, device, clDescPtr, out list), "zeCommandListCreate"))
                    return;
            }
            finally
            {
                Marshal.FreeHGlobal(clDescPtr);
            }

            var zeModuleCreate = loader.GetProc<ZeModuleCreate>("zeModuleCreate");
            foreach (uint version in spirvVersions)
            {
                uint[] candidate = buildSpv(version);
                if (TryCreateModule(loader, zeModuleCreate, context, device, candidate, out module))
                {
                    Console.WriteLine($"    module built OK (SPIR-V {version >> 16}.{version >> 8 & 0xFF})");
                    break;
                }
            }

            if (module == IntPtr.Zero)
            {
                Console.WriteLine("    [FAIL] module build failed with every SPIR-V version tried.");
                failures++;
                return;
            }

            var zeKernelCreate = loader.GetProc<ZeKernelCreate>("zeKernelCreate");
            IntPtr namePtr = Marshal.StringToCoTaskMemUTF8(kernelName);
            var kd = new ZeKernelDesc { stype = L0Constants.ST_KERNEL_DESC, pKernelName = namePtr };
            IntPtr kdPtr = L0Loader.PtrToStructure(ref kd);
            try
            {
                if (!Check(zeKernelCreate(module, kdPtr, out kernel), "zeKernelCreate"))
                    return;
            }
            finally
            {
                Marshal.FreeHGlobal(kdPtr);
                Marshal.FreeCoTaskMem(namePtr);
            }

            nuint bytes = (nuint)n * 4;
            if (!allocShared(bytes, out aBuf) || !allocShared(bytes, out bBuf) || !allocShared(bytes, out cBuf))
                return;

            fill(aBuf, bBuf, cBuf);

            var zeKernelSetArgumentValue = loader.GetProc<ZeKernelSetArgumentValue>("zeKernelSetArgumentValue");
            IntPtr argSlot = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                var args = new[] { aBuf, bBuf, cBuf };
                for (int i = 0; i < argCount; i++)
                {
                    Marshal.WriteIntPtr(argSlot, args[i]);
                    if (!Check(zeKernelSetArgumentValue(kernel, (uint)i, (nuint)IntPtr.Size, argSlot), $"zeKernelSetArgumentValue[{i}]"))
                        return;
                }
            }
            finally
            {
                Marshal.FreeHGlobal(argSlot);
            }

            var groupCount = new ZeGroupCount { groupCountX = groupCountX, groupCountY = 1, groupCountZ = 1 };
            gcPtr = L0Loader.PtrToStructure(ref groupCount);

            var zeAppendLaunch = loader.GetProc<ZeCommandListAppendLaunchKernel>("zeCommandListAppendLaunchKernel");
            var zeCommandListClose = loader.GetProc<ZeCommandListClose>("zeCommandListClose");
            if (!Check(zeAppendLaunch(list, kernel, gcPtr, IntPtr.Zero, 0, IntPtr.Zero), "zeCommandListAppendLaunchKernel") ||
                !Check(zeCommandListClose(list), "zeCommandListClose"))
                return;

            IntPtr listArray = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                Marshal.WriteIntPtr(listArray, list);

                if (!Check(zeCqExecute(queue, 1, listArray, IntPtr.Zero), "zeCommandQueueExecuteCommandLists") ||
                    !Check(zeCqSync(queue, 15_000_000_000UL), "zeCommandQueueSynchronize"))
                    return;

                // Test-specific verification (fill/verify lambdas set expectations).
                bool pass = verify(aBuf, bBuf, cBuf);

                if (timing && pass)
                {
                    var times = new long[8];
                    int runs = 0;
                    for (int i = 0; i < times.Length; i++)
                    {
                        var sw = Stopwatch.StartNew();
                        uint r1 = zeCqExecute(queue, 1, listArray, IntPtr.Zero);
                        uint r2 = zeCqSync(queue, 15_000_000_000UL);
                        sw.Stop();
                        if (r1 != 0 || r2 != 0)
                        {
                            Check(r1 != 0 ? r1 : r2, "timing run");
                            break;
                        }

                        times[runs++] = sw.Elapsed.Ticks * 100; // ns
                    }

                    var valid = times.Take(runs).Where(t => t > 0).OrderBy(t => t).ToArray();
                    if (valid.Length > 0)
                    {
                        long min = valid[0];
                        double median = valid.Length % 2 == 0
                            ? (valid[valid.Length / 2 - 1] + valid[valid.Length / 2]) / 2.0
                            : valid[valid.Length / 2];
                        double opsPerSec = runs > 0 ? opsPerLaunch / (min / 1e9) : 0;
                        Console.WriteLine($"    timing ({runs} launches): min {min / 1000.0:F1} µs, median {median / 1000.0:F1} µs" +
                                          $" -> {opsPerSec / 1e9:F2} GFADD/s (single work item, serial loop)");
                    }
                }
            }
            finally
            {
                Marshal.FreeHGlobal(listArray);
            }
        }
        finally
        {
            if (gcPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(gcPtr);
            if (cBuf != IntPtr.Zero)
                zeMemFree(context, cBuf);
            if (bBuf != IntPtr.Zero)
                zeMemFree(context, bBuf);
            if (aBuf != IntPtr.Zero)
                zeMemFree(context, aBuf);
            if (kernel != IntPtr.Zero)
                loader.GetProc<ZeKernelDestroy>("zeKernelDestroy")(kernel);
            if (module != IntPtr.Zero)
                loader.GetProc<ZeModuleDestroy>("zeModuleDestroy")(module);
            if (list != IntPtr.Zero)
                loader.GetProc<ZeCommandListDestroy>("zeCommandListDestroy")(list);
        }
    }

    private static bool TryCreateModule(
        L0Loader loader,
        ZeModuleCreate zeModuleCreate,
        IntPtr context,
        IntPtr device,
        uint[] spirv,
        out IntPtr module)
    {
        module = IntPtr.Zero;
        GCHandle pinned = GCHandle.Alloc(spirv, GCHandleType.Pinned);
        try
        {
            var desc = new ZeModuleDesc
            {
                stype = L0Constants.ST_MODULE_DESC,
                format = L0Constants.MODULE_FORMAT_IL_SPIRV,
                inputSize = (nuint)(spirv.Length * 4),
                pInputModule = pinned.AddrOfPinnedObject()
            };
            IntPtr descPtr = L0Loader.PtrToStructure(ref desc);
            try
            {
                IntPtr buildLog;
                uint r = zeModuleCreate(context, device, descPtr, out module, out buildLog);
                if (buildLog != IntPtr.Zero)
                {
                    string log = ReadBuildLog(loader, buildLog);
                    if (log.Length > 0)
                        Console.WriteLine(log);
                    loader.GetProc<ZeModuleBuildLogDestroy>("zeModuleBuildLogDestroy")(buildLog);
                }

                if (r != 0)
                {
                    // Callers decide whether a failed build counts as a probe failure
                    // (test 0 / real test kernels) or as an expected driver-bug
                    // diagnostic (bisection variants). Print-only here.
                    Console.WriteLine($"  [FAIL] zeModuleCreate ({spirv.Length * 4} bytes SPIR-V): {L0Constants.ResultName(r)}");
                    return false;
                }

                return true;
            }
            finally
            {
                Marshal.FreeHGlobal(descPtr);
            }
        }
        finally
        {
            pinned.Free();
        }
    }

    private static string ReadBuildLog(L0Loader loader, IntPtr buildLog)
    {
        var getString = loader.GetProc<ZeModuleBuildLogGetString>("zeModuleBuildLogGetString");
        nuint size = 0;
        uint r = getString(buildLog, ref size, IntPtr.Zero);
        if (r != 0 || size == 0 || size > 1_000_000)
            return "";

        IntPtr buf = Marshal.AllocHGlobal((int)size);
        try
        {
            getString(buildLog, ref size, buf);
            return "    build log (" + (int)size + " bytes):\n" + Indent(Marshal.PtrToStringUTF8(buf) ?? "");
        }
        finally
        {
            Marshal.FreeHGlobal(buf);
        }
    }

    private static string Indent(string text) => string.Join("\n", text.Split('\n').Select(line => "      " + line));

    private static bool Check(uint result, string what)
    {
        if (result != 0)
        {
            Console.WriteLine($"  [FAIL] {what}: {L0Constants.ResultName(result)}");
            failures++;
            return false;
        }

        return true;
    }
}