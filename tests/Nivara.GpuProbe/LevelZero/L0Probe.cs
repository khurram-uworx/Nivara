using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// Level Zero enumeration report: drivers, API versions, extension properties,
/// devices (type, PCI id, EU topology, clocks) and module capabilities
/// (SPIR-V version, fp16/fp32/fp64 feature bits, DP4A).
/// </summary>
internal static class L0Probe
{
    public delegate uint ZeInit(uint flags);
    public delegate uint ZeDriverGet(ref uint count, IntPtr drivers);
    public delegate uint ZeDriverGetApiVersion(IntPtr driver, out uint version);
    public delegate uint ZeDriverGetProperties(IntPtr driver, IntPtr props);
    public delegate uint ZeDriverGetExtensionProperties(IntPtr driver, ref uint count, IntPtr props);
    public delegate uint ZeDeviceGet(IntPtr driver, ref uint count, IntPtr devices);
    public delegate uint ZeDeviceGetSubDevices(IntPtr device, ref uint count, IntPtr devices);
    public delegate uint ZeDeviceGetProperties(IntPtr device, IntPtr props);
    public delegate uint ZeDeviceGetModuleProperties(IntPtr device, IntPtr props);
    public delegate uint ZeDeviceGetCommandQueueGroupProperties(IntPtr device, ref uint count, IntPtr props);

    private static int failures;

    public static int Run()
    {
        Console.WriteLine("--- Level Zero stack (l0) ---");
        try
        {
            using var loader = new L0Loader();
            Console.WriteLine($"ze_loader.dll loaded (API target 1.28.x struct layout).");

            var zeInit = loader.GetProc<ZeInit>("zeInit");
            var r = zeInit(0);
            if (r != 0)
            {
                Check(r, "zeInit");
                return failures;
            }

            var zeDriverGet = loader.GetProc<ZeDriverGet>("zeDriverGet");
            uint driverCount = 0;
            r = zeDriverGet(ref driverCount, IntPtr.Zero);
            Check(r, "zeDriverGet (count)");
            if (driverCount == 0)
            {
                Console.WriteLine("  [INFO] no Level Zero drivers found.");
                return failures;
            }

            IntPtr drivers = Marshal.AllocHGlobal((int)driverCount * IntPtr.Size);
            try
            {
                r = zeDriverGet(ref driverCount, drivers);
                Check(r, "zeDriverGet (list)");

                for (int d = 0; d < driverCount; d++)
                {
                    IntPtr driver = Marshal.ReadIntPtr(drivers, d * IntPtr.Size);
                    PrintDriver(loader, driver, d);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(drivers);
            }

            Console.WriteLine($"  enumerated {driverCount} driver(s), {failures} failure(s).");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [FAIL] Level Zero probe crashed: {ex.Message}");
            failures++;
        }

        return failures;
    }

    private static void PrintDriver(L0Loader loader, IntPtr driver, int index)
    {
        Console.WriteLine();
        Console.WriteLine($"  Driver[{index}]: 0x{driver.ToInt64():X}");

        var zeDriverGetApiVersion = loader.GetProc<ZeDriverGetApiVersion>("zeDriverGetApiVersion");
        uint apiVersion = 0;
        Check(zeDriverGetApiVersion(driver, out apiVersion), "zeDriverGetApiVersion");
        Console.WriteLine($"    API version: {apiVersion >> 16}.{apiVersion & 0xFFFF}");

        var zeDriverGetProperties = loader.GetProc<ZeDriverGetProperties>("zeDriverGetProperties");
        var props = ZeDriverProperties.New();
        IntPtr propsPtr = L0Loader.PtrToStructure(ref props);
        try
        {
            Check(zeDriverGetProperties(driver, propsPtr), "zeDriverGetProperties");
            var p = L0Loader.StructureFromPtr<ZeDriverProperties>(propsPtr);
            Console.WriteLine($"    driverVersion: 0x{p.driverVersion:X8}  uuid: {Convert.ToHexString(p.uuid)}");
        }
        finally
        {
            Marshal.FreeHGlobal(propsPtr);
        }

        var zeDriverGetExt = loader.GetProc<ZeDriverGetExtensionProperties>("zeDriverGetExtensionProperties");
        uint extCount = 0;
        Check(zeDriverGetExt(driver, ref extCount, IntPtr.Zero), "zeDriverGetExtensionProperties (count)");
        bool bf16Extension = false;
        if (extCount > 0)
        {
            IntPtr extPtr = Marshal.AllocHGlobal((int)extCount * Marshal.SizeOf<ZeDriverExtensionProperties>());
            try
            {
                Check(zeDriverGetExt(driver, ref extCount, extPtr), "zeDriverGetExtensionProperties (list)");
                for (int i = 0; i < extCount; i++)
                {
                    var e = L0Loader.StructureFromPtr<ZeDriverExtensionProperties>(extPtr + i * Marshal.SizeOf<ZeDriverExtensionProperties>());
                    string name = L0Loader.AnsiString(e.name);
                    Console.WriteLine($"    extension: {name} v{e.version}");
                    if (name == "ZE_extension_bfloat16_conversions")
                        bf16Extension = true;
                }
            }
            finally
            {
                Marshal.FreeHGlobal(extPtr);
            }
        }

        Console.WriteLine(bf16Extension
            ? "    => ZE_extension_bfloat16_conversions present: IGC is contracted to accept SPV_INTEL_bfloat16_conversion modules"
            : "    => ZE_extension_bfloat16_conversions absent: no native BF16 conversion contract; BF16 must be widened on device");

        var zeDeviceGet = loader.GetProc<ZeDeviceGet>("zeDeviceGet");
        uint deviceCount = 0;
        Check(zeDeviceGet(driver, ref deviceCount, IntPtr.Zero), "zeDeviceGet (count)");
        if (deviceCount == 0)
        {
            Console.WriteLine("    [INFO] driver exposes no devices.");
            return;
        }

        IntPtr devices = Marshal.AllocHGlobal((int)deviceCount * IntPtr.Size);
        try
        {
            Check(zeDeviceGet(driver, ref deviceCount, devices), "zeDeviceGet (list)");
            for (int i = 0; i < deviceCount; i++)
            {
                IntPtr device = Marshal.ReadIntPtr(devices, i * IntPtr.Size);
                PrintDevice(loader, device, i);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(devices);
        }
    }

    private static void PrintDevice(L0Loader loader, IntPtr device, int index)
    {
        var zeDeviceGetProperties = loader.GetProc<ZeDeviceGetProperties>("zeDeviceGetProperties");
        var dp = ZeDeviceProperties.New();
        IntPtr dpPtr = L0Loader.PtrToStructure(ref dp);
        string name;
        try
        {
            Check(zeDeviceGetProperties(device, dpPtr), "zeDeviceGetProperties");
            var p = L0Loader.StructureFromPtr<ZeDeviceProperties>(dpPtr);
            name = L0Loader.AnsiString(p.name);
            long euTotal = (long)p.numEUsPerSubslice * p.numSubslicesPerSlice * p.numSlices;
            Console.WriteLine();
            Console.WriteLine($"  Device[{index}]: {name}");
            Console.WriteLine($"    type: {L0Constants.DeviceTypeName(p.type)}  PCI {p.vendorId:X4}:{p.deviceId:X4}" +
                              $"  integrated: {(p.flags & L0Constants.PROP_INTEGRATED) != 0}" +
                              $"  subdevice: {(p.flags & L0Constants.PROP_SUBDEVICE) != 0}" +
                              $"  ecc: {(p.flags & L0Constants.PROP_ECC) != 0}" +
                              $"  onDemandPaging: {(p.flags & L0Constants.PROP_ONDEMANDPAGING) != 0}");
            Console.WriteLine($"    EU topology: {p.numEUsPerSubslice} EU/subslice x {p.numSubslicesPerSlice} subslice/slice x {p.numSlices} slice = {euTotal} EU," +
                              $"  {p.numThreadsPerEU} threads/EU, SIMD-{p.physicalEUSimdWidth}, core clock {p.coreClockRate} MHz");
            Console.WriteLine($"    maxMemAllocSize: {p.maxMemAllocSize / (1024.0 * 1024.0 * 1024.0):F1} GiB, " +
                              $"contexts: {p.maxHardwareContexts}, timerResolution: {p.timerResolution} ns (valid bits {p.timestampValidBits})");
        }
        finally
        {
            Marshal.FreeHGlobal(dpPtr);
        }

        var zeDeviceGetModuleProps = loader.GetProc<ZeDeviceGetModuleProperties>("zeDeviceGetModuleProperties");
        var mp = ZeDeviceModuleProperties.New();
        IntPtr mpPtr = L0Loader.PtrToStructure(ref mp);
        try
        {
            Check(zeDeviceGetModuleProps(device, mpPtr), "zeDeviceGetModuleProperties");
            var p = L0Loader.StructureFromPtr<ZeDeviceModuleProperties>(mpPtr);
            Console.WriteLine($"    SPIR-V: max {L0Loader.SpirvVersion(p.spirvVersionSupported)}" +
                              $"  fp16: {(p.flags & L0Constants.MODULE_FP16) != 0}[{L0Constants.FpFlags(p.fp16flags)}]" +
                              $"  fp32: [{L0Constants.FpFlags(p.fp32flags)}]" +
                              $"  fp64: {(p.flags & L0Constants.MODULE_FP64) != 0}[{L0Constants.FpFlags(p.fp64flags)}]" +
                              $"  DP4A(int8 dot): {(p.flags & L0Constants.MODULE_DP4A) != 0}");
        }
        finally
        {
            Marshal.FreeHGlobal(mpPtr);
        }

        var zeGetCqGroups = loader.GetProc<ZeDeviceGetCommandQueueGroupProperties>("zeDeviceGetCommandQueueGroupProperties");
        uint groupCount = 0;
        Check(zeGetCqGroups(device, ref groupCount, IntPtr.Zero), "zeDeviceGetCommandQueueGroupProperties (count)");
        if (groupCount > 0)
        {
            IntPtr groups = Marshal.AllocHGlobal((int)groupCount * Marshal.SizeOf<ZeCommandQueueGroupProperties>());
            try
            {
                Check(zeGetCqGroups(device, ref groupCount, groups), "zeDeviceGetCommandQueueGroupProperties (list)");
                for (int i = 0; i < groupCount; i++)
                {
                    var g = L0Loader.StructureFromPtr<ZeCommandQueueGroupProperties>(groups + i * Marshal.SizeOf<ZeCommandQueueGroupProperties>());
                    string kind = ((g.flags & L0Constants.CQ_GROUP_COMPUTE) != 0 ? "compute" : "") +
                                  ((g.flags & L0Constants.CQ_GROUP_COPY) != 0 ? "+copy" : "") +
                                  ((g.flags & L0Constants.CQ_GROUP_COOPERATIVE) != 0 ? "+cooperative" : "");
                    Console.WriteLine($"    queue group[{i}]: {(kind.Length == 0 ? "other" : kind)}  queues: {g.numQueues}");
                }
            }
            finally
            {
                Marshal.FreeHGlobal(groups);
            }
        }

        var zeDeviceGetSub = loader.GetProc<ZeDeviceGetSubDevices>("zeDeviceGetSubDevices");
        uint subCount = 0;
        Check(zeDeviceGetSub(device, ref subCount, IntPtr.Zero), "zeDeviceGetSubDevices (count)");
        if (subCount > 0)
            Console.WriteLine($"    subdevices: {subCount}");
    }

    private static bool Check(uint result, string what)
    {
        if (result != 0)
        {
            Console.WriteLine($"    [FAIL] {what}: {L0Constants.ResultName(result)}");
            failures++;
            return false;
        }

        return true;
    }
}