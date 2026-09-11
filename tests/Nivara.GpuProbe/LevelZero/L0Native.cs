using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// Minimal Level Zero bindings. Constants and struct layouts were verified against
/// ze_api.h from the oneapi-src/level-zero repository at tag v1.28.2 (the loader
/// installed on this machine reports API 1.28). Only members actually read by the
/// probe are declared.
/// </summary>
internal static class L0Constants
{
    // ze_structure_type_t
    public const uint ST_DRIVER_PROPERTIES = 0x1;
    public const uint ST_DEVICE_PROPERTIES = 0x3;
    public const uint ST_DEVICE_MODULE_PROPERTIES = 0x5;
    public const uint ST_COMMAND_QUEUE_GROUP_PROPERTIES = 0x6;
    public const uint ST_CONTEXT_DESC = 0xd;
    public const uint ST_COMMAND_QUEUE_DESC = 0xe;
    public const uint ST_COMMAND_LIST_DESC = 0xf;
    public const uint ST_DEVICE_MEM_ALLOC_DESC = 0x15;
    public const uint ST_HOST_MEM_ALLOC_DESC = 0x16;
    public const uint ST_MODULE_DESC = 0x1b;
    public const uint ST_KERNEL_DESC = 0x1d;

    // ze_device_type_t
    public const uint DEVICE_TYPE_GPU = 1;
    public const uint DEVICE_TYPE_CPU = 2;
    public const uint DEVICE_TYPE_FPGA = 3;
    public const uint DEVICE_TYPE_MCA = 4;
    public const uint DEVICE_TYPE_VPU = 5;

    // ze_device_property_flag_t
    public const uint PROP_INTEGRATED = 1u << 0;
    public const uint PROP_SUBDEVICE = 1u << 1;
    public const uint PROP_ECC = 1u << 2;
    public const uint PROP_ONDEMANDPAGING = 1u << 3;

    // ze_device_module_flag_t
    public const uint MODULE_FP16 = 1u << 0;
    public const uint MODULE_FP64 = 1u << 1;
    public const uint MODULE_INT64_ATOMICS = 1u << 2;
    public const uint MODULE_DP4A = 1u << 3;

    // ze_device_fp_flag_t
    public const uint FP_DENORM = 1u << 0;
    public const uint FP_INF_NAN = 1u << 1;
    public const uint FP_ROUND_TO_NEAREST = 1u << 2;
    public const uint FP_ROUND_TO_ZERO = 1u << 3;
    public const uint FP_ROUND_TO_INF = 1u << 4;
    public const uint FP_FMA = 1u << 5;
    public const uint FP_ROUNDED_DIVIDE_SQRT = 1u << 6;
    public const uint FP_SOFT_FLOAT = 1u << 7;

    // ze_command_queue_group_property_flag_t
    public const uint CQ_GROUP_COMPUTE = 1u << 0;
    public const uint CQ_GROUP_COPY = 1u << 1;
    public const uint CQ_GROUP_COOPERATIVE = 1u << 2;
    public const uint CQ_GROUP_METRICS = 1u << 3;

    // ze_module_format_t
    public const uint MODULE_FORMAT_IL_SPIRV = 0;

    public const int MAX_DEVICE_NAME = 256;
    public const int MAX_EXTENSION_NAME = 256;

    public static string DeviceTypeName(uint type) => type switch
    {
        DEVICE_TYPE_GPU => "GPU",
        DEVICE_TYPE_CPU => "CPU",
        DEVICE_TYPE_FPGA => "FPGA",
        DEVICE_TYPE_MCA => "MCA",
        DEVICE_TYPE_VPU => "VPU",
        _ => $"0x{type:X}"
    };

    public static string ResultName(uint r) => r switch
    {
        0 => "ZE_RESULT_SUCCESS",
        1 => "ZE_RESULT_NOT_READY",
        0x70000004 => "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE",
        0x70000005 => "ZE_RESULT_ERROR_MODULE_LINK_FAILURE",
        0x70000006 => "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET",
        0x70010000 => "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS",
        0x70010001 => "ZE_RESULT_ERROR_NOT_AVAILABLE",
        0x78000001 => "ZE_RESULT_ERROR_UNINITIALIZED",
        0x78000002 => "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
        0x78000003 => "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
        0x78000004 => "ZE_RESULT_ERROR_INVALID_ARGUMENT",
        0x78000005 => "ZE_RESULT_ERROR_INVALID_NULL_HANDLE",
        0x78000007 => "ZE_RESULT_ERROR_INVALID_NULL_POINTER",
        0x78000008 => "ZE_RESULT_ERROR_INVALID_SIZE",
        0x78000009 => "ZE_RESULT_ERROR_UNSUPPORTED_SIZE",
        0x7800000a => "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT",
        0x7800000c => "ZE_RESULT_ERROR_INVALID_ENUMERATION",
        0x7800000d => "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION",
        0x78000010 => "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME",
        0x78000011 => "ZE_RESULT_ERROR_INVALID_KERNEL_NAME",
        0x7800001b => "ZE_RESULT_WARNING_ACTION_REQUIRED",
        0x7ffffffe => "ZE_RESULT_ERROR_UNKNOWN",
        _ => $"0x{r:X8}"
    };

    public static string FpFlags(uint flags)
    {
        if (flags == 0) return "none";
        var parts = new List<string>();
        void Add(uint bit, string name) { if ((flags & bit) != 0) parts.Add(name); }
        Add(FP_DENORM, "denorm");
        Add(FP_INF_NAN, "inf/nan");
        Add(FP_ROUND_TO_NEAREST, "rne");
        Add(FP_ROUND_TO_ZERO, "rtz");
        Add(FP_ROUND_TO_INF, "rti");
        Add(FP_FMA, "fma");
        Add(FP_ROUNDED_DIVIDE_SQRT, "div/sqrt");
        Add(FP_SOFT_FLOAT, "soft");
        return string.Join("|", parts);
    }
}

// Note: sequential layouts rely on the .NET marshaller's natural alignment (x64),
// which matches the C layout verified from ze_api.h v1.28.2.

[StructLayout(LayoutKind.Sequential)]
internal struct ZeDriverProperties
{
    public uint stype;
    public IntPtr pNext;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public byte[] uuid;
    public uint driverVersion;

    public static ZeDriverProperties New()
    {
        var s = new ZeDriverProperties { uuid = new byte[16] };
        s.stype = L0Constants.ST_DRIVER_PROPERTIES;
        return s;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeDriverExtensionProperties
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = L0Constants.MAX_EXTENSION_NAME)]
    public byte[] name;
    public uint version;

    public static ZeDriverExtensionProperties New() => new() { name = new byte[L0Constants.MAX_EXTENSION_NAME] };
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeDeviceProperties
{
    public uint stype;
    public IntPtr pNext;
    public uint type;
    public uint vendorId;
    public uint deviceId;
    public uint flags;
    public uint subdeviceId;
    public uint coreClockRate;
    public ulong maxMemAllocSize;
    public uint maxHardwareContexts;
    public uint maxCommandQueuePriority;
    public uint numThreadsPerEU;
    public uint physicalEUSimdWidth;
    public uint numEUsPerSubslice;
    public uint numSubslicesPerSlice;
    public uint numSlices;
    public ulong timerResolution;
    public uint timestampValidBits;
    public uint kernelTimestampValidBits;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public byte[] uuid;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = L0Constants.MAX_DEVICE_NAME)]
    public byte[] name;

    public static ZeDeviceProperties New()
    {
        var s = new ZeDeviceProperties { uuid = new byte[16], name = new byte[L0Constants.MAX_DEVICE_NAME] };
        s.stype = L0Constants.ST_DEVICE_PROPERTIES;
        return s;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeDeviceModuleProperties
{
    public uint stype;
    public IntPtr pNext;
    public uint spirvVersionSupported;
    public uint flags;
    public uint fp16flags;
    public uint fp32flags;
    public uint fp64flags;
    public uint maxArgumentsSize;
    public uint printfBufferSize;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public byte[] nativeKernelSupported;

    public static ZeDeviceModuleProperties New()
    {
        var s = new ZeDeviceModuleProperties { nativeKernelSupported = new byte[16] };
        s.stype = L0Constants.ST_DEVICE_MODULE_PROPERTIES;
        return s;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeCommandQueueGroupProperties
{
    public uint stype;
    public IntPtr pNext;
    public uint flags;
    public UIntPtr maxMemoryFillPatternSize;
    public uint numQueues;

    public static ZeCommandQueueGroupProperties New()
    {
        var s = new ZeCommandQueueGroupProperties();
        s.stype = L0Constants.ST_COMMAND_QUEUE_GROUP_PROPERTIES;
        return s;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeContextDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeCommandQueueDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint ordinal;
    public uint index;
    public uint flags;
    public uint mode;
    public uint priority;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeCommandListDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint commandQueueGroupOrdinal;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeDeviceMemAllocDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint flags;
    public uint ordinal;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeHostMemAllocDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeModuleDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint format;
    public UIntPtr inputSize;
    public IntPtr pInputModule;
    public IntPtr pBuildFlags;
    public IntPtr pConstants;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeKernelDesc
{
    public uint stype;
    public IntPtr pNext;
    public uint flags;
    public IntPtr pKernelName;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ZeGroupCount
{
    public uint groupCountX;
    public uint groupCountY;
    public uint groupCountZ;
}

/// <summary>
/// Loads ze_loader.dll and resolves the Level Zero entry points used by the probe.
/// The loader re-exports the whole core API, so kernel32 LoadLibrary + GetProcAddress
/// is sufficient (no static DllImport needed).
/// </summary>
internal sealed class L0Loader : IDisposable
{
    private const uint LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800;
    private readonly IntPtr handle;

    public L0Loader()
    {
        handle = LoadLibraryEx("ze_loader.dll", IntPtr.Zero, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (handle == IntPtr.Zero)
            throw new InvalidOperationException(
                $"ze_loader.dll not found or failed to load (Win32 error 0x{Marshal.GetLastWin32Error():X8}). " +
                "Level Zero requires an Intel graphics driver with the Level Zero runtime installed.");
    }

    public T GetProc<T>(string name) where T : Delegate
    {
        IntPtr fn = GetProcAddress(handle, name);
        if (fn == IntPtr.Zero)
            throw new EntryPointNotFoundException($"{name} is not exported by ze_loader.dll");
        return Marshal.GetDelegateForFunctionPointer<T>(fn);
    }

    public void Dispose()
    {
        if (handle != IntPtr.Zero)
            FreeLibrary(handle);
    }

    public static string AnsiString(byte[] bytes)
    {
        int end = Array.IndexOf(bytes, (byte)0);
        if (end < 0) end = bytes.Length;
        return Encoding.ASCII.GetString(bytes, 0, end);
    }

    public static string SpirvVersion(uint v) => v == 0 ? "0 (SPIR-V unsupported)" : $"{v >> 16}.{(v >> 8) & 0xFF}";

    public static IntPtr PtrToStructure<T>(ref T value) where T : struct
    {
        IntPtr p = Marshal.AllocHGlobal(Marshal.SizeOf<T>());
        Marshal.StructureToPtr(value, p, false);
        return p;
    }

    public static T StructureFromPtr<T>(IntPtr p) where T : struct => Marshal.PtrToStructure<T>(p);

    [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true)]
    private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
}