using System.Runtime.InteropServices;

namespace Nivara.GpuProbe;

/// <summary>
/// DirectX 12 availability check (option d of the SmolLM-GPU investigation): pure
/// P/Invoke against the inbox dxgi.dll/d3d12.dll — no packages. Proves whether a
/// managed DX12 compute backend (e.g. ComputeSharp) can target this machine's Intel
/// Arc 140T iGPU, which would bypass the buggy IGC OpenCL/Level Zero frontend
/// entirely, and reports the highest D3D feature level and HLSL shader model it
/// reaches. Enumerates every DXGI adapter and probes each in turn.
/// </summary>
internal static class D3d12Check
{
    private static readonly Guid IID_IDXGIFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");
    private static readonly Guid IID_ID3D12Device = new("189819f1-1db6-4b57-be54-1821339b85f7");

    private const uint Level_11_0 = 0xb000, Level_11_1 = 0xb100, Level_12_0 = 0xc000, Level_12_1 = 0xc100, Level_12_2 = 0xc200;
    private const int D3D12FeatureShaderModel = 7; // D3D12_FEATURE_SHADER_MODEL (verified: 7, not 6)
    private const int DxgiErrorNotFound = unchecked((int)0x887A0002U);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct AdapterDesc
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public long DedicatedVideoMemory;
        public long DedicatedSystemMemory;
        public long SharedSystemMemory;
        public long AdapterLuid;
    }

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int EnumAdapters(IntPtr factory, uint adapter, out IntPtr adapterOut);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetAdapterDesc(IntPtr adapter, IntPtr desc);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CheckFeatureSupport(IntPtr device, int feature, IntPtr data, nuint size);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int ReleaseCom(IntPtr obj);

    [DllImport("dxgi.dll", CallingConvention = CallingConvention.StdCall)]
    private static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr factory);

    [DllImport("d3d12.dll", CallingConvention = CallingConvention.StdCall)]
    private static extern int D3D12CreateDevice(IntPtr adapter, uint minFeatureLevel, ref Guid riid, out IntPtr device);

    public static int Run()
    {
        Console.WriteLine("--- DirectX 12 availability (dx12) ---");
        int failures = 0;
        IntPtr factory = IntPtr.Zero;
        try
        {
            int hr = CreateFactory(out factory);
            if (hr < 0)
            {
                Console.WriteLine($"  [FAIL] CreateDXGIFactory1: hr=0x{hr:X8}");
                return 1;
            }

            var enumAdapters = Vtable<EnumAdapters>(factory, 7);
            uint index = 0;
            IntPtr adapter = IntPtr.Zero;
            while (true)
            {
                hr = enumAdapters(factory, index, out adapter);
                if (hr == DxgiErrorNotFound)
                    break;
                if (hr < 0)
                {
                    Console.WriteLine($"  [FAIL] EnumAdapters[{index}]: hr=0x{hr:X8}");
                    failures++;
                    break;
                }

                Console.WriteLine($"  Adapter[{index}]");
                IntPtr descPtr = Marshal.AllocHGlobal(Marshal.SizeOf<AdapterDesc>());
                try
                {
                    hr = Vtable<GetAdapterDesc>(adapter, 8)(adapter, descPtr);
                    if (hr >= 0)
                    {
                        AdapterDesc desc = Marshal.PtrToStructure<AdapterDesc>(descPtr);
                        Console.WriteLine($"    {desc.Description}  PCI {desc.VendorId:X4}:{desc.DeviceId:X4}  " +
                                          $"VRAM {desc.DedicatedVideoMemory / 1048576.0:F0} MiB  " +
                                          $"sys {desc.DedicatedSystemMemory / 1048576.0:F0} MiB  " +
                                          $"shared {desc.SharedSystemMemory / 1048576.0:F0} MiB");
                    }
                    else
                    {
                        Console.WriteLine($"    [FAIL] GetDesc: hr=0x{hr:X8}");
                        failures++;
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(descPtr);
                }

                if (!ProbeDevice(adapter))
                    failures++;

                Release(adapter);
                adapter = IntPtr.Zero;
                index++;
            }

            Console.WriteLine(failures == 0 && index > 0
                ? "  dx12: OK — device creation + shader model queried on all adapters"
                : $"  dx12: {failures} failure(s)");
        }
        finally
        {
            Release(factory);
        }

        return failures;
    }

    private static bool ProbeDevice(IntPtr adapter)
    {
        Guid deviceIid = IID_ID3D12Device;
        foreach (uint level in new[] { Level_12_2, Level_12_1, Level_12_0, Level_11_1, Level_11_0 })
        {
            int hr = D3D12CreateDevice(adapter, level, ref deviceIid, out IntPtr device);
            if (hr < 0)
                continue;

            Console.WriteLine($"    D3D12CreateDevice: OK  feature level {FeatureLevel(level)}");
            IntPtr smPtr = Marshal.AllocHGlobal(4);
            try
            {
                // Prove the call is live before trusting any reply: a bogus feature
                // value is rejected with E_INVALIDARG by the real CheckFeatureSupport
                // and by no other device vtable slot. (Slot 13 = CheckFeatureSupport:
                // ID3D12Device inherits ID3D12Object directly, whose four methods at
                // slots 3-6 sit beneath it, so GetNodeCount starts at 7. The original
                // slot-15 call silently hit GetDescriptorHandleIncrementSize — a UINT
                // getter that writes nothing to our buffer — and "reported" the
                // pre-written value unchanged.)
                int bogusFeature = unchecked((int)0x7FFFFFFF);
                Marshal.WriteInt32(smPtr, 0);
                int hrBogus = Vtable<CheckFeatureSupport>(device, 13)(device, bogusFeature, smPtr, (nuint)4);
                bool live = hrBogus == unchecked((int)0x80070057);

                // Real probe: request shader model 6.8 (the highest clipboard the
                // runtime accepts); it clamps down to the driver's max.
                Marshal.WriteInt32(smPtr, 0x68); // D3D_SHADER_MODEL_6_8
                int smHr = Vtable<CheckFeatureSupport>(device, 13)(device, D3D12FeatureShaderModel, smPtr, (nuint)4);
                int supported = Marshal.ReadInt32(smPtr);
                bool verified = live && smHr >= 0 && supported >= 0x50 && supported <= 0x68;
                Console.WriteLine(verified
                    ? $"    highest shader model: {ShaderModel((uint)supported)} (feature-validated; bogus request rejected with E_INVALIDARG)"
                    : $"    [FAIL] CheckFeatureSupport(shader model): hr=0x{smHr:X8}, bogus-hr=0x{hrBogus:X8}, replied 0x{(uint)supported:X2} — slot/feature probe failed");
            }
            finally
            {
                Marshal.FreeHGlobal(smPtr);
                Release(device);
            }
            return true;
        }

        Console.WriteLine("    D3D12CreateDevice: FAILED for every feature level 11_0..12_2");
        return false;
    }

    internal static T Vtable<T>(IntPtr obj, int slot) where T : Delegate
    {
        IntPtr vtbl = Marshal.ReadIntPtr(obj);
        IntPtr fn = Marshal.ReadIntPtr(vtbl, slot * IntPtr.Size);
        return Marshal.GetDelegateForFunctionPointer<T>(fn);
    }

    internal static void Release(IntPtr obj)
    {
        if (obj != IntPtr.Zero)
            Vtable<ReleaseCom>(obj, 2)(obj);
    }

    /// <summary>Creates the DXGI factory used by both the availability check and
    /// the compute leg.</summary>
    internal static int CreateFactory(out IntPtr factory)
    {
        Guid factoryIid = IID_IDXGIFactory1;
        return CreateDXGIFactory1(ref factoryIid, out factory);
    }

    /// <summary>
    /// Creates a D3D12 device on the first DXGI adapter that accepts one (DXGI
    /// enumerates the primary/Intel adapter first, mirroring the SYCL leg's device-0
    /// selection), trying feature levels 12_0/11_1/11_0. Returns 0 on success; the
    /// device is owned by the caller (release via <see cref="Release"/>). Used by the
    /// DX12 compute leg so it shares the availability check's enumeration plumbing.
    /// </summary>
    internal static int TryCreateDevice(out IntPtr device, out string description)
    {
        device = IntPtr.Zero;
        description = "";
        int hr = CreateFactory(out IntPtr factory);
        if (hr < 0)
            return hr;
        try
        {
            var enumAdapters = Vtable<EnumAdapters>(factory, 7);
            uint index = 0;
            while (true)
            {
                hr = enumAdapters(factory, index, out IntPtr adapter);
                if (hr == DxgiErrorNotFound)
                    return DxgiErrorNotFound;
                if (hr < 0)
                    return hr;
                try
                {
                    Guid deviceIid = IID_ID3D12Device;
                    foreach (uint level in new[] { Level_12_2, Level_12_1, Level_12_0, Level_11_1, Level_11_0 })
                    {
                        int dhr = D3D12CreateDevice(adapter, level, ref deviceIid, out IntPtr dev);
                        if (dhr >= 0)
                        {
                            device = dev;
                            description = ReadAdapterDescription(adapter);
                            return 0;
                        }
                    }
                }
                finally
                {
                    Release(adapter);
                }
                index++;
            }
        }
        finally
        {
            Release(factory);
        }
    }

    private static string ReadAdapterDescription(IntPtr adapter)
    {
        IntPtr descPtr = Marshal.AllocHGlobal(Marshal.SizeOf<AdapterDesc>());
        try
        {
            return Vtable<GetAdapterDesc>(adapter, 8)(adapter, descPtr) >= 0
                ? Marshal.PtrToStructure<AdapterDesc>(descPtr).Description
                : "";
        }
        finally
        {
            Marshal.FreeHGlobal(descPtr);
        }
    }

    private static string FeatureLevel(uint level) => level switch
    {
        Level_11_0 => "11_0",
        Level_11_1 => "11_1",
        Level_12_0 => "12_0",
        Level_12_1 => "12_1",
        Level_12_2 => "12_2",
        _ => $"0x{level:X4}"
    };

    private static string ShaderModel(uint sm) => sm switch
    {
        0x51 => "5.1",
        0x60 => "6.0",
        0x61 => "6.1",
        0x62 => "6.2",
        0x63 => "6.3",
        0x64 => "6.4",
        0x65 => "6.5",
        0x66 => "6.6",
        0x67 => "6.7",
        0x68 => "6.8",
        _ => $"unknown 0x{sm:X2}"
    };
}