using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using Nivara.GpuProbe.Kernels;
using static Nivara.GpuProbe.D3d12Check;

namespace Nivara.GpuProbe.D3d12;

/// <summary>
/// Hand-rolled D3D12 compute leg: pure P/Invoke against the inbox d3d12.dll / dxgi.dll /
/// d3dcompiler_47.dll — no packages, no COM wrappers. Drives the complete compute pipeline
/// on the Arc 140T iGPU (device → queue → command allocator/list → HLSL <c>sm_5_1</c> DXBC →
/// root signature → shader-visible descriptor heap → committed upload/default/readback
/// buffers → PSO → dispatch → fence → readback) and gates the three production SmolLM
/// kernels against the CPU leg exactly like the SYCL leg does. This proves managed .NET
/// can drive DX12 compute directly, and is the fallback backend if IGC's OpenCL/Level Zero
/// frontend stays broken (it bypasses IGC only at the driver's DX12 frontend — same UMD
/// binary, different entry point).
///
/// Every vtable slot, GUID and struct constant below was read from this machine's SDK
/// d3d12.h, not from memory — memory-derived GUIDs were wrong in three of seven cases, and
/// D3d12Check's original slot-15 CheckFeatureSupport was a false positive (it hit
/// GetDescriptorHandleIncrementSize, a UINT getter that never writes the probe buffer, and
/// read back a pre-written value). Device vtable slots are +4 over the naive method list:
/// ID3D12Device inherits ID3D12Object directly (slots 3–6), so GetNodeCount starts at 7.
/// Struct layouts are validated against the C sizes at leg start (<see cref="ValidateLayouts"/>).
/// </summary>
internal static class D3d12Compute
{
    private static readonly Guid IID_ID3D12CommandQueue = new("0ec870a6-5d7e-4c22-8cfc-5baae07616ed");
    private static readonly Guid IID_ID3D12CommandAllocator = new("6102dee4-af59-4b09-b999-b44d73f09b24");
    private static readonly Guid IID_ID3D12GraphicsCommandList = new("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455");
    private static readonly Guid IID_ID3D12DescriptorHeap = new("8efb471d-616c-4f49-90f7-127bb763fa51");
    private static readonly Guid IID_ID3D12Fence = new("0a753dcf-c4d8-4b91-adf6-be5a60d95a76");
    private static readonly Guid IID_ID3D12RootSignature = new("c54a6b66-72df-4ee8-8be5-a946a1429214");
    private static readonly Guid IID_ID3D12PipelineState = new("765a30f3-f624-4c6f-a828-ace948622445");
    private static readonly Guid IID_ID3D12Resource = new("696442be-a72e-4059-bc79-5b5c98040fad");
    private static readonly Guid IID_ID3D12Device = new("189819f1-1db6-4b57-be54-1821339b85f7");

    // D3D12 constants (values read from the SDK d3d12.h on this machine).
    private const int HeapTypeDefault = 1, HeapTypeUpload = 2, HeapTypeReadback = 3;
    private const int ResourceStateUnorderedAccess = 0x8;
    private const int ResourceStateCopyDest = 0x400;
    private const int ResourceStateCopySource = 0x800;
    private const int ResourceStateGenericRead = 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800; // GENERIC_READ
    private const int ResourceDimensionBuffer = 1;
    private const int TextureLayoutRowMajor = 1;
    private const int FormatUnknown = 0;
    private const int ResourceFlagsAllowUnorderedAccess = 0x4;
    private const int CommandListTypeDirect = 0;
    private const int DescriptorHeapTypeCbvSrvUav = 0;
    private const int DescriptorHeapFlagShaderVisible = 0x1;
    private const int DescriptorRangeTypeSrv = 0, DescriptorRangeTypeUav = 1;
    private const int RootParameterTypeDescriptorTable = 0;
    private const int ShaderVisibilityAll = 0;
    private const int RootSignatureFlagNone = 0;
    private const int RootSignatureVersion1 = 1;
    private const int SrvDimensionBuffer = 1, UavDimensionBuffer = 1;
    private const int DefaultShader4ComponentMapping = 0x1688; // D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0,1,2,3)
    private const int BarrierTypeTransition = 0;
    private const int FenceFlagNone = 0;
    private const int PipelineStateFlagNone = 0;
    private const uint D3DCompileOptimizationLevel3 = 0x4000;
    private const uint InfiniteWait = 0xFFFFFFFF;
    private const int EInvalidArg = unchecked((int)0x80070057);

    // --- Native structs (LayoutKind.Sequential = the C declaration; field order is authoritative) ---

    [StructLayout(LayoutKind.Sequential)]
    private struct HeapProperties { public int Type; public int CpuPageProperty; public int MemoryPoolPreference; public int CreationNodeMask; public int VisibleNodeMask; }

    [StructLayout(LayoutKind.Sequential)]
    private struct SampleDesc { public int Count; public int Quality; }

    [StructLayout(LayoutKind.Sequential)]
    private struct ResourceDesc
    {
        public int Dimension;              // D3D12_RESOURCE_DIMENSION_BUFFER
        public long Alignment;             // 0 (auto)
        public long Width;                 // buffer size in bytes
        public int Height;                 // 1
        public ushort DepthOrArraySize;    // 1
        public ushort MipLevels;           // 1
        public int Format;                 // DXGI_FORMAT_UNKNOWN
        public SampleDesc SampleDesc;      // {1, 0}
        public int Layout;                 // D3D12_TEXTURE_LAYOUT_ROW_MAJOR
        public int Flags;                  // 0 or ALLOW_UNORDERED_ACCESS
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DescriptorRange
    {
        public int RangeType;
        public int NumDescriptors;
        public int BaseShaderRegister;
        public int RegisterSpace;
        public int OffsetInDescriptorsFromTableStart;
    }

    // RootParameter carries a union in C; explicit pads reproduce the union layout
    // (ParameterType | {NumDescriptorRanges, pDescriptorRanges} | ShaderVisibility).
    [StructLayout(LayoutKind.Sequential)]
    private struct RootParameter
    {
        public int ParameterType;              // 0
        public int Pad0;                       // 4  (union alignment)
        public int NumDescriptorRanges;        // 8
        public int Pad1;                       // 12 (union member: 4 + padded ptr)
        public IntPtr PDescriptorRanges;       // 16
        public int ShaderVisibility;           // 24
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RootSignatureDesc
    {
        public int NumParameters;              // 0
        public int Pad0;                       // 4
        public IntPtr PParameters;             // 8
        public int NumStaticSamplers;          // 16
        public int Pad1;                       // 20
        public IntPtr PStaticSamplers;         // 24
        public int Flags;                      // 32
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DescriptorHeapDesc { public int Type; public int NumDescriptors; public int Flags; public int NodeMask; }

    [StructLayout(LayoutKind.Sequential)]
    private struct CpuDescriptorHandle { public long Ptr; }

    [StructLayout(LayoutKind.Sequential)]
    private struct GpuDescriptorHandle { public long Ptr; }

    [StructLayout(LayoutKind.Sequential)]
    private struct CommandQueueDesc { public int Type; public int Priority; public int Flags; public int NodeMask; }

    [StructLayout(LayoutKind.Sequential)]
    private struct ShaderResourceViewDesc
    {
        public int Format;                     // 0
        public int ViewDimension;              // 4
        public int Shader4ComponentMapping;    // 8
        public int Pad0;                       // 12
        public long FirstElement;              // 16
        public int NumElements;                // 24
        public int StructureByteStride;        // 28
        public int Flags;                      // 32
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct UnorderedAccessViewDesc
    {
        public int Format;                     // 0
        public int ViewDimension;              // 4
        public long FirstElement;              // 8
        public int NumElements;                // 16
        public int StructureByteStride;        // 20
        public long CounterOffsetInBytes;      // 24
        public int Flags;                      // 32
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ResourceBarrierDesc
    {
        public int Type;                       // 0
        public int Flags;                      // 4
        public IntPtr PResource;               // 8
        public int Subresource;                // 16
        public int StateBefore;                // 20
        public int StateAfter;                 // 24
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ComputePipelineStateDesc
    {
        public IntPtr PRootSignature;          // 0
        public IntPtr PShaderBytecode;         // 8
        public long ShaderBytecodeLength;      // 16
        public int NodeMask;                   // 24
        public int Pad0;                       // 28
        public IntPtr PCachedBlob;             // 32
        public long CachedBlobSizeInBytes;     // 40
        public int Flags;                      // 48
    }

    // --- COM delegates (filtered through D3d12Check.Vtable; slots in method tabs) ---

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateCommandQueue(IntPtr device, IntPtr pDesc, ref Guid riid, out IntPtr ppQueue);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateCommandAllocator(IntPtr device, int type, ref Guid riid, out IntPtr ppAllocator);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateComputePipelineState(IntPtr device, IntPtr pDesc, ref Guid riid, out IntPtr ppPipelineState);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateCommandList(IntPtr device, uint nodeMask, int type, IntPtr allocator, IntPtr initialState, ref Guid riid, out IntPtr ppCommandList);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateDescriptorHeap(IntPtr device, IntPtr pDesc, ref Guid riid, out IntPtr ppHeap);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate uint GetDescriptorHandleIncrementSize(IntPtr device, int heapType);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateRootSignature(IntPtr device, uint nodeMask, IntPtr pBlob, nuint blobLength, ref Guid riid, out IntPtr ppRootSignature);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void CreateShaderResourceView(IntPtr device, IntPtr resource, IntPtr pDesc, CpuDescriptorHandle destination);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void CreateUnorderedAccessView(IntPtr device, IntPtr resource, IntPtr pCounterResource, IntPtr pDesc, CpuDescriptorHandle destination);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateCommittedResource(IntPtr device, IntPtr pHeapProperties, int heapFlags, IntPtr pDesc, int initialState, IntPtr pClearValue, ref Guid riid, out IntPtr ppResource);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateFence(IntPtr device, long initialValue, int flags, ref Guid riid, out IntPtr ppFence);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int ExecuteCommandLists(IntPtr queue, uint numLists, IntPtr ppCommandLists);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int SignalQueue(IntPtr queue, IntPtr fence, long value);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int ResetCommandAllocator(IntPtr allocator);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CloseCommandList(IntPtr list);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int ResetCommandList(IntPtr list, IntPtr allocator, IntPtr initialState);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void Dispatch(IntPtr list, uint x, uint y, uint z);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void CopyBufferRegion(IntPtr list, IntPtr dst, long dstOffset, IntPtr src, long srcOffset, long numBytes);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void SetPipelineState(IntPtr list, IntPtr pipelineState);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void ResourceBarrier(IntPtr list, uint numBarriers, IntPtr pBarriers);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void SetDescriptorHeaps(IntPtr list, uint numHeaps, IntPtr ppHeaps);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void SetComputeRootSignature(IntPtr list, IntPtr rootSignature);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void SetComputeRootDescriptorTable(IntPtr list, uint rootParameterIndex, GpuDescriptorHandle gpuHandle);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void GetCpuDescriptorHandleForHeapStart(IntPtr heap, out long handle);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void GetGpuDescriptorHandleForHeapStart(IntPtr heap, out long handle);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetDevice(IntPtr obj, ref Guid riid, out IntPtr device);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void GetHeapDesc(IntPtr heap, out DescriptorHeapDesc desc);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int SetEventOnCompletion(IntPtr fence, long value, IntPtr hEvent);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int MapResource(IntPtr resource, uint subresource, IntPtr pReadRange, out IntPtr ppData);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void UnmapResource(IntPtr resource, uint subresource, IntPtr pWrittenRange);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate IntPtr GetBufferPointer(IntPtr blob);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate nuint GetBufferSize(IntPtr blob);

    [DllImport("d3dcompiler_47.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
    private static extern int D3DCompile(
        IntPtr pSrcData, nuint srcDataSize, string pSourceName,
        IntPtr pDefines, IntPtr pInclude,
        string pEntrypoint, string pTarget,
        uint flags1, uint flags2,
        out IntPtr ppCode, out IntPtr ppErrorMsgs);

    [DllImport("d3d12.dll", CallingConvention = CallingConvention.StdCall)]
    private static extern int D3D12SerializeRootSignature(IntPtr pRootSignature, int version, out IntPtr ppBlob, out IntPtr ppErrorBlob);

    [DllImport("kernel32.dll")]
    private static extern IntPtr CreateEvent(IntPtr lpEventAttributes, bool bManualReset, bool bInitialState, IntPtr lpName);

    [DllImport("kernel32.dll")]
    private static extern uint WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr hObject);

    /// <summary>Entry point behind the <c>dx12</c> CLI mode: availability check +
    /// the compute leg through the standard gate table.</summary>
    public static int Run()
    {
        Console.WriteLine();
        return KernelGate.Run(KernelFixtures.Generate(), "DX12 (hand-rolled)", RunLeg);
    }

    /// <summary>The DX12 leg over the fixed fixture set: dot16 + silu + gemv through
    /// HLSL <c>sm_5_1</c> compute shaders, gated later by <see cref="KernelGate"/>.</summary>
    public static LegResults? RunLeg(KernelFixtures fixtures)
    {
        Console.WriteLine("--- DX12 compute leg (hand-rolled P/Invoke: d3d12.dll + dxgi.dll + d3dcompiler_47.dll) ---");

        string[] layoutErrors = ValidateLayouts();
        if (layoutErrors.Length > 0)
        {
            foreach (string error in layoutErrors)
                Console.WriteLine($"  [FAIL] layout check: {error}");
            return null;
        }

        int hr = D3d12Check.TryCreateDevice(out IntPtr device, out string deviceName);
        if (hr < 0 || device == IntPtr.Zero)
        {
            Console.WriteLine($"  [FAIL] D3D12CreateDevice: hr=0x{hr:X8} (DX12 compute unavailable)");
            return null;
        }

        try
        {
            Console.WriteLine($"  device: {deviceName}");
            using var ctx = CreateContext(device);
            if (ctx is null)
                return null;

            var dot16 = RunKernel(ctx, GemvKernels.Dot16(KernelFixtures.Dot16Length), "CsDot16",
                ConcatPacked(fixtures.Dot16A, fixtures.Dot16B), outCount: 1,
                groupsX: 1, out double dot16Us);
            var silu = RunKernel(ctx, GemvKernels.Silu(KernelFixtures.HiddenSize), "CsSilu",
                GemvKernels.PackBf16(fixtures.SiluX), outCount: KernelFixtures.HiddenSize,
                groupsX: GroupsFor(KernelFixtures.HiddenSize), out double siluUs);
            var gemv = RunKernel(ctx, GemvKernels.Gemv(KernelFixtures.IntermediateSize, KernelFixtures.HiddenSize), "CsGemv",
                ConcatPacked(fixtures.GemvW, fixtures.GemvX), outCount: KernelFixtures.IntermediateSize,
                groupsX: GroupsFor(KernelFixtures.IntermediateSize), out double gemvUs);

            if (dot16 is null || silu is null || gemv is null)
                return null;

            return new LegResults(dot16[0], silu, gemv, dot16Us, siluUs, gemvUs);
        }
        finally
        {
            D3d12Check.Release(device);
        }
    }

    private static uint GroupsFor(int elements) => (uint)((elements + GemvKernels.ThreadsPerGroup - 1) / GemvKernels.ThreadsPerGroup);

    private static uint[] ConcatPacked(ReadOnlySpan<BFloat16> first, ReadOnlySpan<BFloat16> second)
    {
        var all = new BFloat16[first.Length + second.Length];
        first.CopyTo(all);
        second.CopyTo(all.AsSpan(first.Length));
        return GemvKernels.PackBf16(all);
    }

    /// <summary>Everything shared across the three kernel dispatches: one device, queue,
    /// command allocator/list, fence+event, root signature, and the shader-visible
    /// descriptor heap (slot 0 = SRV t0, slot 1 = UAV u0). Fences advance monotonically.</summary>
    private sealed class ComputeContext : IDisposable
    {
        public IntPtr Device;
        public IntPtr Queue;
        public IntPtr Allocator;
        public IntPtr CommandList;
        public IntPtr Fence;
        public IntPtr Event;
        public IntPtr RootSignature;
        public IntPtr DescriptorHeap;
        public uint DescriptorIncrement;
        public CpuDescriptorHandle SrvCpuSlot;
        public CpuDescriptorHandle UavCpuSlot;
        public GpuDescriptorHandle SrvGpuSlot;
        public GpuDescriptorHandle UavGpuSlot;
        public long FenceValue;

        public long NextFenceValue() => ++FenceValue;

        public void Dispose()
        {
            D3d12Check.Release(CommandList);
            D3d12Check.Release(Allocator);
            D3d12Check.Release(Queue);
            D3d12Check.Release(DescriptorHeap);
            D3d12Check.Release(RootSignature);
            D3d12Check.Release(Fence);
            if (Event != IntPtr.Zero) CloseHandle(Event);
        }
    }

    private static ComputeContext? CreateContext(IntPtr device)
    {
        IntPtr queue = IntPtr.Zero, allocator = IntPtr.Zero, list = IntPtr.Zero,
               fence = IntPtr.Zero, rootSig = IntPtr.Zero, heap = IntPtr.Zero, evt = IntPtr.Zero;
        using var scratch = new NativeScratch();

        ComputeContext? Fail(string what, int hresult)
        {
            Console.WriteLine($"  [FAIL] {what}: hr=0x{hresult:X8}");
            D3d12Check.Release(queue);
            D3d12Check.Release(allocator);
            D3d12Check.Release(list);
            D3d12Check.Release(fence);
            D3d12Check.Release(rootSig);
            D3d12Check.Release(heap);
            if (evt != IntPtr.Zero) CloseHandle(evt);
            return null;
        }

        var queueDesc = new CommandQueueDesc { Type = CommandListTypeDirect };
        Guid cmdQueueIid = IID_ID3D12CommandQueue;
        int hr = Vtable<CreateCommandQueue>(device, 8)(device, scratch.Copy(ref queueDesc), ref cmdQueueIid, out queue);
        if (hr < 0) return Fail("CreateCommandQueue", hr);

        Guid allocIid = IID_ID3D12CommandAllocator;
        hr = Vtable<CreateCommandAllocator>(device, 9)(device, CommandListTypeDirect, ref allocIid, out allocator);
        if (hr < 0) return Fail("CreateCommandAllocator", hr);

        Guid listIid = IID_ID3D12GraphicsCommandList;
        hr = Vtable<CreateCommandList>(device, 12)(device, 0, CommandListTypeDirect, allocator, IntPtr.Zero, ref listIid, out list);
        if (hr < 0) return Fail("CreateCommandList", hr);

        Guid fenceIid = IID_ID3D12Fence;
        hr = Vtable<CreateFence>(device, 36)(device, 0, FenceFlagNone, ref fenceIid, out fence);
        if (hr < 0) return Fail("CreateFence", hr);

        evt = CreateEvent(IntPtr.Zero, false, false, IntPtr.Zero);
        if (evt == IntPtr.Zero) return Fail("CreateEvent", -1);

        hr = BuildRootSignature(device, out rootSig);
        if (hr < 0) return Fail("D3D12SerializeRootSignature/CreateRootSignature", hr);

        var heapDesc = new DescriptorHeapDesc { Type = DescriptorHeapTypeCbvSrvUav, NumDescriptors = 2, Flags = DescriptorHeapFlagShaderVisible, NodeMask = 0 };
        Guid heapIid = IID_ID3D12DescriptorHeap;
        hr = Vtable<CreateDescriptorHeap>(device, 14)(device, scratch.Copy(ref heapDesc), ref heapIid, out heap);
        if (hr < 0) return Fail("CreateDescriptorHeap", hr);

        // Runtime identity self-check: the heap must report back the creating
        // device and its own descriptor-layout description. Catches vtable
        // mis-slotting loudly instead of crashing or corrupting later.
        Guid devIid = IID_ID3D12Device;
        int identityHr = Vtable<GetDevice>(heap, 7)(heap, ref devIid, out IntPtr heapDevice);
        if (identityHr < 0 || heapDevice != device)
            return Fail($"heap->GetDevice identity (0x{(long)heapDevice:X} vs 0x{(long)device:X})", identityHr);
        Vtable<GetHeapDesc>(heap, 8)(heap, out var actualDesc);
        if (actualDesc.Type != DescriptorHeapTypeCbvSrvUav
            || actualDesc.NumDescriptors != 2
            || actualDesc.Flags != DescriptorHeapFlagShaderVisible)
            return Fail("heap->GetDesc identity", EInvalidArg);

        uint increment = Vtable<GetDescriptorHandleIncrementSize>(device, 15)(device, DescriptorHeapTypeCbvSrvUav);

        // Descriptor handles are returned through a hidden pointer (like GetDesc):
        // the callers below must pass an out slot, not read a register return.
        Vtable<GetGpuDescriptorHandleForHeapStart>(heap, 10)(heap, out long gpuStart);
        Vtable<GetCpuDescriptorHandleForHeapStart>(heap, 9)(heap, out long cpuStart);

        return new ComputeContext
        {
            Device = device,
            Queue = queue, Allocator = allocator, CommandList = list,
            Fence = fence, Event = evt, RootSignature = rootSig, DescriptorHeap = heap,
            DescriptorIncrement = increment,
            SrvCpuSlot = new CpuDescriptorHandle { Ptr = cpuStart },
            UavCpuSlot = new CpuDescriptorHandle { Ptr = cpuStart + increment },
            SrvGpuSlot = new GpuDescriptorHandle { Ptr = gpuStart },
            UavGpuSlot = new GpuDescriptorHandle { Ptr = gpuStart + increment }
        };
    }

    /// <summary>Root signature (version 1.0): parameter 0 = SRV table at t0, parameter 1 =
    /// UAV table at u0. Serializes to a blob with d3d12.dll and creates the signature.</summary>
    private static int BuildRootSignature(IntPtr device, out IntPtr rootSig)
    {
        rootSig = IntPtr.Zero;
        using var scratch = new NativeScratch();

        var rangeSrv = new DescriptorRange { RangeType = DescriptorRangeTypeSrv, NumDescriptors = 1, BaseShaderRegister = 0, RegisterSpace = 0, OffsetInDescriptorsFromTableStart = 0 };
        var rangeUav = new DescriptorRange { RangeType = DescriptorRangeTypeUav, NumDescriptors = 1, BaseShaderRegister = 0, RegisterSpace = 0, OffsetInDescriptorsFromTableStart = 0 };
        IntPtr pRangeSrv = scratch.Copy(ref rangeSrv);
        IntPtr pRangeUav = scratch.Copy(ref rangeUav);

        IntPtr pParams = scratch.Alloc(2 * Marshal.SizeOf<RootParameter>());
        var param0 = new RootParameter { ParameterType = RootParameterTypeDescriptorTable, NumDescriptorRanges = 1, PDescriptorRanges = pRangeSrv, ShaderVisibility = ShaderVisibilityAll };
        var param1 = new RootParameter { ParameterType = RootParameterTypeDescriptorTable, NumDescriptorRanges = 1, PDescriptorRanges = pRangeUav, ShaderVisibility = ShaderVisibilityAll };
        Marshal.StructureToPtr(param0, pParams, false);
        Marshal.StructureToPtr(param1, IntPtr.Add(pParams, Marshal.SizeOf<RootParameter>()), false);

        var rootDesc = new RootSignatureDesc { NumParameters = 2, PParameters = pParams, NumStaticSamplers = 0, PStaticSamplers = IntPtr.Zero, Flags = RootSignatureFlagNone };
        IntPtr pRootDesc = scratch.Copy(ref rootDesc);

        int hr = D3D12SerializeRootSignature(pRootDesc, RootSignatureVersion1, out IntPtr blob, out IntPtr errorBlob);
        if (hr >= 0)
        {
            IntPtr code = Vtable<GetBufferPointer>(blob, 3)(blob);
            nuint codeSize = Vtable<GetBufferSize>(blob, 4)(blob);
            Guid rsIid = IID_ID3D12RootSignature;
            hr = Vtable<CreateRootSignature>(device, 16)(device, 0, code, codeSize, ref rsIid, out rootSig);
            D3d12Check.Release(blob);
        }
        else
        {
            D3d12Check.Release(errorBlob);
        }
        return hr;
    }

    /// <summary>Runs one kernel: compile → PSO → buffers (upload input / default output /
    /// readback) → SRV+UAV views → 1 warmup + 3 timed [record, execute, wait] passes →
    /// readback f32. Returns the results and the best-of-3 kernel-only time in µs.</summary>
    private static float[]? RunKernel(ComputeContext ctx, string hlsl, string entry, uint[] input, int outCount, uint groupsX, out double minUs)
    {
        minUs = 0;
        using var scratch = new NativeScratch();
        using var com = new ComScope();

        byte[] srcBytes = System.Text.Encoding.UTF8.GetBytes(hlsl);
        IntPtr pSrc = scratch.AllocBytes(srcBytes);
        int hr = D3DCompile(pSrc, (nuint)srcBytes.Length, "kernel.hlsl", IntPtr.Zero, IntPtr.Zero, entry, "cs_5_1", D3DCompileOptimizationLevel3, 0, out IntPtr codeBlob, out IntPtr errorBlob);
        if (hr < 0)
        {
            Console.WriteLine($"  [FAIL] {entry} D3DCompile: hr=0x{hr:X8}");
            if (errorBlob != IntPtr.Zero)
            {
                Console.WriteLine($"         {ReadBlobText(errorBlob).ReplaceLineEndings(" | ")}");
                D3d12Check.Release(errorBlob);
            }
            return null;
        }
        IntPtr codePtr = Vtable<GetBufferPointer>(codeBlob, 3)(codeBlob);
        nuint codeSize = Vtable<GetBufferSize>(codeBlob, 4)(codeBlob);

        var psoDesc = new ComputePipelineStateDesc
        {
            PRootSignature = ctx.RootSignature,
            PShaderBytecode = codePtr,
            ShaderBytecodeLength = (long)codeSize,
            NodeMask = 0, PCachedBlob = IntPtr.Zero, CachedBlobSizeInBytes = 0, Flags = PipelineStateFlagNone
        };
        Guid psoIid = IID_ID3D12PipelineState;
        hr = Vtable<CreateComputePipelineState>(ctx.Device, 11)(ctx.Device, scratch.Copy(ref psoDesc), ref psoIid, out IntPtr pso);
        D3d12Check.Release(codeBlob);
        if (hr < 0)
        {
            Console.WriteLine($"  [FAIL] {entry} CreateComputePipelineState: hr=0x{hr:X8}");
            return null;
        }
        com.Add(pso);

        uint inputBytes = (uint)(input.Length * sizeof(uint));
        uint outputBytes = (uint)(outCount * sizeof(float));
        IntPtr inputBuf = CreateBuffer(ctx.Device, HeapTypeUpload, ResourceStateGenericRead, inputBytes, allowUav: false);
        IntPtr outputBuf = CreateBuffer(ctx.Device, HeapTypeDefault, ResourceStateUnorderedAccess, outputBytes, allowUav: true);
        IntPtr readback = CreateBuffer(ctx.Device, HeapTypeReadback, ResourceStateCopyDest, outputBytes, allowUav: false);
        if (inputBuf == IntPtr.Zero || outputBuf == IntPtr.Zero || readback == IntPtr.Zero)
        {
            Console.WriteLine($"  [FAIL] {entry} CreateCommittedResource failed");
            return null;
        }
        com.Add(inputBuf);
        com.Add(outputBuf);
        com.Add(readback);

        var srvDesc = new ShaderResourceViewDesc
        {
            Format = FormatUnknown, ViewDimension = SrvDimensionBuffer, Shader4ComponentMapping = DefaultShader4ComponentMapping,
            FirstElement = 0, NumElements = input.Length, StructureByteStride = 4, Flags = 0
        };
        Vtable<CreateShaderResourceView>(ctx.Device, 18)(ctx.Device, inputBuf, scratch.Copy(ref srvDesc), ctx.SrvCpuSlot);

        var uavDesc = new UnorderedAccessViewDesc
        {
            Format = FormatUnknown, ViewDimension = UavDimensionBuffer,
            FirstElement = 0, NumElements = outCount, StructureByteStride = 4, CounterOffsetInBytes = 0, Flags = 0
        };
        Vtable<CreateUnorderedAccessView>(ctx.Device, 19)(ctx.Device, outputBuf, IntPtr.Zero, scratch.Copy(ref uavDesc), ctx.UavCpuSlot);

        int mapHr = Vtable<MapResource>(inputBuf, 8)(inputBuf, 0, IntPtr.Zero, out IntPtr mapped);
        if (mapHr < 0)
        {
            Console.WriteLine($"  [FAIL] {entry} Map(upload): hr=0x{mapHr:X8}");
            return null;
        }
        CopyUint32sToNative(mapped, input);
        Vtable<UnmapResource>(inputBuf, 9)(inputBuf, 0, IntPtr.Zero);

        IntPtr pHeapSlot = scratch.Alloc(IntPtr.Size);
        Marshal.WriteIntPtr(pHeapSlot, ctx.DescriptorHeap);
        IntPtr pListSlot = scratch.Alloc(IntPtr.Size);
        Marshal.WriteIntPtr(pListSlot, ctx.CommandList);

        var barrier = new ResourceBarrierDesc { Type = BarrierTypeTransition, Flags = 0, PResource = outputBuf, Subresource = 0, StateBefore = ResourceStateUnorderedAccess, StateAfter = ResourceStateCopySource };
        IntPtr pBarrier = scratch.Copy(ref barrier);

        var timer = new Stopwatch();
        double bestUs = double.MaxValue;
        for (int pass = 0; pass < 4; pass++)
        {
            timer.Restart();

            Vtable<ResetCommandAllocator>(ctx.Allocator, 8)(ctx.Allocator);
            Vtable<ResetCommandList>(ctx.CommandList, 10)(ctx.CommandList, ctx.Allocator, IntPtr.Zero);
            Vtable<SetComputeRootSignature>(ctx.CommandList, 29)(ctx.CommandList, ctx.RootSignature);
            Vtable<SetPipelineState>(ctx.CommandList, 25)(ctx.CommandList, pso);
            Vtable<SetDescriptorHeaps>(ctx.CommandList, 28)(ctx.CommandList, 1, pHeapSlot);
            Vtable<SetComputeRootDescriptorTable>(ctx.CommandList, 31)(ctx.CommandList, 0, ctx.SrvGpuSlot);
            Vtable<SetComputeRootDescriptorTable>(ctx.CommandList, 31)(ctx.CommandList, 1, ctx.UavGpuSlot);
            Vtable<Dispatch>(ctx.CommandList, 14)(ctx.CommandList, groupsX, 1, 1);
            Vtable<ResourceBarrier>(ctx.CommandList, 26)(ctx.CommandList, 1, pBarrier);
            Vtable<CopyBufferRegion>(ctx.CommandList, 15)(ctx.CommandList, readback, 0, outputBuf, 0, outputBytes);
            Vtable<CloseCommandList>(ctx.CommandList, 9)(ctx.CommandList);

            Vtable<ExecuteCommandLists>(ctx.Queue, 10)(ctx.Queue, 1, pListSlot);
            long value = ctx.NextFenceValue();
            Vtable<SignalQueue>(ctx.Queue, 14)(ctx.Queue, ctx.Fence, value);
            Vtable<SetEventOnCompletion>(ctx.Fence, 9)(ctx.Fence, value, ctx.Event);
            WaitForSingleObject(ctx.Event, InfiniteWait);

            timer.Stop();
            if (pass > 0)
            {
                double us = timer.Elapsed.TotalMilliseconds * 1000.0;
                if (us < bestUs)
                    bestUs = us;
            }
        }
        minUs = bestUs;

        int readHr = Vtable<MapResource>(readback, 8)(readback, 0, IntPtr.Zero, out IntPtr readPtr);
        if (readHr < 0)
        {
            Console.WriteLine($"  [FAIL] {entry} Map(readback): hr=0x{readHr:X8}");
            return null;
        }
        float[] result = KernelFixtures.ReadF32(readPtr, outCount);
        Vtable<UnmapResource>(readback, 9)(readback, 0, IntPtr.Zero);
        return result;
    }

    private static IntPtr CreateBuffer(IntPtr device, int heapType, int initialState, uint widthBytes, bool allowUav)
    {
        using var scratch = new NativeScratch();
        var heapProps = new HeapProperties { Type = heapType, CpuPageProperty = 0, MemoryPoolPreference = 0, CreationNodeMask = 1, VisibleNodeMask = 1 };
        var resDesc = new ResourceDesc
        {
            Dimension = ResourceDimensionBuffer,
            Alignment = 0,
            Width = widthBytes,
            Height = 1,
            DepthOrArraySize = 1,
            MipLevels = 1,
            Format = FormatUnknown,
            SampleDesc = new SampleDesc { Count = 1, Quality = 0 },
            Layout = TextureLayoutRowMajor,
            Flags = allowUav ? ResourceFlagsAllowUnorderedAccess : 0
        };
        Guid resIid = IID_ID3D12Resource;
        int hr = Vtable<CreateCommittedResource>(device, 27)(
            device, scratch.Copy(ref heapProps), 0, scratch.Copy(ref resDesc), initialState, IntPtr.Zero, ref resIid, out IntPtr resource);
        return hr >= 0 ? resource : IntPtr.Zero;
    }

    private static void CopyUint32sToNative(IntPtr destination, ReadOnlySpan<uint> values)
    {
        byte[] bytes = MemoryMarshal.AsBytes(values).ToArray();
        Marshal.Copy(bytes, 0, destination, bytes.Length);
    }

    private static string ReadBlobText(IntPtr blob)
    {
        IntPtr p = Vtable<GetBufferPointer>(blob, 3)(blob);
        nuint size = Vtable<GetBufferSize>(blob, 4)(blob);
        var bytes = new byte[(int)size];
        Marshal.Copy(p, bytes, 0, bytes.Length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    /// <summary>Validates every marshaled struct against the C size it must match; a
    /// mismatch here would corrupt pipeline arguments silently.</summary>
    private static string[] ValidateLayouts()
    {
        var errors = new List<string>();
        void Check<T>(int expected) where T : struct
        {
            int actual = Marshal.SizeOf<T>();
            if (actual != expected)
                errors.Add($"{typeof(T).Name} Marshal.SizeOf={actual}, C requires {expected}");
        }
        Check<HeapProperties>(20);
        Check<ResourceDesc>(56);
        Check<DescriptorRange>(20);
        Check<RootParameter>(32);
        Check<RootSignatureDesc>(40);
        Check<DescriptorHeapDesc>(16);
        Check<CpuDescriptorHandle>(8);
        Check<GpuDescriptorHandle>(8);
        Check<CommandQueueDesc>(16);
        Check<ShaderResourceViewDesc>(40);
        Check<UnorderedAccessViewDesc>(40);
        Check<ResourceBarrierDesc>(32);
        Check<ComputePipelineStateDesc>(56);
        return errors.ToArray();
    }

    /// <summary>Native memory scratch pad: frees every block at scope exit.</summary>
    private sealed class NativeScratch : IDisposable
    {
        private readonly List<IntPtr> blocks = new();

        public IntPtr Copy<T>(ref T value) where T : struct
        {
            IntPtr p = Marshal.AllocHGlobal(Marshal.SizeOf<T>());
            Marshal.StructureToPtr(value, p, false);
            blocks.Add(p);
            return p;
        }

        public IntPtr Alloc(int bytes)
        {
            IntPtr p = Marshal.AllocHGlobal(bytes);
            blocks.Add(p);
            return p;
        }

        public IntPtr AllocBytes(byte[] bytes)
        {
            IntPtr p = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, p, bytes.Length);
            blocks.Add(p);
            return p;
        }

        public void Dispose() => blocks.ForEach(Marshal.FreeHGlobal);
    }

    /// <summary>Releases COM objects created for one kernel run (PSO + buffers).</summary>
    private sealed class ComScope : IDisposable
    {
        private readonly List<IntPtr> owned = new();

        public void Add(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
                owned.Add(ptr);
        }

        public void Dispose()
        {
            foreach (IntPtr ptr in owned)
                D3d12Check.Release(ptr);
        }
    }
}