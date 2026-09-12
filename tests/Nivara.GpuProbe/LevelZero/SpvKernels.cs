using System.Text;

namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// Hand-authored SPIR-V modules targeting the "Kernel" capability + OpenCL memory
/// model (the same shape clang emits for OpenCL C), so the Intel Level Zero driver's
/// built-in compiler (IGC) can build them without any local SPIR-V toolchain.
///
/// Opcode/operand lists below follow the SPIR-V 1.2 specification; every value used
/// (MemoryModel.OpenCL=2, ExecutionModel.Kernel=6, Capability.Addresses=4,
/// Capability.Kernel=6, StorageClass.CrossWorkgroup=5/Input=1, Decoration.BuiltIn=11,
/// BuiltIn.GlobalInvocationId=28) was double-checked against the spec.
/// </summary>
internal static class SpvKernels
{
    private const uint SPIRV_MAGIC = 0x07230203;

    /// <summary>SPIR-V 1.0 version word (= 0x00010000).</summary>
    public const uint Version10 = 0x00010000;

    /// <summary>SPIR-V 1.2 version word (= 0x00010200).</summary>
    public const uint Version12 = 0x00010200;

    public static byte[] ToBytes(uint[] words)
    {
        var result = new byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++)
        {
            result[i * 4] = (byte)(words[i] & 0xFF);
            result[i * 4 + 1] = (byte)((words[i] >> 8) & 0xFF);
            result[i * 4 + 2] = (byte)((words[i] >> 16) & 0xFF);
            result[i * 4 + 3] = (byte)((words[i] >> 24) & 0xFF);
        }

        return result;
    }

    /// <summary>
    /// Parallel add: 256 work items (LocalSize 256, one work-group). This driver's
    /// IGC crashes (access violation) on any OpAccessChain / GEP, so per-element
    /// scatter is impossible; the kernel instead proves real SIMD parallelism with
    /// a device-side atomic reduction: every work item reads its lane index from
    /// BuiltIn.GlobalInvocationId and does OpAtomicIAdd on the single counter
    /// argument. Expected result for 256 lanes: 0+1+...+255 = 32640. The atomic
    /// targets the kernel argument pointer directly — no access chain involved.
    /// </summary>
    public static uint[] AddParallel(uint spirvVersion = Version12, uint localSize = 256)
    {
        uint lanes = localSize == 0 ? 256 : localSize;
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 11, "add_parallel", 9);
        w.Op(16, 11, 17, lanes, 1, 1);   // OpExecutionMode %11 LocalSize <lanes> 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(11, "add_parallel");      // OpName %11 "add_parallel"
        w.Name(9, "gid");                // OpName %9 "gid"
        w.Op(19, 1);                     // %1 = void
        w.Op(21, 2, 32, 0);              // %2 = uint
        w.Op(23, 3, 2, 3);               // %3 = v3uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, uint>
        w.Op(32, 5, 1, 3);               // %5 = ptr<Input, v3uint>
        w.Op(33, 6, 1, 4);               // %6 = fn(void, ptr)
        w.Op(43, 2, 7, 1);               // %7 = const uint Scope Device
        w.Op(43, 2, 8, 0x200);            // %8 = const uint memsem CrossWorkgroupMemory|Relaxed
        w.Op(71, 9, 11, 28);             // OpDecorate %9 BuiltIn GlobalInvocationId
        w.Op(59, 5, 9, 1);               // %9 = var Input (builtin global id)
        w.Op(54, 1, 11, 0, 6);           // OpFunction %1 None %6 %11
        w.Op(55, 4, 12);                 // %12 = param counter
        w.Op(248, 13);                   // entry block
        w.Op(61, 3, 14, 9);              // %14 = OpLoad v3uint %9
        w.Op(81, 2, 15, 14, 0);          // %15 = OpCompositeExtract uint %14 0
        w.Op(234, 2, 16, 12, 7, 8, 15);  // %16 = OpAtomicIAdd counter Device Relaxed %15
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 17);
    }

    /// <summary>
    /// Serial add: one work item runs 1,000,000 iterations of acc += *a using
    /// OpPhi value-flow (loop counter + accumulator as header phis, no Private
    /// variables and no access chains — Private OpVariables mis-link on this
    /// driver with "undefined reference to gVar", access chains access-violate).
    /// Result *c = acc is written via a direct store to the kernel-argument
    /// pointer. Host sets a[0] = 1.0f, so c[0] must equal 1,000,000.0 exactly.
    /// </summary>
    public static uint[] AddLoop(uint spirvVersion = Version12)
    {
        const uint N = 1_000_000;
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 11, "add_loop");
        w.Op(16, 11, 17, 1, 1, 1);       // OpExecutionMode %11 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(11, "add_loop");          // OpName %11 "add_loop"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(20, 4);                     // %4 = bool
        w.Op(32, 5, 5, 2);               // %5 = ptr<CrossWorkgroup, float>
        w.Op(33, 6, 1, 5, 5, 5);         // %6 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 7, 0);               // %7 = const uint 0
        w.Op(43, 3, 8, 1);               // %8 = const uint 1
        w.Op(43, 3, 9, N);               // %9 = const uint N
        w.Op(43, 2, 10, 0);              // %10 = const float 0.0 (acc init)
        w.Op(54, 1, 11, 0, 6);           // OpFunction %1 None %6 %11
        w.Op(55, 5, 12);                 // %12 = param a
        w.Op(55, 5, 13);                 // %13 = param b
        w.Op(55, 5, 14);                 // %14 = param c
        w.Op(248, 15);                   // entry block
        w.Op(249, 16);                   // branch to header
        w.Op(248, 16);                   // header block
        w.Op(246, 19, 18, 0);            // OpLoopMerge exit(19) continue(18) None
        w.Op(245, 3, 20, 7, 15, 23, 18); // %20 = phi i: (0 from 15), (23 from 18)
        w.Op(245, 2, 21, 10, 15, 26, 18);// %21 = phi acc: (0.0 from 15), (26 from 18)
        w.Op(176, 4, 22, 20, 9);         // %22 = i < N
        w.Op(250, 22, 17, 19);           // branch %22 ? body(17) : exit(19)
        w.Op(248, 17);                   // body block
        w.Op(61, 2, 25, 12);             // %25 = load float %12 (*a)
        w.Op(129, 2, 26, 21, 25);        // %26 = acc + *a
        w.Op(249, 18);                   // branch to continue
        w.Op(248, 18);                   // continue block
        w.Op(128, 3, 23, 20, 8);         // %23 = i + 1
        w.Op(249, 16);                   // branch back to header
        w.Op(248, 19);                   // exit block
        w.Op(62, 14, 21);                // store *c = acc
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 27);
    }

    /// <summary>
    /// Absolute minimal OpenCL-style kernel: no args, empty body (entry + return).
    /// Used to bisect IGC build failures — if this module also crashes, the problem
    /// is in the prologue (capabilities/memory model/source), not the body logic.
    /// </summary>
    public static uint[] Minimal(uint spirvVersion = Version12)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 5, "minimal");
        w.Op(16, 5, 17, 1, 1, 1);        // OpExecutionMode %5 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(5, "minimal");            // OpName %5 "minimal"
        w.Op(19, 1);                     // %1 = void
        w.Op(33, 2, 1);                  // %2 = fn(void)
        w.Op(54, 1, 5, 0, 2);            // OpFunction %1 None %2 %5
        w.Op(248, 6);                    // label %6
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 7);
    }

    /// <summary>
    /// Bisection variants used to isolate which construct crashes IGC's SPIR-V
    /// frontend: each adds exactly one layer beyond <see cref="Minimal"/>.
    /// </summary>
    public static (string Name, uint[] Module)[] BisectVariants(uint spirvVersion) =>
    [
        ("params: 3 CrossWorkgroup ptr params, empty body", ParamsModule(spirvVersion)),
        ("builtin: Input v3uint gid load + extract, no params", BuiltinModule(spirvVersion)),
        ("straight: params + c[0] = a[0] + b[0] (no loop/phi)", StraightModule(spirvVersion)),
        ("scalar params: float x3, empty body", ScalarParamsModule(spirvVersion)),
        ("generic ptr params: ptr<Generic,float> x3, empty body", GenericPtrParamsModule(spirvVersion)),
        ("consts: params + 2 float consts + FAdd (unused), return", ConstsModule(spirvVersion)),
        ("chain: params + accesschain a[0] (unused), return", ChainModule(spirvVersion)),
        ("chain_u64: accesschain with uint64 index 0", ChainUInt64Module(spirvVersion)),
        ("chain_int: accesschain with signed int32 index 0", ChainIntModule(spirvVersion)),
        ("global_chain: accesschain off module-scope global var", GlobalChainModule(spirvVersion)),
        ("loop_nogep: OpPhi loop (no accesschain, no private vars) -> *c = acc", LoopNoGepModule(spirvVersion)),
        ("ibchain: params + inbounds accesschain a[0] (unused)", InBoundsChainModule(spirvVersion)),
        ("ptrchain: params + ptraccesschain a[0] (unused)", PtrChainModule(spirvVersion)),
        ("chain_decor: chain + Restrict/NoAlias decors on params", ChainDecorModule(spirvVersion)),
        ("direct: *c = *a + *b via direct loads/stores (no accesschain)", DirectModule(spirvVersion)),
        ("load: params + accesschain + load float, return", LoadModule(spirvVersion)),
        ("store: params + store const to c[0], return", StoreModule(spirvVersion)),
    ];

    private static uint[] ParamsModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 5, "params");
        w.Op(16, 5, 17, 1, 1, 1);        // OpExecutionMode %5 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(5, "params");             // OpName %5 "params"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(32, 3, 5, 2);               // %3 = ptr<CrossWorkgroup, float>
        w.Op(33, 4, 1, 3, 3, 3);         // %4 = fn(void, ptr, ptr, ptr)
        w.Op(54, 1, 5, 0, 4);            // OpFunction %1 None %4 %5
        w.Op(55, 3, 6);                  // %6 = param a
        w.Op(55, 3, 7);                  // %7 = param b
        w.Op(55, 3, 8);                  // %8 = param c
        w.Op(248, 9);                    // entry block
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 10);
    }

    private static uint[] BuiltinModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 10, "builtin", 9);
        w.Op(16, 10, 17, 1, 1, 1);       // OpExecutionMode %10 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(10, "builtin");           // OpName %10 "builtin"
        w.Name(9, "gid");                // OpName %9 "gid"
        w.Op(19, 1);                     // %1 = void
        w.Op(21, 2, 32, 0);              // %2 = uint
        w.Op(23, 3, 2, 3);               // %3 = v3uint
        w.Op(32, 4, 1, 3);               // %4 = ptr<Input, v3uint>
        w.Op(33, 5, 1);                  // %5 = fn(void)
        w.Op(43, 2, 6, 0);               // %6 = const uint 0
        w.Op(59, 4, 9, 1);               // %9 = var Input
        w.Op(71, 9, 11, 28);             // OpDecorate %9 BuiltIn GlobalInvocationId
        w.Op(54, 1, 10, 0, 5);           // OpFunction %1 None %5 %10
        w.Op(248, 11);                   // entry block
        w.Op(61, 3, 12, 9);              // %12 = OpLoad v3uint %9
        w.Op(81, 2, 13, 12, 0);          // %13 = OpCompositeExtract uint %12 0
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    private static uint[] StraightModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 6, "straight");
        w.Op(16, 6, 17, 1, 1, 1);        // OpExecutionMode %6 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(6, "straight");           // OpName %6 "straight"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 12, 0);              // %12 = const uint 0
        w.Op(54, 1, 6, 0, 5);            // OpFunction %1 None %5 %6
        w.Op(55, 4, 7);                  // %7 = param a
        w.Op(55, 4, 8);                  // %8 = param b
        w.Op(55, 4, 9);                  // %9 = param c
        w.Op(248, 10);                   // entry block
        w.Op(65, 4, 13, 7, 12);          // %13 = OpAccessChain a[0]
        w.Op(65, 4, 14, 8, 12);          // %14 = OpAccessChain b[0]
        w.Op(61, 2, 15, 13);             // %15 = load float %13
        w.Op(61, 2, 16, 14);             // %16 = load float %14
        w.Op(129, 2, 17, 15, 16);        // %17 = %15 + %16
        w.Op(65, 4, 18, 9, 12);          // %18 = OpAccessChain c[0]
        w.Op(62, 18, 17);                // store %17 -> %18
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 19);
    }

    private static uint[] ScalarParamsModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 4, "scalar");
        w.Op(16, 4, 17, 1, 1, 1);        // OpExecutionMode %4 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(4, "scalar");             // OpName %4 "scalar"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(33, 3, 1, 2, 2, 2);         // %3 = fn(void, float, float, float)
        w.Op(54, 1, 4, 0, 3);            // OpFunction %1 None %3 %4
        w.Op(55, 2, 5);                  // %5 = param
        w.Op(55, 2, 6);                  // %6 = param
        w.Op(55, 2, 7);                  // %7 = param
        w.Op(248, 8);                    // entry block
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 9);
    }

    private static uint[] GenericPtrParamsModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 5, "generic");
        w.Op(16, 5, 17, 1, 1, 1);        // OpExecutionMode %5 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(5, "generic");            // OpName %5 "generic"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(32, 3, 8, 2);               // %3 = ptr<Generic, float>
        w.Op(33, 4, 1, 3, 3, 3);         // %4 = fn(void, ptr, ptr, ptr)
        w.Op(54, 1, 5, 0, 4);            // OpFunction %1 None %4 %5
        w.Op(55, 3, 6);                  // %6 = param a
        w.Op(55, 3, 7);                  // %7 = param b
        w.Op(55, 3, 8);                  // %8 = param c
        w.Op(248, 9);                    // entry block
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 10);
    }

    /// <summary>
    /// A real serial loop with NO access chains and NO private variables: loop
    /// counter and accumulator flow through OpPhi value edges (header-block phi
    /// with incoming values from entry and continue). Body runs N iterations of
    /// acc += 1.0f, then a direct store into the c kernel argument (*c = acc).
    /// Private OpVariables mis-link on this driver ("undefined reference to
    /// gVar"), so the phi form is the only loop shape that can build.
    /// </summary>
    private static uint[] LoopNoGepModule(uint spirvVersion)
    {
        const uint N = 1_000_000;
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 11, "loop_nogep");
        w.Op(16, 11, 17, 1, 1, 1);       // OpExecutionMode %11 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(11, "loop_nogep");        // OpName %11 "loop_nogep"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(20, 4);                     // %4 = bool
        w.Op(32, 5, 5, 2);               // %5 = ptr<CrossWorkgroup, float>
        w.Op(33, 6, 1, 5, 5, 5);         // %6 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 7, 0);               // %7 = const uint 0
        w.Op(43, 3, 8, 1);               // %8 = const uint 1
        w.Op(43, 3, 9, N);               // %9 = const uint N
        w.Op(43, 2, 10, 0x3F800000);     // %10 = const float 1.0
        w.Op(54, 1, 11, 0, 6);           // OpFunction %1 None %6 %11
        w.Op(55, 5, 12);                 // %12 = param a
        w.Op(55, 5, 13);                 // %13 = param b
        w.Op(55, 5, 14);                 // %14 = param c
        w.Op(248, 15);                   // entry block
        w.Op(249, 16);                   // branch to header
        w.Op(248, 16);                   // header block
        w.Op(246, 19, 18, 0);            // OpLoopMerge exit(19) continue(18) None
        w.Op(245, 3, 20, 7, 15, 23, 18); // %20 = phi i: (0 from 15), (23 from 18)
        w.Op(245, 2, 21, 10, 15, 24, 18);// %21 = phi acc: (1.0 from 15), (24 from 18)
        w.Op(176, 4, 22, 20, 9);         // %22 = i < N
        w.Op(250, 22, 17, 19);           // branch %22 ? body(17) : exit(19)
        w.Op(248, 17);                   // body block
        w.Op(249, 18);                   // branch to continue
        w.Op(248, 18);                   // continue block
        w.Op(128, 3, 23, 20, 8);         // %23 = i + 1
        w.Op(129, 2, 24, 21, 10);        // %24 = acc + 1.0
        w.Op(249, 16);                   // branch back to header
        w.Op(248, 19);                   // exit block
        w.Op(62, 14, 21);                // store *c = acc (direct store)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 25);
    }

    /// <summary>
    /// Params + two float constants + an OpFAdd whose result is unused. Isolates
    /// whether constants and/or float arithmetic crash IGC's OpenCL frontend.
    /// </summary>
    private static uint[] ConstsModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 9, "consts");
        w.Op(16, 9, 17, 1, 1, 1);        // OpExecutionMode %9 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(9, "consts");             // OpName %9 "consts"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 2, 6, 0x3F800000);      // %6 = const float 1.0
        w.Op(43, 2, 7, 0x40000000);      // %7 = const float 2.0
        w.Op(54, 1, 9, 0, 5);            // OpFunction %1 None %5 %9
        w.Op(55, 4, 10);                 // %10 = param a
        w.Op(55, 4, 11);                 // %11 = param b
        w.Op(55, 4, 12);                 // %12 = param c
        w.Op(248, 13);                   // entry block
        w.Op(129, 2, 8, 6, 7);           // %8 = %6 + %7 (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    /// <summary>
    /// Params + a single OpAccessChain a[0] whose result is unused. Isolates whether
    /// access chains crash IGC's OpenCL frontend.
    /// </summary>
    private static uint[] ChainModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "chain");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "chain");              // OpName %7 "chain"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(65, 4, 12, 8, 6);           // %12 = OpAccessChain a[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// Access chain with a 64-bit (uint64) constant index, matching OpenCL's
    /// size_t-based array indexing on 64-bit targets. Tests whether the crash is
    /// specific to 32-bit index types in IGC's SPIR-V frontend.
    /// </summary>
    private static uint[] ChainUInt64Module(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 11);                    // OpCapability Int64
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 8, "chain_u64");
        w.Op(16, 8, 17, 1, 1, 1);        // OpExecutionMode %8 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(8, "chain_u64");          // OpName %8 "chain_u64"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint (32)
        w.Op(21, 4, 64, 0);              // %4 = uint64
        w.Op(32, 5, 5, 2);               // %5 = ptr<CrossWorkgroup, float>
        w.Op(33, 6, 1, 5, 5, 5);         // %6 = fn(void, ptr, ptr, ptr)
        w.Op(43, 4, 7, 0);               // %7 = const uint64 0
        w.Op(54, 1, 8, 0, 6);            // OpFunction %1 None %6 %8
        w.Op(55, 5, 9);                  // %9 = param a
        w.Op(55, 5, 10);                 // %10 = param b
        w.Op(55, 5, 11);                 // %11 = param c
        w.Op(248, 12);                   // entry block
        w.Op(65, 5, 13, 9, 7);           // %13 = OpAccessChain a[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    /// <summary>
    /// Access chain with a signed int32 constant index (clang's default "int"
    /// index type). Tests whether index signedness matters to IGC.
    /// </summary>
    private static uint[] ChainIntModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "chain_int");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "chain_int");          // OpName %7 "chain_int"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 1);              // %3 = int (signed 32)
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const int 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(65, 4, 12, 8, 6);           // %12 = OpAccessChain a[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// Access chain whose base is a module-scope CrossWorkgroup global variable
    /// (no kernel parameters). Tests whether the crash is specific to access
    /// chains rooted at kernel argument pointers.
    /// </summary>
    private static uint[] GlobalChainModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 8, "global_chain");
        w.Op(16, 8, 17, 1, 1, 1);        // OpExecutionMode %8 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(8, "global_chain");       // OpName %8 "global_chain"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1);                  // %5 = fn(void)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(59, 4, 7, 5);               // %7 = var CrossWorkgroup (module global)
        w.Op(54, 1, 8, 0, 5);            // OpFunction %1 None %5 %8
        w.Op(248, 9);                    // entry block
        w.Op(65, 4, 10, 7, 6);           // %10 = OpAccessChain %7[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 11);
    }

    /// <summary>
    /// Same as <see cref="ChainModule"/> but with OpInBoundsAccessChain (66), whose
    /// code path in IGC's OpenCL frontend may differ from OpAccessChain.
    /// </summary>
    private static uint[] InBoundsChainModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "ibchain");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "ibchain");            // OpName %7 "ibchain"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(66, 4, 12, 8, 6);           // %12 = OpInBoundsAccessChain a[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// Same as <see cref="ChainModule"/> but with OpPtrAccessChain (67) + the
    /// VariablePointers capability. Different opcode path in IGC's frontend.
    /// </summary>
    private static uint[] PtrChainModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 4442);                  // OpCapability VariablePointers
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "ptrchain");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "ptrchain");           // OpName %7 "ptrchain"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(67, 4, 12, 8, 6, 6);        // %12 = OpPtrAccessChain a[0] elems=0 (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// Same as <see cref="ChainModule"/> but with clang-style Restrict + NoAlias
    /// decorations on every pointer parameter, matching real OpenCL kernel output.
    /// Tests whether the missing decorations, not the access chain itself, crash IGC.
    /// </summary>
    private static uint[] ChainDecorModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "chain_decor");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "chain_decor");        // OpName %7 "chain_decor"
        w.Op(71, 8, 19);                 // OpDecorate %8 Restrict
        w.Op(71, 8, 38, 4);              // OpDecorate %8 FuncParamAttr NoAlias
        w.Op(71, 9, 19);                 // OpDecorate %9 Restrict
        w.Op(71, 9, 38, 4);              // OpDecorate %9 FuncParamAttr NoAlias
        w.Op(71, 10, 19);                // OpDecorate %10 Restrict
        w.Op(71, 10, 38, 4);             // OpDecorate %10 FuncParamAttr NoAlias
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(65, 4, 12, 8, 6);           // %12 = OpAccessChain a[0] (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// Params + direct loads/stores with no access chain: *c = *a + *b. This is
    /// exactly what clang emits for pointer dereference with no index — isolates
    /// whether IGC can lower loads/stores from CrossWorkgroup kernel args at all.
    /// </summary>
    private static uint[] DirectModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 6, "direct");
        w.Op(16, 6, 17, 1, 1, 1);        // OpExecutionMode %6 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(6, "direct");             // OpName %6 "direct"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(54, 1, 6, 0, 5);            // OpFunction %1 None %5 %6
        w.Op(55, 4, 7);                  // %7 = param a
        w.Op(55, 4, 8);                  // %8 = param b
        w.Op(55, 4, 9);                  // %9 = param c
        w.Op(248, 10);                   // entry block
        w.Op(61, 2, 11, 7);              // %11 = load float %7 (a)
        w.Op(61, 2, 12, 8);              // %12 = load float %8 (b)
        w.Op(129, 2, 13, 11, 12);        // %13 = %11 + %12
        w.Op(62, 9, 13);                 // store %13 -> %9 (c)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    /// <summary>
    /// Params + OpAccessChain a[0] + OpLoad float. Isolates whether loads from
    /// CrossWorkgroup pointers crash IGC's OpenCL frontend.
    /// </summary>
    private static uint[] LoadModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "load");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "load");               // OpName %7 "load"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(54, 1, 7, 0, 5);            // OpFunction %1 None %5 %7
        w.Op(55, 4, 8);                  // %8 = param a
        w.Op(55, 4, 9);                  // %9 = param b
        w.Op(55, 4, 10);                 // %10 = param c
        w.Op(248, 11);                   // entry block
        w.Op(65, 4, 12, 8, 6);           // %12 = OpAccessChain a[0]
        w.Op(61, 2, 13, 12);             // %13 = load float %12 (unused)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    /// <summary>
    /// Params + OpAccessChain c[0] + OpStore of a float constant. Isolates whether
    /// stores into CrossWorkgroup pointers crash IGC's OpenCL frontend.
    /// </summary>
    private static uint[] StoreModule(uint spirvVersion)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 8, "store");
        w.Op(16, 8, 17, 1, 1, 1);        // OpExecutionMode %8 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(8, "store");              // OpName %8 "store"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, float>
        w.Op(33, 5, 1, 4, 4, 4);         // %5 = fn(void, ptr, ptr, ptr)
        w.Op(43, 3, 6, 0);               // %6 = const uint 0
        w.Op(43, 2, 7, 0x40400000);      // %7 = const float 3.0
        w.Op(54, 1, 8, 0, 5);            // OpFunction %1 None %5 %8
        w.Op(55, 4, 9);                  // %9 = param a
        w.Op(55, 4, 10);                 // %10 = param b
        w.Op(55, 4, 11);                 // %11 = param c
        w.Op(248, 12);                   // entry block
        w.Op(65, 4, 13, 11, 6);          // %13 = OpAccessChain c[0]
        w.Op(62, 13, 7);                 // store %7 -> %13
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 14);
    }

    /// <summary>
    /// BF16 -&gt; F32 conversion using Intel's native instruction OpConvertBF16ToFINTEL
    /// (SPV_INTEL_bfloat16_conversion, capability BFloat16ConversionINTEL = 6115).
    /// The BF16 value is a 16-bit integer bit pattern — exactly the layout of
    /// System.Numerics.BFloat16 (SafeTensorsLoader keeps BF16 weights that way) —
    /// held in CrossWorkgroup memory and OpLoad-ed directly (no access chain).
    /// Build success answers whether IGC honors the ZE_extension_bfloat16_conversions
    /// contract, which requires accepting modules that declare this capability.
    /// </summary>
    public static uint[] Bf16Native(uint spirvVersion = Version10)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(17, 6115);                  // OpCapability BFloat16ConversionINTEL
        w.Extension("SPV_INTEL_bfloat16_conversion");
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 7, "bf16_native");
        w.Op(16, 7, 17, 1, 1, 1);        // OpExecutionMode %7 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(7, "bf16_native");        // OpName %7 "bf16_native"
        w.Name(8, "a");                  // OpName %8 "a"
        w.Name(9, "c");                  // OpName %9 "c"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 16, 0);              // %3 = u16 (bfloat16 bit pattern storage)
        w.Op(32, 4, 5, 3);               // %4 = ptr<CrossWorkgroup, u16>
        w.Op(32, 5, 5, 2);               // %5 = ptr<CrossWorkgroup, float>
        w.Op(33, 6, 1, 4, 5);            // %6 = fn(void, ptr u16, ptr f32)
        w.Op(54, 1, 7, 0, 6);            // OpFunction %1 None %6 %7
        w.Op(55, 4, 8);                  // %8 = param a (u16 BF16 bits)
        w.Op(55, 5, 9);                  // %9 = param c (f32 out)
        w.Op(248, 10);                   // entry block
        w.Op(61, 3, 11, 8);              // %11 = OpLoad u16 %8
        w.Op(6117, 2, 12, 11);           // %12 = OpConvertBF16ToFINTEL float %11
        w.Op(62, 9, 12);                 // store %12 -> %9 (c)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 13);
    }

    /// <summary>
    /// BF16 -&gt; F32 widening with only safe-subset instructions (no Intel extension):
    /// OpUConvert (113) widens the loaded u16 to u32, then OpShiftLeftLogical (196)
    /// shifts 16 — bfloat16 is the top 16 bits of IEEE754 single, so the widened
    /// value is exactly the float bit pattern. Stored to a u32 output; the host
    /// reinterprets the pattern. Proves BF16 weights can be widened on-device even
    /// if IGC rejects the native conversion instruction.
    /// </summary>
    public static uint[] Bf16EmulWiden(uint spirvVersion = Version10)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 8, "bf16_emul");
        w.Op(16, 8, 17, 1, 1, 1);        // OpExecutionMode %8 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(8, "bf16_emul");          // OpName %8 "bf16_emul"
        w.Name(9, "a");                  // OpName %9 "a"
        w.Name(10, "c");                 // OpName %10 "c"
        w.Op(19, 1);                     // %1 = void
        w.Op(21, 2, 16, 0);              // %2 = u16
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(32, 4, 5, 2);               // %4 = ptr<CrossWorkgroup, u16>
        w.Op(32, 5, 5, 3);               // %5 = ptr<CrossWorkgroup, uint>
        w.Op(33, 6, 1, 4, 5);            // %6 = fn(void, ptr u16, ptr uint)
        w.Op(43, 3, 7, 16);              // %7 = const uint 16
        w.Op(54, 1, 8, 0, 6);            // OpFunction %1 None %6 %8
        w.Op(55, 4, 9);                  // %9 = param a (u16 BF16 bits)
        w.Op(55, 5, 10);                 // %10 = param c (u32 widened bits)
        w.Op(248, 11);                   // entry block
        w.Op(61, 2, 12, 9);              // %12 = OpLoad u16 %9
        w.Op(113, 3, 13, 12);            // %13 = OpUConvert uint %12
        w.Op(196, 3, 14, 13, 7);         // %14 = %13 << 16 (f32 bit pattern)
        w.Op(62, 10, 14);                // store %14 -> %10
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 15);
    }

    /// <summary>
    /// The dot-product shape without indexing: an OpPhi loop runs N iterations of
    /// acc += OpConvertBF16ToFINTEL(*a), i.e. BF16 widen + fp32 accumulate, exactly
    /// what a per-element GEMV would do in its inner loop (indexing excepted — that
    /// is blocked by the access-chain IGC bug). Host sets a[0] to BF16 1.0 (0x3F80),
    /// so c[0] must equal N exactly. Same shape as <see cref="AddLoop"/>.
    /// </summary>
    public static uint[] Bf16NativeAccumulate(uint spirvVersion = Version10, uint iterations = 1_000_000)
    {
        var w = new SpvWriter();
        w.Op(17, 6);                     // OpCapability Kernel
        w.Op(17, 4);                     // OpCapability Addresses
        w.Op(17, 5);                     // OpCapability Linkage
        w.Op(17, 6115);                  // OpCapability BFloat16ConversionINTEL
        w.Extension("SPV_INTEL_bfloat16_conversion");
        w.Op(14, 2, 2);                  // OpMemoryModel Physical64 OpenCL
        w.EntryPoint(6, 13, "bf16_native_acc");
        w.Op(16, 13, 17, 1, 1, 1);       // OpExecutionMode %13 LocalSize 1 1 1
        w.Op(3, 3, 102000);              // OpSource OpenCL_C 102000
        w.Name(13, "bf16_native_acc");   // OpName %13 "bf16_native_acc"
        w.Op(19, 1);                     // %1 = void
        w.Op(22, 2, 32);                 // %2 = float
        w.Op(21, 3, 32, 0);              // %3 = uint
        w.Op(20, 4);                     // %4 = bool
        w.Op(21, 5, 16, 0);              // %5 = u16
        w.Op(32, 6, 5, 5);               // %6 = ptr<CrossWorkgroup, u16>
        w.Op(32, 7, 5, 2);               // %7 = ptr<CrossWorkgroup, float>
        w.Op(33, 8, 1, 6, 7);            // %8 = fn(void, ptr u16, ptr f32)
        w.Op(43, 3, 9, 0);               // %9 = const uint 0
        w.Op(43, 3, 10, 1);              // %10 = const uint 1
        w.Op(43, 3, 11, iterations);     // %11 = const uint N
        w.Op(43, 2, 12, 0);              // %12 = const float 0.0 (acc init)
        w.Op(54, 1, 13, 0, 8);           // OpFunction %1 None %8 %13
        w.Op(55, 6, 14);                 // %14 = param a (u16 BF16 bits)
        w.Op(55, 7, 15);                 // %15 = param c (f32 out)
        w.Op(248, 16);                   // entry block
        w.Op(249, 17);                   // branch to header
        w.Op(248, 17);                   // header block
        w.Op(246, 20, 19, 0);            // OpLoopMerge exit(20) continue(19) None
        w.Op(245, 3, 21, 9, 16, 27, 19); // %21 = phi i: (0 from 16), (27 from 19)
        w.Op(245, 2, 22, 12, 16, 26, 19);// %22 = phi acc: (0.0 from 16), (26 from 19)
        w.Op(176, 4, 23, 21, 11);        // %23 = i < N
        w.Op(250, 23, 18, 20);           // branch %23 ? body(18) : exit(20)
        w.Op(248, 18);                   // body block
        w.Op(61, 5, 24, 14);             // %24 = OpLoad u16 %14 (BF16 bits)
        w.Op(6117, 2, 25, 24);           // %25 = OpConvertBF16ToFINTEL float %24
        w.Op(129, 2, 26, 22, 25);        // %26 = acc + %25
        w.Op(249, 19);                   // branch to continue
        w.Op(248, 19);                   // continue block
        w.Op(128, 3, 27, 21, 10);        // %27 = i + 1
        w.Op(249, 17);                   // branch back to header
        w.Op(248, 20);                   // exit block
        w.Op(62, 15, 22);                // store %15 (c) = %22 (acc)
        w.Op(253);                       // OpReturn
        w.Op(56);                        // OpFunctionEnd
        return w.Finish(spirvVersion, 28);
    }

    private sealed class SpvWriter
    {
        private readonly List<uint> words = new();

        public void Op(uint opcode, params uint[] operands)
        {
            words.Add(opcode | ((uint)(operands.Length + 1) << 16));
            words.AddRange(operands);
        }

        public void EntryPoint(uint executionModel, uint functionId, string name, params uint[] interfaceIds)
        {
            int start = words.Count;
            words.Add(0); // patched with opcode + word count below
            words.Add(executionModel);
            words.Add(functionId);
            words.AddRange(StringWords(name));
            words.AddRange(interfaceIds);
            words[start] = 15 | ((uint)(words.Count - start) << 16); // OpEntryPoint
        }

        public void Name(uint target, string name)
        {
            // OpName (5): IdRef + LiteralString (variable-length)
            words.Add(5 | ((uint)(StringWords(name).Length + 2) << 16));
            words.Add(target);
            words.AddRange(StringWords(name));
        }

        public void Extension(string name)
        {
            // OpExtension (10): LiteralString (variable-length)
            var sw = StringWords(name);
            words.Add(10 | ((uint)(sw.Length + 1) << 16));
            words.AddRange(sw);
        }

        public uint[] Finish(uint spirvVersion, uint bound)
        {
            var result = new List<uint>(words.Count + 5)
            {
                SPIRV_MAGIC,
                spirvVersion,
                0, // generator
                bound,
                0  // schema
            };
            result.AddRange(words);
            return result.ToArray();
        }

        private static uint[] StringWords(string s)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(s + "\0");
            int count = (bytes.Length + 3) / 4;
            var result = new uint[count];
            for (int i = 0; i < count; i++)
            {
                uint word = 0;
                for (int b = 0; b < 4; b++)
                {
                    int idx = i * 4 + b;
                    if (idx < bytes.Length)
                        word |= (uint)bytes[idx] << (8 * b);
                }

                result[i] = word;
            }

            return result;
        }
    }
}