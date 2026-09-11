# Nivara.GpuProbe

Probe: can .NET access Intel GPU compute via Level Zero? Pure P/Invoke against the
inbox `ze_loader.dll` (no packages, no bindings, no CUDA/OpenCL) driving
**hand-authored SPIR-V 1.0** kernels that do real compute, with host-verified
results — not just device enumeration.

Host: **Intel Core Ultra 7 255H (Arrow Lake-H)** with the **Arc 140T iGPU**
(128 EU, PCI 8086:7DD1) and an on-package **Intel AI Boost NPU** (8086:7D1D).

## Build & Run

```bash
dotnet run -c Release --project tests/Nivara.GpuProbe -- list   # enumerate L0 drivers/devices/extensions
dotnet run -c Release --project tests/Nivara.GpuProbe -- spv    # dump hand-authored SPIR-V to %TEMP%\opencode\spv
dotnet run -c Release --project tests/Nivara.GpuProbe -- run    # build + launch kernels, verify results (exit 0 = gates pass)
dotnet run -c Release --project tests/Nivara.GpuProbe -- l0     # list + run
dotnet run -c Release --project tests/Nivara.GpuProbe -- ocl    # OpenCL diagnostic (loader only — dead end, see below)
dotnet run -c Release --project tests/Nivara.GpuProbe # default: l0
```

`run` is the real probe: it builds modules with `zeModuleCreate` (logs any
`zeModuleBuildLogGetString` output), launches kernels on a compute queue, and
**verifies results on the host**. Exit code is the number of real failures;
expected driver-bug diagnostics are reported separately and do not fail the run.

This mirrors the `tests/Nivara.SimdProbe` convention — a standalone,
run-manually console app, **not** part of the NUnit suite.

## What the probe proves (this machine, driver L0 API 1.15, SPIR-V max 1.0)

| gate | kernel | result |
|------|--------|--------|
| test 1 `add_parallel` | 256 work items (1-lane workgroups), each `OpAtomicIAdd`s its global id into a shared counter | **PASS** — counter = 32640 = Σ(0..255), verified exactly |
| test 2 `add_loop` | 1 work item, 1,000,000 serial fp32 adds in an OpPhi loop, `*c = acc` | **PASS** — c[0] = 1,000,000.0, min ≈ 5.03 ms → ≈ **0.20 GFADD/s** |

Both kernels round-trip host↔device memory through `zeMemAllocShared`, launch via
`zeCommandListAppendLaunchKernel + zeCommandQueueExecuteCommandLists +
zeCommandQueueSynchronize`, and results are marshaled back and compared with
exact expectations. This is real iGPU compute verified on the host.

## Verified driver findings

### Working constructs (SPIR-V 1.0, OpenCL kernel model)

- Kernel params of `ptr<CrossWorkgroup, T>` (both raw pointer and scalar args).
- Direct `OpLoad`/`OpStore` **through kernel-argument pointers** (`*c = *a + *b`).
- Constants, `OpFAdd`/`OpIAdd`/`OpULessThan`, compare-and-branch control flow.
- `BuiltIn.GlobalInvocationId` (Input storage-class variable + load + extract).
- **OpPhi value-flow loops** (header phis for counter + accumulator, `OpLoopMerge`
  + `OpBranchConditional`, continue block with `OpIAdd`) — the basis of `add_loop`.
- `OpAtomicIAdd` on a kernel-argument pointer when each work item is its **own
  1-lane workgroup** (compile the atomic as simd1 and every lane lands).

### Broken constructs on this driver build (IGC OpenCL frontend) — `[FAIL]` diagnostics

These are **driver bugs at the IGC level**, reproducible with minimal hand-authored
SPIR-V and confirmed via a step-by-step bisection (`run` prints all 17 variants):

1. **Any access-chain opcode** (`OpAccessChain` 65, `OpInBoundsAccessChain` 66,
   `OpPtrAccessChain` 67) → `IGC: Internal Compiler Error: Access violation`,
   regardless of index type (u32/i32/u64), base pointer (param/global), or
   `Restrict`/`NoAlias` decorations. This is the same Intel IGC bug class as
   IGCIT #1144 (Blender AV on Arc B580), so it cannot be fixed from the SPIR-V
   side — only a driver update changes this. It rules out indexed memory access
   (`c[i] = a[i] + b[i]`) entirely on this build.
2. **Private storage-class `OpVariable`s** → linker error
   `undefined reference to `gVar``. Worked around with OpPhi value flow (no
   variables at all).
3. **Generic-pointer kernel args** (`ptr<Generic,...>`) → clean validation
   error (`GenericPointers are not allowed as kernel argument storage class!`);
   rejected, not a crash.
4. **Fat `LocalSize` (8/16-vectorized) atomics drop the upper half of each
   SIMD vector.** With `LocalSize ≥ 8` the driver's atomic codegen only lands
   the low half of the vectorized `OpAtomicIAdd`, deterministically
   (`LocalSize 256` gives 15808 instead of 32640; the missing amount is exactly
   `Σ lanes with (lane % 16) ≥ 8`). One-lane workgroups avoid the
   vectorization and pass exactly.

The probe's `[test 1]` sweep documents bug #4 (localSize × groups matrix, all
`simd8`/`simd16` shapes fail); the (1, 256) config is the pass gate.
Bisection variants documenting #1/#2/#3 are reported as **expected
driver-bug diagnostics** (15 on this machine) and do not fail the run.

### SPIR-V language ceiling

The driver reports **SPIR-V max 1.0**; loading a 1.2 header hangs `zeModuleCreate`
(observed). All kernels are emitted as 1.0. The NPU (`Intel(R) AI Boost`,
DDI driver[1], API 1.14) exposes **no SPIR-V support at all** (`spirvVersionSupported
= 0`) and cannot run these kernels.

### Kernel binary export (follow-up)

The iGPU driver advertises `ZE_extension_kernel_binary_exp` (export the compiled
native binary), `ZE_extension_float_atomics`, `ZE_extension_bfloat16_conversions`
(BF16 → the BF16 inference story), `ZE_extension_kernel_max_group_size_properties`,
and `ZEX_intel_experimental_queue_copy_operations_offload_hint`. None are used by
this probe — the hand-authored SPIR-V is the source of truth.

## Why hand-authored SPIR-V?

No compiler toolchain is available on this machine (no `clang`, no SPIR-V tools),
so every kernel is written word-by-word by `SpvKernels.cs` against the
`spirv.core.grammar.json` opcode tables. Correctness was validated by a local
Python disassembler (`spvdis.py`) — the key SPIR-V encoding lesson: results are
emitted **result-first** by intuition but the format is
`(opcode) (word-count) (result-type) (result-id) <operands>`, i.e. the result
type precedes the result id (this was the first bug found). The dump loop (`spv`
mode) writes `.spv` files for external validation.

## Level Zero P/Invoke surface

Structs, constants, and proc addresses are taken only from the official
`ze_api_v1282.h` (API target 1.28 layout) — see `LevelZero/L0Native.cs` for the
hand-verified constant/layout values (`ST_MODULE_DESC = 0x1b`,
`MODULE_FORMAT_IL_SPIRV = 0`, `ZeModuleDesc` layout, etc.). The loader resolves
entry points by name from `ze_loader.dll`.

## Files

- `Program.cs` — CLI dispatch (`list` / `run` / `spv` / `ocl` / `l0` / `all`).
- `LevelZero/L0Probe.cs` — enumeration: drivers, API versions, extensions, devices
  (type, PCI id, EU topology, clocks), module caps (SPIR-V version, fp16/fp32/fp64
  flags, DP4A), queue groups.
- `LevelZero/L0Run.cs` — the execution probe: module build, kernel create, shared
  memory buffers, launch, sync, **host verification**, timing loop, and the
  failure/diagnostic accounting.
- `LevelZero/SpvKernels.cs` — the hand-authored SPIR-V writer plus all kernels
  (`add_parallel`, `add_loop`) and the 17-variant IGC bisection set.
- `LevelZero/SpvDump.cs` — dumps modules to `%TEMP%\opencode\spv`.
- `LevelZero/OclProbe.cs` — OpenCL loader diagnostic (dead end: an OpenCL ICD
  dispatch path is not pursued; kept only for the Intel-export function discovery).

## Recommendations

- On this driver build, **Nivara GPU kernels must stay inside the working subset**:
  no access chains (no general scatter/gather), no private variables, serial or
  per-lane-atom free vector reductions via 1-lane workgroups, OpPhi loops for
  iteration, direct loads/stores through args only.
- The practical shape for real Nivara work is **large serial/phi reductions per
  work item over one scalar value per arg** (the `add_loop` pattern) combined
  with **one-lane-group atomic accumulation** for cross-work-item aggregation.
- Re-test these kernel shapes after any Intel driver update: the GEP/private-var
  access violations are version-fixed upstream (IGCIT). `run` bisects the whole
  surface in seconds, so a driver bump can be revalidated cheaply.
- The NPU (second L0 driver) is out of scope for SPIR-V compute — it exposes no
  SPIR-V; the graph-based NPU extensions (`ZE_extension_graph*`) would be the
  only path there, if pursued at all.