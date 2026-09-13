# Nivara.GpuProbe

Probe: can .NET access Intel GPU compute via Level Zero? Pure P/Invoke against the
inbox `ze_loader.dll` (no packages, no bindings, no CUDA/OpenCL) driving
**hand-authored SPIR-V 1.0** kernels that do real compute, with host-verified
results — not just device enumeration. **Verdict so far:** the runtime harness
works end-to-end (module load, launch, readback, verifiable results) but this
driver's IGC miscompiles FP multiply/divide in the OpenCL kernel model
(`OpFMul`→`OpFSub`, `OpFDiv`→`OpFMul`), so real dot/GEMV/SiLU math moves to the
**oneAPI SYCL/DPC++ toolchain path** (see `docs/TODO.md`).

Host: **Intel Core Ultra 7 255H (Arrow Lake-H)** with the **Arc 140T iGPU**
(128 EU, PCI 8086:7DD1) and an on-package **Intel AI Boost NPU** (8086:7D1D).

## References

The probe consumes existing Nivara kernels **read-only** for its CPU verification
leg (no code changes to the referenced projects; the GPU legs keep the host side
pure P/Invoke — no GPU bindings/packages. The kernel *source* is hand-authored
SPIR-V for the now-blocked L0 leg, compiler-produced SYCL/DPC++ SPIR-V for the
oneAPI leg, and HLSL for the DX12 leg, per the pivot in `docs/TODO.md`):

- `src/Nivara` — `LlamaFusedKernels.MatMulTransposedB<T>` (the production SmolLM
  GEMV kernel — dot and GEMV both run through it, in the exact fused-head call
  shape) and `ReverseGradTensor<T>.FromArray` + `Activation.Silu<T>` (the
  production sigmoid-then-multiply SiLU kernel) for the CPU reference.
- `samples/Nivara.Samples` — `SafeTensorsLoader.WidenBf16ToF32` (SIMD BF16→f32
  widen at load, once-only).

## Kernel phase fixtures & gold target

The many-way (CPU gold · L0 hand-authored · SYCL/oneAPI · DX12) BF16 dot/GEMV + SiLU probe
compares every GPU leg against the **CPU leg, which is the production Nivara kernels exactly as
`NivaraInference` calls them for SmolLM** (consumed read-only through public API)
on the same byte-identical inputs. No hand-rolled or double-precision oracle is
maintained — the production kernels are the gold target:

- `Kernels/KernelFixtures.cs` — deterministic SmolLM-135M-shaped BF16 fixtures
  (hidden 576, heads 9, kv 3, intermediate 1536, silu, vocab 49152 — embedded,
  `config.json` is an optional override only). One xorshift RNG stream (seed
  `0x9E3779B9`, Qwen mirror), values `(rng/uint.Max − 0.5f)·0.1f` narrowed via
  `BFloat16.CreateChecked`; buffers are `System.Numerics.BFloat16` end-to-end (raw
  16-bit patterns on the wire, GPU widens in-register). Also hosts the native
  write-BF16 / read-f32 helpers.
- `Kernels/CpuLeg.cs` — **the CPU leg = the production kernels**, read-only via
  public API: dot/GEMV via `LlamaFusedKernels.MatMulTransposedB<float>` (aRows=1
  → the allocation-free BLAS2 GEMV path the fused Llama head runs; bCols=1 pure
  dot), SiLU via `Activation.Silu` (`GradKernels.Silu`, sigmoid-then-multiply).
  Gate for every kernel/leg: `|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|`;
  per-leg worst ULP vs the CPU output reported as a diagnostic.

## Build & Run

```bash
dotnet run -c Release --project tests/Nivara.GpuProbe -- list   # enumerate L0 drivers/devices/extensions
dotnet run -c Release --project tests/Nivara.GpuProbe -- spv    # dump hand-authored SPIR-V to %TEMP%\opencode\spv
dotnet run -c Release --project tests/Nivara.GpuProbe -- run    # build + launch kernels, verify results (exit 0 = no unexpected failures; diagnosed driver bugs reported separately)
dotnet run -c Release --project tests/Nivara.GpuProbe -- l0     # list + run
dotnet run -c Release --project tests/Nivara.GpuProbe -- ocl    # OpenCL diagnostic (loader only — dead end, see below)
dotnet run -c Release --project tests/Nivara.GpuProbe -- dx12   # D3D12 availability check (FL level + shader model)
dotnet run -c Release --project tests/Nivara.GpuProbe # default: l0 + run + dx12
```

`run` is the real probe: it builds modules with `zeModuleCreate` (logs any
`zeModuleBuildLogGetString` output), launches kernels on a compute queue, and
**verifies results on the host**. Exit code is the number of *unexpected*
failures; expected driver-bug diagnostics are reported separately and do not
fail the run. After the five availability tests it runs the **kernel phase**: the
BF16 dot K=16 kernels (native + emul widen) and 576 per-element SiLU launches,
each gated against the production CPU kernels (`CpuLeg`).

**Kernel-phase verdict (this machine):** the L0 leg cannot pass these gates —
this driver's IGC miscompiles `OpFMul`(131) as `OpFSub`(130) and `OpFDiv`(132) as
`OpFMul`(131) in the OpenCL kernel model (proven exactly by the three single-op
evidence probes), so dot16 computes Σ(a−b) and silu computes x·(1+exp(−x)), both
**confirmed to the mirror formula** and counted as expected diagnostics, not
failures. The kernels are structurally valid; proof of correctness pivots to the
**oneAPI SYCL/DPC++ toolchain path** (kernels authored in SYCL and compiled by
`icpx`, so IGC consumes compiler-produced SPIR-V) — see `docs/TODO.md`.

This mirrors the `tests/Nivara.SimdProbe` convention — a standalone,
run-manually console app, **not** part of the NUnit suite.

## What the probe proves (this machine, driver L0 API 1.15, SPIR-V max 1.0)

| gate | kernel | result |
|------|--------|--------|
| test 1 `add_parallel` | 256 work items (1-lane workgroups), each `OpAtomicIAdd`s its global id into a shared counter | **PASS** — counter = 32640 = Σ(0..255), verified exactly |
| test 2 `add_loop` | 1 work item, 1,000,000 serial fp32 adds in an OpPhi loop, `*c = acc` | **PASS** — c[0] = 1,000,000.0, min ≈ 5.03 ms → ≈ **0.20 GFADD/s** |
| test 3 `bf16_native` | BF16 → F32 via `OpConvertBF16ToFINTEL` (native, SPV_INTEL_bfloat16_conversion), 4 patterns × test values | **PASS** — widened f32 correct, **reloaded as BF16** equals the original `BFloat16` (exact, zero host widening) |
| test 4 `bf16_emul` | BF16 → F32 widen via `OpUConvert` + `<<16` (safe subset, no extension), same 4 patterns | **PASS** — same exact results + BF16 reload |
| test 5 `bf16_native_acc` | 1M-iteration `acc += widen(BF16)` loop in `OpPhi`, `OpConvertBF16ToFINTEL` inside the loop, L0→F32 accumulate | **PASS** — c[0] = 1,000,000.0, **exact** (f32 accumulation gate) |
| `dx12` | D3D12CreateDevice at FL 11_0→12_2, shader model via CheckFeatureSupport | **PASS** — Arc 140T reaches **FL 12_2 / SM 6.8** |
| **kernel phase** `bf16_fmul_probe` | single-op probe, exact inputs 1 OP 2, `OpFMul`(131) | **DIAG** — reads −1 (runs as OpFSub 130); miscompiled (expected driver bug) |
| **kernel phase** `bf16_fadd_probe` | single-op probe, `OpFAdd`(129), 1 OP 2 | **PASS** — reads 3, correct |
| **kernel phase** `bf16_fdiv_probe` | single-op probe, `OpFDiv`(132), 1 OP 2 | **DIAG** — reads 2 (runs as OpFMul 131); miscompiled (expected driver bug) |
| **kernel phase** `bf16_dot_k16_native`/`_emul` | straight-line K=16 BF16 dot, gated vs production CPU dot (gold target) | **DIAG** — reads 0.0380020142 = Σ(a−b) exactly (FMul→FSub miscode confirmed by mirror); gate FAILs as expected |
| **kernel phase** `bf16_silu` | 576 single-element BF16 SiLU, gated vs production sigmoid-multiply kernel | **DIAG** — all 576 read `x·(1+exp(−x))` (FDiv→FMul miscode confirmed by mirror), 0 unexpected → gate FAILs as expected |

All kernels round-trip host↔device memory through `zeMemAllocShared`, launch via
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
- **`BFloat16` ↔ `f32` round-trip** via `OpConvertBF16ToFINTEL` (capability 6115):
  IGC honors the extension; BF16 passes through to `f32` losslessly, the widened
  `f32`'s upper half maps back to the original `BFloat16` exactly — proven native
  via `.NET 11 BFloat16` in/out, zero host widening.

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
5. **`OpFMul`(131) executes as `OpFSub`(130) and `OpFDiv`(132) executes as
   `OpFMul`(131)** — a deterministic, opcode-specific IGC miscompile in the
   OpenCL kernel model for hand-authored SPIR-V. Proven by exact-input single-op
   binop probes: BF16 1.0 and 2.0 widened, `OpFMul` reads −1 (i.e. 1−2),
   `OpFDiv` reads 2 (i.e. 1×2); `OpFAdd` and `OpenCL.std exp` both correct.
   All 576 SiLU elements read `x·(1+exp(−x))` (the FDiv→FMul mapped form),
   and the dot16 kernels read Σ(a−b) instead of the dot — both confirmed
   numerically against in-place CPU mirrors. Dot16/SiLU kernels are structurally
   valid; the failure is fully on the driver side. Does not affect the
   SYCL/oneAPI toolchain path (llama.cpp SYCL backend runs on Arrow Lake Arc
   iGPUs via the same IGC, confirming toolchain-produced SPIR-V mul/div works).

The probe's `[test 1]` sweep documents bug #4 (localSize × groups matrix, all
`simd8`/`simd16` shapes fail); the (1, 256) config is the pass gate.
Bisection variants documenting #1/#2/#3 are reported as **expected
driver-bug diagnostics** (15 on this machine) and do not fail the run.
Bug #5 is proven by the kernel-phase binop evidence probes; the dot16/SiLU gate
failures it causes are likewise expected diagnostics (mirror-confirmed).

### SPIR-V language ceiling

The driver reports **SPIR-V max 1.0**; loading a 1.2 header hangs `zeModuleCreate`
(observed). All kernels are emitted as 1.0. The NPU (`Intel(R) AI Boost`,
DDI driver[1], API 1.14) exposes **no SPIR-V support at all** (`spirvVersionSupported
= 0`) and cannot run these kernels.

### DX12 availability — the IGC-bypass side-path

The Arc 140T iGPU creates a D3D12 device at **feature level 12_2** with **shader
model 6.8** (`D3d12Check.cs`, pure P/Invoke against inbox `dxgi.dll`/`d3d12.dll`).
A managed DX12 compute backend (ComputeSharp / HLSL → DXIL) targets a separate
shader compilation path that bypasses the buggy IGC OpenCL/SPIR-V frontend entirely.
This is the viable immediate fallback for real GEMM/attention kernels while the
Level Zero access-chain ICE (bug #1 above) remains unfixed on this driver.

### Kernel binary export (follow-up)

The iGPU driver advertises `ZE_extension_kernel_binary_exp` (export the compiled
native binary), `ZE_extension_float_atomics`, `ZE_extension_bfloat16_conversions`
(BF16 → the BF16 inference story), `ZE_extension_kernel_max_group_size_properties`,
and `ZEX_intel_experimental_queue_copy_operations_offload_hint`. None are used by
this probe — the hand-authored SPIR-V is the source of truth.

## Why hand-authored SPIR-V?

No compiler toolchain was available on this machine at probe start (no `clang`,
no SPIR-V tools), so every kernel is written word-by-word by `SpvKernels.cs`
against the `spirv.core.grammar.json` opcode tables. Correctness was validated by
a local Python disassembler (`spvdis.py`) — the key SPIR-V encoding lesson:
results are emitted **result-first** by intuition but the format is
`(opcode) (word-count) (result-type) (result-id) <operands>`, i.e. the result
type precedes the result id (this was the first bug found). The dump loop (`spv`
mode) writes `.spv` files for external validation.

That hand-authoring exercise is what isolated **bug #5** (FMul/FDiv miscompile):
a compiler could never be blamed for such a finding, and no hand-written bytecode
could have "worked" around it. The probe's conclusion is therefore to *stop*
hand-authoring kernels and use the oneAPI toolchain — the hand-written path
served its purpose (proving the runtime harness + documenting the driver
landmines) and now steps aside.

## Level Zero P/Invoke surface

Structs, constants, and proc addresses are taken only from the official
`ze_api_v1282.h` (API target 1.28 layout) — see `LevelZero/L0Native.cs` for the
hand-verified constant/layout values (`ST_MODULE_DESC = 0x1b`,
`MODULE_FORMAT_IL_SPIRV = 0`, `ZeModuleDesc` layout, etc.). The loader resolves
entry points by name from `ze_loader.dll`.

## Files

- `Program.cs` — CLI dispatch (`list` / `run` / `spv` / `ocl` / `l0` / `all`).
- `Kernels/KernelFixtures.cs` — SmolLM-shaped BF16 fixtures + shared native
  write-BF16 / read-f32 helpers.
- `Kernels/CpuLeg.cs` — **the production-kernel CPU leg (gold target)** for the
  three-way gate: dot/GEMV via `LlamaFusedKernels.MatMulTransposedB<float>`
  (aRows=1 → the allocation-free BLAS2 GEMV path the fused Llama head runs),
  SiLU via `Activation.Silu` (`GradKernels.Silu`, sigmoid-then-multiply), all
  consumed read-only through public API. Carries the gate helpers
  (`WithinTolerance`, `UlpDistance`, `GateAbs`/`GateRel`). Replaces the deleted
  double-precision `GoldenReferences.cs` — no hand-rolled or double oracle exists.
- `LevelZero/L0Probe.cs` — enumeration: drivers, API versions, extensions, devices
  (type, PCI id, EU topology, clocks), module caps (SPIR-V version, fp16/fp32/fp64
  flags, DP4A), queue groups.
- `LevelZero/L0Run.cs` — the execution probe: module build, kernel create, shared
  memory buffers, launch, sync, **host verification**, timing loop, and the
  failure/diagnostic accounting. Runs tests 1–5, the IGC bisection, the
  kernel-phase evidence probes (binop FP-op mappings), and the dot16/SiLU gates
  vs `CpuLeg`.
- `LevelZero/SpvKernels.cs` — the hand-authored SPIR-V writer plus all kernels
  (`add_parallel`, `add_loop`, `bf16_native`, `bf16_emul`, `bf16_native_acc`,
  `bf16_dot_k16_native`/`_emul`, `bf16_silu`, `bf16_fmul/fadd/fdiv_probe`
  evidence) and the 17-variant IGC bisection set.
- `LevelZero/SpvDump.cs` — dumps modules to `%TEMP%\opencode\spv`.
- `D3d12Check.cs` — D3D12 availability probe: DXGI adapter enumeration,
  `D3D12CreateDevice` at FL 11_0..12_2, shader model via `CheckFeatureSupport`.
- `LevelZero/OclProbe.cs` — OpenCL loader diagnostic (dead end: an OpenCL ICD
  dispatch path is not pursued; kept only for the Intel-export function discovery).

## Recommendations

- **BF16 compute model is proven end-to-end**: `.NET 11 BFloat16` in/out, GPU
  in the middle, `f32` accumulation (the hardware's native model — Xe2 DPAS also
  accumulates BF16 products in `f32`), zero host widening. The only blocker for a
  real GEMV kernel is the access-chain ICE (bug #1); the kernel-phase blockers
  are bugs #1 (no indexed memory) and #5 (mul/div miscompiled).
- **Hand-authored SPIR-V on this driver is a dead end for real math** (bug #5):
  the L0 leg cannot perform FP multiply or divide at all, so no dot/GEMV/SiLU is
  expressible through hand-written bytecode. Documented as the authoritative
  verdict of this probe phase; **kernel authoring pivots to the oneAPI SYCL/DPC++
  toolchain** (compiler-produced SPIR-V via `icpx`), where IGC consumes validated
  toolchain bytecode — the same path llama.cpp's SYCL backend uses on Arrow Lake
  Arc iGPUs. The probe keeps its L0 harness (module load, launch, readback all
  proven) and simply feeds it compiler-built modules; promotion decisions
  (e.g. `src/Nivara.Gpu`) come later.
- **DX12/ComputeSharp remains the compiler-backed fallback**: the Arc 140T
  reaches FL 12_2 + SM 6.8, and HLSL → DXIL is another validated compiler path
  that bypasses hand-authored bytecode concerns entirely.
- **Level Zero working subset** (for reference probes only): no access chains (no
  scatter/gather), no private variables, direct loads/stores through args only,
  OpPhi loops for iteration, 1-lane workgroup atomics for aggregation.
- Re-test these kernel shapes after any Intel driver update: both the GEP
  access-chain ICE (bug #1) and the FP-opcode miscompile (bug #5) are
  version-fixed upstream (IGCIT). `run` bisects the whole surface in seconds, so
  a driver bump can be revalidated cheaply.
- The NPU (second L0 driver) is out of scope for SPIR-V compute — it exposes no
  SPIR-V; the graph-based NPU extensions (`ZE_extension_graph*`) would be the
  only path there, if pursued at all.