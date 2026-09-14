# Nivara.GpuProbe

Probe: can .NET access Intel GPU compute? Pure P/Invoke — **no packages, no
bindings, no CUDA, no toolchain** — with one deliberate exception: the phase-4a
ILGPU leg (issue #431) is the probe's first NuGet reference, because there the
package *is* the toolchain (a pure-managed C#→OpenCL JIT runtime). The probe
drives the same SmolLM-shaped BF16 kernels (dot16 · silu · gemv) through **five
independent GPU paths**, each gated against the production Nivara CPU kernels:

| path | route | verdict (Arc 140T) |
|---|---|---|
| **Level Zero** | hand-authored SPIR-V 1.0 on inbox `ze_loader.dll` | runtime harness proven, but this driver's IGC miscompiles FP multiply/divide in the OpenCL kernel model — **dead end for real math** (see [docs/SPIRV.md](../../docs/SPIRV.md)) |
| **SYCL/oneAPI** | `icpx`-compiled DPC++ (SPIR-V over L0) | **proven PASS** (compiler-produced bytecode is handled correctly by IGC); toolchain no longer installed on this machine → row UNBUILT |
| **DX12** | hand-rolled HLSL `cs_5_1` via inbox `d3dcompiler_47.dll` | **proven PASS** — FL 12_2 / SM 6.8, no external tooling |
| **OpenVINO** | pip-installed `openvino_c.dll` + tuned GPU plugin, IR v11 models | **proven PASS** — first-party, zero compiler, ~26–34× CPU on gemv; BF16 silu honestly F16-tier |
| **ILGPU (phase 4a)** | NuGet `ILGPU 1.5.3` — pure-managed JIT of C# kernels to OpenCL C (in-box ICD + Intel driver) | **proven PASS** — all three gates on the real iGPU, no CPU fallback; fastest GPU leg on silu (see [docs/ILGPU.md](../../docs/ILGPU.md)) |

The Level Zero leg got this series started by proving the harness end-to-end
(module load, launch, readback, verifiable results) and then isolating a
**driver-bug class** (IGC OpenCL frontend: `OpFMul`→`OpFSub`, `OpFDiv`→`OpFMul`,
access-chain ICE) that only affects hand-authored bytecode — the reason real
math pivoted to the compiler-backed and driver-backed paths above. Full write-ups
in [docs/SPIRV.md](../../docs/SPIRV.md) · [docs/SYCL.md](../../docs/SYCL.md) · [docs/DX12.md](../../docs/DX12.md) · [docs/OPENVINO.md](../../docs/OPENVINO.md) ·
[docs/ILGPU.md](../../docs/ILGPU.md).

Host: **Intel Core Ultra 7 255H (Arrow Lake-H)** with the **Arc 140T iGPU**
(128 EU, PCI 8086:7DD1) and an on-package **Intel AI Boost NPU** (8086:7D1D).

## References

The probe consumes existing Nivara kernels **read-only** for its CPU verification
leg (no code changes to the referenced projects; the GPU legs keep the host side
pure P/Invoke — no GPU bindings/packages, except the phase-4a ILGPU leg whose
NuGet packages *are* the toolchain). The kernel *source* is hand-authored
SPIR-V for the now-blocked L0 leg, compiler-produced SYCL/DPC++ SPIR-V for the
oneAPI leg, HLSL for the DX12 leg, hand-emitted OpenVINO IR v11 XML for the
OpenVINO leg, and C# device methods for the ILGPU leg, per the case-study docs [docs/SPIRV.md](../../docs/SPIRV.md)
(L0), [docs/SYCL.md](../../docs/SYCL.md) (oneAPI/SYCL), [docs/DX12.md](../../docs/DX12.md) (DX12),
[docs/OPENVINO.md](../../docs/OPENVINO.md) (OpenVINO), and [docs/ILGPU.md](../../docs/ILGPU.md) (ILGPU phase 4a):

- `src/Nivara` — `LlamaFusedKernels.MatMulTransposedB<T>` (the production SmolLM
  GEMV kernel — dot and GEMV both run through it, in the exact fused-head call
  shape) and `ReverseGradTensor<T>.FromArray` + `Activation.Silu<T>` (the
  production sigmoid-then-multiply SiLU kernel) for the CPU reference.
- `samples/Nivara.Samples` — `SafeTensorsLoader.WidenBf16ToF32` (SIMD BF16→f32
  widen at load, once-only).

## Kernel phase fixtures & gold target

The many-way (CPU gold · L0 hand-authored · SYCL/oneAPI · DX12 · OpenVINO-bf16 ·
OpenVINO-f32 · ILGPU/OpenCL) BF16 dot/GEMV + SiLU probe compares every GPU leg against the **CPU leg, which is the production Nivara kernels exactly as
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

## Backend comparison — the numbers, side by side

The single place all measured figures live (µs per invocation, kernel-only —
process/device setup excluded). The ranges span runs across the whole series
(best-of-3 in SYCL/DX12 runs, best-of-25 for OpenVINO); CPU times vary with
clock/power state so treat them as directional, not spec. Per-leg detail and
methodology live in the case-study docs — the table here is the decision aid.

| kernel | CPU (produ. Nivara) | SYCL/oneAPI | DX12 (hand-rolled) | OpenVINO bf16 | OpenVINO f32 | ILGPU (OpenCL) | gate |
|---|---|---|---|---|---|---|---|
| `dot16` (K=16) | 1.2–4.8 | 13–53 | 117–560 | 75.8 | 68.5 | **11.4–11.7** | launch-bound — CPU wins (ILGPU fastest GPU leg) |
| `silu` (576) | 29–99 | 12–34 | 86–364 | 58.9 | 51.2 | **6.9** | **ILGPU ~5–14× CPU** — best GPU leg |
| `gemv` (1536×576) | 2291–4745 | 170–198 | 152–525 | **137.9** | **181.7** | 133.4–135.8 | **~10–34× GPU win** (ILGPU ~27×) |

Correctness (same gate as above, vs the production CPU kernels; L0 cannot
express the kernels at all):

| leg | `dot16` | `silu` (576) | `gemv` (1536×576) |
|---|---|---|---|
| L0 hand-authored SPIR-V | — (IGC `OpFMul`→`FSub`/`OpFDiv`→`FMul` miscompile, [docs/SPIRV.md](../../docs/SPIRV.md)) | same | same |
| SYCL/oneAPI | PASS (0.0 ULP) | PASS (4.0 ULP) | PASS |
| DX12 | PASS (0.0 ULP) | PASS (4.0 ULP) | PASS |
| OpenVINO bf16 | PASS (0.0 ULP) | **honest FAIL** — F16 silu, 402/576 ([docs/OPENVINO.md](../../docs/OPENVINO.md) §3) | PASS |
| OpenVINO f32 | PASS (0.0 ULP) | PASS (4.0 ULP) | PASS |
| ILGPU (OpenCL) | PASS (0.0 ULP) | PASS (4.0 ULP) | PASS |

Reading: the **gemv is the deliverable** — every SmolLM decode token is dominated
by `[1536×576]·[576]` GEMVs, and all live GPU legs run it at 133–525 µs vs
~2.3–4.7 ms CPU. OpenVINO and ILGPU trade the gemv win run-to-run (86.8 µs tuned
gemm vs 133.4 µs naive one-thread-per-row); ILGPU is decisively **fastest on
silu** (6.9 µs, ~5–14× CPU) and on dot16's launch-bound floor (11.4 µs). silu
splits the GPU legs (ILGPU/SYCL/OV ≈ 2–14× CPU, DX12 still launch-bound at 576
elements). dot16 exists only as the smallest correctness probe and stays
CPU-fastest everywhere.

## Build & Run

```bash
dotnet run -c Release --project tests/Nivara.GpuProbe -- list    # enumerate L0 drivers/devices/extensions
dotnet run -c Release --project tests/Nivara.GpuProbe -- spv     # dump hand-authored SPIR-V to %TEMP%\opencode\spv
dotnet run -c Release --project tests/Nivara.GpuProbe -- run     # build + launch kernels, verify results (exit 0 = no unexpected failures; diagnosed driver bugs reported separately)
dotnet run -c Release --project tests/Nivara.GpuProbe -- l0      # list + run
dotnet run -c Release --project tests/Nivara.GpuProbe -- ocl     # OpenCL diagnostic (loader only — dead end, see below)
dotnet run -c Release --project tests/Nivara.GpuProbe -- dx12    # D3D12 availability check + the full DX12 compute leg gates (in-process HLSL→DXBC cs_5_1→PSO→dispatch, no toolchain)
dotnet run -c Release --project tests/Nivara.GpuProbe -- sycl    # SYCL leg gates (needs Sycl/build.cmd first, see below)
dotnet run -c Release --project tests/Nivara.GpuProbe -- ov      # OpenVINO availability (runtime + GPU readback) + both precision configs' gates
dotnet run -c Release --project tests/Nivara.GpuProbe -- ilgpu  # ILGPU availability (OpenCL devices + accelerator) + gates (phase 4a)
dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels # six-way gate harness: CPU gold + SYCL + DX12 + OV-bf16 + OV-f32 + ILGPU (exit = failed cells)
dotnet run -c Release --project tests/Nivara.GpuProbe # default: l0 + run + dx12 + openvino
```

`kernels` (and `sycl`/`ov`/`ilgpu`, which route through the same harness) is the
**correctness gate**: the CPU leg (production Nivara kernels, see
`Kernels/CpuLeg.cs`) is the gold target, and each wired GPU leg's `dot16` /
`silu` / `gemv` output is gated against it with
`|leg − cpuNivara| ≤ 1e-6 + 1e-5·|cpuNivara|`. Exit code = number of failed
cells. On this machine it is **405** = 402 (the *honest* OV-bf16 silu row:
F16-elementwise sigmoid/multiply on a BF16-declared model, see
[docs/OPENVINO.md](../../docs/OPENVINO.md) §3) + 3 (SYCL UNBUILT baseline). The ILGPU row (phase 4a)
passes all three — exit unchanged. Every failing cell is an
explicitly-flagged honest one — never a miscode.

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
`icpx`, so IGC consumes compiler-produced SPIR-V) — **proven** — and then to the
DX12 ([docs/DX12.md](../../docs/DX12.md)) and OpenVINO ([docs/OPENVINO.md](../../docs/OPENVINO.md)) legs, all gated below.

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

### DX12 — the IGC-bypass side-path (now compute-proven, not just enumerated)

The Arc 140T iGPU creates a D3D12 device at **feature level 12_2** with **shader
model 6.8** (`D3d12Check.cs`, pure P/Invoke against inbox `dxgi.dll`/`d3d12.dll`).
The DX12 **compute** leg (`D3d12/D3d12Compute.cs` + `D3d12/GemvKernels.cs`) is a
hand-rolled HLSL→DXBC `cs_5_1` path: P/Invoke + vtable dispatch (no D3D12.NET /
ComputeSharp / TerraFX), compiling in-process with `d3dcompiler_47.dll`, root
signature with two descriptor tables (SRV t0 / UAV u0), SHADER_VISIBLE
CBV_SRV_UAV heap, UPLOAD→DEFAULT(→COPY_SOURCE)→READBACK buffers, fence+event
wait per iteration, 1 warmup + 3 timed best-of-3 — the same methodology as the
SYCL leg. This bypasses the buggy IGC OpenCL/SPIR-V frontend entirely (HLSL →
DXBC/DXIL → the driver's compute pipeline), so real GEMM/attention kernels are
expressible while the Level Zero access-chain ICE (bug #1 above) remains
unfixed on this driver. The `kernels` mode runs the **six-way** CPU·SYCL·DX12·
OV·ILGPU gate table with a per-GPU-leg timing column; full workflow notes in
[docs/DX12.md](../../docs/DX12.md).

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

## oneAPI SYCL/DPC++ leg (commit 5 — the pivot, proven)

`Sycl/sycl_runner.cpp` is a DPC++ runner with the three production-shape kernels
(bf16→f32 emul-widen `uint16 << 16` on device, exactly like the L0 leg):
`dot16` (single task, serial K=16 f32 accumulate), `gemv` (one work-item per
output row, serial K=576 — real indexed memory, which L0 couldn't do), `silu`
(per-element, `x / (1 + exp(−x))`). Raw BF16 fixture bytes in → raw f32
results out; builds with `icpx -fsycl -O2` (`Sycl/build.cmd` sources VS 2022
Build Tools `VsDevCmd` for the MSVC host tools, then oneAPI `setvars`).

The kernel is the production toolchain shape llama.cpp's SYCL backend uses
(SYCL runtime on Level Zero underneath), so this answers the decisive question
commit 4 left open: **does IGC execute BF16 FP mul/div correctly on
toolchain-produced bytecode? — Yes.**

| kernel | gate vs CpuLeg (production Nivara) | worst | result |
|---|---|---|---|
| `dot16` (K=16) | `\|leg − cpu\| ≤ 1e-6 + 1e-5·\|cpu\|` | **0.0 ULP** (bit-exact) | PASS |
| `silu` (576) | tolerance gate per element | 4.0 ULP | PASS (576/576) |
| `gemv` (1536×576) | tolerance gate per row | 14 336 ULP @ row 1508 (`\|diff\| = 1.6e-9`, near-zero ref row) | PASS (1536/1536) |

Note on the gemv "worst ULP" figure: it is the *worst relative* row — after
cancellation some rows land near zero where an f32 ULP is ~1.9e-9, so a perfectly
good row reads thousands of ULP while sitting far inside the `1e-6` absolute
gate bound (`\|diff\| = 1.63e-9` on the worst row). The threshold row used for
diagnostics, not pass/fail.

Load path decision (recorded): commit 5 ships the **SYCL-runtime shim** path —
`SyclLeg.cs` spawns `run.cmd` (sources `setvars` so `sycl8.dll`/`ur_loader.dll`
resolve; a direct spawn fails `STATUS_DLL_NOT_FOUND`), gates all three kernels
against `CpuLeg`, and exits nonzero on any real failure. The alternative
(extract compiler-produced SPIR-V and load through the probe's `zeModuleCreate`
harness) is a follow-up for the `kernels` gate mode; the SPIR-V 1.0 ceiling
noted below may force native-image loading — the runtime shim is the safe
default because it also exercises the production UR adapter path end-to-end.

### Performance (commit 7 — kernel-only, steady-state)

The `kernels` mode now prints a per-kernel timing table: CPU times come from
`CpuLeg.ComputeLeg` (warmed once to pay JIT, then timed via `Stopwatch`);
SYCL times come from the runner's own `std::chrono` best-of-3 `submit→wait`
(`TIME` lines in `sycl_runner`), excluding process/queue setup. Measured on
the Arc 140T (driver 1.15.37858). **The figures live in the consolidated
side-by-side table above** — the reading: `dot16` is launch-bound, `silu` and
`gemv` are the real SmolLM decode shapes and both are decisive GPU wins (gemv
~23× on this leg). The ~50 ms subprocess launch per kernel is not included — a
production native `.dll` with a long-lived queue eliminates it entirely. Full
notes in [docs/SYCL.md](../../docs/SYCL.md); the DX12 workflow lives in [docs/DX12.md](../../docs/DX12.md);
the L0/hand-authored-SPIR-V verdict and safe subset live in [docs/SPIRV.md](../../docs/SPIRV.md).
Together these five case-study docs — [docs/SPIRV.md](../../docs/SPIRV.md) · [docs/SYCL.md](../../docs/SYCL.md) ·
[docs/DX12.md](../../docs/DX12.md) · [docs/OPENVINO.md](../../docs/OPENVINO.md) · [docs/ILGPU.md](../../docs/ILGPU.md) — are Nivara's GPU-backend
decision records.

## DX12 compute leg (commit 8 — hand-rolled, proven)

`D3d12/D3d12Compute.cs` is a standalone hand-rolled DX12 compute path — no
bindings, no ComputeSharp/TerraFX, no `#if WINDOWS` — just `DllImport` +
vtable dispatch against inbox `d3d12.dll`/`dxgi.dll`/`d3dcompiler_47.dll`
(all struct layouts, GUIDs, enum values and vtable slots taken from the SDK
header `10.0.26100.0\um\d3d12.h`; `ValidateLayouts()` asserts `Marshal.SizeOf`
== C sizes before running). BF16 fixtures are packed 2-per-`uint` (even
element in the HIGH half, odd in the LOW) and widened in-shader by
`Widen(packed, element)`; each kernel gets its inputs concatenated into one
`StructuredBuffer<uint>` SRV and writes a `RWStructuredBuffer<uint>` UAV. The
leg runs `dx12` CLI mode through the same `KernelGate` harness (CPU gold
first, then the DX12 leg, timing table).

| kernel | gate vs CpuLeg (production Nivara) | worst | result |
|---|---|---|---|
| `dot16` (K=16) | `\|leg − cpu\| ≤ 1e-6 + 1e-5·\|cpu\|` | **0.0 ULP** (bit-exact) | PASS |
| `silu` (576) | tolerance gate per element | 4.0 ULP | PASS (576/576) |
| `gemv` (1536×576) | tolerance gate per row | 14 336 ULP @ row 1508 (`\|diff\| = 1.6e-9`, near-zero ref row) | PASS (1536/1536) |

Same gemv worst-ULP caveat as SYCL: the diagnostic row lands near zero where an
f32 ULP is tiny, far inside the `1e-6` absolute gate.

### Performance (Arc 140T, driver 1.15.37858; best-of-3 per run, best observed across runs)

**The figures live in the consolidated side-by-side table above.**

Reading: same profile as SYCL — tiny kernels drown in per-dispatch overhead
(PSO+wall clock), while the real SmolLM decode shape (`gemv`) is a decisive
GPU win approaching the SYCL figure (the gap is the per-iteration
copy/transition overhead of the probe's one-shot-transient-buffer design;
a production `src/Nivara.Gpu` would keep persistent buffers). Descriptor-handle
structs return through a **hidden pointer**, not RAX — `GetCpu/GpuDescriptorHandleForHeapStart`
take an `out` slot (the first ABI bug this leg hit, fixed after a heap-vtable crash).

## OpenVINO leg (phase 3 — first-party, proven)

`OpenVino/` is a pure P/Invoke leg against the pip-installed `openvino_c.dll`
(the official Intel wheel `pip install openvino==2026.2.1`; no NuGet — no
official runtime package exists, only rejected third-party repacks). The runtime
dir is discovered at runtime (`NIVARA_OPENVINO_DIR` env > python-package probe >
documented `site-packages\openvino\libs` default) and loaded with
`LOAD_WITH_ALTERED_SEARCH_PATH` so `openvino.dll` + plugins + TBB resolve. All
C-API entry points are `GetProcAddress`-by-name `Cdecl` delegates; struct/enum
constants come from the pinned release's `openvino.h` (the DX12 lesson — never
from memory).

`OvIrModels.cs` hand-emits the three kernels as dependency-free IR v11 XML +
raw-BF16 `.bin` (dot16 = `MatMul[1,16]·[16,1]`, silu = `Sigmoid` then `Multiply`
over 576, gemv = `MatMul[1536,576]·[576]`), each with a `ConvertToF32` + `Result`
tail so the plugin hands back F32 tensors for exact gating. The leg compiles on
explicit device `"GPU"` (never `AUTO`), **asserts the read-back
`EXECUTION_DEVICES` contains `GPU`** (observed `GPU.0`), and runs two precision
configs: the **bf16 row** (BF16-declared IR, compiled with *no*
`INFERENCE_PRECISION_HINT` — `=BF16` is a CPU-only token the GPU plugin rejects;
it resolves internally to F16) and the **f32 row** (F32-declared IR + mandatory
`INFERENCE_PRECISION_HINT=f32`, else the plugin silently FXs to F16). A fresh
temp `CACHE_DIR` is used per (config, run) — a reused one across configs crashed
the process (0xC0000005).

Gates vs the production Nivara CPU kernels:

| config | kernel | gate | result |
|---|---|---|---|
| `bf16` | `dot16` | `\|leg − cpu\| ≤ 1e-6 + 1e-5·\|cpu\|` | **PASS — 0.0 ULP (bit-exact)** |
| `bf16` | `silu` (576) | tolerance gate per element | **honest FAIL** — 402/576; F16-elementwise sigmoid/multiply on a BF16-declared model (~1.25e-5 @ 0.014). Precision finding, not a miscode ([docs/OPENVINO.md](../../docs/OPENVINO.md) §3) |
| `bf16` | `gemv` (1536×576) | tolerance gate per row | **PASS** — 1536/1536 (F32-accumulated) |
| `f32` | `dot16` | tolerance | **PASS — 0.0 ULP (bit-exact)** |
| `f32` | `silu` (576) | tolerance | **PASS** — 576/576, worst 4.0 ULP |
| `f32` | `gemv` (1536×576) | tolerance | **PASS** — 1536/1536 |

Readings: the tuned GPU plugin does **not** carry the hand-authored SPIR-V bug
class (no FMul/FDiv miscompile, no access-chain ICE); the f32 config is the
direct-IGC-class correctness proof and passes everything. The bf16 row keeps
the F32-tier matmul win (reductions accumulate in F32) while honestly surfacing
the F16-tier elementwise silu rounding. Timings live in the consolidated
side-by-side table above.

gemv — the dominating SmolLM decode shape — runs ~26–34× the CPU on the OV
plugin with zero compiler and zero packages. Full notes in [docs/OPENVINO.md](../../docs/OPENVINO.md).

## ILGPU leg (phase 4a — pure-managed C#→OpenCL JIT, proven)

`Ilgpu/` is the probe's **first NuGet-based backend** (issue #431): `ILGPU 1.5.3`
+ `ILGPU.Algorithms 1.5.3` (NCSA, pure managed) JIT-compile ordinary C# static
methods to OpenCL C, executed on the Arc 140T iGPU through the in-box
`OpenCL.dll` ICD + Intel driver — no SDK, no toolchain, no native install. Kernels
are implicitly-grouped `Index1D` methods over the same packed-2-per-uint BF16
transport as DX12 (in-shader widen via `Interop.IntAsFloat`, silu via
`XMath.Exp` — the OpenCL `exp` intrinsic; core ILGPU has no exp, hence the
Algorithms package). The context is OpenCL-only (`builder.OpenCL()`), the device
is selected by `CL_DEVICE_TYPE_GPU` + Intel vendor, and the leg **asserts the GPU
type — a CPU fallback is impossible** (unreachable ⇒ UNBUILT row, never a run on
CPU). Setup (accelerator creation + first sync ~9 ms; per-kernel JIT ~1.3–3.8 ms) is timed
separately from steady state (1 warmup + best-of-25 synchronized dispatches,
persistent buffers — the OpenVINO methodology).

Gates vs the production Nivara CPU kernels:

| kernel | gate vs CpuLeg (production Nivara) | worst | result |
|---|---|---|---|
| `dot16` (K=16) | `\|leg − cpu\| ≤ 1e-6 + 1e-5·\|cpu\|` | **0.0 ULP** (bit-exact) | PASS — 11.4 µs steady |
| `silu` (576) | tolerance gate per element | 4.0 ULP | PASS (576/576) — **6.9 µs, fastest GPU leg** |
| `gemv` (1536×576) | tolerance gate per row | 14 336 ULP @ row 1508 (`\|diff\| = 1.6e-9`, near-zero ref row) | PASS (1536/1536) — 133.4 µs |

Same gemv worst-ULP caveat as the other legs (diagnostic row near zero, far
inside the `1e-6` absolute gate). **IGC verdict: PASS** — ILGPU-generated OpenCL
C is handled correctly by the same frontend that mangles hand-authored SPIR-V
([docs/SPIRV.md](../../docs/SPIRV.md) §3), matching the SYCL finding: *compiler*-produced code is what
IGC runs right. Readings: silu and dot16 are the fastest GPU-leg figures measured
anywhere in the series; gemv trails only OpenVINO's *tuned* gemm (86.8 µs bf16)
with a deliberately naive one-thread-per-row shape. Native BF16 kernel types
don't exist in ILGPU 1.5.3 (upstream PR #1221 open), so the packed-widen path is
the primary one — byte-identical to DX12. Full notes in [docs/ILGPU.md](../../docs/ILGPU.md).

## Level Zero P/Invoke surface

Structs, constants, and proc addresses are taken only from the official
`ze_api_v1282.h` (API target 1.28 layout) — see `LevelZero/L0Native.cs` for the
hand-verified constant/layout values (`ST_MODULE_DESC = 0x1b`,
`MODULE_FORMAT_IL_SPIRV = 0`, `ZeModuleDesc` layout, etc.). The loader resolves
entry points by name from `ze_loader.dll`.

## Files

- `Program.cs` — CLI dispatch (`list` / `run` / `spv` / `ocl` / `l0` / `dx12` /
  `sycl` / `ov` / `kernels` / `all`).
- `Kernels/KernelFixtures.cs` — SmolLM-shaped BF16 fixtures + shared native
  write-BF16 / read-f32 helpers.
- `Kernels/LegResults.cs` — one leg's results over the fixture set
  (`dot16` / `silu` / `gemv`), produced by every leg for the gate, plus
  per-kernel wall time in µs for the timing table.
- `Kernels/KernelGate.cs` — the multi-leg correctness gate harness: CPU gold +
  every wired GPU leg, per-kernel gate rows with worst-ULP diagnostics, and a
  CPU-vs-GPU timing table (µs); exit code = failed cells. `kernels` CLI mode
  runs it six-way (SYCL + DX12 + OV-bf16 + OV-f32 + ILGPU; SYCL row prints
  UNBUILT on this machine).
- `Kernels/CpuLeg.cs` — **the production-kernel CPU leg (gold target)** for the
  gate: dot/GEMV via `LlamaFusedKernels.MatMulTransposedB<float>`
  (aRows=1 → the allocation-free BLAS2 GEMV path the fused Llama head runs),
  SiLU via `Activation.Silu` (`GradKernels.Silu`, sigmoid-then-multiply), all
  consumed read-only through public API, plus `ComputeLeg` over the full
  fixtures (per-kernel `Stopwatch` timings). Carries the gate helpers
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
  `D3D12CreateDevice` at FL 11_0..12_2, shader model via `CheckFeatureSupport`
  (feature-validated: a bogus feature value is rejected with E_INVALIDARG to
  prove the call is live). Shared P/Invoke helpers: `Vtable<T>` (vtable-slot
  dispatch), `Release`, `CreateFactory`, `TryCreateDevice`.
- `D3d12/D3d12Compute.cs` — the hand-rolled DX12 compute leg: full pipeline
  (layout validation → device → queue/allocator/list/fence/root-signature/heap
  → per-kernel D3DCompile + PSO + UPLOAD/DEFAULT/READBACK buffers + SRV/UAV
  views → 1 warmup + 3 timed dispatch passes → readback f32 → `LegResults`).
  All vtable slots/GUIDs/structs taken from the SDK header, with runtime
  identity self-checks on the descriptor heap.
- `D3d12/GemvKernels.cs` — HLSL `cs_5_1` source builders for the three kernels
  (`dot16`, `silu` = `x/(1+exp(−x))`, `gemv`), the in-shader `Widen(packed,
  element)` BF16→f32 unpack, and the host-side `PackBf16` 2-per-uint packing.
- `LevelZero/OclProbe.cs` — OpenCL loader diagnostic (dead end: an OpenCL ICD
  dispatch path is not pursued; kept only for the Intel-export function discovery).
- `Sycl/sycl_runner.cpp` — DPC++ runner: `dot16`/`gemv`/`silu` kernels
  (`sycl::queue` + USM, device-side `uint16 << 16` BF16→f32 emul-widen), raw
  BF16 in → raw f32 out; prints `TIME <mode> = X us` (best-of-3 `submit→wait`,
  post-JIT steady state). Built by `Sycl/build.cmd` (`icpx -fsycl -O2`, sources
  VsDevCmd + oneAPI setvars).
- `Sycl/build.cmd` / `Sycl/run.cmd` — build entry point; launcher that sources
  oneAPI `setvars` so the child process resolves `sycl8.dll`/`ur_loader.dll`
  (a direct spawn dies with `STATUS_DLL_NOT_FOUND`).
- `Sycl/SyclLeg.cs` — the .NET SYCL leg (transport only): writes the fixtures,
  spawns `run.cmd`, reads the f32 outputs + `TIME` lines, returns a `LegResults`
  with per-kernel timings. Gating lives in `KernelGate`, not here.
- `Sycl/.gitignore` — keeps `sycl_runner.exe` / LLVM objects out of git.
- `OpenVino/OpenVinoRunner.cs` — runtime-dir discovery (env > python probe >
  pip default) + `LoadLibraryEx` altered-search-path load + `GetProc<T>`.
- `OpenVino/OpenVinoNative.cs` — the `openvino_c.dll` C-API surface: core
  create/properties/read/compile, compiled-model property read-back, infer
  request + tensor lifecycle, element-type/shape helpers, `ov_get_last_err_msg`
  wired into every status check.
- `OpenVino/OvIrModels.cs` — dependency-free IR v11 XML + raw-BF16 `.bin`
  writers for `dot16` / `silu` / `gemv`, `bool bf16` flavor param (BF16-declared
  vs F32-declared), fixed `ConvertToF32`+`Result` tails.
- `OpenVino/OpenVinoLeg.cs` — the dual-config leg: `RunBf16` (no precision hint;
  plugin resolves F16) and `RunF32` (mandatory `INFERENCE_PRECISION_HINT=f32`),
  each compiling on `"GPU"`, asserting `EXECUTION_DEVICES` contains `GPU`, fresh
  `CACHE_DIR` per (config, run), 1 warmup + best-of-25 timed infers.
- `OpenVino/Availability.cs` — `ov` mode banner: runtime version +
  `FULL_DEVICE_NAME` read-back.
- `Ilgpu/IlgpuKernels.cs` — the three production kernels as ILGPU device methods
  (implicitly-grouped `Index1D`, bounds-checked for padded grids), in-shader
  `Widen` via `Interop.IntAsFloat`, silu via `XMath.Exp` (ILGPU.Algorithms).
- `Ilgpu/IlgpuLeg.cs` — the phase-4a leg runner: OpenCL-only O2 context,
  Intel-GPU `CL_DEVICE_TYPE_GPU` device select + assert (no CPU fallback),
  persistent packed-BF16 buffers, `LoadAutoGroupedKernel` launchers, setup/JIT vs
  best-of-25 steady-state split, `LegResults`.
- `Ilgpu/Availability.cs` — `ilgpu` mode banner: OpenCL devices + chosen
  accelerator.

## Recommendations

- **BF16 compute model is proven end-to-end on the SYCL leg**: `.NET 11 BFloat16`
  in/out, GPU in the middle, `f32` accumulation (the hardware's native model — Xe2
  DPAS also accumulates BF16 products in `f32`), zero host widening. `dot16` is
  bit-exact against the production CPU kernel and the full 1536×576 GEMV and 576
  SiLU gates pass inside the tolerance bound.
- **OpenVINO is the lowest-infrastructure proven path (phase 3)**: pip install +
  pure P/Invoke, no compiler, no subprocess, no packages. dot16 bit-exact, gemv
  ~26–34× CPU; the bf16 row's silu honestly reports F16-elementwise rounding
  (precision, not correctness — the f32 row passes all gates). Live candidate
  for a managed `src/Nivara.Gpu`.
- **ILGPU is the managed-JIT proven path (phase 4a)**: NuGet-only install,
  kernels in plain C#, in-box OpenCL ICD. All three gates PASS on the real iGPU
  (no CPU fallback); fastest GPU leg on silu (6.9 µs ≈ 5–14× CPU) and on the
  dot16 floor. The IGC question is answered — ILGPU-generated OpenCL C runs
  correctly. Naive one-thread-per-row gemv trails OpenVINO's tuned gemm; a tiled
  gemv is the next lever for a `src/Nivara.Gpu` promotion decision.
- **Hand-authored SPIR-V on this driver is a dead end for real math** (bug #5):
  the L0 leg cannot perform FP multiply or divide at all, so no dot/GEMV/SiLU is
  expressible through hand-written bytecode. Documented as the authoritative
  verdict of this probe phase; **kernel authoring has pivoted to the oneAPI
  SYCL/DPC++ toolchain and is proven correct** — `icpx`-compiled DPC++ kernels
  (the same stack llama.cpp's SYCL backend uses on Arrow Lake Arc iGPUs) gate
  PASS against the production Nivara kernels while running through the real SYCL
  runtime on Level Zero. Whether a real `src/Nivara.Gpu` should ride the SYCL
  runtime or load compiler-produced native images through the probe's L0 harness
  is a promotion decision for later; both paths are now live options.
- **DX12 is a proven second compiler-backed path (commit 8)**: the Arc 140T
  reaches FL 12_2 + SM 6.8, and the hand-rolled HLSL→DXIL→PSO pipeline gates all
  three kernels PASS against the production Nivara kernels with zero external
  tooling (in-process `d3dcompiler_47`). Readies the option of a managed-native
  GPU backend (HLSL kernels + P/Invoke bindings) without the oneAPI subprocess.
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