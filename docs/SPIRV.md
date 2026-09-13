# Hand-authored SPIR-V on Intel Level Zero — Lessons from the GpuProbe

This is the first of the GPU-backend case-study docs. It documents what the
`tests/Nivara.GpuProbe` L0 leg (commits 1–4 of the kernel-probe phase) found
when it tried to write kernels **by hand** — `spirv.core.grammar.json` opcode by
opcode — against the Intel **Level Zero** API, and why that path is a **dead
end for real math on this driver**, plus what still works (the safe subset) and
what the findings imply for every compiler-backed path that followed. The
series continues with `docs/SYCL.md` (oneAPI DPC++ → SPIR-V), `docs/DX12.md`
(HLSL `cs_5_1` DXBC), and the fourth, `docs/OPENVINO.md`
(Intel OpenVINO GPU plugin — issue #428).

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU,
driver **1.15.37858**, Level Zero API **1.15**) running Windows 10.0.26200 and
.NET 11.0 RC1. The L0 leg is pure P/Invoke against the inbox
`ze_loader.dll` + `ze_intel_gpu.dll` — no packages, no toolchain.

---

## 1. Why hand-author SPIR-V at all?

At probe start this machine had **no compiler toolchain** (no `clang`, no SPIR-V
tools), and the L0 availability probe had already shown the `Arc 140T` exposes
an Intel GPU that can load SPIR-V modules. Hand-authoring the bytes is the
minimal-dependency way to answer "can this GPU run compute kernels at all, and
what is the safe instruction subset?" — before committing to any toolchain.
Every kernel in `LevelZero/SpvKernels.cs` is emitted word-by-word against the
`spirv.core.grammar.json` opcode tables (results are encoded **result-first**,
not operand-order — the classic first-page bug, caught by a local Python
disassembler `spvdis.py`).

The probe has three entry points relevant here:

```bash
dotnet run -c Release --project tests/Nivara.GpuProbe -- list   # enumerate L0 drivers/devices/extensions
dotnet run -c Release --project tests/Nivara.GpuProbe -- spv    # dump hand-authored SPIR-V to %TEMP%\opencode\spv
dotnet run -c Release --project tests/Nivara.GpuProbe -- run    # build + launch all kernels; 15 expected driver-bug diagnostics, exit 0
```

## 2. What works — the L0 safe subset

| gate | kernel | result |
|---|---|---|
| `add_parallel` | 256 work items, each a **1-lane workgroup**, each `OpAtomicIAdd`s its global id | **PASS** — counter = 32640 = Σ(0..255) exactly |
| `add_loop` | 1 work item, 1,000,000 serial `fp32` adds in an `OpPhi` loop | **PASS** — c[0] = 1,000,000.0; ≈5.03 ms → **≈0.20 GFADD/s** |
| `bf16_native` | BF16→f32 via `OpConvertBF16ToFINTEL` (SPV_INTEL_bfloat16_conversion) | **PASS** — exact, and reloading the widened f32 as BF16 returns the original `BFloat16` (native round-trip, zero host widening) |
| `bf16_emul` | BF16→f32 via `OpUConvert` + `<<16` (safe subset, no extension) | **PASS** — identical results to native |
| `bf16_native_acc` | 1M-iteration `acc += widen(BF16)` in an `OpPhi` loop | **PASS** — c[0] = 1,000,000.0 **exact** (f32 accumulate) |

The safe constructs, in one line each:

- Kernel params of `ptr<CrossWorkgroup, T>` (raw pointer and scalar args).
- Direct `OpLoad`/`OpStore` **through kernel-argument pointers** (`*c = *a + *b`).
- `OpFAdd`/`OpIAdd`/`OpULessThan`, constants, compare-and-branch control flow.
- `BuiltIn.GlobalInvocationId` (Input storage class).
- **`OpPhi` value-flow loops** — loops without any `OpVariable`.
- `OpAtomicIAdd` when every work item is its **own 1-lane workgroup**.
- **BFloat16 ↔ f32 round-trip** (native or emulated widen) — GPUs widen
  in-register from raw 16-bit patterns; the host never widens.

That last bullet is why the whole phase kept **BF16 on the wire** everywhere:
the transport (raw `BFloat16` bits → GPU-side widen) is validated on every path
that followed (SYCL, DX12 — and OpenVINO proved it again, with raw-BF16 `.bin`
weights consumed by the plugin).

## 3. What this driver cannot do (IGC OpenCL-frontend bugs)

Five driver bugs, each reproduced with minimal hand-authored SPIR-V and proven
by bisection (the `run` sweep prints all 17 variants). These are **driver bugs
at the IGC level** — not kernel-authoring errors (each "broken" kernel was
cross-checked against an in-place CPU mirror):

1. **Any access-chain opcode** (`OpAccessChain` 65, `OpInBoundsAccessChain` 66,
   `OpPtrAccessChain` 67) → `IGC: Internal Compiler Error: Access violation`,
   regardless of index type (u32/i32/u64), base pointer kind, or
   `Restrict`/`NoAlias` decorations. Same IGC bug class as IGCIT #1144 (Blender
   AV on Arc B580). **Rules out indexed memory access entirely** —
   `c[i] = a[i] + b[i]` is impossible on this build. Not fixable from SPIR-V;
   only a driver update changes it.
2. **Private storage-class `OpVariable`s** → linker error
   `undefined reference to 'gVar'`. Worked around with OpPhi value flow.
3. **Generic-pointer kernel args** (`ptr<Generic,...>`) → clean validation error
   (`GenericPointers are not allowed as kernel argument storage class!`).
4. **Fat `LocalSize` (8/16-vectorized) atomics drop the upper half of each SIMD
   vector** — at `LocalSize ≥ 8` only the low half of each `simd8/simd16`
   atomic lands (`LocalSize 256` gives 15808 instead of 32640). 1-lane
   workgroups compile to simd1 and land everything.
5. **`OpFMul`(131) executes as `OpFSub`(130) and `OpFDiv`(132) executes as
   `OpFMul`(131)** — a deterministic, opcode-specific IGC miscompile. Proven by
   exact-input single-op probes: widen BF16 `1.0`/`2.0`, `OpFMul` reads **−1**
   (i.e. 1−2), `OpFDiv` reads **2** (i.e. 1×2); `OpFAdd` reads 3 and
   `OpenCL.std exp` are correct. Consequence: the straight-line K=16 BF16 dot
   (structurally valid!) reads **0.0380020142 = Σ(a−b)** instead of the dot,
   and all 576 SiLU elements read **x·(1+exp(−x))** (the FDiv→FMul-mapped
   form) — both mirror-confirmed. **This kills real math on L0 for
   hand-authored SPIR-V.**

## 4. Language ceiling

- Driver reports **SPIR-V max 1.0**; a 1.2 header hangs `zeModuleCreate`
  (observed). All kernels are emitted as 1.0.
- The **NPU** (`Intel(R) AI Boost`, DDI driver, API 1.14) exposes **no SPIR-V
  support at all** (`spirvVersionSupported = 0`) — it cannot run these kernels.

## 5. The verdict and the pivot

**Hand-authored SPIR-V on Level Zero is a dead end for any real math on this
driver** — bug #5 alone means dot products, GEMV and SiLU (the SmolLM kernels
the phase targets) cannot be expressed correctly, and bug #1 rules out indexed
access that any real GEMM needs. The design decision this data forced:

- **Proof-of-correctness pivots to the oneAPI SYCL/DPC++ toolchain** (`icpx`-produced
  SPIR-V over Level Zero — the same path llama.cpp's SYCL backend uses on Arrow
  Lake Arc iGPUs). Toolchain-produced bytecode is handled differently by IGC
  than hand-authored SPIR-V: the SYCL leg **proven** all three kernels PASS
  (`docs/SYCL.md`).
- **DX12 remains the IGC-bypass side-path**: HLSL → DXBC `cs_5_1` goes through
  the driver's D3D12 compute frontend, entirely bypassing the buggy OpenCL
  frontend — and was later **compute-proven** end-to-end (`docs/DX12.md`).
- **OpenVINO (issue #428)** was the third, first-party option whose GPU plugin
  historically also sits on the OpenCL/IGC stack — its tuned, precompiled
  kernel set may behave differently from hand-authored SPIR-V. Measured: it
  does (**PASS all three kernels on the F32 config; the BF16 row's silu is an
  honest F16-precision finding, `docs/OPENVINO.md`**).

## 6. Lessons for Nivara's GPU strategy

- **Never ship hand-authored SPIR-V kernels assuming correctness on Intel
  Windows iGPUs.** The safe-subset win (BF16 transport, OpPhi loops, simd1
  atomics) is real, but FP mul/div being silently remapped to other opcodes is
  a correctness landmine with no SPIR-V-side defense.
- **The single-op, exact-input evidence probe is the reusable verification
  pattern.** `1 OP 2` through each opcode against a CPU mirror is how bug #5
  was pinned in minutes; any future driver/hardware bring-up (OpenVINO, Arc
  dGPGPU, newer drivers) should run the same battery before trusting kernels.
- **BF16 wire format + in-register widen is validated on every path.** Raw
  `BFloat16` bits in, GPU-side `<<16`/native widen — proven exact on L0
  (native + emulated), SYCL and DX12. Design debt zero.
- **Decision input for a future `src/Nivara.Gpu`:** on this
  Arc iGPU/Windows/driver, the proven compute paths are compiler-backed
(SYCL/oneMKL-shaped) and driver-shader-backed (DX12); L0 hand-authored is
   excluded; OpenVINO's verdict is in (#428, `docs/OPENVINO.md`). The four way
   numbers the case-study series ends up with (PR #427, `docs/SYCL.md`,
   `docs/DX12.md`, `docs/OPENVINO.md`) are the rough end-to-end estimate
   inputs.

## Reference links

- [Intel Level Zero spec (oneAPI spec)](https://spec.oneapi.io/level-zero/latest/)
- [SPIR-V specification (Khronos)](https://registry.khronos.org/SPIR-V/)
- [SPIR-V 1.0 core grammar](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [SPV_INTEL_bfloat16_conversion extension (Khronos)](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_bfloat16_conversion.asciidoc)
- [IGCIT #1144 — Blender access violation on Arc B580 (driver bug class)](https://github.com/intel/intel-graphics-compiler/issues/1144)
- Series: `docs/SYCL.md` · `docs/DX12.md` · `docs/OPENVINO.md` — issue #428