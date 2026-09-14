# ILGPU on the Arc 140T from .NET — Lessons from the GpuProbe

This is the fifth GPU-backend case-study doc (after `docs/SPIRV.md`, `docs/SYCL.md`,
`docs/DX12.md`, `docs/OPENVINO.md`) and the deliverable for issue #431 (GPU probe
phase 4a). The previous phases proved three compile/driver paths (hand-authored
SPIR-V over Level Zero — honest dead end; the oneAPI SYCL/DPC++ toolchain over
Level Zero; hand-rolled HLSL `cs_5_1` over D3D12) and one first-party runtime
(OpenVINO's precompiled GPU plugin). ILGPU is the **first fully-managed option on
the stack**: a pure-C# JIT kernel runtime (NCSA license, no native dependencies)
that compiles ordinary C# static methods to *OpenCL C* and runs them through the
in-box Windows `OpenCL.dll` ICD + Intel graphics driver. The open empirical
question, exactly parallel to the SYCL phase: **does Intel's IGC frontend — the
same frontend that mangles hand-authored SPIR-V — handle *ILGPU-generated* OpenCL
C correctly?** This document records the measured answer.

Companion phase 4b (#432) applies the same contract to ComputeSharp (DXIL/DX12
kernels) with its own doc; this doc covers only the ILGPU/OpenCL leg. All findings
below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU, driver
**1.15.37858**) running Windows 10.0.26200 and .NET 11.0 (Release), ILGPU
**1.5.3** + ILGPU.Algorithms **1.5.3** from NuGet.

---

## 1. Toolchain / setup (the "managed JIT" path)

OpenVINO needed a pip install; SYCL needed the oneAPI toolchain; DX12 needed a
compiler DLL. ILGPU needs **none of that**:

```xml
<PackageReference Include="ILGPU" Version="1.5.3" />
<PackageReference Include="ILGPU.Algorithms" Version="1.5.3" />
```

ILGPU 1.5.3 (published 2025-07-12; there is no 1.5.4/2.0) is a pure-managed
library: `dotnet restore` alone makes the backend available. The GPU itself is
reached through the **in-box Windows OpenCL ICD loader** (`OpenCL.dll`, shipped
with Windows) + the Intel graphics driver's OpenCL platform — **no SDK, no CUDA,
no custom runtime install**. `ILGPU.Algorithms` is the optional sibling package
that supplies `XMath.Exp` — see §4; core `IntrinsicMath` has no exp.

Footprint (measured, `bin/Release/net11.0`):

| assembly | size |
|---|---|
| `ILGPU.dll` | 1946 KB |
| `ILGPU.Algorithms.dll` | 1502 KB |
| transitive (System.Collections.Immutable, System.Memory, System.Reflection.Metadata, System.Runtime.CompilerServices.Unsafe) | resolve from the net11.0 shared framework — **no extra copies** |

The probe's **first** NuGet package references (everything else is pure P/Invoke)
— a deliberate, documented exception: ILGPU is the one backend where the package
*is* the toolchain.

## 2. API shape & kernel model (what the leg actually does)

The leg (`tests/Nivara.GpuProbe/Ilgpu/`) is small: three C# kernel methods, one
runner, one availability banner.

```csharp
using var context = Context.Create(builder => builder.OpenCL().Optimize(OptimizationLevel.O2));
CLDevice? device = IlgpuLeg.SelectGPU(context);                    // CL_DEVICE_TYPE_GPU + Intel name
using CLAccelerator accelerator = device.CreateCLAccelerator(context);
using AcceleratorStream stream = accelerator.CreateStream();

using var input  = accelerator.Allocate1D<uint>(packed.Length);    // persistent buffers
using var output = accelerator.Allocate1D<float>(n);
input.CopyFromCPU(packed);

var kernel = accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<uint>, ArrayView<float>, int>(Kernels.Dot16Kernel);
kernel(stream, 1, input.View, output.View, 16);                     // high-level launcher, no boxing
stream.Synchronize();
float[] result = output.AsContiguous().GetAsArray();
```

Kernels are ordinary static methods (`Index1D` index first, then views/params):

```csharp
static void Dot16Kernel(Index1D index, ArrayView<uint> input, ArrayView<float> output, int k)
{
    int i = index;
    if (i > 0) return;                        // implicit grouping pads the grid — must bounds-check
    float acc = 0f;
    for (int j = 0; j < k; j++)
        acc += Widen(input[j >> 1], (uint)j) * Widen(input[(k + j) >> 1], (uint)(k + j));
    output[0] = acc;
}
```

Key facts verified against the 1.5.3 sources (many are non-obvious, all cost
compile-fix cycles during the leg build):

- The 1D index type is **`Index1D`** in 1.5.3 (the `Index1` of older docs no
  longer exists — `Index2D`/`Index3D`/`LongIndex1D` follow the same scheme).
- `LoadAutoGroupedKernel<TIndex, T1, …>` returns an
  `Action<AcceleratorStream, TIndex, …>`; the high-level launcher means **no
  boxing** and no manual grid/group math.
- Implicit grouping rounds the grid up to group-size multiples, so **every kernel
  must bounds-check its padded index** — identical reason to the HLSL leg's
  `if (dt.x >= n) return;`.
- `CLDeviceType` exposes the raw OpenCL names: `CL_DEVICE_TYPE_GPU`,
  `CL_DEVICE_TYPE_CPU`, … (not `GPU`/`CPU`).
- Buffer readback is `view.AsContiguous().GetAsArray()` — the generic
  `GetAsArray<T>(ArrayView<T>)` extension does not infer through the implicit
  `ArrayView1D<T, Dense>` → `ArrayView<T>` conversion, so `.View.GetAsArray()`
  fails to compile (CS0411); `AsContiguous()` returns the non-generic view
  directly.
- Setup is split honestly: context+accelerator creation **and** kernel JIT +
  first dispatch are timed separately from steady state (see §5).

## 3. The IGC verdict — measured, not assumed

The SYCL phase proved *compiler-produced* SPIR-V mul/div survives IGC while
hand-authored SPIR-V does not. ILGPU emits **OpenCL C**, which the same IGC
frontend then compiles. The phase question: does that path work on Arc 140T?

**Verdict: PASS.** The leg ran on the actual Arc 140T iGPU
(`Intel(R) Graphics`, `CL_DEVICE_TYPE_GPU`, vendor `Intel(R) Corporation` — the
leg asserts the device type and refuses to fall back to any CPU device), IGC
compiled every ILGPU-generated kernel, and all three production-shape gates
passed against the production Nivara CPU kernels (the multi-leg gold target,
`|leg − cpu| ≤ 1e-6 + 1e-5·|cpu|`):

| kernel | gate vs CpuLeg (production Nivara) | worst | result |
|---|---|---|---|
| `dot16` (K=16) | tolerance gate | **0.0 ULP** (bit-exact) | PASS |
| `silu` (576) | tolerance gate per element | 4.0 ULP | PASS (576/576) |
| `gemv` (1536×576) | tolerance gate per row | 14 336 ULP @ row 1508 (`\|diff\| = 1.63e-9`, near-zero ref row) | PASS (1536/1536) |

Same gemv worst-ULP caveat as the SYCL/DX12/OV legs: the diagnostic row lands
near zero where an f32 ULP is tiny, far inside the `1e-6` absolute gate — a
*diagnostic*, not the pass/fail bound. The `kernels` exit code on this machine
stays the documented 405 = 402 (honest OV-bf16 silu) + 3 (SYCL UNBUILT); the
ILGPU row contributes **0** failures.

Reading: the bug class that blocks hand-authored bytecode (§3 of
`docs/SPIRV.md`) does **not** affect ILGPU — as with SYCL, **compiler-produced
code is what IGC handles correctly**. For ILGPU the "compiler" is a .NET
library, not a toolchain install.

## 4. BF16 — the packed-widen path (no native BF16 in 1.5.3)

ILGPU 1.5.3 has **no native BF16 kernel type** (BF16/FP8 support is an open PR,
#1221; the v2.0 branch carries PR #1577 — both unmerged at the time of writing).
`Half` cannot carry BF16 bit patterns, so the issue's specified primary path is
used, byte-identical to the DX12 leg's transport:

- Host: `D3d12.GemvKernels.PackBf16` packs BF16 pairs 2-per-`uint` (element 2k in
  the HIGH half, 2k+1 in the LOW). Concatenated fixtures are packed *after*
  concatenation so absolute element parity is preserved across buffer segments.
- Device: the in-shader `Widen(packed, element)` reads the correct half and
  reinterprets with ILGPU's kernel-safe bit intrinsic
  `Interop.IntAsFloat(bits << 16)` — the managed mirror of HLSL `asfloat`.

```csharp
private static float Widen(uint packed, uint element) =>
    (element & 1u) == 0u
        ? Interop.IntAsFloat(packed & 0xFFFF0000u)
        : Interop.IntAsFloat((packed & 0xFFFFu) << 16);
```

The widen is exact (BF16→f32 is a left-shift of 16, no rounding), which is why
`dot16` gates at **0.0 ULP** and the accumulated GEMV/SiLU results sit inside
tolerance. This is the proven production BF16 wire story from the open-weights
pipeline: `.NET 11 BFloat16` in/out, GPU in the middle, f32 accumulation — the
hardware's native Xe2 model. Upstream alternatives (natively typed BF16 kernels,
which would remove the pack/widen entirely) remain tracked:
`SpawnDev.ILGPU` (fork with native BF16), `ILGPU.Lightning` (sample lib), and
ILGPU v2.0.

## 5. Setup vs steady state (the honest split)

The three timings that matter, measured on the Arc 140T (µs; six-way `kernels`
run unless noted):

| kernel | jit (load + first dispatch) | steady (1 warmup + best-of-25) | CPU (production Nivara) | margin |
|---|---|---|---|---|
| `dot16` (K=16) | 2668 | **11.4** | 1.2–4.8 | launch-bound — CPU still wins; ~3–6× faster than the other GPU legs (32–560 µs) |
| `silu` (576) | 1317 | **6.9** | 29–99 | **~4.6–14× GPU — fastest GPU leg measured on silu** |
| `gemv` (1536×576) | 1371 | **133.4** | 2291–4745 | **~17–35× GPU** |

Context + OpenCL accelerator creation: ~8.8–9.8 ms once. Per-kernel JIT costs
land between ~1.3 ms (silu) and ~2.7–3.8 ms (dot16) and are one-time; steady
state is a single `kernel(stream, extent, …)` + `stream.Synchronize()` per
dispatch on persistent buffers. `dot16`'s 11.4 µs steady beat every other GPU leg
(DX12 189 µs, OV 32–43 µs) despite remaining CPU-bound-class — a small,
uniformly-sized kernel where launch round-trip dominates.

gemv perspective: the naive shape (one thread per output row, serial K=576 f32
accumulate — the exact production formula) runs 133.4 µs, ~1.8× faster than the
equally-naive DX12 kernel (246 µs) and behind only OpenVINO's *tuned* gemm bf16
row (86.8 µs). The obvious levers (thread-per-8-rows FMA loops, work-group
tiling, `LoadAutoGroupedKernel` group-size tuning, `v2.0`-era bf16) are all
preserved for a real `src/Nivara.Gpu`. Default float IEEE math was kept
(`MathMode.Fast` was deliberately **not** used — the gate demands honesty).

## 6. PRO/CON + gotchas matrix (ILGPU row)

| aspect | ILGPU (OpenCL) |
|---|---|
| **delivery** | NuGet: `ILGPU 1.5.3` + `ILGPU.Algorithms 1.5.3` (NCSA, pure C#) — first package ref in the probe |
| **toolchain** | none — JIT-compiles C# kernels to OpenCL C in-process; in-box `OpenCL.dll` + Intel driver only |
| **IGC/OpenCL verdict** | **PASS** — compiler-produced OpenCL C handled correctly (same frontend that mangles hand-authored SPIR-V) |
| **BF16** | no native BF16 kernel type in 1.5.3 (PR #1221 open) → packed-2-per-uint + in-shader widen, byte-identical to DX12 transport |
| **correctness** | dot16 **0.0 ULP** · silu worst 4.0 ULP · gemv within gate — **3/3 PASS** |
| **setup cost** | ~9 ms context+accelerator; ~1.3–3.8 ms kernel JIT each |
| **steady state** | dot16 **11.4 µs** · silu **6.9 µs** (fastest GPU leg) · gemv **133.4 µs** |
| **footprint** | +3.4 MB (ILGPU 1946 KB + Algorithms 1502 KB); BCL transitives inbox on net11 |
| **risk** | JIT + ICD layer = two moving compilers (ILGPU → OpenCL C → IGC); no native BF16 today; OpenCL ICD required at runtime |
| **gotchas (hit during the leg)** | `Index1D` naming (not `Index1`); implicit grouping pads the grid → **bounds-check every kernel**; `CLDeviceType` uses `CL_DEVICE_TYPE_*` names; `XMath.Exp` requires the Algorithms package (core has no exp); readback via `AsContiguous().GetAsArray()` (generic inference skips the implicit view conversion); device-select assert (`CL_DEVICE_TYPE_GPU`) prevents silent CPU fallback |

## 7. Conclusions & decision records

- **Phase question answered**: ILGPU-generated OpenCL C survives Intel IGC on the
  Arc 140T — all three production kernels gate PASS on the real iGPU, with no
  CPU fallback and the accelerator name/type printed.
- ILGPU is the **lowest-friction compiled path**: `dotnet restore` is the whole
  install; kernels are plain C# (the same language as the host), high-level
  launchers avoid boxing, and BF16 rides the proven packed-widen transport.
- It is also the **fastest measured GPU leg on silu** (6.9 µs, ~5–14× CPU) and
  within 1.5× of OpenVINO's tuned gemv with a naive kernel shape.
- As a promotion candidate for a managed `src/Nivara.Gpu`, ILGPU's trade is
  **open source + managed kernels vs OpenVINO's closed tuned runtime**; the
  deciding measure for real workloads will be a tiled gemv (next step), where
  OV's tuned gemm currently leads.
- Re-test after any driver update: the whole kernel surface re-measures in
  seconds via `dotnet run … -- kernels`. The IGC bug class is driver-versioned
  (see `docs/SPIRV.md`); ILGPU's OpenCL C path is equally exposed to future
  frontend changes.