# ILGPU on the Arc 140T from .NET — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md) · [SYCL / oneAPI](../tests/Nivara.GpuProbe/SYCL.md) · [DX12](../tests/Nivara.GpuProbe/DX12.md) · [OpenVINO](../tests/Nivara.GpuProbe/OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](../tests/Nivara.GpuProbe/COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)

This is the fifth GPU-backend case-study doc (after [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md), [SYCL / oneAPI](../tests/Nivara.GpuProbe/SYCL.md),
[DX12](../tests/Nivara.GpuProbe/DX12.md), [OpenVINO](../tests/Nivara.GpuProbe/OPENVINO.md)) and the deliverable for issue #431 (GPU probe
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

This document has two parts:

- **Part 1 — the spike.** Issue **#431** (GpuProbe phase 4a): does Intel's IGC frontend handle
  *ILGPU-generated* OpenCL C correctly? The verdict and the measurements, from `tests/Nivara.GpuProbe`.
- **Part 2 — the production implementation.** The ILGPU code in `samples/Nivara.Samples/Gpu/` built
  on that verdict — the runners, the kernels, the geometry decisions and the lessons found while
  writing them. The Silk backend that was ported from it now lives in `src/Nivara.Extensions/Silk/`
  ([SILK.md](SILK.md) Part 2). Written from the source, as the map for adding the next model.

---

# Part 1 — the spike (GpuProbe, #431)

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
*diagnostic*, not the pass/fail bound. The #606-gate `kernels` exit code on
this machine is **3** = 3 (SYCL UNBUILT) + **0** unexpected: the 402 OV-bf16
silu failures are diagnosed-honest (all ≤ 2 half-ULP from the F16 mirror —
pre-#606 the documented exit was 405 = 402 + 3); the ILGPU row contributes
**0** failures.

Reading: the bug class that blocks hand-authored bytecode (§3 of
[SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md)) does **not** affect ILGPU — as with SYCL, **compiler-produced
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

The three timings that matter, measured on the Arc 140T (µs; the six-way
`kernels` run this leg first shipped with). The "fastest GPU leg" margins below
were true **of that run**; the two-host profiles in #552 (Arc 140T) and #553
(Iris Xe) supersede them as orderings: on both hosts ILGPU ties Silk.NET at the
top of the small kernels and is disjointly slower than it on `gemv`, and its
`gemv` rank against DX12 flips between hosts — device-dependent. This section
keeps the leg's own six-way figures; the joint reading lives in the GpuProbe
`README.md` two-host tables.

| kernel | jit (load + first dispatch) | steady (1 warmup + best-of-25) | CPU (production Nivara) | margin |
|---|---|---|---|---|
| `dot16` (K=16) | 2668 | **11.4** | 4.1 [2.1–18.4] | launch-bound — CPU-class: ties the CPU on the Arc, loses to it on the Iris Xe (7.1 vs 78.8) |
| `silu` (576) | 1317 | **6.9** | 28.1 [16.1–50.2] | ~4× CPU on the Arc (median 8.6 vs 28.1); ties Silk.NET on both hosts; **no GPU win** on the Iris Xe (median 72.9 vs CPU 45.7) |
| `gemv` (1536×576) | 1371 | **133.4** | 2348.9 [817.1–2961.5] | ~23× CPU on the Arc (median 144.2 vs 2348.9); Silk.NET beats it on both hosts; DX12 rank flips between hosts |

Context creation and accelerator creation are timed as part of the leg's setup
split. Accelerator creation + first device sync measures ~8.8–9.8 ms once
(context creation costs a one-time few-ms more; both excluded from the
steady-state table). Per-kernel JIT costs
land between ~1.3 ms (silu) and ~2.7–3.8 ms (dot16) and are one-time; steady
state is a single `kernel(stream, extent, …)` + `stream.Synchronize()` per
dispatch on persistent buffers. `dot16`'s 11.4 µs steady beat every GPU leg in
that run (DX12 189 µs, OV 32–43 µs) despite remaining CPU-bound-class — a
small, uniformly-sized kernel where launch round-trip dominates. The two-host
profile ties the CPU on the Arc (4.1 [2.1–18.4] vs 11.4 [8.7–17.4]) and has the
CPU win on the Iris Xe (7.1 vs 78.8), so "CPU still wins" is host-dependent.

gemv perspective: the naive shape (one thread per output row, serial K=576 f32
accumulate — the exact production formula) ran 133.4 µs in this leg's run; the
#552/#553 profile puts the Arc median at 144.2 µs. Against the equally-naive
DX12 kernel the pair flips between hosts (Arc ILGPU 144.2 < DX12 250.4; Iris Xe
DX12 705.8 < ILGPU 859.7) — device-dependent, not a property. On the Arc,
ILGPU's gemv ties OpenVINO's *tuned* gemm bf16 row (114.0 [92.8–201.0]
overlaps 144.2 [135.8–167.4]) rather than trailing it. The obvious levers
(thread-per-8-rows FMA loops, work-group
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
| **setup cost** | ~9 ms accelerator creation + first sync (context costs a one-time few-ms more); ~1.3–3.8 ms kernel JIT each |
| **steady state** | dot16 **11.4 µs** · silu **6.9 µs** (this leg's six-way run; the #552/#553 profile ties silu with Silk.NET on both hosts) · gemv **133.4 µs** |
| **footprint** | +3.4 MB (ILGPU 1946 KB + Algorithms 1502 KB); BCL transitives inbox on net11 |
| **risk** | JIT + ICD layer = two moving compilers (ILGPU → OpenCL C → IGC); no native BF16 today; OpenCL ICD required at runtime |
| **gotchas (hit during the leg)** | `Index1D` naming (not `Index1`); implicit grouping pads the grid → **bounds-check every kernel**; `CLDeviceType` uses `CL_DEVICE_TYPE_*` names; `XMath.Exp` requires the Algorithms package (core has no exp); readback via `AsContiguous().GetAsArray()` (generic inference skips the implicit view conversion); device-select assert (`CL_DEVICE_TYPE_GPU`) prevents silent CPU fallback; **at most one `SharedMemory` allocation per kernel** — two `Allocate2D` calls with disagreeing extents get the second tile mis-placed by the OpenCL lowering (#468), and the 1-D `Allocate<float>` form costs 10–20% against one wide `Allocate2D` |

## 7. Conclusions & decision records

- **Phase question answered**: ILGPU-generated OpenCL C survives Intel IGC on the
  Arc 140T — all three production kernels gate PASS on the real iGPU, with no
  CPU fallback and the accelerator name/type printed.
- ILGPU is the **lowest-friction compiled path**: `dotnet restore` is the whole
  install; kernels are plain C# (the same language as the host), high-level
  launchers avoid boxing, and BF16 rides the proven packed-widen transport.
- On the six-way run it was the **fastest measured GPU leg on silu** (6.9 µs);
  the two-host profile ties it with Silk.NET on both hosts and makes the CPU
  comparison host-dependent: ~4× CPU on the Arc (8.6 vs 28.1), while on the
  Iris Xe the CPU is faster than every GPU leg on silu (45.7 vs 72.9). On
  `gemv` it is disjointly beaten by Silk.NET on both hosts and its rank against
  DX12 flips between hosts — #552 / #553 are the profile that fixes both
  readings as an ordering.
- As a promotion candidate for a managed `src/Nivara.Gpu`, ILGPU's trade is
  **open source + managed kernels vs OpenVINO's closed tuned runtime**; the
  deciding measure for real workloads will be a tiled gemv (next step). On the
  Arc, OV's tuned gemm currently *ties* ILGPU's naive kernel (band overlap) —
  not the "leads" the six-way run implied — while on the Iris Xe ILGPU's gemv
  at worst ties the CPU's noisy band (845.5–1212.0 vs 1127.7–11229.6).
- Re-test after any driver update: the whole kernel surface re-measures in
  seconds via `dotnet run … -- kernels`. The IGC bug class is driver-versioned
  (see [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md)); ILGPU's OpenCL C path is equally exposed to future
  frontend changes.
- **The `--gemm` f32 fingerprint is the thing to re-record first after a driver bump**, and its
  baseline is keyed by device *and* toolchain (`OpenCL <version>, driver <CL_DRIVER_VERSION>`),
  so a bump reports one `Fingerprint NOT VERIFIED` line naming both toolchains instead of 171
  numeric failures that read like a kernel regression. Two traps found the hard way:
  `CLDevice.DeviceVersion` is the **OpenCL** version, not the driver version — ILGPU 1.5.3
  exposes no driver string, so it needs a `clGetDeviceInfo(CL_DEVICE_DRIVER_VERSION)` P/Invoke —
  and re-recording the reference is a separate explicit flag, because a gate that can rewrite
  its own reference is a gate whose green means nothing. A *missing* baseline fails the run; with
  nothing to be exact against, the run verified nothing.

---

# Part 2 — the production implementation

Where Part 1 asked *whether* ILGPU's OpenCL path is trustworthy on this device, Part 2 is the code
built on the answer. The ILGPU half is written from `samples/Nivara.Samples/Gpu/`; the Silk half
that was ported from it now lives in `src/Nivara.Extensions/Silk/` ([SILK.md](SILK.md) Part 2).
Read this part as the map for adding a model to the ILGPU path.

## 8. Layout: what is in `samples/Nivara.Samples/Gpu/`

| file | role |
|---|---|
| `IlgpuRuntime.cs` | OpenCL-only `Context`/`CLAccelerator`/stream at `OptimizationLevel.O2`; a device select that **asserts a GPU and has no CPU fallback**; `Allocate1D`, `Synchronize` |
| `GpuBuffers.cs` | ILGPU upload, readback and launch maths. `Transposed` / `QkvPacked` / `Req` delegate to `Nivara.Silk.WeightLayout` so Silk produces byte-identical weights from the same arithmetic. Also `Ensure`, `Readback`, `Cfg1D`/`GemmCfg`, `ValidateGemmSharedMemory`, `ValidateAttentionLocalMemory`; constants `LinearGroupSize` = 256, `AttentionGroupSize` = 64, `MaxHeadDim` = 64, `GlobalAttentionBand` = −1 |
| `Nivara.AI.IBertEncoderRunner` | the backend seam, in `src/Nivara.Extensions/AI/GpuSeam.cs`: `DeviceName` + `Forward(int[] tokenIds, float[] mask, int batch, int seqLen)`. `BertEncoderGpuRunner` implements it |
| `GemmKernels.cs` | `TiledGemmKernelRow4` (the keystone) and the 1×1 baseline; the fused-epilogue siblings `Row4Bias`/`Row4Gelu`/`Row4Relu`/`Row4Qkv`; the four #440 register-blocked geometries, kept as a labelled negative; `TiledGemm` |
| `AttentionKernels.cs` | `BatchedAttention` — score + scale + mask + row-softmax + weighted-V in three passes, banded |
| `ElementwiseKernels.cs` | `Gather`, `AddBias`, `SplitColumns`, `Add`, `LayerNormResidual1D`, `EmbeddingSum`, `LayerNorm1D`, `GeluExact`, `Gelu`, `Relu`, `Rotary`, `GeGlu` |
| `BertEncoderGpuRunner.cs` | BERT-family encoder (MiniLM, DistilBERT) + the optional SST-2 head; `BertGpuNaming`/`BertGpuKeys` make it naming-driven |
| `ModernBertGpuRunner.cs` | ModernBERT encoder; exposes `HiddenOnDevice`/`HiddenRows`/`ForwardOnDevice` |
| `LayaHeadGpuRunner.cs` | the Laya typed head, hung off the encoder's on-device hidden state |
| (Silk) | the second backend moved to `src/Nivara.Extensions/Silk/`; documented in [SILK.md](SILK.md) Part 2 |

## 9. Runtime and the no-CPU-fallback contract

`IlgpuRuntime` builds the context OpenCL-only (`builder.OpenCL().Optimize(OptimizationLevel.O2)`,
`IlgpuRuntime.cs:36`). `SelectGpu` (`:57–69`) returns the first device that is `CL_DEVICE_TYPE_GPU`
**and** whose name contains "Intel"; if none, construction throws ("--ilgpu requires an OpenCL GPU
device…"). It then asserts the *accelerator's* device type too (`:47–51`) — a driver that answers the
enumeration with something else must fail loudly rather than become a CPU measurement. `XMath` from
`ILGPU.Algorithms` is available to kernels without an explicit `EnableAlgorithms` call
(probe-verified). `Device` is exposed for its `DeviceId` (the native `cl_device_id`);
`CLDevice.DeviceVersion` is the **OpenCL** version, not `CL_DEVICE_DRIVER_VERSION`, which ILGPU
1.5.3 does not surface.

## 10. The GEMM family

`TileSize = 16` and groups are 16×16 everywhere. The keystone is **`TiledGemmKernelRow4`**
(`GemmKernels.cs:68`): each thread owns a **1×4** register block — 4 adjacent output columns of one
row — so a 16×16 group covers a 16-row × 64-col tile and each K step reads `BlockRows + BlockCols =
5` shared floats for 4 MACs, against the 1×1 baseline's 2 reads per 1 MAC. It was measured at
303–379 GMAC/s on all DistilBERT shapes against the naive gemv's 6.6 GMAC/s, and its ascending-K
order is identical to the 1×1 kernel, so the results are **bit-identical** — which is what kept
parity simple. The 1×1 form itself is local-memory bound at ~150 GMAC/s flat, regardless of shape.

Both Row4 and the 1×1 kernel stage A and B in **one wide `Allocate2D`** (`RowStride = TileSize +
TileCols = 80`; A in columns `[0,16)`, B in `[16,80)`), which measured ~20% faster than the earlier
two-allocation form (#468, below).

The fused-epilogue siblings keep Row4's 1×4 geometry, single allocation and ascending-K order and
differ **only** in the epilogue — so each is bit-identical to the unfused GEMM + add sequence:

- `Row4Bias` (`:514`) — `y = A·Bt + bias`, the lean identity epilogue.
- `Row4Gelu` (`:584`) — `y = gelu(A·Bt + bias)`, calling the shared `ElementwiseKernels.GeluExact`.
- `Row4Relu` (`:654`) — `y = max(A·Bt + bias, 0)`, written `(r < 0 ? 0 : r)`.
- `Row4Qkv` (`:737`) — takes an extra `blockWidth` and scatters the result into a **block-separable**
  layout `[q-block | k-block | v-block]` via `QkvDest` (`:808`), so attention reads each block as a
  dense `[rows × hidden]` view. Still bit-identical to three separate GEMM + add sequences.

Three siblings rather than one kernel with a runtime activation byte is deliberate: a shared kernel
would inline all three paths and bloat registers on the hot lean launches.

### 10.1 Labelled negative: the #440 register-blocked geometries

`GemmGeometries` (`GpuBuffers.cs:134`) holds four wider-block geometries (`2x2@KT16`, `2x2@KT32`,
`4x2@KT32`, `1x8@KT16`) whose kernels live in `GemmKernels.cs:197–501`, each flagged **"Measured
slower than Row4 — not adopted."** All four were gated and measured on 2026-09-29 and every one lost
on every shape: best cell 1.01× (a wash inside run-to-run noise), typical 0.6–0.85×, and the
launch-bound narrow shapes worst at 0.25–0.42×. The full ratio table is in
`tests/Nivara.PerformanceTests/README.md` ("GEMM gate baseline (2026-09-29)").

They lose because **the premise was wrong**: they widen the per-thread output block to cut shared
floats read per MAC by 20–40% on paper, but the binding constraint on this iGPU is shared-memory
*capacity per group*, not *traffic*. The isolating pair — `2x2@KT16` against `2x2@KT32` — has
identical blocking factors, hence identical reads per MAC, and differs only in footprint (4 KB vs
8 KB); on the nine largest shapes the 8 KB one is 18–25% slower, every one of them, and `4x2@KT32`
agrees with the mechanism (best traffic ratio 0.75 reads/MAC, worst footprint 12 KB, mid-pack). The
scoped claim is "footprint dominates at saturation", not "footprint predicts the ordering" — the
unscoped version is false on 6 of 15 shapes. Row4 therefore stands, and the geometries stay in the
file so the negative is reproducible.

All four keep the incumbent's group, K-tile staging and ascending-K accumulation, so each is
bit-identical to Row4 — and the `--gemm` gate asserts that byte-identity alongside its
double-precision truth check, which is what caught #468: an ILGPU OpenCL lowering bug that returns
plausible wrong values and **cannot be seen in review**.

### #784 candidates (2×4 / 4×4 @ K16)

Issue #784 asks to revisit row blocking with **2×4** and **4×4** geometries (keeping 4 output
columns per thread but 2 or 4 rows). Implemented as `TiledGemmKernelReg2x4K16` and
`TiledGemmKernelReg4x4K16` (`GemmKernels.cs`) and registered in `GpuBuffers.GemmGeometries`
indices 4/5, following the same row-striping + single-allocation pattern as the #440 family.
Both are plain (no fused epilogue) so the byte-identity claim reduces to the accumulation;
epilogues are ported only for the winner. Shared footprint: 2×4 ≈ 5 KB, 4×4 ≈ 5 KB (KT-wide
A strip), both under the 64 KB device limit. Measurement pending — do not route a runner at
either until the `--gemm` gate shows a win.

### 10.2 Lesson #468: at most one shared allocation per kernel body

Every #440 kernel that staged a KT-wide A tile against a TileCols-wide B tile returned *plausible*
wrong values (`maxAbs` 20–119 — the right order of magnitude for a real dot product taken against the
wrong elements, so it does not look like an out-of-bounds smear), and one failed at `LoadKernel`
with **`CL_OUT_OF_RESOURCES`** on a device with shared memory to spare. ILGPU's OpenCL lowering
mis-places the second allocation when the two extents disagree. The narrow rule first written — "the
two allocations need matching extents" — was **wrong**: the `1x8@KT16` row had a square 16×16 A tile
and still would not build, so Row4's survival was a property of this build and driver, not of its
source. The rule that holds is **stage every tile through one allocation**;
`tests/Nivara.Tests/Gpu/SharedMemoryAllocationTests.cs` enforces it on the compiled IL.

After #470 the two families are on the same wide `Allocate2D` form — the #440 kernels were ported
from 1D `Allocate<float>` with hand-computed 2-D offsets so the family can be re-measured on one
addressing form. The 2D form is 10–20% faster on this device. Leave a kernel's form alone unless it
is being changed for a measured reason.

## 11. Attention

`BatchedAttention` (`samples/Nivara.Samples/Gpu/AttentionKernels.cs:36`) runs one work item per `(b, h, q)` and computes the
softmax row in **three passes, each recomputing the score** — max, exp-sum, then the
probability-weighted V accumulation into a shared-memory tile. It never spills the row: `[B, H, S, S]`
would be 268 MB at S=2048 and 4.3 GB at S=8192. The accumulator tile is `[MaxHeadDim ×
AttentionGroupSize]` with `DenseX(MaxHeadDim)` stride — column `Group.IdxX` is the thread's private
slice, so two work items in a group cannot race — and it is over-allocated to compile-time constants
because ILGPU needs a statically known allocation size while `headDim` is a runtime argument.
`GpuBuffers.ValidateAttentionLocalMemory` rejects a larger `headDim` at construction; every encoder
on this path has `headDim` 64, so the 16 KB tile costs no extra local memory.

Two numerics rules:

- **`max == -inf → zeros`** (`:86–91`): a row with every key suppressed would otherwise compute
  `s - max = NaN` and poison the residual stream below it. It mirrors the CPU safe-softmax clamp; the
  BERT-family runners pass `GlobalAttentionBand = −1` and never hit it, but a sliding band over a
  padded batch does.
- **The band restricts the loop bounds** (`:70–71`), computed once and reused across all three
  passes, so a sliding layer's cost tracks its window rather than `seqLen` (#447) — before this the
  band only overwrote an already-computed score with `-inf` and bought no arithmetic. `Keep` (`:152`)
  stays as the *authority* on the window even though the bounds now agree with it: the cost is one
  integer compare against a `headDim`-long dot product, and collapsing the test into the bounds would
  make the kernel silently wrong the first time someone widened a bound without thinking about it.

## 12. Elementwise kernels

Implicitly-grouped 1D kernels over flattened buffers: `Gather`; `AddBias`; `SplitColumns` (walks
rows, because a pre-fused projection is interleaved per row — lesson 13 below); `Add`; `EmbeddingSum`
(word/position/token-type gathers + sum in one launch, with `includeTt` gating the token-type read
because a view cannot be null); `LayerNorm1D`; `LayerNormResidual1D` (residual folded into the LN
row reduction; `y` may alias `a`); `Relu`; `Rotary` (half-split RoPE, one work item per
`(row, head, frequency)` writing **both** halves so `rotate_half` needs no cross-work-item
communication); `GeGlu` (activates the *first* half of the fused `Wi` projection,
`gelu(Wi[0:I]) * Wi[I:2I]`).

The exact-GELU polynomial has **one definition** on the GPU path: `internal static float
GeluExact(float v)` (`ElementwiseKernels.cs:171`), a direct A–S 7.1.26 erf port with `XMath.Exp`
(ILGPU has no `erf`). Both `Gelu` and the GELU GEMM epilogues call it, so the polynomial cannot
drift.

## 13. The BERT-family runner and the seam

`BertEncoderGpuRunner` uploads the loader's weight key set once (GEMM weights transposed
`[out,in] → Bt[in,out]` so `C = A·Bt` is plain row-major and global reads coalesce along K), then runs
the whole forward with every intermediate resident on device and only the final hidden / logits read
back (`BertEncoderGpuRunner.cs:76`). It is config-driven from `BertConfig` and naming-driven via
`BertGpuNaming { DistilBert, Bert }` + `BertGpuKeys`, so MiniLM and DistilBERT share it:
`embeddings`/`encoder.layer.N.*` for Bert against `distilbert.embeddings`/`distilbert.transformer.layer.N.*`
for DistilBert (Bert keys are pre-stripped of the `bert.` prefix during provisioning).

Per layer: fused QKV GEMM (`Row4Qkv`) → three sub-views → attention → o-projection (`Row4Bias`) →
`LayerNormResidual` → fc1 (`Row4Gelu`) → fc2 (`Row4Bias`) → `LayerNormResidual`. Workspace is
cap-sized once (`rowsCap = 8*128`; `Forward` rejects `batch > 8 || seqLen > 128`), posIds are cached
per `seqLen` (`row % seqLen`, so the repeat-forward benchmark path skips the copy), and token-type
embeddings are added only when the key is present.

The optional SST-2 head runs entirely on device: `Gather` the CLS row → `Row4Relu` (`pre_classifier`)
→ `Row4Bias` (`classifier`), producing `[batch, 2]` logits. `IBertEncoderRunner` is the seam, and the
*only* difference between the two backends is what sits behind `Forward`. The padding contract is
stated on the interface: mask `0` at padding, positional ids `0..seqLen−1` repeated per batch row.

## 14. ModernBERT

`ModernBertGpuRunner` mirrors `ModernBertEncoder<T>.Forward` operation for operation: rotary instead
of learned positions, a gated feed-forward instead of a plain FFN, pre-norm instead of post-norm, and
**per-layer attention geometry** (band + rope theta). It is the first runner sharing no structure
with the BERT family.

- **RoPE tables come from the CPU** (`RotaryEmbedding<T>.GetPositionTables`, now public) rather than
  device `XMath.Cos` — one host-libm-vs-`XMath` ulp per entry would otherwise be the largest single
  error term in the gate (lesson 14).
- **Bias-free GEMMs**: all projections use plain `TiledGemmKernelRow4` and adds are the separate
  `Add`; norms pass a shared zero-filled `zeroBeta` because a view cannot be null.
- The fused `Wi` GEMM writes `2*intermediateDim`; `SplitColumns` then separates the per-row
  interleaved q/k/v; `GeGlu` reads the fused gate/up buffer directly.
- `LayerWeights.Band`/`.Full` come from `config.Band(index)`/`config.IsFullAttention(index)`; layer
  0's `attn_norm` is an identity.

Its seam for a downstream head is additive: `HiddenOnDevice` (the final-normed hidden, still on
device), `HiddenRows`, `ForwardOnDevice`. The buffer is the runner's own and grown-never-shrunk, so a
head must run before the next `Forward`.

## 15. Laya head, and the on-device seam

`LayaHeadGpuRunner` needed **zero new kernels**: its layer is `nn.TransformerEncoderLayer` defaults —
pre-norm, *biased* LayerNorm, ReLU FFN, fused *biased* QKV, no RoPE, no band — all already present.
It reads `ModernBertGpuRunner.HiddenOnDevice` and only the marker/act logits cross to the host. Two
easy-to-invert details are called out in the code: the residual lives in the head layer (adding it
inside attention would double-count), and every norm here is **biased** where every ModernBERT norm
is not. The act head's four features are computed on the host by `LayaHeadScoring` (device would need
a softmax, top-k and entropy kernel) and the assembled `[hidden + 4]` row is uploaded.

## 16. Design decisions worth keeping

- **Correctness first**: one kernel per operation, one stream, one sync before readback. Fusion and
  stream tricks were deferred until the thing was trivially verifiable.
- **Weights transposed once at upload** (`WeightLayout.Transposed`, which `GpuBuffers.Transposed`
  delegates to, so Silk produces byte-identical weights from the same arithmetic rather than a
  second copy). A *transpose-free*
  in-kernel GEMM was dropped for first-cut correctness; it is worth revisiting only if upload time
  ever matters, which it does not on a shared-DRAM iGPU.
- **Persistent cap-sized workspace** with `Ensure`-only growth; payload copies go through
  explicit-length `View.SubView(0, n)`.
- **Mask semantics copied exactly from the CPU**: per-batch column-`j` suppressed when `mask[b][j] <
  0.5`, identical across query rows.
- **Same-process CPU reference** beats fixture-only gating.

## 17. What we learned (the durable lessons)

1. **`ArrayView.CopyFromCPU<T>(T[])` fills the *whole* view** — its guard is `data.Length ≥
   view.Length`, and a shorter payload throws. Copy into `View.SubView(0, n)`; there is no
   "copy prefix" overload.
2. **Stream ordering is your responsibility.** Buffer-level `CopyFromCPU` uses the default stream,
   kernels use yours — route every copy through the kernel stream and sync after bulk ctor uploads.
3. **Launch overhead dominates at inference shapes — but the per-kernel cost is ~30 µs, not the
   ~0.4–0.7 ms first estimated.** M2 removed ~70 of ~100 dispatches and moved MiniLM 26.8 → 24.3 ms
   and DistilBERT 65.3 → 63.1 ms; launch overhead was ~3 ms of the baseline, not ~21 ms.
4. **A tiled GEMM is a requirement, not an optimization** — the naive gemv's 6.6 GMAC/s would take
   ~825 ms for DistilBERT's matmuls; Row4 hit 303–379 GMAC/s with the per-output-column accumulation
   order unchanged.
5. **Double-precision-truth GEMM gates catch real bugs** — a Row4 `colBase` bug (global thread-Y
   instead of group index) produced maxAbs ~40–77 and was caught by the bounds check.
6. **f32 parity between summation orders has a floor** — ~4.4e-5…1.6e-4 (K=768…3072), so a `1e-6`
   gate is impossible; the `1e-3·(1+|x|)` bound is right, and real bugs still trip it at ~40-class
   outliers.
7. **`XMath` has no `erf`** — GELU needs an in-kernel polynomial; the direct A–S 7.1.26 port with
   `XMath.Exp` matched CPU `GeluExact` inside hidden-state parity.
8. **Battery throttles the iGPU flat** — every shape drops to ~150–160 GMAC/s regardless of size.
   **All GPU perf claims require AC power.**
9. **ILGPU deployment is the whole story** — NuGet package = the entire install, in-box Windows
   `OpenCL.dll` + Intel ICD, kernels are C# static methods. Assert `CL_DEVICE_TYPE_GPU` and re-run the
   gates after every driver bump (two compilers in the path: ILGPU JIT + IGC).
10. **The iGPU shares DRAM** — the weight upload is memcpy-class, so transposing once at upload costs
    nothing and buys GEMM coalescing.
11. **Workspace hygiene** — allocate cap buffers once and reuse (`Ensure` only grows); integer
    payloads need an explicit integer-buffer alloc. `GetAsArray()` returns the full cap; always slice
    by payload length.
12. **Same-process CPU reference beats fixture-only gating** — fixtures may be absent.
13. **A pre-fused projection is interleaved per row, not block-concatenated.** Row `r` holds
    `[q(r) | k(r) | v(r)]`, so contiguous `SubView` offsets are right for row 0 and wrong for every
    row after it — the first row looks perfect. `SplitColumns` walks rows. The gate/up buffer needs
    no split because `GeGlu` indexes the fused buffer directly: check which layout the consumer
    actually assumes before reaching for a split.
14. **RoPE tables should come from the CPU, not device `XMath`.** Recomputing them on device is one
    host-libm-vs-`XMath` ulp per entry, a three-order-of-magnitude regression against a `maxRel`
    ~3e-6 — for a difference unrelated to whether the port is correct. Promote the authoritative
    definition to an accessor rather than duplicating the formula.
15. **ILGPU's `SharedMemory` is per work *group*, not per work item, and its `Allocate` size must be
    statically known.** A first attempt gave each thread its own `Allocate<float>(64)` and every
    thread raced on the same 64 floats — structurally wrong, not rounding-level. The tile is 2-D with
    the size over-allocated to a compile-time constant. Query `MaxSharedMemoryPerGroup`, **not**
    `MaxLocalMemorySize`, and size the single allocation for the whole group (lesson 19).
16. **Verify a suspected kernel defect with an isolated probe before blaming the runner.** A GEMM and
    an uploaded-weight bit-compare cost one run and ruled out the two most expensive hypotheses.
    Build the reference with the *exact* layout the device buffer has, every time.
17. **Diagnostic references are code and they rot like code.** Of five "kernel bugs" chased for #449,
    three were faults in the throwaway stage diagnostic (block-major vs row-major, per-row interleave
    vs block-contiguous, and a missing `GeluExact` on a gated activation — the last showing as a
    suspiciously perfect 100% relative error).
18. **The ILGPU stream is in order; do not go looking for a race.** Lesson 2 is the real ordering
    rule; "the default stream is unordered" is not one of its failure modes.
19. **At most one `SharedMemory` allocation per kernel body** — see §10.2 for how the narrower rule
    ("matching extents") was first written, and how the `1x8@KT16` row refuted it.

## 18. Adding the next model

1. Work from the CPU implementation and mirror it **operation for operation**; the parity gate is the
   contract, so restructuring a CPU op is a decision to make deliberately.
2. Decide which existing runner it resembles. If it shares the BERT-family shape, extend
   `BertEncoderGpuRunner` (it is config- and naming-driven) — do not fork it.
3. Reuse the seam for a head (`HiddenOnDevice`/`ForwardOnDevice`) before writing kernels: the Laya
   head needed none.
4. Check the fused weights' layout before slicing them — per-row interleaved needs `SplitColumns`,
   block-concatenated does not.
5. Take any reference tables (RoPE, and anything `libm` computes) from the CPU, not the device.
6. Ask whether any kernel needs a second shared allocation; if so, restructure to one (#468).
7. Gate it: same-process CPU compare at the `1e-3·(1+|x|)` bound first, then fixtures.
8. Sizes are hard caps today (`batch ≤ 8`, `seqLen ≤ 128`). Raise them with the scenario, and say so
   in the error message when you do.

---

**Series:** [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md) · [SYCL / oneAPI](../tests/Nivara.GpuProbe/SYCL.md) · [DX12](../tests/Nivara.GpuProbe/DX12.md) · [OpenVINO](../tests/Nivara.GpuProbe/OPENVINO.md) · [docs/ILGPU.md](ILGPU.md) · [ComputeSharp](../tests/Nivara.GpuProbe/COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md) — the GPU-backend case-study series