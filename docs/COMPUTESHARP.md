# ComputeSharp on the Arc 140T from .NET — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · **ComputeSharp**

This is the sixth GPU-backend case-study doc and the deliverable for issue #432
(GPU probe phase 4b). The previous phases proved four compile/driver paths
(hand-authored SPIR-V over Level Zero — honest dead end; the oneAPI SYCL/DPC++
toolchain over Level Zero; hand-rolled HLSL `cs_5_1` over D3D12) one first-party
runtime (OpenVINO's precompiled GPU plugin) and one fully-managed JIT
(ILGPU → OpenCL C). ComputeSharp is the **second fully-managed option and the
first *managed-D3D12* path**: a pure-C# library (MIT, no native dependencies of
its own) that source-generates HLSL from ordinary C# structs, compiles it to
**DXIL** in-process via the **DXC binaries bundled by `ComputeSharp.Dxc`**, and
dispatches it on the iGPU through D3D12. The open empirical question: **can the
source-gen → bundled-DXC → D3D12 pipeline match the hand-rolled DX12 leg's
correctness with a fraction of the plumbing** (no vtable dispatch, no root
signature, no descriptor heaps), and where does its steady-state perf land?
This doc records the measured answer.

The hand-rolled DX12 leg ([docs/DX12.md](DX12.md)) proved in phase 2 that managed
.NET can drive the iGPU's D3D12 compute pipeline directly. ComputeSharp is the
library wrapper around exactly that idea — the same `d3d12.dll`/`dxgi.dll`
surface, reached through a NuGet package instead of hand-pinned vtable slots.

**Machine status — read this before the numbers.** The measurement target this
series documents is the **Intel Arc 140T** (8086:7DD1, 128 EU, driver
1.15.37858). The phase-4b leg itself was validated live on a **second machine**
during this phase — an **Intel Iris Xe iGPU** over D3D12, Windows 10.0.26200,
.NET 11.0 (Release), ComputeSharp **3.2.0** + ComputeSharp.Dxc **3.2.0** from
NuGet — where all three production-shape gates PASSed with no CPU fallback
(WARP rejection exercised, not hit). Measured figures from that run are recorded
below in *clearly labeled Iris Xe* rows; the Arc 140T figures remain **pending**
until a `kernels` run happens on that machine. The two iGPUs are different
hardware — treat the Iris Xe numbers as functional proof, not as the Arc 140T
performance answer.

---

## 1. Toolchain / setup (the "managed D3D12" path)

SYCL needed the oneAPI toolchain; DX12 needed `d3dcompiler_47.dll` + hand-rolled
vtable dispatch; OpenVINO needed a pip install; ILGPU needed `dotnet restore`
alone. ComputeSharp needs **`dotnet restore` alone** — for both halves of the
compiler story:

```xml
<PackageReference Include="ComputeSharp" Version="3.2.0" />
<PackageReference Include="ComputeSharp.Dxc" Version="3.2.0" />
```

The base `ComputeSharp` package depends only on `ComputeSharp.Core` — the
HLSL→DXIL compilation at runtime is handled by the DXC binaries
(`dxcompiler.dll` + `dxil.dll`) **bundled inside `ComputeSharp.Dxc`** (the
package's runtimes/ assets are deployed to the output), so no external compiler
or toolchain install exists. Both packages are MIT and target net8.0
(compatible with the probe's net11.0).

Footprint (measured, `bin/Release/net11.0`):

| assembly | size |
|---|---|
| `ComputeSharp.dll` | 516 KB |
| `ComputeSharp.Core.dll` | 809 KB |
| `ComputeSharp.Dxc.dll` | 29 KB |
| `ComputeSharp.Dxc` native `dxcompiler.dll` (win-x64) + `dxil.dll` | 16.56 MB + 1.36 MB |

The probe's **second** NuGet package references (everything else is pure
P/Invoke) — the same deliberate, documented exception as ILGPU: ComputeSharp is
a backend where the package *is* the toolchain (source generator + embedded
DXC).

## 2. API shape & kernel model (what the leg actually does)

The leg (`tests/Nivara.GpuProbe/ComputeSharp/`) is small: three shader structs,
one runner, one availability banner.

```csharp
using GraphicsDevice device = GraphicsDevice.GetDefault();   // hardware first, WARP last
Console.WriteLine(device.Name);                              // adapter description
if (!device.IsHardwareAccelerated) return null;              // honest UNBUILT row, never CPU

using ReadOnlyBuffer<uint>  input  = device.AllocateReadOnlyBuffer(packed);   // packed BF16, 2-per-uint
using ReadWriteBuffer<float> output = device.AllocateReadWriteBuffer<float>(n);

device.For(n, new SiluShader(input, output, n));             // synchronous: submit + wait
float[] result = output.ToArray();                           // readback
```

Shaders are ordinary `readonly partial struct`s implementing `IComputeShader`,
annotated `[GeneratedComputeShaderDescriptor]` + `[ThreadGroupSize(...)]` — the
source generator lowers them to HLSL:

```csharp
[ThreadGroupSize(256, 1, 1)]
[GeneratedComputeShaderDescriptor]
internal readonly partial struct SiluShader(
    ReadOnlyBuffer<uint> input, ReadWriteBuffer<float> output, int n) : IComputeShader
{
    public void Execute()
    {
        int i = ThreadIds.X;
        if (i >= n) return;                                   // D3D12 pads the grid to group-size multiples
        float x = Widen(input[i >> 1], (uint)i);
        output[i] = x / (1f + Hlsl.Exp(-x));
    }
}
```

Key facts verified against the rel/3.2.0 sources (non-obvious, all validated
during G1 grounding):

- **`GraphicsDevice.GetDefault()`** is the 3.2.0 API (the 2.x-era
  `GetDefaultDevice` no longer exists); it walks adapters with WARP last, so
  the default device is hardware on a normal machine — and we additionally
  assert `IsHardwareAccelerated` (WARP detection), returning an honest
  KernelGate UNBUILT row instead of ever running on the software rasterizer.
- Kernel index is **`ThreadIds.X`** (`Int3`, mapped to `SV_DispatchThreadID`);
  group geometry is `[ThreadGroupSize(x, y, z)]`, and the X-dispatch form is
  **`device.For(int x, in T shader)`** — one call = one full round-trip
  (the compute context executes the command list and waits for completion on
  dispose; verified in `ComputeContext.Dispose` → `ExecuteAndWaitForCompletion`),
  so timing semantics match the ILGPU `dispatch + stream.Synchronize()` idiom.
- Buffers are **`ReadOnlyBuffer<T>` / `ReadWriteBuffer<T>`** (`T : unmanaged`),
  allocated via `AllocateReadOnlyBuffer(uint[])` /
  `AllocateReadWriteBuffer<float>(n)` extension methods; readback is
  `ToArray()` (BufferExtensions) / `CopyTo(span)`.
- `[GeneratedComputeShaderDescriptor]` is the descriptor attribute in 3.x
  (the source generator also requires the containing type be declared
  `partial` — the generator emits a partial for shader descriptor plumbing;
  `[AutoConstructor]` is 2.x-era and gone).
- The compute shaders compile to **SM 6.x HLSL** (not DX12's hand-authored
  `cs_5_1`) — DXC produces real DXIL, the non-deprecated shader-model path.
- Mutable shader fields are not allowed — the struct is `readonly`; captured
  scalar params (`int n`, etc.) ride along as root constants.

## 3. The correctness verdict — measured, not assumed

The hand-rolled DX12 leg proved HLSL→DXBC over D3D12 is correct on the Arc 140T.
ComputeSharp produces **DXIL** (SM 6.x) via its bundled DXC — a different
compiler backend and shader model than `d3dcompiler_47`'s `cs_5_1` DXBC. The
phase question: does the source-gen + bundled-DXC pipeline produce correct
results for the three production shapes on a real Intel iGPU?

**Verdict: PASS** (validated on the Iris Xe machine, §machine-status above).
The leg ran on a real D3D12 hardware device (`device.Name` printed;
`IsHardwareAccelerated` asserted), DXC compiled every generated shader (SM 6.x
DXIL), and all three production-shape gates passed against the production
Nivara CPU kernels (the multi-leg gold target,
`|leg − cpu| ≤ 1e-6 + 1e-5·|cpu|`):

| kernel | gate vs CpuLeg (production Nivara) | worst (Iris Xe run, measured) | result |
|---|---|---|---|
| `dot16` (K=16) | tolerance gate | **0.0 ULP** (bit-exact) | PASS |
| `silu` (576) | tolerance gate per element | 4.0 ULP @ index 410 | PASS (576/576) |
| `gemv` (1536×576) | tolerance gate per row | 12 288 ULP @ row 1508 (`\|diff\| = 1.4e-9`, near-zero ref row) | PASS (1536/1536) |

*Arc 140T worst-ULP values pending a `kernels` run on that machine.*

Same gemv worst-ULP caveat as every other leg: the diagnostic row lands near
zero where an f32 ULP is tiny, far inside the `1e-6` absolute gate — a
*diagnostic*, not the pass/fail bound. On the Iris Xe validation machine the
`kernels` exit code was **9** (3 SYCL UNBUILT + 6 OpenVINO UNBUILT — neither
toolchain installed on that machine), **0 unexpected**; the ComputeSharp row
contributes **0** failures. On the Arc 140T machine the documented exit stays
405 (402 honest OV-bf16 silu + 3 SYCL UNBUILT) with ComputeSharp adding 0 —
re-checked when the Arc 140T run happens.

Reading: DXIL from ComputeSharp's bundled DXC is exact to the same bound as the
hand-rolled `cs_5_1` DXBC — yet another evidence point that **compiler-produced
bytecode is what the driver handles correctly**, whether it is SYCL SPIR-V,
OpenCL C (ILGPU), or DXIL (ComputeSharp).

## 4. BF16 — the packed-widen path (same transport as DX12/ILGPU)

ComputeSharp has no `BFloat16` buffer element type worth using here (its
`Half` is fp16, not BF16), so the issue's unmodified directive applies — the
byte-identical packed transport shared with the DX12 and ILGPU legs:

- Host: `D3d12.GemvKernels.PackBf16` packs BF16 pairs 2-per-`uint` (element 2k
  in the HIGH half, 2k+1 in the LOW). As with ILGPU, fixtures are concatenated
  *first* then packed, so absolute element parity is preserved across buffer
  segments.
- Device: the in-shader `Widen(packed, element)` reads the correct half and
  reinterprets with **`Hlsl.AsFloat(bits << 16)`** — ComputeSharp's intrinsic
  for HLSL `asfloat`, the managed mirror of both `Interop.IntAsFloat` (ILGPU)
  and the DX12 leg's raw HLSL:

```csharp
internal static float Widen(uint packed, uint element) =>
    (element & 1u) == 0u
        ? Hlsl.AsFloat(packed & 0xFFFF0000u)
        : Hlsl.AsFloat((packed & 0xFFFFu) << 16);
```

The widen is exact (BF16→f32 is a left-shift of 16, no rounding) — the reason
`dot16` gates at **0.0 ULP**. This is now the **same proven grain of
transport across four independent GPU toolchains** (SYCL, DX12, ILGPU,
ComputeSharp), which is itself a strong finding: the packed-widen wire format
is backend-agnostic. silu uses **`Hlsl.Exp`** (HLSL `exp` intrinsic — the
ComputeSharp counterpart of ILGPU's `XMath.Exp`).

## 5. Setup vs steady state (the honest split)

**Arc 140T target figures — pending** (record from the first `kernels` run on
that machine; the CPU reference ranges are the series-wide documented values):

| kernel | dxc (HLSL→DXIL + first dispatch) | steady (1 warmup + best-of-25) | CPU (production Nivara) | margin |
|---|---|---|---|---|
| `dot16` (K=16) | _pending_ | _pending_ | 1.2–4.8 | _pending_ |
| `silu` (576) | _pending_ | _pending_ | 29–99 | _pending_ |
| `gemv` (1536×576) | _pending_ | _pending_ | 2291–4745 | _pending_ |

**Measured on the Iris Xe validation machine** (§machine-status above; µs,
`kernels` run — CPU and ComputeSharp from the same run, so the margins are
self-consistent):

| kernel | dxc (HLSL→DXIL + first dispatch) | steady (1 warmup + best-of-25) | CPU (production Nivara, same run) | margin |
|---|---|---|---|---|
| `dot16` (K=16) | 53 431 | 215.7 | 3.8 | 0.018× (launch-bound; CPU wins) |
| `silu` (576) | 11 013 | 206.5 | 46.8 | 0.23× (CPU wins on this iGPU) |
| `gemv` (1536×576) | 26 746 | **624.2** | 2824.8 | **4.5× CPU** |

For-probe comparison on the same machine/run: the hand-rolled DX12 leg measured
303.5 / 224.7 / 882.1 µs and ILGPU 74.8 / 84.3 / 871.9 µs on the same three
kernels — so ComputeSharp was **fastest of the D3D12 legs on gemv** (624.2 vs
882.1 hand-rolled DX12, 871.9 ILGPU) on this iGPU, and its managed per-dispatch
round-trip dominates the tiny launch-bound kernels (dot16 215.7 µs vs ILGPU
74.8 µs). None of these Iris Xe figures are the Arc 140T answer.

Device creation (`GraphicsDevice.GetDefault()`) is the one-time setup (tens of
ms through the D3D12 runtime). Per-kernel **DXC compile is split out honestly**:
the first dispatch of each shader pays HLSL source-gen → DXC → DXIL → PSO
creation (the `dxc` column, one-time; measured 11–53 ms on Iris Xe); steady
state is repeated `device.For(...)` + wait on **persistent buffers** — nothing
is recompiled or re-uploaded between timed passes. Expected profile:
launch-bound `dot16`, GPU-win `gemv` — confirmed (624.2 vs 2824.8 µs CPU) — and
silu caught in the managed per-dispatch overhead on the small 576-element
shape.

## 6. PRO/CON + gotchas matrix (ComputeSharp row)

| aspect | ComputeSharp (DXIL/D3D12) |
|---|---|
| **delivery** | NuGet: `ComputeSharp 3.2.0` + `ComputeSharp.Dxc 3.2.0` (MIT, pure C# + bundled DXIL compiler) — second package ref in the probe |
| **toolchain** | none — source-generates HLSL from C# structs; in-process DXC (bundled `dxcompiler`/`dxil`) → DXIL → D3D12; **no d3dcompiler_47, no hand-rolled vtable dispatch** |
| **shader model** | SM 6.x DXIL via DXC (the non-deprecated path) vs the DX12 leg's `cs_5_1` DXBC via FXC |
| **IGC/D3D12 verdict** | **PASS** — DXIL handled correctly on a real D3D12 Intel iGPU (validated on Iris Xe: 3/3 gates; Arc 140T verdict pending its `kernels` run) |
| **BF16** | no BF16 buffer element type → packed-2-per-uint + in-shader `Hlsl.AsFloat` widen, byte-identical to DX12/ILGPU transport |
| **correctness** | dot16 **0.0 ULP** · silu ≤4 ULP · gemv within gate — **3/3 PASS** |
| **setup cost** | one-time device creation; per-kernel DXC compile (the `dxc` split) then PSO — compiled once, reused (measured 11–53 ms on Iris Xe) |
| **steady state** | Iris Xe measured: dot16 215.7 µs · silu 206.5 µs · gemv **624.2 µs (4.5× CPU)** — managed per-dispatch overhead dominates launch-bound kernels; **Arc 140T pending** |
| **footprint** | _pending_ (ComputeSharp.dll + Core.dll + Dxc.dll + native dxcompiler/dxil win-x64 bundle) |
| **risk** | source generator + bundled DXC = two moving compilers; per-dispatch `ComputeContext` creation is a managed overhead on tiny kernels; Windows/D3D12-only (CA1416 platform-annotated, honest) |
| **gotchas (hit during the leg)** | `GraphicsDevice.GetDefault()` (not `GetDefaultDevice`); container type must be **`partial`** for the shader descriptor generator; `[SupportedOSPlatform("windows6.2")]` needed on the leg (ComputeSharp APIs are platform-annotated → CA1416 without it); shaders are `readonly partial struct` implementing `IComputeShader` (no mutable fields, `[AutoConstructor]` is 2.x-era); `Hlsl.AsFloat(uint)` (not `AsUInt`) is the exact asfloat intrinsic; each `For()` is a full submit+wait round-trip (timing semantics identical to ILGPU's `Synchronize`); grid padded to group-size multiples → bounds-check every kernel (`if (i >= n) return;`) |

## 7. Conclusions & decision records

- **Phase question answered**: ComputeSharp's source-gen → bundled-DXC → DXIL →
  D3D12 pipeline produces correct results for all three production kernels on a
  real Intel iGPU (validated on the Iris Xe machine) — the same gate outcome as
  the hand-rolled DX12 leg, at a
  fraction of the plumbing (~100 lines of pure C# shaders vs ~460 lines of
  vtable/descriptor/heap code). The Arc 140T perf target remains open — this
  doc's Iris Xe figures are functional proof, and the steady-state numbers on
  the series' target machine are re-measured via a single `kernels` run.
- ComputeSharp is the **lowest-friction D3D12 path**: `dotnet restore` is the
  whole install, kernels are plain C# structs in the same language as the host,
  and the bundled DXC keeps the DXIL compiler out of any external install story.
- As a promotion candidate for a managed `src/Nivara.Gpu`, the managed-D3D12
  trio (ComputeSharp / hand-rolled DX12 / TerraFX) is now bracketed on both
  ends; ComputeSharp brings the HLSL↔C# ergonomics that the hand-rolled leg
  lacks, while both D3D12 legs sidestep the IGC OpenCL/SPIR-V frontend entirely.
- Re-test after any driver update: the whole kernel surface re-measures in
  seconds via `dotnet run … -- kernels`. DXIL handling is driver-versioned like
  every other path in the series.

**Series:** [docs/SPIRV.md](SPIRV.md) · [docs/SYCL.md](SYCL.md) · [docs/DX12.md](DX12.md) · [docs/OPENVINO.md](OPENVINO.md) · [docs/ILGPU.md](ILGPU.md) · **ComputeSharp** — the GPU-backend case-study series