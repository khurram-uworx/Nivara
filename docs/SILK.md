# Silk.NET OpenCL GPU Probe — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md) · [SYCL / oneAPI](../tests/Nivara.GpuProbe/SYCL.md) · [DX12](../tests/Nivara.GpuProbe/DX12.md) · [OpenVINO](../tests/Nivara.GpuProbe/OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](../tests/Nivara.GpuProbe/COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU; the reported driver **1.15.37858** is the Level Zero `driverVersion`, not a Windows GPU driver) running Windows 10.0.26200 and .NET 11.0. The Silk.NET OpenCL leg of `tests/Nivara.GpuProbe` uses **Silk.NET.OpenCL 2.23.0** (thin .NET Foundation bindings to the OpenCL C API) to discover platforms/devices and to run OpenCL C kernels directly on the in-box OpenCL ICD + Intel driver. The goal is a vendor-neutral baseline (especially Intel/AMD iGPUs) for apples-to-apples comparisons with other backends. Whether this device can do native BF16 or FP16 arithmetic — and whether FP16 is faster — is measured in [Precision on this iGPU](#precision-on-this-igpu-bf16-and-fp16), not inferred from the extension string.

---

This document has two parts:

- **Part 1 — the spike.** The `tests/Nivara.GpuProbe` Silk.NET leg (`silk` mode): Silk.NET/OpenCL
  setup, the measured results, and the BF16/FP16 answer for this iGPU.
- **Part 2 — the production implementation.** The `--silk` hand-authored OpenCL C backend in
  `src/Nivara.Extensions/Silk/` (`namespace Nivara.Silk`) — every kernel, the runtime, the runners
  and the gate. Written from the source, as the map for adding the next model to this backend.

---

# Part 1 — the spike (GpuProbe, `silk` mode)

## 1. What the leg does (current state)

Everything lives in **`OpenCl/SilkLeg.cs`** (mode `silk`, and a row in `kernels`):

- Enumerates OpenCL platforms and GPU devices via `CL.GetApi()`, logging each platform with its GPU device count and the one selected (`*` marks the pick) — so a machine with no Intel/AMD iGPU still shows what OpenCL exposed.
- Selects an **Intel/AMD OpenCL GPU** preferentially (case-insensitive vendor match), falling back to the first GPU on any platform, then asserts `CL_DEVICE_TYPE_GPU` — a silent CPU fallback is impossible.
- Creates an in-order context and command queue on the selected device.
- Compiles three hand-authored OpenCL C kernels (`dot16`, `silu`, `gemv`) **in-process** via `clCreateProgramWithSource` + `clBuildProgram`. No external toolchain, no subprocess, no compiler binaries.
- Reuses the **byte-identical packed-BF16 transport** (`D3d12.GemvKernels.PackBf16`, 2 elements per uint) that the DX12, ComputeSharp and ILGPU legs upload, widening to f32 in-shader by the same element-parity `as_float` trick — so a widening difference can never masquerade as a kernel difference.
- Splits setup from steady state: program build and each kernel's first dispatch are reported separately; the timed passes are 1 warmup + **best-of-25** synchronized dispatches, matching the ILGPU/OpenVINO/ComputeSharp methodology.
- Prints the driver `CL_PROGRAM_BUILD_LOG` on a build failure (the single most useful Intel/AMD driver diagnostic).

This mirrors the spirit of the existing `LevelZero/OclProbe.cs` (direct P/Invoke) but uses Silk.NET's managed bindings instead of manual delegate loading. The leg is probe-scoped only (no changes to `src/Nivara`).

An earlier revision of this branch had a separate minimal `OpenCl/SilkProbe.cs` discovery-only leg (modes `opencl` / `silk-opencl`). It has been **deleted**: `SilkLeg` subsumes it — it selects the same Intel/AMD-first device, reports more caps, and actually runs kernels — so keeping both meant two device-selection paths that could disagree.

### Measured results (Arc 140T, `kernels` mode)

Ranges across two eight-way runs on the same machine and fixtures; the legs are internally
best-of-25, but single-run figures move (the CPU `gemv` column alone spanned 2785–4158 µs),
so ranges are the honest unit here.

**Two-host supersession (#599):** the repeated five-sweep profiles #552 (Arc 140T) and
#553 (Iris Xe) are the authoritative source for Silk's steady-state orderings — Arc medians
dot16 7.0 · silu 6.7 · gemv 100.6 µs, Iris Xe 71.4 · 69.2 · 486.2 µs, with every leg's bands
in [tests/Nivara.GpuProbe/README.md](../tests/Nivara.GpuProbe/README.md) → "Backend comparison — the numbers, side by
side". Where a claim in this section came from the two eight-way runs only, the two-host read
wins.

| kernel | CPU (production Nivara) | Silk.NET OpenCL | next best | gate |
|---|---|---|---|---|
| dot16 | 1.5–1.7 µs | **6.4–6.7 µs** | ILGPU 10.0–11.6 | PASS — bit-exact, 0.0 ULP |
| silu | 31.2–41.6 µs | **6.1–7.1 µs** | ILGPU 7.0 (tie) | PASS — 576/576, worst 4.0 ULP |
| gemv | 2785–4158 µs | 90.1–100.1 µs | **OV bf16 87.9–93.4** (tie on the #552 profile) | PASS — 1536/1536 |

Silk.NET is the fastest **hand-authored-kernel** leg, and the fastest row on the launch-bound
`dot16` (~1.5x faster than ILGPU, the nearest hand-authored competitor). It ties ILGPU on
`silu`. On `gemv` it **ties OpenVINO's tuned bf16 kernel** on the #552 profile (Silk band
90.3–103.7 µs, median 100.6, vs OV bf16 92.8–201.0 µs, median 114.0 — overlapping bands);
the earlier single-run "second, behind OV bf16" reading is retired, and since a naive OpenCL
gemm matches a tuned one there is no binding-overhead question to answer.

Careful with "fastest OpenCL row": `OV (bf16)` and `OV (f32)` are themselves OpenCL rows.
OPENVINO.md records the GPU plugin as OpenCL/IGC-frontened, running on the same Intel ICD,
so counting them as non-OpenCL to flatter this leg would be dishonest. Against every OpenCL
row, Silk.NET leads on `dot16`, ties on `silu`, and ties OV-bf16 on `gemv` on the #552
profile (bands overlap; medians 100.6 vs 114.0).

Why it leads the hand-authored legs: it is the only one that dispatches the compiled binary
directly, with no ILGPU runtime abstraction between the NDRange and the queue. Build cost is
~4.4–5.2 ms (in-process, driver JIT, highly variable); first dispatch adds 66–498 µs.

**Note on the observed ULP figures:** `gemv`'s worst-case 14336 ULP and `silu`'s 4.0 ULP are *not* Silk.NET-specific — the DX12, OV-f32, ILGPU and ComputeSharp legs report identical values, because it is f32 rounding at those magnitudes against a slightly different summation order, not a kernel difference. `|diff|` is ~1.6e-09 for gemv and ~3.7e-09 for silu, both far inside the `1e-6 + 1e-5·|ref|` gate. `OV (bf16 IR)` is the only silu leg outside the f32 gate (402/576) — a real OpenVINO bf16 IR precision issue, filed as issue **#606** and now a *diagnosed* gate row: all 402 failing elements match the F16-elementwise reference within 2 half-ULP, so it contributes 0 to the exit code.

---

## 2. Why Silk.NET.OpenCL (vs direct P/Invoke)

- **Thin and explicit**: Close to the raw OpenCL C API (platform/device query, buffers, programs/kernels, events). Good for seeing launch/transfer overhead without heavy abstraction.
- **Cross-vendor**: Targets Intel/AMD iGPUs well (matches "edge/embedded GPUs that sit unused" goal). Doesn't prioritize discrete NVIDIA.
- **Stable API surface**: Silk.NET 2.23.0 exposes the common 1.2 surface used on most iGPUs; easier to reason about than hand-rolled delegate tables.
- **.NET Foundation blessed**: Actively maintained, minimal deps, probe-scoped.
- **Parity-friendly**: Writing OpenCL C kernels lets us match shapes/sizes with `Nivara.SimdProbe` (transpose/GEMM) for apples-to-apples comparisons later.

---

## 3. Setup & usage

Add dependency (already done in this branch):
```xml
<PackageReference Include="Silk.NET.OpenCL" Version="2.23.0" />
```

Build and run:
```bash
dotnet build tests/Nivara.GpuProbe/Nivara.GpuProbe.csproj -c Release
dotnet run -c Release --project tests/Nivara.GpuProbe -- silk      # the three production kernels, gated vs CPU
dotnet run -c Release --project tests/Nivara.GpuProbe -- kernels   # all legs, Silk.NET included as a row
```

Expected (on Intel Arc 140T) for `silk`:

```
--- Silk.NET OpenCL leg (managed bindings, in-process OpenCL C compile) ---
  platform: Intel(R) OpenCL Graphics | GPU devices 1 | best: * Intel(R) Graphics
  device: Intel(R) Graphics (Intel(R) Corporation) | max WG 1024 | OpenCL OpenCL 3.0 NEO
  program: OpenCL C built in-process | build    4715 µs
  [dot16] first      415 µs | steady      6.8 µs
  ...
  Silk.NET OpenCL              dot16 ... -> PASS (0.0 ULP)
  kernels mode exit: 0 unexpected failed kernel(s), 0 diagnosed honest, 5 sweeps
```

Exit code 0 means all three gates passed. Always build `Release` — in `Debug` the probe
measures an unoptimized host against optimized driver code.

---

## 4. Implementation notes (Silk.NET 2.23.0)

These are the **verified** signatures for 2.23.0 — the bindings expose many generic
overloads per entry point, so picking the wrong shape fails to compile. Verify against
the shipped assembly before assuming:

- **Getting the API**: `CL cl = CL.GetApi();` — instance methods on `CL`, not statics.
- **Program source**: prefer the **managed** overload
  `CreateProgramWithSource(context, count, string[] strings, UIntPtr* lengths, out int err)`
  and pass `null` for `lengths`. The `byte**` overloads require `fixed` pinning and
  several variants differ between `UIntPtr*` and `UIntPtr&`.
- **Build**: `BuildProgram(program, 0, null, (string?)null, null, null)`. `num_devices: 0`
  means "all devices in the context", which is exactly the single device we asked for.
  The `(string?)null` cast is needed to pick the `string` options overload over `byte*`.
- **Kernel**: `CreateKernel(program, "name", out int err)` — again the `string` overload
  disambiguates from the `byte*`/`byte&` ones.
- **Buffers**: `CreateBuffer(context, MemFlags.ReadWrite, (nuint)bytes, null, out int err)`.
  `MemFlags` is `ulong`-backed; `MemFlags.ReadWrite` is the correct member (not a raw
  `(1 << 1)` cast).
- **Scalar args**: `SetKernelArg(kernel, index, sizeof(uint), &value)` — buffer args use
  `(nuint)sizeof(nint)`. **Check the returned status**: an argument-count mismatch fails
  here, and if ignored the kernel reads uninitialized scalars — which surfaces as a
  *fast* wrong answer rather than an error. (This exact failure mode bit the first
  `dot16` run: a stray leading `1` in the scalar list made `k` = 1 instead of 16.)
- **Enqueue**: the method is spelled **`EnqueueNdrangeKernel`** (lowercase `r` in
  `range`), not `EnqueueNDRangeKernel`.
- **Info queries**: `GetDeviceInfo`/`GetPlatformInfo`/`GetProgramBuildInfo` return
  strings as raw NUL-terminated bytes into a buffer sized by a first query with
  `param_value_size = 0`. Trim at the first NUL — do not assume `size - 1`.
  `ProgramBuildInfo` for the log is named **`BuildLog`**, not `Log`.
- **Enums are strongly typed**: `DeviceType.Gpu`, `MemFlags.ReadWrite`,
  `CommandQueueProperties.None`. Passing a bare `uint` will not bind.
- **Device type query**: `GetDeviceInfo(device, DeviceInfo.Type, sizeof(DeviceType), &type, null)`.
- **Resource cleanup**: Release in reverse order (mem objects, kernel, program, queue, context).

---

## 5. OpenCL C notes for iGPUs (Intel/AMD)

- **Target 1.2**: Safe common denominator for Intel/AMD iGPUs. Avoid 2.0/3.0-only features unless gated by device queries.
- **Workgroup size**: Start with power-of-two (e.g. 256). Query `CL_DEVICE_MAX_WORK_GROUP_SIZE` and clamp. Some older drivers behave better with 64/128.
- **Local memory**: Use `__local` sparingly; iGPUs often have small local memory vs discrete. For transpose/GEMM tiles, 16x16 is usually reasonable; measure.
- **Coalescing**: Write kernels with coalesced global accesses where possible (critical for integrated memory bandwidth).
- **Kernel strings**: Keep simple and deterministic. For parity with SimdProbe, match the same reference logic (row-major indexing, same shapes).
- **Build logs**: On build failure, fetch `CL_PROGRAM_BUILD_LOG` via `cl.GetProgramBuildInfo` — invaluable for Intel/AMD driver diagnostics.

## Precision on this iGPU (BF16 and FP16)

This is the section to read before trying a lower-precision path through raw OpenCL on this
device. Both answers were measured on the Arc 140T (8086:7DD1, driver 1.15.37858), not
inferred from the extension string. The leg prints the full `CL_DEVICE_EXTENSIONS` list
(71 entries here) because a count is not enough — see the naming trap below.

| | BF16 | FP16 (`half`) |
|---|---|---|
| Extension | `cl_intel_bfloat16_conversions` | `cl_khr_fp16` |
| Advertised here | yes | yes |
| Type | **no** `bfloat16` type; value rides in a `ushort` | **yes**, `half` / `half2` / `half4` / … |
| Convert | yes, lossless BF16→f32 (`intel_convert_as_bfloat16_float`, plus vector forms) | yes (`half` load, or `vload_half`) |
| Arithmetic | **no** — every function is `float ↔ bfloat16` | **yes** — mul/add/div/`exp` on `half` |
| Proved on this device | extension string + Khronos spec v1.0.0 | program built and ran; see below |
| Faster than the f32 kernels at these shapes | not reachable | **no** — tie, 0.97–1.07× |

### BF16: conversion yes, arithmetic no

```c
float  intel_convert_as_bfloat16_float(ushort source);   // + ushort2/4/8/16
ushort intel_convert_bfloat16_as_ushort(float source);
```

Backed by `OpConvertBF16ToFINTEL` / `OpConvertFToBF16INTEL` and SPIR-V capability
`Bfloat16ConversionINTEL` — the same instruction `SpvKernels.Bf16Native` hand-authors for the
Level Zero leg. Conversions are lossless in the BF16→float direction. There is no BF16
multiply, FMA, or dot product.

Consequences, so they don't get re-derived:

- The `as_float` shift in `Widen` can be swapped for the native convert. It should be
  bit-identical, which makes it a correctness cross-check, not a speed path. **Not done** —
  the production kernels keep the shift so the transport stays byte-identical with DX12,
  ILGPU, and ComputeSharp.
- `dot16` cannot become a packed BF16 dot on this device. The multiply stays f32 whichever
  convert is used. A packed BF16 dot needs a different toolchain (HLSL `__bf16` on DX12, or
  SYCL `bf16`). OpenVINO's GPU plugin does reach Xe2 DPAS BF16 for GEMM; that is a closed
  tuned kernel, not something this leg can author, and the same bf16 IR leg sits outside
  the f32 gate on `silu` (402/576 — all diagnosed-honest within 2 half-ULP of the F16
  mirror, #606).
- The f32 widening in this leg is a property of the platform, not of Silk.NET.

**Naming trap.** A substring search for `bf16` reports this device as having no BF16 support,
because Intel spells it `bfloat16`. The first version of the detector did exactly that and
printed "0 mentioning bf16". The matcher tests both `bf16` and `bfloat`, and the leg prints
the full extension list so a spelling variant cannot hide a capability.

### FP16: arithmetic yes, no speed win at these shapes

`cl_khr_fp16` adds a real `half` type, which the BF16 extension does not. `silk` builds a
**separate** translation unit so the f32 kernels are untouched:

```c
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT off
```

`FP_CONTRACT off` is there so a contraction cannot hide a promotion. This is a diagnostic,
not a gate row: half numerics cannot pass the f32 `1e-6` bound, and a build or dispatch
failure here is printed and swallowed. `silk` still exits 0 if only the f32 gates pass.

The compiler did **not** promote the arithmetic back to f32. `gemv_f16` matched a CPU
`System.Half` accumulator at **0 half-ULP across all 1536 rows**, and sat **32427 half-ULP**
from the same products accumulated in f32 and rounded once. That gap is the accumulator
width, and the GPU is on the half side of it. `silu_f16` is 2 half-ULP from f32 `exp`
rounded to half. The BCL has no `Half` exp, so silu cannot separate promotion from native
half; gemv can, and it says native.

Two runs, each f16 dispatch timed in the same process as the f32 kernel above it:

| kernel | f32 steady | f16 steady | ratio |
|---|---|---|---|
| silu (576) | 6.5 / 7.6 µs | 6.7 / 7.4 µs | 1.03× / 0.97× |
| gemv (1536×576) | 97.8 / 112.9 µs | 104.8 / 109.2 µs | 1.07× / 0.97× |

The ratio flips across the run-to-run band, so this is a tie. silu is launch-bound. gemv is
a one-thread-per-row loop over 576, which is not bandwidth-bound, so halving the traffic
does not show up. Native FP16 is reachable from raw OpenCL. It is not, on this evidence, a
reason to leave f32 for these shapes.

ILGPU 1.5.3 cannot drive this. Its `Half` type is software-emulated, and its OpenCL path
does not enable `cl_khr_fp16`. That is a binding limitation, not a device one — the opposite
of the BF16 situation, where the device itself has no arithmetic to drive.

### Still advertised, not measured

`cl_khr_integer_dot_product` (an int8 route, not BF16) and
`cl_intel_subgroup_matrix_multiply_accumulate` (subgroup MMA, a plausible gemv lever) are
both in the extension list. Neither has a kernel here. Do not treat "advertised" as
"usable" — the BF16 string was advertised and still has no arithmetic, and the FP16 string
was only believed once a program built and matched a half accumulator.

Timings in this section are one host, two runs. They are not the cross-host ordering.
That profile is #552 (this Arc 140T) and #553 (Iris Xe).

## 6. Parity status

**Achieved:** the three production SmolLM kernels (`dot16`, `silu`, `gemv`) run on the
iGPU through Silk.NET and gate clean against the CPU reference, with the same packed-BF16
transport and the same timing discipline as the other legs.

**Still deferred** (not attempted — no existing probe mode covers them, so they would need
a new harness rather than an addition to this leg):

- **Vector add**: 1M elements, correctness vs CPU reference, median timing.
- **Transpose**: row-major in/out, CPU reference bit-exact, shapes matching
  `Nivara.SimdProbe`: (1024,1024), (512,512), (2048,1024), (1024,2048), (129,257).
- **A/B style**: keep the same timing discipline (warmups, best-of-N) so the numbers stay
  comparable with the other backends.

---

## 7. Comparison notes vs other backends

| Aspect | Silk.NET OpenCL | ILGPU (OpenCL) | SYCL/DPC++ | DX12 (hand-rolled) | ComputeSharp (DXIL) |
|---|---|---|---|---|---|
| Abstraction | Thin (raw OpenCL) | High-level C# kernels | Compiler-backed SYCL | Hand-rolled D3D12 | Source-gen HLSL/DXIL |
| Control | High (explicit) | Medium | Medium-High | Very High | Medium |
| Boilerplate | Moderate (C API) | Low | Low (DPC++) | High (P/Invoke) | Low |
| Intel/AMD iGPU focus | Excellent | Good | Excellent (oneAPI) | Excellent (D3D12) | Excellent |
| Subprocess/toolchain | None (in-process) | None | Often run.cmd + icpx | None | None |
| Best for | Overhead visibility, baseline | Quick C# GPU | Production SYCL | Deep plumbing insight | Managed DX12 ease |

Silk.NET gives the "true OpenCL" baseline: launch cost, buffer transfers, and kernel overhead visible with minimal indirection.

---

## 8. Known caveats

- **Silent-wrong-answer hazard**: OpenCL reports a scalar-argument-count mismatch as a
  `clSetKernelArg` status, not as a dispatch error. Ignoring that status produces a
  kernel that reads uninitialized scalars — a *fast* wrong answer rather than a failure.
  Every status in this leg is checked and a bad dispatch aborts the measurement (the leg
  then reports UNBUILT instead of gating a value the driver never produced).
- **One shared `err` slot**: the buffer/kernel creation helpers all write to the same
  status out-param, so checking only after the last of them silently discards an earlier
  failure — the leg would then run on a half-created set of buffers. Each status is
  checked where it is produced. For the same reason statuses are never OR'd together
  (CL codes are not a bitmask; OR'ing two failures yields a number that names neither).
- **Failure paths must not throw**: `KernelGate` does not guard its legs, so an exception
  escaping a leg aborts the whole multi-leg `kernels` run instead of marking one leg
  UNBUILT. Readback failure therefore returns `null` and prints a `[FAIL]` line.
- **Build time is highly variable**: the in-process `clBuildProgram` measured 4.7 ms,
  236 ms and ~500 ms across runs on the same machine and driver — the driver's JIT cache
  state dominates. Steady-state kernel timings are stable (±15%); build time is not, so it
  is reported separately and should not be compared across runs.
- **An advertised extension is not a working kernel**: `cl_intel_bfloat16_conversions` being
  present means the builtins exist, not that a kernel using them compiles and runs. Only a
  real build-and-gate proves it.
- **Global size rounding**: OpenCL requires `global_work_size` to be a multiple of
  `local_work_size`, so `silu` (576 elements) dispatches 768 work-items and the extra
  ones return early. This is inside the kernel, not host-side padding.
- **In-order queue**: chosen for clarity; out-of-order + events would be needed to
  overlap transfers with execution. The timing is therefore a per-kernel synchronized
  dispatch, not a pipelined throughput measure.
- **Driver variance**: Intel vs AMD OpenCL drivers differ in caps, build-log format and
  whether `cl_khr_fp64`/2.0 features are exposed; discovery prefers Intel/AMD explicitly
  and the kernels stay within OpenCL 1.2.
- **Silk.NET API shape**: the bindings' many generic overloads make the *correct* overload
  non-obvious; see §4 for the verified signatures. A Silk.NET upgrade may require
  revisiting them.

---

## 9. Reproducibility

Verified on:
- Host: Intel Core Ultra 7 255H (Arrow Lake-H), Arc 140T iGPU
- OS: Windows 10.0.26300
- .NET: 11.0
- Silk.NET.OpenCL: 2.23.0
- Driver: Intel Graphics driver with in-box OpenCL ICD (`OpenCL OpenCL 3.0 NEO` reported)

Run (Release):

```
dotnet run -c Release --project tests/Nivara.GpuProbe -- silk     # exit 0 = all three gates pass
```

The `kernels` mode includes this leg alongside SYCL/DX12/OpenVINO/ILGPU/ComputeSharp.
Note that its overall exit code is **not** 0 on this machine even when the Silk.NET row is
fully green — SYCL is unavailable (no oneAPI on this box) and the OpenVINO bf16-IR SiLU leg
is outside the f32 gate 402/576, each failure within 2 half-ULP of the F16 mirror (a
diagnosed-honest #606 row, 0 in the exit code) — pre-existing conditions unrelated to
Silk.NET. The #606-gate exit on this machine is 3 = SYCL-unavailable + 0 unexpected. Verify
the Silk.NET row's own PASS lines, not just the process exit code.

---

# Part 2 — the production `--silk` implementation

Part 1 measured the Silk.NET/OpenCL route on microkernels. Part 2 is the same route on a real model:
the `--silk` backend, OpenCL C written by hand and compiled in-process by the driver. It is written
from `src/Nivara.Extensions/Silk/` (`namespace Nivara.Silk`) — the map for adding the next model to
this backend. ILGPU stays in `samples/Nivara.Samples/Gpu/`.

## 10. What the path is, and what is in it

`--silk` runs encoders, causal decoders, and vision convolution through hand-authored OpenCL C
(`Silk.NET.OpenCL` 2.23.0, the same version pin as the probe leg). It is a vendor-neutral route on
real models. Scope today: **MiniLM, DistilBERT, DistilBERT+SST-2, ModernBERT, Laya, SmolLM,
Qwen, MobileNetV2, and ResNet-18**, F32 only. Qwen and the vision models expose their own mode
sets (see §14); the BERT-family models use `default` / `benchmark` / `compare`. F32 only (no
bf16/fp16 translation unit).

| file | role |
|---|---|
| `SilkKernels.cs` | the OpenCL C translation unit (`Source`) and all geometry, as pure data so it can be unit-checked without a device; `KernelNames` is the build order. `internal` |
| `SilkRuntime.cs` | `CL.GetApi()`, platform/device select, in-process build with the build log decoded on failure, buffers/sub-buffers, per-argument status checking, blocking dispatch, reverse-order dispose |
| `SilkLauncher.cs` | typed dispatch shared by the three runners: argument order and grid geometry, so a runner cannot drift from the kernel. `internal` |
| `WeightLayout.cs` | `Transposed` / `QkvPacked` / `Req` — the host math both backends share. `GpuBuffers` delegates to it |
| `SilkBertEncoderRunner.cs` | the Silk counterpart of `BertEncoderGpuRunner` — same weight layout, same operation order, same fused epilogues |
| `SilkModernBertGpuRunner.cs` | the Silk counterpart of `ModernBertGpuRunner` — bias-free GEMM, row-interleaved QKV split, CPU-uploaded RoPE, per-layer band |
| `SilkLayaHeadGpuRunner.cs` | the Silk counterpart of `LayaHeadGpuRunner` — biased norms, fused biased QKV, host-side act-head features |
| `SilkSmolLMGpuRunner.cs` | SmolLM causal decode — RMSNorm, SwiGLU, causal GQA attention, cached KV, argmax |
| `SilkQwenGpuRunner.cs` | Qwen causal decode, benchmark, and tool paths |
| `SilkMobileNetV2GpuRunner.cs` | vision inference — grouped conv, batchnorm, pooling |
| `SilkResNet18GpuRunner.cs` | vision inference — grouped conv, batchnorm, pooling |
| `SilkModernBertSession.cs` / `SilkLayaSession.cs` / `SilkSmolLMSession.cs` / `SilkQwenSession.cs` | the Silk session implementations. The CLI sees `Forward` or the session's `Run`, not a device buffer |
| `src/Nivara.Extensions/AI/GpuSeam.cs` | `IBertEncoderRunner`, `IModernBertGpuSession`, `ILayaGpuSession` and the open delegates (`namespace Nivara.AI`). The ILGPU session classes stay in `samples/Nivara.Samples/Gpu/` and implement them |

`Nivara.Extensions.csproj` pins `Silk.NET.OpenCL` 2.23.0. `SilkKernels` and `SilkLauncher` are
`internal`, with `InternalsVisibleTo` to `Nivara.Tests` and `Nivara.PerformanceTests`, so the gate
and the unit checks can reach them. Samples no longer pins Silk.NET; it arrives transitively.

## 11. The translation unit

### 11.1 The kernel set

`KernelNames` (`SilkKernels.cs::SilkKernels`) is the build/creation order:

```csharp
[EmbedSum, GatherRows, LayerNormResidual, LayerNorm, Add, AddBias, SplitColumns, Rotary, GeGlu,
 Gemm, GemmBias, GemmRelu, GemmGelu, GemmQkv, GemmReg2x4, GemmReg4x4, Attention, AttentionSliding,
 RmsNorm, RmsNormLmHead, RmsNormGemv, Silu, SwiGlu, CausalGqaAttention, DecodeGqaAttention, ArgMax,
 GemmAdd, GemmGemvAdd, GemmGemvAddBf16, MlpBlock, MlpBlockBf16, RmsNormGemvBf16, GemmBf16,
 DecodeGqaAttentionOprojFused,
 Conv2d, BatchNormEval, Relu, Relu6, MaxPool2d, AdaptiveAvgPool2d,
 GroupedHashReducePartial,
 GemmNt, GemmTn, ReluBackward, ColSum, Fill,
 Conv2dBackwardWeight, Conv2dBackwardInput, Conv2dBackwardBias, AdamUpdate,
 AttentionBackwardPrep, AttentionBackwardDq, AttentionBackwardDkv, Gelu,
 EmbeddingBackwardRows, ScatterRows,
FusionForward, FusionWgrad]
```

i.e. the families by origin: the original eight BERT encoder kernels (`embed_sum`, `gather_rows`,
`layer_norm_residual`, the four Row4 epilogues, `batched_attention`); the seven ModernBERT/Laya
ports (`layer_norm`, `add`, `add_bias`, `split_columns`, `rotary`, `geglu`, `gemm_row4`); the
Qwen/SmolLM decode kernels (`rms_norm`, `rms_norm_lm_head`, `rms_norm_gemv`, `silu`, `swiglu`,
`causal_gqa_attention`, `decode_gqa_attention`, `argmax`, `gemm_row4_add`) plus their fused and
measured-candidate derivatives (`gemm_reg2x4`, `gemm_reg4x4`, `batched_attention_sliding`,
`gemm_gemv_add`, `mlp_block`, `decode_gqa_attention_oproj_fused`) and the packed-BF16 transport
quartet (`gemm_row4_bf16`, `gemm_gemv_add_bf16`, `rms_norm_gemv_bf16`, `mlp_block_bf16`); the
vision kernels (`conv2d`, `batchnorm_eval`, `relu`, `relu6`, `maxpool2d`, `adaptive_avgpool2d`);
the analytics kernel (`grouped_hash_reduce_partial`); and the classifier-backward family (#900):
the transposed-operand GEMMs, the elementwise/reduction backward pieces and the fused Adam step.
The array order is the creation list, not the source order.

Each is a port of the same-shape ILGPU kernel with the operation order preserved — same strictly
ascending K-reduction, same tile geometry where the kernel is tiled, same fused epilogues:

| kernel | signature (abridged) | purpose |
|---|---|---|
| `embed_sum` | `ids, posIds, wordEmb, posEmb, tokenType, y, n, hidden, includeTt` | fused word + position (+ token-type row, gated by `includeTt`) gather/sum for the embedding step |
| `gather_rows` | `table, ids, y, n, hidden` | gather rows — the CLS-row selection in the classifier head |
| `layer_norm_residual` | `a, b, gamma, beta, y, rows, cols, eps` | `LayerNorm(a + b)` in one kernel; a zero `b` reproduces the embedding LayerNorm exactly |
| `gemm_row4_bias` | `a, b, c, bias, aRows, aCols, bCols` | Row4 GEMM + bias epilogue (o-projection, fc2) |
| `gemm_row4_relu` | as above | Row4 GEMM + ReLU epilogue (head `pre_classifier`); clamp written `(r < 0 ? 0 : r)`, not `fmax` |
| `gemm_row4_gelu` | as above | Row4 GEMM + exact-GELU epilogue (fc1) |
| `gemm_row4_qkv` | as above **+ `blockWidth`** | Row4 GEMM writing the block-separable `[q | k | v]` layout via `qkv_dest` |
| `batched_attention` | `q, k, v, mask, attnOut, batch, seqLen, numHeads, headDim, band, scale` | score + scale + mask + row-softmax + weighted-V, three passes, banded |
| `layer_norm` | `x, gamma, beta, y, rows, cols, eps` | `LayerNorm(x)`, not `LayerNorm(a+b)`. ModernBERT pre-norm and the Laya head. A zero partner in `layer_norm_residual` is a different reduction |
| `add` | `a, b, y, n` | residual add; `y` may alias `a` |
| `add_bias` | `x, bias, y, rows, cols` | Laya type-embedding broadcast, in place |
| `gemm_row4` | `a, b, c, aRows, aCols, bCols` | bias-free Row4 store. Not `gemm_row4_bias` with a zero bias: `acc + 0.0f` canonicalizes `-0` |
| `rms_norm` | `x, gamma, y, rows, cols, eps` | `RMSNorm(x)` |
| `rms_norm_lm_head` | `x, gamma, weight, logits, aRows, aCols, bCols, eps` | fused `RMSNorm(x) @ weight` (prefill, including the last-row-only prefill head; the cached decode head uses `rms_norm_gemv`) |
| `rms_norm_gemv` | `x, gamma, w0, bias0, out0, w1, bias1, out1, w2, bias2, out2, aCols, cols0, cols1, cols2, hasBias, eps` | cached-decode projections: one `RMSNorm(x)` feeding up to three GEMMs (Q/K/V, gate/up, tied head). Decode-shaped — single row, so there is no row dimension to leave idle: a flat 1-D launch, one thread per `BLOCK` columns, no shared-memory staging. A slot of zero columns is unreachable, and `hasBias` 0 stores the plain accumulator — a zero bias would be `acc + 0.0f`, the same distinction `gemm_row4` draws |
| `gemm_row4_add` | `a, b, c, residual, aRows, aCols, bCols` | Row4 GEMM plus residual add in the store (O/down projections in cached decode) |
| `split_columns` | `src, dst, rows, blockCols, parts, part` | ModernBERT QKV is row-interleaved; a contiguous sub-view is right for row 0 and wrong after it |
| `rotary` | `x, posIds, cosTab, sinTab, y, rows, numHeads, headDim` | half-split RoPE, in place on q and k. Tables are inputs. The pair width is not named `half`: this driver treats that as a type name and rejects the program |
| `geglu` | `x, y, rows, intermediate` | `gelu(first half) * second half`. Reversing the halves still looks plausible |
| `silu` | `x, y, n` | elementwise SiLU (swish) activation |
| `swiglu` | `gate, up, y, rows, intermediate` | fused SwiGLU: `silu(gate) * up` |
| `causal_gqa_attention` | `q, k, v, out, seqLen, numHeads, numKvHeads, headDim, scale` | causal GQA attention for prefill (attends j=0..qPos, KV-repeat) |
| `decode_gqa_attention` | `q, kCache, vCache, out, kvLen, numHeads, numKvHeads, headDim, scale` | single-query decode attention over the KV cache (online softmax; one work-group per head, kvLen split across lanes) |
| `gemm_gemv_add` | `a, b, c, residual, aCols, bCols` | flat single-row GEMV + residual add for cached-decode O-projection (#672, #681). Silk-only: no ILGPU counterpart; correctness pinned by exact equality to `gemm_row4_add` (aRows=1) and the CPU-reference gate cell, not by the ILGPU byte-identity list. |
| `decode_gqa_attention_oproj_fused` | `q, kCache, vCache, Wo, residual, out, kvLen, numHeads, numKvHeads, headDim, hiddenDim, scale` | true single-work-group fused decode attention + O-projection (#691). Only runs when `MaxWorkGroupSize >= numHeads * AttentionGroupSize`; otherwise `DecodeGqaAttentionOproj` falls back to `decode_gqa_attention` + `gemm_gemv_add`. |
| `mlp_block` | `x, gamma, Wgate, Wup, Wdown, out, residual, hidden, intermediate, eps` | fused single-token MLP block for cached decode: RMSNorm, gate/up, SwiGLU, down-projection, residual (#672) |
| `argmax` | `input, partialIndex, partialValue, n` | device-side partial argmax reduction (256 elements per work-group) |
| `conv2d` | `input, weight, bias, output, N, C, H, W, outC, kH, kW, strideH, strideW, pad*, groups, hasBias, oH, oW` | NCHW convolution (grouped) |
| `batchnorm_eval` | `x, mean, var, gamma, beta, y, N, C, HW, eps, hasAffine` | batch norm in eval mode |
| `relu` | `x, y, n` | elementwise ReLU |
| `relu6` | `x, y, n` | elementwise clamp to [0, 6] |
| `maxpool2d` | `input, output, N, C, H, W, kH, kW, strideH, strideW, pad*, oH, oW` | 2D max pooling |
| `adaptive_avgpool2d` | `input, output, N, C, H, W` | global average pool to 1x1 |
| `gemm_reg2x4` | `a, b, c, aRows, aCols, bCols` | measured 2×4-register-tile GEMM candidate (#784); plain ascending-K, kept for the geometry A/B |
| `gemm_reg4x4` | as above | the 4×4-register-tile candidate from the same A/B |
| `batched_attention_sliding` | same args as `batched_attention` | sliding-window attention the launcher picks when the band is finite and narrow enough for the local cache (`band ≥ 0` and `2·band + 1 ≤ SlidingAttentionCacheSize`) |
| `gemm_row4_bf16` | `a, b, c, aRows, aCols, bCols` (B packed BF16) | Row4 GEMM with B transported as packed BF16 ushort, widened in-shader; same accumulation core |
| `gemm_gemv_add_bf16` | `a, b, c, residual, aCols, bCols` | the decode GEMV + residual add over packed-BF16 B |
| `rms_norm_gemv_bf16` | as `rms_norm_gemv` | the fused decode projections over packed-BF16 weights |
| `mlp_block_bf16` | as `mlp_block` | the fused single-token MLP block over packed-BF16 weights |
| `grouped_hash_reduce_partial` | `serviceIds, statusCodes, durations, partialCounts, partialErrors, partialSums, n, groupCount` | incident analytics: per-work-group partial (count, 5xx errors, duration sum) per service group, merged on the host |
| `gemm_row4_nt` | `a, b, c, aRows, aCols, bCols` (B stored `[N,K]`) | backward GEMM `A·Bᵀ` with the transpose in the tile load (`dH = dZ·W2ᵀ`, `dPooled = dH·W1ᵀ`); same accumulation core as the forward Row4 GEMMs, so the host replay stays byte-identical (#900) |
| `gemm_row4_tn` | `a, b, c, aRows, aCols, bCols` (A stored `[K,M]`) | backward GEMM `Aᵀ·B` with the transpose in the tile load (`dW2 = hᵀ·dZ`, `dW1 = pooledᵀ·dH`) (#900) |
| `relu_backward` | `d, act, y, n` | ReLU mask backward: `y = act > 0 ? d : 0`; `act` is the post-activation buffer (ReLU(x) > 0 iff x > 0) |
| `col_sum` | `x, y, rows, cols, scale` | `y[j] = scale·Σ_i x[i·cols + j]` — FC bias gradients; the scale is applied once after the ascending accumulation |
| `fill` | `y, value, n` | constant fill — zeroes the dX workspace before the per-branch `conv2d_backward_input` launches accumulate into it |
| `conv2d_backward_weight` | `dA, input, dW, N, C, H, W, outC, kH, kW, stride*, pad*, groups, oH, oW` | convolution weight gradient; same NCHW/stride/pad/groups geometry and weight layout as `conv2d` (padBottom/padRight do not enter the index math), one work-item per weight element |
| `conv2d_backward_input` | `dA, weight, dX, ...` (same geometry) | convolution input gradient; **accumulates** (`dX +=`) — zero via `fill` first, the text CNN launches it once per branch over the same dX |
| `conv2d_backward_bias` | `dA, db, N, outC, oH, oW` | convolution bias gradient |
| `adam_update` | `w, g, m, v, lr, bc1, bc2, eps, n` | fused Adam step (β1 0.9, β2 0.999, ε 1e-8 — the host `AdamState` constants); `bc1`/`bc2` are the bias corrections precomputed host-side, gradients arrive already batch-averaged |
| `attention_backward_prep` | `qkv, mask, dAttn, probs, delta, batch, seqLen, numHeads, headDim, band, scale` | re-derives the softmax and `delta_i = Σ_j p_ij (dO_i·V_j)`; writes `probs` and `delta`; fully masked query rows write zeros (no 0/0 NaN) |
| `attention_backward_dq` | `qkv, probs, delta, dAttn, dQkv, batch, seqLen, numHeads, headDim, scale` | `dQ_i = scale·Σ_j p_ij (dotV_ij − delta_i) K_j`; writes the q block of the row-major `[rows, 3·hidden]` gradient |
| `attention_backward_dkv` | same args | `dV_j = Σ_i p_ij dO_i`, `dK_j = scale·Σ_i p_ij (dotV_ij − delta_i) Q_i`; writes the k and v blocks |
| `gelu` | `x, y, n` | elementwise `gelu_exact(x)` — the split form of `gemm_row4_gelu`, bit-identical to it, so the training forward can keep the pre-activation for `gelu_backward` |
| `embedding_backward_rows` | `ids, dX, gradEmb, rows, embedDim` | row-major embedding scatter `gradEmb[ids[r]] += dX[r]` for a `[rows, hidden]` activation; accumulates, so the gradient buffer must be zeroed first |
| `scatter_rows` | `dCls, dX, batchSize, seqLen, embedDim` | `gather_rows` backward: writes `dCls[b]` into row `b·seqLen` of a pre-zeroed `[batch·seqLen, hidden]` buffer |

### 11.2 Geometry

`GemmTileSize` 16, `GemmBlockCols` 4, `GemmTileCols` 64, `GemmRowStride` 80, `LinearGroupSize` 256,
`AttentionGroupSize` 64, `MaxHeadDim` 64, `GlobalAttentionBand` −1 — the same numbers the ILGPU
`GpuBuffers` carries. The decode shape adds `GemvGroupSize` 32, `GemvBlockCols` 4 and
`GemvGroupCols` 128: a flat launch, because a one-row problem has no row dimension to fill and the
tile form left 240 of a group's 256 threads discarding results. The helpers are pure functions
(`GemmGlobalRows`, `GemmGlobalCols`, `GemvGlobalCols`, `LinearGlobal`, `AttentionGlobal`, `RoundUp`)
so they unit-check without a device.
`GemmGlobalCols` counts *groups*, not threads: a group is 16 threads wide but covers 64 output
columns, so returning `RoundUp(bCols, GemmTileCols)` would over-launch by `GemmBlockCols`× and the
surplus groups would run the whole K-loop and then discard every result. `GemvGlobalCols` is that
same arithmetic one group wider (`RoundUp(cols, GemvGroupCols) / GemvBlockCols`), and a zero width
returns zero — which is how a trailing output slot is omitted rather than launched and discarded.

### 11.3 Bit-identity: the macro and the pragma

The Row4 GEMM family, including the bias-free `gemm_row4` store and the #900 backward NT/TN
variants, shares **one** accumulation macro, `GEMM_ACCUMULATE_CORE` (`SilkKernels.cs::Source`)
deliberately: the byte-identity claim rests on every variant adding the same f32
values to the same accumulators in the same order, so the text must not be able to drift between
them. The tile loads before it and the store after it are the only per-variant differences. The inner product is a plain `acc += a * b` — **not** `fma()` — and `#pragma OPENCL FP_CONTRACT
OFF` (uppercase; the lowercase spelling the spec also admits is not honoured here) keeps the driver
from fusing it, because that is what ILGPU produces. The gate replays each dot product three ways in
the same ascending-K order and reports the hit counts: ILGPU matches `mul+add` on 100% of elements
and `fma` on ~19%. An explicit `fma()` here would defeat the pragma and diverge from ILGPU on ~81% of
outputs at rounding scale, systematically wrong and invisible to any tolerance.

**A-tile vector loads and alignment (#807).** The A-staging loops load four consecutive f32 values with
load4(0, a + aBase). OpenCL load/store are the designated unaligned access primitives: load4
requires alignment to the *element* type (loat � four bytes), not to loat4. Since  is a
__global const float*,  + aBase is always 4-byte aligned by construction (float index into a float
array), so the base is never misaligned at load time � even when Cols or the row stride is not a
multiple of four. The old (aBase % 4u) == 0u guard was therefore over-conservative (it demanded
16-byte alignment) and denied vectorization on non-multiple-of-four K or hidden dimensions.
It has been dropped; the in-range guards (srcRow < aRows, ColBase + 3 < aCols) and the scalar
tail remain. Partial tiles at the K or batch edge still use the scalar loop, so correctness (including
byte-identity under #pragma OPENCL FP_CONTRACT OFF) is unchanged, and the same treatment is
*not* applied to the ILGPU side: there ArrayView<float>.Cast<Float4>() is an unchecked
alignment assertion with no safe unaligned counterpart.

**B-tile vector loads and alignment (#822).** The B-staging loops load four consecutive values with
vload4(0, b + bBase). The base  is a __global const float* for the gemm_row4* kernels and
a __global const ushort* for gemm_row4_bf16, so  + bBase is always aligned to the element
type by construction (4 bytes and 2 bytes respectively) - even when Cols or the row stride is not
a multiple of four. The old (bBase % 4u) == 0u guard was over-conservative and has been dropped;
the in-range guard Row < aCols && (outCol0 + 3u) < bCols and the scalar tail remain. Partial
tiles at the edge still use the scalar loop, so correctness, accumulation order and byte-identity
(including #pragma OPENCL FP_CONTRACT OFF) are unchanged. The same treatment is *not* applied to
the ILGPU side, as with the A-tile (#807).


**Unroll form (issue #796).** The inner 16-iteration accumulation is now expressed as a single
`_Pragma("unroll 16") for (uint k = 0; k < TILE; ++k)` loop rather than 16 hand-written blocks.
An A/B on the target GPU compiled the pragma form successfully, produced bit-identical output to the
manual form on 100% of elements, and measured ~3.5% faster (manual 128.70 us, pragma 124.15 us,
128x128x128 tile, Release, 2026-10-22). Manual unrolling is therefore not canonical: the pragma
form is shorter, less error-prone to edit, and slightly faster, while preserving the ascending-K
`mul+add` operation order and the `#pragma OPENCL FP_CONTRACT OFF` invariant the parity gate depends on.

Transcendentals (`exp` in the softmax and the GELU erf) are **not** assumed identical to ILGPU's
`XMath.Exp`; those cells gate on the model's numeric bound instead (§14).

`gemm_gemv_add` (#681) is a Silk-only decode shape with no ILGPU counterpart, so it is
deliberately **not** in the ILGPU byte-identity list. Its contraction is the same ascending-K
`mul+add` as `gemm_row4_add` with `aRows == 1`, and that equivalence is pinned by
`SilkGemmGemvAddTests` (bit-identity) plus the `MeasureCpuReference` cell in
`SilkKernelsGate`.

The #900 backward GEMMs (`gemm_row4_nt`, `gemm_row4_tn`) extend the same claim to the transposed
operands: they invoke the same `GEMM_ACCUMULATE_CORE` and transpose only the tile **load** (B
row-major `[N,K]` for NT, A row-major `[K,M]` for TN). The transposed loads are scalar — `vload4`
needs elements contiguous along the load direction, which a transposed operand destroys — but the
load is a small fraction of the MAC loop. No operand copies are made and no transposed weight
buffers exist. Like `gemm_gemv_add` they are Silk-only and deliberately **not** in the ILGPU
byte-identity list; `SilkBackwardKernelTests` pins each bit-for-bit against a CPU replay of the
same ascending order. The Adam step (`adam_update`) is the one backward kernel that cannot claim
byte identity: fp32 divide and sqrt are not guaranteed correctly rounded without the
`-cl-fp32-correctly-rounded-divide-sqrt` build option, and `BuildProgram` passes no options, so
`SilkBackwardKernelTests` pins it by device-vs-device bitwise determinism plus a tolerance
against the host formula instead.

## 12. The runtime (`SilkRuntime`)

- **Acquisition:** `CL.GetApi()` — instance methods on `CL`, not statics.
- **Device select:** two-call platform/device enumeration, `DeviceType.Gpu` only, with Intel/AMD
  vendors preferred and the first candidate kept as a fallback; then a GPU **re-assert** (`IsGpu`),
  because "a driver that answers with something else must fail loudly rather than silently turn this
  into a CPU measurement." Construction throws rather than falling back.
- **Build:** `CreateProgramWithSource` + `BuildProgram` with `num_devices = 0` ("every device in the
  context" — the one device asked for). On failure the driver's build log is decoded
  (first-NUL trim, UTF-8) and thrown — the single most useful diagnostic on Intel/AMD drivers.
- **Buffers:** `CreateBuffer(ReadWrite)`; `Allocate(floatCount)` and `AllocateBytes(bytes)`.
  `Ensure(SilkBuffer, floatCount)` grows a buffer the way `GpuBuffers.Ensure` does: allocate on
  first use, release and replace when the next forward is longer, never shrink. The BERT runner
  does not use it; ModernBERT does, up to `max_position_embeddings`.
- **Sub-buffers:** `CreateSubview(parent, elementOffset, elementCount)` builds a `BufferRegion` and
  `CreateSubBuffer(ReadOnly, …)`. The view shares the parent's storage, so the parent must outlive it
  and the caller releases it.
- **Upload/download:** blocking `EnqueueWriteBuffer` / `EnqueueReadBuffer` of exactly the payload.
- **Argument setting:** `SetBuffer` / `SetUInt32` / `SetInt32` / `SetFloat32`, **each call checked
  individually** and the failure names the kernel and argument index. This is the deliberate
  divergence from the probe's OR-ed status: a mismatched argument count otherwise makes the kernel
  read uninitialized scalars and answer fast and wrong rather than erroring.
- **Dispatch:** `EnqueueNdrangeKernel` only (non-blocking). The command queue is in-order, so
  releases/completion follow enqueue order; consumers synchronize via blocking Upload/Download or
  an explicit `SilkRuntime.Finish()`. The global size must already be a whole multiple of the
  local size (the geometry helpers guarantee it), because OpenCL rejects a non-multiple and the
  in-kernel guard would otherwise be dead.
- **Cleanup:** kernels → program → command queue → context, reverse of construction. A failed
  partial construction disposes what it built (`catch { Dispose(); throw; }`).

## 13. The runner (`SilkBertEncoderRunner`)

The constructor takes a **`BertGpuNaming`** (`DistilBert` or `Bert`) — the parameter that lets
DistilBERT reuse the MiniLM runner — and resolves embeddings + per-layer q/k/v/o, ffn and LayerNorm
keys through the shared `BertGpuKeys` resolver (`src/Nivara.Extensions/AI/BertEncoderGpuTypes.cs`).

- **Uploads.** Embeddings plain. q/k/v through `WeightLayout.QkvPacked`, one pre-transposed
  `[in × 3·out]` `[Bt_q | Bt_k | Bt_v]` plus a `[3·out]` bias — byte-identical to ILGPU's, produced
  from the same arithmetic rather than a second copy. `O`/`W1`/`W2` transposed; biases and LayerNorms
  plain. Token-type: only row 0 is read (all-zero segment ids), so one row is uploaded rather than the
  checkpoint's two.
- **Head weights** are allocated unconditionally — so `Dispose` is unconditional — and uploaded only
  when the checkpoint ships `pre_classifier`/`classifier`.
- **Workspace** is cap-sized (`batch ≤ 8`, `seqLen ≤ 128`, the scenario's actual maxima) with every
  layer intermediate resident on device. A `zeroResidual` buffer lets `layer_norm_residual` reproduce
  the encoder's one non-residual norm exactly (`a + 0.0f == a`), which is why there is no dedicated
  embedding-norm kernel.
- **Forward.** Upload mask + ids; rebuild posIds only when `seqLen` changes (`row % seqLen`), so the
  repeat-forward benchmark path skips the copy; `embed_sum` then `layer_norm_residual(x, zeroResidual,
  …)`; then per layer fused-QKV → attention → o → LN-residual → fc1 GELU → fc2 → LN-residual. The
  optional SST-2 head runs `gather_rows` (CLS row `b*seqLen`) → `gemm_row4_relu` → `gemm_row4_bias`
  for `[batch, 2]` logits. Only the final hidden state (and logits) are downloaded.
- **q/k/v sub-buffers** are created once per forward from the packed `qkv` result — the three
  contiguous blocks, so the already-gated fused attention kernel is untouched — and released in a
  `finally`, because they share the parent's storage.

`IBertEncoderRunner` is the only thing the CLI's BERT-family entry points see, so the two backends are
the same code with a different `Forward` behind it. The padding contract is stated on the interface:
mask `0` at padding, positional ids `0..seqLen−1` repeated per batch row.

## 13a. ModernBERT and Laya

`SilkModernBertGpuRunner` mirrors `ModernBertGpuRunner`. Prefix `"model"` for the stock checkpoint,
`"encoder"` for Laya. Workspace grows via `SilkRuntime.Ensure` up to `max_position_embeddings`; it
is not the BERT cap of batch ≤ 8, seqLen ≤ 128. RoPE cos/sin are uploaded from the CPU
`RotaryEmbedding<float>` for both thetas. Layer 0 has no `attn_norm` and is an identity, not a zero
gamma. Each layer passes `config.Band(index)` into `batched_attention`; global layers pass `-1`.
QKV is a bias-free `gemm_row4` plus `split_columns`. GeGLU activates the first half.

`SilkLayaHeadGpuRunner` mirrors `LayaHeadGpuRunner` and adds no kernels. Norms are biased, eps is
`1e-5` (not `config.NormEps`), QKV is the existing `gemm_row4_qkv` with its real bias, attention is
the global band, and the four act-head features stay on `LayaHeadScoring`. A checkpoint without
`type_emb.weight` throws at head construction; the session disposes the encoder on that path.

The CLI sees `IModernBertGpuSession` / `ILayaGpuSession`. `HiddenOnDevice` stays backend-private.

## 13b. Causal decoders and vision runners

`SilkSmolLMGpuRunner` and `SilkQwenGpuRunner` run the cached causal-LM decode path rather than an
encoder. They use the same `SilkRuntime`, plus `rms_norm`, `swiglu`, `causal_gqa_attention`,
`decode_gqa_attention`, `decode_gqa_attention_oproj_fused`, `mlp_block`, `gemm_gemv_add`, and `argmax`.
Qwen exposes `benchmark`, plain, tools, and KV-cache modes through `SilkQwenSession`; SmolLM exposes
`benchmark` and `generate`. `NivaraChat --smollm --silk` now also serves the same SmolLM decoder as an
`IChatClient` for chat/plain generation through `SilkSmolLMChatClient`.

`SilkMobileNetV2GpuRunner` and `SilkResNet18GpuRunner` run the vision stack: `conv2d`, `batchnorm_eval`,
`relu`, `relu6`, `maxpool2d`, and `adaptive_avgpool2d`. They share the same runtime and host-side
geometry helpers, and are the kernels listed in `KernelNames` under the vision family.

As with the encoders, these are F32 only.

## 13c. Text classifiers (NivaraClassifier sample)

`GpuTextConvClassifier` (`src/Nivara.Extensions/Silk/GpuTextConvClassifier.cs`) trains the
`ConvTextClassifierModel` (three Conv1d branches k=3/5/7 → concat → FC1 → ReLU → FC2) end to end
on the device (#900):

- **Forward** is fully on-device: three `Conv2d` (H=1 spelling of conv1d) launches into branch-major subviews of one `convOutBuf`, a single in-place `Relu`, then `GemmEpilogue` (branch 0 seeds FC1 with bias) + two `GemmAdd` (branches 1–2 accumulate into the same `hBuf`) + one `Relu`, and a final `GemmEpilogue` for FC2. No concat copy and no transpose: FC1 is three accumulated GEMMs over branch-major activation blocks.
- **Backward** is device-side too: the host computes the batch-averaged softmax dZ from the
  downloaded logits (`SilkLoss.InputGradient`, the `1/n` of `CrossEntropyLoss`'s mean reduction
  folded in) and uploads it; then per branch `gemm_row4_tn` writes the FC1 weight-block gradient
  into a writable row block of `dFc1`, `gemm_row4_nt` maps dH back onto the branch, `relu_backward`
  masks it, and `conv2d_backward_weight`/`conv2d_backward_bias` close the branch; `fill` +
  three accumulating `conv2d_backward_input` launches produce dX; the device scatter
  (`embedding_backward_input`) reduces dX into `dEmbBuf` deterministically
  (the forward is a plain embedding lookup, so no pooling factor applies).
- **Optimizer** is the fused `adam_update` kernel for every device parameter (per-tensor moment
  buffers), including the device-resident embedding table and its moments. The embedding scatter
  is a deterministic on-device reduction (`embedding_backward_input`/`embedding_backward_pool`),
  not an atomic CAS. Device buffers are authoritative during
  training; host mirrors refresh when `Train` returns, so the trained model still serializes
  through the normal `ModelSerializer`/`ConvTextClassifierModel` path via `ExportCpuLayout`.
  `Dropout` is disabled on the GPU path (`Eval`) to match the deterministic AutoDiff reference
  used for parity.

`GpuTextLinearClassifier` (`src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs`) is the same
shape without convolutions: gather + mean pool on the host (dX scatter then divides by
`max(1, seqLen)`), `gemm_row4_relu`/`gemm_row4_bias` forward on the device, and the identical
backward chain (`gemm_row4_tn` → `gemm_row4_nt` → `relu_backward` → `col_sum`) with the fused
Adam step.

Parity is pinned in `tests/Nivara.Tests/Gpu/GpuTextConvClassifierTests.cs` and
`tests/Nivara.Tests/Gpu/GpuTextLinearClassifierTests.cs`: forward logits within 2e-4 and batch
gradients within 5e-4 against reverse-mode AutoDiff on identical weights (the device GEMMs
reassociate, so these are tolerance bounds, not bit-identity claims), a loss-decreasing smoke run
per classifier, and the CPU-layout round trip exact.

- **`SilkModes`** is the allow-list: `minilm`, `distilbert`, `distilbert_sst`, `modernbert`, and
  `laya` (empty, `benchmark`, `compare`); `smollm` (empty, `benchmark`, `generate`); `qwen` (empty,
  `benchmark`, `tools`, `distill`); `mobilenet_v2`, `resnet18`, and `modernbert` also carry
  `compare_diag`, which routes to the existing CPU/AutoDiff diagnostic path rather than to a Silk
  runner (the Silk runners expose only final outputs). SmolLM and Qwen are causal decoders (not
  encoders) and own their runners, as do the MobileNetV2 and ResNet-18 vision runners.
- **Mutual exclusion** (`--ilgpu` + `--silk`) is checked **before** the model-scoped guards, so
  `minilm --ilgpu --silk` names the conflict rather than one flag's model scope.
- **The model-scope guard runs before any model-file lookup** — pinned by
  `Silk_RejectedBeforeModelFileLookup`, whose vehicle is `minilm --silk similarity`: a mode the CPU
  path carries and Silk does not, so a "Model file not found" would otherwise mask the real reason.
  (`qwen --silk distill` held that role until distill became a Silk mode.)
- **F32-only**: `--silk --precision bf16|fp16` is rejected (there is no bf16/fp16 translation unit).
- **`--gpu` is rejected, not aliased** to `--ilgpu`, so an old invocation fails loudly.
- **`OpenSilkEncoder`** implements the `OpenEncoder` delegate shared by the MiniLM/DistilBERT
  SST-2 entry points, so the CPU and Silk paths stay structurally identical.
- **The gate** is `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
  (`SilkKernelsGate.cs`). It runs the CPU reference and the Silk kernels **in one process** and
  requires **byte-identity** for `gemm_row4`, `gemm_row4_bias`, `gemm_row4_qkv`, `gemm_row4_relu`,
  `gather_rows`, `add`, `add_bias`, `split_columns`, and `rotary` (pure multiply-and-add, so a bit
  difference means the accumulation order or the contraction changed — a design bug, not rounding),
  and the numeric bound `|silk − cpu| ≤ 1e-3·(1 + |cpu|)` for the rest (`gemm_row4_gelu`,
  `geglu`, `layer_norm`, `batched_attention`, and the elementwise cells, whose `exp` or `sqrt` may
  differ by an ulp). One attention cell launches `band = 64`; the BERT cells stay on the global band. Timing is best-of-N after warmups and is
  **explicitly not a verdict** — a ratio outside x0.8..x1.2 is a red flag to report, not a failure to
  encode, and the exit code counts correctness failures only. Shapes: MiniLM (seq 128, hidden 384, 12
  heads, headDim 32) and DistilBERT (seq 128, hidden 768, 12 heads, headDim 64) — DistilBERT drives
  `headDim` to `MaxHeadDim`, the attention kernel's compiled local-memory bound. An unavailable device
  or program exits `UNBUILT` (10). The gate covers the forward kernels shared by the runners; the
  trainer's backward/Adam kernels are exercised by `SilkDistilBertTrainerTests` (gradient parity)
  rather than byte-identity.
- **End-to-end compares are the other half.** `minilm --silk compare`, `distilbert --silk compare` and
  `distilbert_sst --silk compare` each check the GPU against the **same-process CPU** at
  `|gpu−cpu| ≤ 1e-3·(1+|cpu|)` — with pooled-embedding cosine for MiniLM, and the PyTorch 3-way
  (`argmax 8/8`) for SST-2. `modernbert --silk compare` and `laya --silk compare` are the same bound
  against the in-process CPU encoder and head. **Both pass (AC, 2026-10-13):** ModernBERT at maxAbs
  2.861E-005 / maxRel 5.577E-004 / cosine 1.0000001 — the identical figures to the ILGPU gate — and
  Laya at 1.110E-005 = 1.1% of the 1e-3 bound with act matching to 0.0 and mask clamp PASS on both.
  The `--silk-kernels` cells are what localise a failure; the compares are what say the
  assembled forward is right. **The benchmark A/B has been run too** — ModernBERT and Laya timing
  (back-to-back `--silk` / `--ilgpu benchmark`, AC, same session, 1 warmup + 3 timed median) is in
  the cross-backend rows are now historical; the current acceleration results and Silk-only scope are in
  [docs/ACCELERATION.md](ACCELERATION.md), and earlier backend comparisons remain in [docs/ILGPU.md](ILGPU.md).
  The compare and the benchmark remain separate verdicts.

## 13e. SilkDistilBertTrainer (#956)

`SilkDistilBertTrainer` (`src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`) trains a
DistilBERT sequence-classification head end to end on the device:

- **Forward** materialises every layer intermediate (`qkv`, `attn`, pre-norm sums `p1`/`p2`,
  `h1`, `f1pre`, `f1post`) instead of reusing one set of buffers, so the reverse chain has
  something to differentiate. It runs `gemm_row4_bias` + `gelu` (bit-identical to `gemm_row4_gelu`)
  and `layer_norm` over a materialised `add` (bit-identical to `layer_norm_residual` with a zero
  partner), so its logits are byte-identical to `SilkBertEncoderRunner`.
- **Backward** is the composed chain: head (classifier → ReLU → pre_classifier → CLS scatter) →
  per layer (LN2 → FC2 → GELU → FC1 → LN1 → attention → qkv) → embedding LN → word/position
  embedding scatter. Weight gradients are written directly in device layout (`gemm_row4_tn` with
  the same packed q/k/v column blocks), bias and LayerNorm-parameter gradients via `col_sum`
  scaled by 1.0 (the mean loss already scaled dLogits by 1/batch). The attention backward writes
  its gradient row-major `[rows, 3·hidden]` in q|k|v column order even though the forward's packed
  q/k/v output is block-major, so `dW`/`dB`/`dX` reuse the plain GEMM path with no unpack pass.
- **Optimizer** is one `adam_update` per parameter with the caller's weight decay (0.01 by
  default, matching the CPU path's single parameter group); the two moment buffers are zeroed at
  the start of `Train`.
- **Materialization** is `ExportCheckpoint()` (HF-keyed, CPU layout) + `model.LoadWeights(...)` +
  `ModelSerializer.Save` — no device checkpoint format. `ExportGradients()` inverts the same layout
  so the device gradients can be compared key-for-key with a reverse-mode pass.
- Caps come from the constructor (`maxBatch`, `maxSeqLen`); the trainer throws when the
  checkpoint has no `pre_classifier`/`classifier` head.

Parity is pinned in `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs`: forward logits within
2e-4 and every parameter gradient within 5e-4 against `DistilBertForSequenceClassification<float>`
driven through `CrossEntropyLoss`, on a 2-layer / hidden-32 / 4-head / vocab-64 / seqLen-8 model.

## 13f. Retrieval fusion (`RagLearnMode`, #915)

`SilkRetrievalTrainer` (`samples/NivaraChat/Helpers/SilkRetrievalTrainer.cs`) trains the
#915 late-fusion weights (and the stats/embed query gate) on the device when `--rag-learn --silk`
is requested, reusing the existing kernels rather than a new model runner:

- **Forward** launches `fusion_forward` over the per-query candidate pool (`[rows, 4]` scores
  plus presence mask, weights `w`): `f[i] = (Σ_k p[i,k] s[i,k] w[k]) / (Σ_k p[i,k] w[k])`.
  The denominator is never substituted; a zero divisor throws, preserving the CPU path's parity.
- **Gradient** runs `fusion_wgrad` to reduce `dθ[k] = Σ_i g[i]·p[i,k]·(s[i,k] − f[i]) / d[i]`
  and then reweights by `w[k]·(contrib − Σ_v w[v]·contrib)` to match `ReverseGradOperations`'s
  softmax Jacobian (the weights are `w = softmax(theta)`).
- **Optimizer** is the fused `adam_update` kernel over the 4 (global) / 4 (gate bias) plus gate
  weights, with the host bias corrections precomputed. Momentum/v buffers are zeroed per run.
- **Gate** combines a CPU feature vector (8 stats features or the 384-dim MiniLM embedding) with
  a hand-written GEMM and reuses the same fusion path.
- **Parity** is the mode's printed `Init parity: max |learned − fixed|` (target ~1e-7) and the
  held-out comparison; `--loss pairwise` deliberately falls back to the CPU `FusionModel` because
  the Silk trainer is listwise-only.

## 13d. Wiring coverage (user-facing entry points)

Silk (`--silk`) is honoured by the following user-facing entry points as of this release:

- **NivaraChat**
  - `--embed --silk` — embedding search (MiniLM CLS vectors; `SilkEmbeddingGenerator` wrapper).
  - `--intent --silk` — intent router's factual-retrieval MiniLM generator.
  - `--rag* --silk`, `--rag-multiview --silk` — RAG ingestion/search embeddings
    (already wired via `MiniLMEmbedder.Create`).
  - `--rag-learn --silk` — differentiable retrieval fusion training via
    `SilkRetrievalTrainer` (listwise loss; pairwise falls back to CPU).
  - `--smollm --silk` (and `--smollm --precision bf16 --silk`) — SmolLM chat/plain
    generation on `SilkSmolLMChatClient`.
  - `--qwen --silk …` — tools/plain/benchmark/distill through `SilkQwenSession`.
- **NivaraInference**
  - `minilm {default, benchmark, compare, similarity}`; `distilbert …`; `distilbert_sst
    {default, benchmark, compare, predict}`; `modernbert {default, benchmark, compare, compare_diag}`;
    `laya {default, benchmark, compare}`; `smollm {default, benchmark, generate}`; `qwen …`;
    `mobilenet_v2`, `resnet18 {default, benchmark, compare, compare_diag}`.
- **NivaraClassifier** — `train/predict/benchmark` for the conv and linear text classifiers
  (`GpuTextConvClassifier` / `GpuTextLinearClassifier`).
- **NivaraIncident.Cli** — `--gpu` grouped-aggregation path (`grouped_hash_reduce_partial`).
- **NivaraFineTuning** — `--mode {train,eval,predict} --silk`: training (forward, backward, AdamW)
  runs on the device via `SilkDistilBertTrainer`; eval and predict use `SilkBertEncoderRunner`.
  F32-only; exits 1 when OpenCL is unavailable (no CPU fallback).

Deliberately out of scope (no Silk seam exists yet):
- `NivaraChat --embed`/`--intent` without a model directory present (CPU fallback only).
- `minilm --silk compare_diag` is not a mode (diagnostics is an AutoDiff/CPU path).

## 15. Adding a model to the Silk path

1. The runtime is model-agnostic. A BERT-shaped model changes `SilkKernels.Source` and
   `SilkBertEncoderRunner`. A model that is not BERT-shaped adds a runner, the way ModernBERT and
   Laya did. Do not fold those into the BERT runner.
2. Add a kernel only if the operation is not already in `KernelNames`; reuse
   `GEMM_ACCUMULATE_CORE` for any new f32 GEMM — through the forward macro, or through the
   NT/TN backward variants when an operand is stored transposed — so the byte-identity claim
   keeps holding.
3. Keep the accumulation order identical to the CPU reference of the same shape — the gate compares
   them directly.
4. Keep `KernelNames` in sync with the translation unit (add the name constant).
5. Take `Transposed`, `QkvPacked` and `Req` from `WeightLayout`, and dispatch from `SilkLauncher`.
   The CPU reference assumed by the gate consumes the same host layout, so these come from
   `WeightLayout` rather than a per-backend re-derivation: the drift would be a silently wrong
   attention rather than a crash.
6. Register every buffer so `Dispose` is unconditional; release sub-buffers in a `finally`.
7. Add the model to `SilkModes` and the CLI dispatch, and extend `InferenceCliArgumentTests` for each
   new rejection.
8. Add the shape to `SilkKernelsGate.Shapes`; mark any pure multiply-and-add cell `requireBytes:
   true`.
9. Watch `headDim`: `MaxHeadDim` is a compile-time bound, so a model above 64 needs the constant
   raised — `EnsureAttentionLocalMemory` rejects it at construction and says so.

---

## 14. Follow-up issues (Silk fine-tuning)

Open follow-ups from the #956 implementation, in priority order. All are tracked as GitHub
issues; status and discussion live there.

| # | Priority | Title | Notes |
|---|----------|-------|-------|
| [#962](https://github.com/khurram-uworx/Nivara/issues/962) | High | test(silk): add loss-decreasing smoke test for `SilkDistilBertTrainer` | Gradient parity proves the math; this proves the AdamW wiring |
| [#963](https://github.com/khurram-uworx/Nivara/issues/963) | High | feat(samples): add per-batch timing to `RunTrainSilk` | Landed (khurram/963); unblocks #960 |
| [#964](https://github.com/khurram-uworx/Nivara/issues/964) | Medium | perf(samples): use `SilkBertEncoderRunner` for dev eval | Avoids per-epoch weight export (66M params × 2 conversions) |
| [#965](https://github.com/khurram-uworx/Nivara/issues/965) | Medium | fix(samples): consistent `--max-len` clamping in `RunTrainSilk` | Training vs dev-eval shape mismatch risk |
| [#966](https://github.com/khurram-uworx/Nivara/issues/966) | Low | refactor(samples): simplify `RunTrainSilk` training loop | Single `Train` call + per-epoch eval |
| [#960](https://github.com/khurram-uworx/Nivara/issues/960) | Measurement | bench: add Silk column to NivaraFineTuning benchmark table | Harness + docs landed (khurram/960); AC-power readings pending |
| [#961](https://github.com/khurram-uworx/Nivara/issues/961) | Investigation | perf(silk): low GPU utilization during training | <10% GPU on Intel Graphics; profile first |

---

## References

- [Silk.NET OpenCL](https://github.com/dotnet/Silk.NET)
- [OpenCL 1.2 Specification](https://registry.khronos.org/OpenCL/specs/opencl-1.2.pdf)
- [Khronos OpenCL](https://www.khronos.org/opencl/)
- Series: [SPIR-V / Level Zero](../tests/Nivara.GpuProbe/SPIRV.md) · [SYCL / oneAPI](../tests/Nivara.GpuProbe/SYCL.md) · [DX12](../tests/Nivara.GpuProbe/DX12.md) · [OpenVINO](../tests/Nivara.GpuProbe/OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](../tests/Nivara.GpuProbe/COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)

## Running device-dependent tests locally

Device-dependent (OpenCL/Silk) tests are categorized [Category("Gpu")] and are excluded from the main CI filter.

- With an OpenCL GPU available:
  ``powershell
  dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "Category=Gpu"
  ``n- On CI (no GPU) these tests are excluded by Category!=Gpu and skip via Assert.Ignore when run manually.

