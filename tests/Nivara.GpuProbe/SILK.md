# Silk.NET OpenCL GPU Probe — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU, driver **1.15.37858**) running Windows 10.0.26200 and .NET 11.0. The Silk.NET OpenCL leg of `tests/Nivara.GpuProbe` uses **Silk.NET.OpenCL 2.23.0** (thin .NET Foundation bindings to the OpenCL C API) to discover platforms/devices and to run OpenCL C kernels directly on the in-box OpenCL ICD + Intel driver. The goal is a vendor-neutral baseline (especially Intel/AMD iGPUs) for apples-to-apples comparisons with other backends.

---

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

| kernel | CPU (production Nivara) | Silk.NET OpenCL | gate |
|---|---|---|---|
| dot16 | 1.6 µs | **6.8 µs** | PASS — bit-exact, 0.0 ULP |
| silu | 32.4 µs | **6.1 µs** | PASS — 576/576, worst 4.0 ULP |
| gemv | 2963.6 µs | **90.0 µs** | PASS — 1536/1536 |

Silk.NET is the fastest OpenCL/OpenCL-C row in the gate (ILGPU 10.4 / 7.0 / 135.9 µs on the same run) because it is the only leg that dispatches the compiled binary directly — no ILGPU runtime abstraction between the NDRange and the queue. Build cost is ~5–6 ms (in-process, driver JIT); first dispatch adds 62–415 µs.

**Note on the observed ULP figures:** `gemv`'s worst-case 14336 ULP and `silu`'s 4.0 ULP are *not* Silk.NET-specific — the DX12, OV-f32, ILGPU and ComputeSharp legs report identical values, because it is f32 rounding at those magnitudes against a slightly different summation order, not a kernel difference. `|diff|` is ~1.6e-09 for gemv and ~3.7e-09 for silu, both far inside the `1e-6 + 1e-5·|ref|` gate. `OV (bf16 IR)` is the only silu leg that genuinely fails (402/576) — a real OpenVINO bf16 IR precision issue, pre-existing and tracked separately.

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
  kernels mode exit: 0 failed kernel(s), 0 unexpected
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

---

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
fails 402/576, which are pre-existing conditions unrelated to Silk.NET. Verify the Silk.NET
row's own PASS lines, not just the process exit code.

---

## References

- [Silk.NET OpenCL](https://github.com/dotnet/Silk.NET)
- [OpenCL 1.2 Specification](https://registry.khronos.org/OpenCL/specs/opencl-1.2.pdf)
- [Khronos OpenCL](https://www.khronos.org/opencl/)
- Series: [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)
