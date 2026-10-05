# Silk.NET OpenCL GPU Probe — Lessons from the GpuProbe

> **Series:** [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)

All findings below were verified on an **Intel Arc 140T** (8086:7DD1, 128 EU, driver **1.15.37858**) running Windows 10.0.26200 and .NET 11.0. The Silk.NET OpenCL leg of `tests/Nivara.GpuProbe` uses **Silk.NET.OpenCL 2.23.0** (thin .NET Foundation bindings to the OpenCL C API) to discover platforms/devices and to run OpenCL C kernels directly on the in-box OpenCL ICD + Intel driver. The goal is a vendor-neutral baseline (especially Intel/AMD iGPUs) for apples-to-apples comparisons with other backends.

---

## 1. What the leg does (current state)

The current Silk.NET leg (`OpenCl/SilkProbe.cs`) is intentionally minimal and stable:

- Enumerates OpenCL platforms and devices via `CL.GetApi()`.
- Logs platform names and GPU device names/vendors; prefers **Intel** and **AMD** GPUs when present (case-insensitive). Falls back to the first GPU if none match.
- Reports basic device caps: `CL_DEVICE_MAX_COMPUTE_UNITS`, `CL_DEVICE_LOCAL_MEM_SIZE`, `CL_DEVICE_MAX_WORK_GROUP_SIZE`.
- Creates an OpenCL context and command queue for the selected device.
- Runs a small **vector add** (1M elements) kernel written in OpenCL C as a correctness + timing sanity check.

This mirrors the spirit of the existing `LevelZero/OclProbe.cs` (direct P/Invoke) but uses Silk.NET's managed bindings instead of manual delegate loading. The leg is probe-scoped only (no changes to `src/Nivara`).

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
dotnet run -c Release --project tests/Nivara.GpuProbe -- opencl
# or
dotnet run -c Release --project tests/Nivara.GpuProbe -- silk-opencl
```

Expected (on Intel Arc 140T): platform `Intel(R) OpenCL Graphics` detected, preferred device marked `*`, caps logged, vector add sampled PASS with median timing.

---

## 4. Implementation notes (Silk.NET 2.23.0)

- **Getting the API**: `var cl = CL.GetApi();` returns the loaded OpenCL API.
- **Platform/device enumeration**: Use `cl.GetPlatformIDs` and `cl.GetDeviceIDs(DeviceType.Gpu, ...)`. Prefer vendors containing `Intel` or `AMD` (case-insensitive).
- **Info queries**: `cl.GetPlatformInfo`/`cl.GetDeviceInfo` fill byte buffers; trim trailing NUL. Use `PlatformInfo.Name`, `DeviceInfo.Name`, `DeviceInfo.Vendor`, `DeviceInfo.MaxComputeUnits`, `DeviceInfo.LocalMemSize`, `DeviceInfo.MaxWorkGroupSize`.
- **Context/queue**: `cl.CreateContext` with selected device; `cl.CreateCommandQueue` with `CommandQueueProperties.None` (in-order). Simpler for correctness/timing.
- **Program/kernel**: `cl.CreateProgramWithSource` from UTF8 kernel string, `cl.BuildProgram` (pass `(byte*)null` for options if none), `cl.CreateKernel`.
- **Buffers**: `cl.CreateBuffer` with `(uint)CL_MEM_READ_WRITE`. Read/write via `cl.EnqueueWriteBuffer`/`cl.EnqueueReadBuffer` with `blocking=true` for simplicity in probe.
- **Launch**: `cl.EnqueueNDRangeKernel` with global/local work sizes; `cl.Finish(queue)` to sync before readback/timing.
- **Resource cleanup**: Release in reverse order (mem objects, kernel, program, queue, context).
- **Pointers/unsafe**: Silk.NET uses pointer parameters for many calls; build with `AllowUnsafeBlocks=true` (already set). Keep unsafe blocks minimal and localized.

---

## 5. OpenCL C notes for iGPUs (Intel/AMD)

- **Target 1.2**: Safe common denominator for Intel/AMD iGPUs. Avoid 2.0/3.0-only features unless gated by device queries.
- **Workgroup size**: Start with power-of-two (e.g. 256). Query `CL_DEVICE_MAX_WORK_GROUP_SIZE` and clamp. Some older drivers behave better with 64/128.
- **Local memory**: Use `__local` sparingly; iGPUs often have small local memory vs discrete. For transpose/GEMM tiles, 16x16 is usually reasonable; measure.
- **Coalescing**: Write kernels with coalesced global accesses where possible (critical for integrated memory bandwidth).
- **Kernel strings**: Keep simple and deterministic. For parity with SimdProbe, match the same reference logic (row-major indexing, same shapes).
- **Build logs**: On build failure, fetch `CL_PROGRAM_BUILD_LOG` via `cl.GetProgramBuildInfo` — invaluable for Intel/AMD driver diagnostics.

---

## 6. Parity plan (deferred, tracked)

Full vecadd/transpose parity with `Nivara.SimdProbe` transpose shapes is planned but deferred to avoid churn with Silk.NET 2.23.0 API binding details. When implemented:

- **Vector add**: 1M elements, correctness vs CPU reference, median timing (already present in spirit; can be expanded to match OclProbe's reporting).
- **Transpose**: row-major in/out, CPU reference bit-exact, test shapes matching SimdProbe: (1024,1024), (512,512), (2048,1024), (1024,2048), (129,257). Report correctness + timing.
- **A/B style**: Keep consistent timing discipline (warmups, best-of/median) to compare meaningfully against other backends.

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

- **API signature nuances**: Different Silk.NET versions can vary slightly in pointer types (byte*/sbyte*/nint*/nuint*). Keep casts minimal and match working patterns.
- **In-order queue**: Chosen for clarity; out-of-order + events possible later if needed for overlapping transfers/exec.
- **32-bit buffer sizes**: For very large allocations on old drivers, watch size limits; 1M elements (4MB) is fine.
- **Driver variance**: Intel vs AMD OpenCL drivers differ slightly in caps/log messages; discovery prefers Intel/AMD explicitly.
- **Build stability**: Current minimal discovery is stable; full kernel execution paths should be added incrementally with careful error checking.

---

## 9. Reproducibility

Verified on:
- Host: Intel Core Ultra 7 255H (Arrow Lake-H), Arc 140T iGPU
- OS: Windows 10.0.26300
- .NET: 11.0
- Silk.NET.OpenCL: 2.23.0
- Driver: Intel Graphics driver with in-box OpenCL ICD

Run: `dotnet run -c Release --project tests/Nivara.GpuProbe -- opencl` — platform/device detected, caps printed.

---

## References

- [Silk.NET OpenCL](https://github.com/dotnet/Silk.NET)
- [OpenCL 1.2 Specification](https://registry.khronos.org/OpenCL/specs/opencl-1.2.pdf)
- [Khronos OpenCL](https://www.khronos.org/opencl/)
- Series: [SPIR-V / Level Zero](SPIRV.md) · [SYCL / oneAPI](SYCL.md) · [DX12](DX12.md) · [OpenVINO](OPENVINO.md) · [ILGPU](ILGPU.md) · [ComputeSharp](COMPUTESHARP.md) · [Silk.NET OpenCL](SILK.md)
