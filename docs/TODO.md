# Plan: Add Silk.Net.OpenCL to Nivara.GpuProbe (Intel/AMD iGPU focus)

## Problem statement recap
We created tests/Nivara.GpuProbe and tried a few things; missed Silk.Net (the .NET Foundation blessed GPU library). Only interested in Silk.Net's OpenCL side (edge devices / embedded GPUs - Intel iGPU and AMD GPU). Goal is parity with what other probes did so comparisons are apples-to-apples.

## Grounding (G1) - to do before implementation
- Ground plan via microsoft-learn MCP (OpenCL basics, .NET interop patterns). Check relevance to Intel/AMD iGPU paths.
- Navigate codebase with code-memory MCP for probe patterns (Program.cs modes, KernelGate, existing backends).
- Blast radius: tests/Nivara.GpuProbe only (no src/Nivara changes). Files affected: csproj, new OpenCl/Silk backend, Program.cs, README.md.

## Changes

### 1) Add Silk.Net.OpenCL dependency
Add PackageReference to `tests/Nivara.GpuProbe/Nivara.GpuProbe.csproj`:
- `Silk.NET.OpenCL` (latest stable). Keep probe-scoped only.

### 2) Create OpenCL backend (Silk.Net)
Create `tests/Nivara.GpuProbe/OpenCl/` folder (or `Backends/OpenCl/`? follow existing layout; see LevelZero/D3d12/OpenVino/Sycl/Kernels structure). Existing has LevelZero/OclProbe.cs (P/Invoke direct). Better to add `OpenCl/SilkOpenClProbe.cs` (or `OpenCl/SilkProbe.cs`) to distinguish from the P/Invoke OclProbe.

Add `SilkOpenClProbe.cs` with:
- Device discovery: enumerate platforms/devices, filter CL_DEVICE_TYPE_GPU. Prefer Intel/AMD (case-insensitive) by default; allow override via args (platform/device index).
- Log: platform/vendor, device name/vendor, OpenCL version, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE, preferred vector widths.
- Context + command queue (in-order initially).
- Vector add (correctness + timing) - step A.
- Transpose (row-major) matching Nivara.SimdProbe transpose shapes - step B (parity).
- Helpers: program build with build log on failure, buffer create/write/read, error checking.

### 3) Wire into Program.cs
Add mode `opencl` / `opencl-silk`. Extend switch to include:
- "opencl" or "opencl-silk" → run SilkOpenClProbe.Run() (vector add + device query)
- maybe add "opencl-correctness" and "opencl-benchmark" submodes later? Start simple.

### 4) Documentation
Update `tests/Nivara.GpuProbe/README.md`:
- Add Silk.Net OpenCL mode, focus Intel/AMD iGPUs, parity goal, how to run, requirements (OpenCL drivers), caveats.

## Verification
- Build: `dotnet build Nivara.slnx --no-restore` on probe project or full
- Run: `dotnet run -c Release --project tests/Nivara.GpuProbe -- opencl` 
- Correctness: vector add 10^6 elements passes. Transpose compares against CPU reference for test shapes.
- Compare with existing probes as shapes added.

## Commits (logical units)
1. deps: add Silk.NET.OpenCL to Nivara.GpuProbe
2. feat(gpu-probe): add Silk.Net OpenCL backend with device discovery and vector add
3. feat(gpu-probe): add OpenCL transpose with CPU reference parity
4. feat(gpu-probe): wire opencl mode in Program.cs
5. docs(gpu-probe): document Silk.Net OpenCL mode (Intel/AMD focus)

## Blast radius
tests/Nivara.GpuProbe/* only. No public API changes. No src changes.

## GitHub issues log
- [ ] (to be filled if deferred work found during execution)

## Notes
- Target OpenCL 1.2 for broad iGPU support.
- In-order queue for clarity. Timing via Stopwatch around enqueue+finish.
- Prefer Intel/AMD devices by default. Let user override via platform/device args if needed.

## Grounding results (G1)
[fill after grounding]

## G2 checklist (pre-deletion review)
- [ ] Branch work as a whole reviewed vs issue/plan
- [ ] Branch vs TODO.md item-by-item
- [ ] Both clear, ready to delete TODO.md
