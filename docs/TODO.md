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
Create `tests/Nivara.GpuProbe/OpenCl/` folder. Add `SilkProbe.cs` with device discovery, prefer Intel/AMD GPUs, log caps, basic vector add correctness+timing.

### 3) Wire into Program.cs
Add mode `opencl` and `silk-opencl`.

### 4) Documentation
Update `tests/Nivara.GpuProbe/README.md` to document Silk.NET OpenCL mode.

## Verification
- Build succeeded. Ran `dotnet run -c Release -- opencl` and `silk-opencl` - both detect Intel iGPU.

## Commits (logical units)
1. deps: add Silk.NET.OpenCL to Nivara.GpuProbe ✓
2. feat(gpu-probe): add Silk.NET OpenCL backend with device discovery and vector add ✓ (discovery+vecadd done)
3. feat(gpu-probe): add OpenCL transpose with CPU reference parity (next)
4. feat(gpu-probe): wire opencl mode in Program.cs ✓
5. docs(gpu-probe): document Silk.Net OpenCL mode (Intel/AMD focus)

## Blast radius
tests/Nivara.GpuProbe/* only. No public API changes. No src changes.

## GitHub issues log
- [ ] (to be filled if deferred work found during execution)

## Notes
- Target OpenCL 1.2 for broad iGPU support. Silk.NET 2.23.0.
- Intel iGPU detected successfully on Arc 140T.

## Grounding results (G1)
Basic structure in place; following existing probe patterns.

## G2 checklist (pre-deletion review)
- [ ] Branch work as a whole reviewed vs issue/plan
- [ ] Branch vs TODO.md item-by-item
- [ ] Both clear, ready to delete TODO.md
