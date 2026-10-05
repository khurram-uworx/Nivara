# Plan: Add Silk.Net.OpenCL to Nivara.GpuProbe (Intel/AMD iGPU focus)

## Status
- Added Silk.NET.OpenCL (2.23.0) to probe csproj.
- Added OpenCl/SilkProbe.cs with platform discovery (works; detects Intel iGPU).
- Wired modes "opencl" and "silk-opencl" in Program.cs.
- Build stable.

## Next
- Add device selection (prefer Intel/AMD), caps logging, context+queue, vector add with correctness+timing (incrementally, matching existing probe style).
- Add transpose in OpenCL C with CPU reference parity to SimdProbe shapes.
- Document in README.md.

## Commits
1. deps ✓, 2. discovery ✓, 4. wire ✓, 3/5 pending.

## Blast radius
tests/Nivara.GpuProbe only.

## GitHub issues log
- [ ]

## G2 checklist
- [ ] Branch as whole reviewed
- [ ] Against TODO reviewed
- [ ] Clear to delete
