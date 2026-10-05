# Plan: Add Silk.Net.OpenCL to Nivara.GpuProbe (Intel/AMD iGPU focus)

## Status
- Added Silk.NET.OpenCL (2.23.0) to probe csproj. 
- Added OpenCl/SilkProbe.cs with platform discovery; detects Intel iGPU. 
- Wired modes "opencl" and "silk-opencl" in Program.cs.
- Build stable; basic discovery works.

## Next steps
- Extend SilkProbe to select preferred Intel/AMD device, log caps (CUs, maxWG, localMem), create context/queue.
- Add vector add (1M elements) with correctness check + timing (parity with existing OclProbe).
- Add transpose kernel in OpenCL C matching Nivara.SimdProbe transpose shapes for apples-to-apples comparison.
- Update README.md with Silk.NET OpenCL mode, Intel/AMD focus, usage.

## Commits
1. deps ✓, 2. feat(discovery) ✓, 3. feat(vecadd/transpose) pending, 4. wire ✓, 5. docs pending

## Blast radius
tests/Nivara.GpuProbe only. No src changes.

## GitHub issues log
- [ ]

## G2 checklist
- [ ] Reviewed as whole
- [ ] Reviewed against TODO
- [ ] Ready to delete
