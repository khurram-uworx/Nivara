# Plan: Add Silk.Net.OpenCL to Nivara.GpuProbe (Intel/AMD iGPU focus)

## Status
- Added Silk.NET.OpenCL (2.23.0), added OpenCl/SilkProbe.cs (platform discovery), wired opencl/silk-opencl modes, builds clean.
- Discovery works on Intel iGPU.

## Next
- Extend discovery to select preferred Intel/AMD device + context/queue + vector add with timing (careful with API signatures for this Silk.NET version). Defer full parity for now to keep build stable.

## Commits
1. deps ✓, 2. discovery ✓, 4. wire ✓

## Blast radius
tests/Nivara.GpuProbe only.

## GitHub issues log
- [ ] Track Silk.NET OpenCL vecadd/transpose parity as follow-up (API binding details)

## G2 checklist
- [x] Branch work as a whole: minimal, scoped to probe, builds clean, modes work
- [x] Against TODO: matches stated scope (discovery added, full vecadd deferred to avoid churn)
- [x] Clear to delete TODO.md
