# Plan: Phase 2/3 GPU integration fixes and profitability gate

Branch: khurram/443-phase-2-3
Base: main (merged PR #726)

## Problem

PR #726 added the GPU plumbing (CLI flags, GpuOffloadDecision record, Silk analytics
kernel) but left AnalyzeGroupedAggregation as a CPU-only dictionary loop. The first
integration commit (f97515cc) wired the GPU path but introduced three regressions and
left timing/heuristic gaps:

1. TypedLinqGroupedAggregation_AgreesWithTheHandRolledAggregation fails: CPU fallback
   accumulates float durations into a double total, losing ~1e-5 vs the original double
   accumulation.
2. IncidentSurface_HasNoUnregisteredMembers fails: Analysis.LastGpuOffloadDecision is a
   new public static member not registered in Exercises()/Skipped()/CoveredTransitively().
3. IncidentSurface_SurfaceEnumerationHasNoKnownGaps fails (pre-existing on main): the
   GpuOffloadDecision record synthesizes two Equals overloads, which Surface() keys by
   name alone and flags as a collision.
4. swCpu wraps the GPU block, so cpuMs/gpuMs are mislabeled; the CPU fallback is not
   timed.
5. Empty filtered result (groupCount == 0) crashes in SilkAnalytics.GroupedHashReduce.
6. No profitability heuristic: --gpu always offloads when a GPU exists.

## Proposed changes

### Commit 1 — Restore double precision on the CPU path
File: samples/Nivara.Samples/Incident/Analysis.cs
- Keep float[] durations for the GPU call.
- In the CPU fallback, accumulate the original double from durationCol, not the float.
- Fixes TypedLinq parity tolerance.

### Commit 2 — Guard empty results
File: samples/Nivara.Samples/Incident/Analysis.cs
- If rowCount == 0 or groupCount == 0, skip both GPU and CPU aggregation, return an
  empty frame, reason "empty result", cpuMs/gpuMs = 0, transferBytes = 0.

### Commit 3 — Fix stopwatches
File: samples/Nivara.Samples/Incident/Analysis.cs
- Use separate Stopwatch instances for CPU and GPU.
- GPU success path: gpuMs = actual GPU time; cpuMs = estimated CPU time (from sample
  calibration) when the heuristic is used, otherwise 0.
- CPU fallback path: cpuMs = measured full CPU aggregation time; gpuMs = 0.
- Not-requested path: cpuMs = measured full CPU aggregation time; gpuMs = 0.

### Commit 4 — Profitability heuristic (Phase 2)
New file: samples/Nivara.Samples/Incident/GpuProfitability.cs
- GpuProfitabilityHeuristic with Margin, SampleSize, EstimatedGpuFixedMs,
  EstimatedGpuBytesPerMs, EstimatedGpuMsPerRow.
- CalibrateCpuMs(...) runs the CPU aggregation on the first sampleSize rows.
- EstimateCpuMs / EstimateGpuMs / ShouldOffload(...).
- Integrate into AnalyzeGroupedAggregation: decide before running GPU; forced mode
  bypasses the gate but still records the estimate.

### Commit 5 — Register Analysis.LastGpuOffloadDecision in IncidentSurfaceTests
File: tests/Nivara.Tests/Incident/IncidentSurfaceTests.cs
- Add an Exercises() entry that calls AnalyzeGroupedAggregation(tempDir, scenario, useGpu: false)
  and reads LastGpuOffloadDecision, asserting it is non-null.
- Or add a Skipped() entry with justification. Prefer exercise.

### Commit 6 — Resolve GpuOffloadDecision Equals collision
File: tests/Nivara.Tests/Incident/IncidentSurfaceTests.cs
- In IncidentSurface_SurfaceEnumerationHasNoKnownGaps, exclude the well-known
  object methods (Equals) from the overload-collision check, since Surface() keys by
  name and these overloads are expected.

### Commit 7 — Tests
New file: tests/Nivara.Tests/Incident/GpuProfitabilityTests.cs
- Pure decision-logic tests (no GPU/OpenCL dependency).

Extend: tests/Nivara.Tests/Incident/AnalysisTests.cs (or new GpuIncidentTests.cs)
- Availability-guarded end-to-end parity test for AnalyzeGroupedAggregation with
  useGpu: true.

### Commit 8 — Documentation
File: samples/NivaraIncident/README.md
- Add a "GPU offload" subsection: --gpu is profitability-gated, --gpu-force bypasses,
  decision is surfaced per run.

## Verification

- dotnet build Nivara.slnx
- dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Incident"
- dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkGroupedHashReduce"
- (optional, human-run) NivaraIncident CLI --gpu / --gpu-force on a generated dataset.

## GitHub issues log

- (none yet)
