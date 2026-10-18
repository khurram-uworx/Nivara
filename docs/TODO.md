# Plan: Silk/Laya attention-route performance (khurram/typesafe-laya-perf)

## Problem
The Silk Laya path's dominant cost is attention: `BatchedAttention` is ~57% of the forward (#447).
GEMM tiling is currently measured-null on this iGPU, and CPU-side gaps (#458/#457) are a separate track.
Before changing kernels, establish a measurable attention profile/baseline and reduce avoidable
per-request overhead in the type-safe Nivara route.

## Proposed changes
1. Add a profiling/perf scenario in `tests/Nivara.PerformanceTests` for the Laya attention path
   (packed QKV, cache pressure, sequence lengths, median timing), rather than a scratch project.
2. Reduce per-request overhead in `NivaraTypeSafeClient.SystemOneAsync`:
   - reuse prompt-building/state-tokenization where safe
   - keep device work batched
   - avoid preventable host synchronization
3. Recondition GEMM tiling only after attention measurements are captured on one addressing form.
4. Record any out-of-scope findings as GitHub issues immediately.

## Verification
- Silk `compare`/parity remains exact for any kernel experiment.
- `GpuAttentionBandTests` and related attention tests pass.
- All perf runs use Release and the repository A/B methodology.
- `dotnet format` whitespace check before PR.

## Planned commits
1. docs: plan silk-laya-perf in TODO.md
2. perf: add attention scenario coverage
3. perf: reduce per-request overhead in type-safe Laya route
4. (optional, evidence-gated) attention experiment
5. docs: remove TODO.md — plan executed

## GitHub issues log
- #706 — Track Silk Laya attention-path performance work

## Blast radius
- `samples/NivaraInference/NivaraTypeSafeClient.cs`
- `src/Nivara.Extensions/Silk/SilkLayaSession.cs`, `SilkLayaHeadGpuRunner.cs`
- `tests/Nivara.PerformanceTests`
- Silk compare/parity tests remain the guardrails.

