# Plan: issue #824 — drop ILGPU DistilBERT GPU path; keep Silk

## Problem
DistilBERT still exposes an ILGPU entry point (`--ilgpu`) alongside the Silk backend.
Remove ILGPU references for DistilBERT so `--silk` is the only GPU backend for
that model family, while leaving MiniLM/ModernBERT/Laya ILGPU paths intact.

## Blast radius
- `samples/NivaraInference/Program.cs`: `GpuModes`, dispatch `switch` (distilbert, distilbert_sst), six wrapper methods.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: `GpuModeCases` (one distilbert_sst row).
- `samples/NivaraInference/README.md`: two ILGPU command lines (optional).
- No `src/` changes; `OpenIlgpuEncoder` stays (MiniLM still uses it).

## Proposed changes
1. Remove `distilbert`/`distilbert_sst` from `GpuModes`.
2. Delete the `useGpu` branches in `distilbert` and `distilbert_sst` cases.
3. Delete `RunDistilBertGpuInference`, `BenchmarkDistilBertGpu`, `RunDistilBertGpuCompare`,
   `RunDistilBertSstGpuInference`, `BenchmarkDistilBertSstGpu`, `RunDistilBertSstGpuCompare`.
4. Update help/usage GPU model list accordingly.
5. Fix now-invalid test row; rebuild sample.

## Verification
- `dotnet build Nivara.slnx` (sample).
- `InferenceCliArgumentTests`.
- `distilbert --ilgpu` exits 1 (generic guard); `distilbert --silk` unchanged.

## Planned commits
1. docs: plan #824 in TODO.md
2. chore: drop ILGPU DistilBERT entry points
3. test: update InferenceCliArgumentTests GpuModeCases
4. docs: remove TODO.md after review

## GitHub issues log
(none)