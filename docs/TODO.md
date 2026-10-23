# Plan: chore: remove ILGPU package and --ilgpu backend from samples (#814)

## Problem

The Silk/OpenCL path is the shipped GPU backend, but `--ilgpu` and its ILGPU
package remain in `samples/Nivara.Samples` / `NivaraInference` plus ILGPU-dependent
perf probes, even though DistilBERT's GPU path was already dropped (10862a5a).

## Blast radius

- `samples/Nivara.Samples/Gpu/*` (10 files) — all ILGPU.
- `samples/Nivara.Samples/Nivara.Samples.csproj` package refs.
- `samples/NivaraInference/Program.cs` (useGpu, GpuModes, guards, modernbert/laya
  dispatch), comments in ModernBert.cs/Laya.cs, README.
- `tests/Nivara.Tests/Gpu/*` (3 files).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` (--ilgpu cases).
- `tests/Nivara.PerformanceTests` (GemmBenchmark, GemmLegBenchmark, GpuAllocProbe,
  ClDriverVersion).
- `AGENTS.md` one line.

No `src/` changes. `tests/Nivara.GpuProbe` keeps its ILGPU leg (documented history).

## Planned commits

1. `docs: plan #814 in TODO.md`
2. `chore: drop ILGPU from Nivara.Samples`
3. `chore: remove --ilgpu backend from NivaraInference CLI`
4. `test: remove ILGPU probe tests and ILGPU --ilgpu CLI cases`
5. `chore: trim ILGPU from PerformanceTests`
6. `docs: update AGENTS.md for Silk-only backend`
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)