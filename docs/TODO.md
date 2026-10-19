# Plan: Phase 4 of issue #443 — GPU offload measurement and documentation

## Problem

Phase 0-3 landed the `--gpu` / `--gpu-force` flags, the Silk grouped hash-reduction
kernel, the profitability gate, and correctness parity tests. Phase 4 must make the
pilot *measurable* and *documented*: a CPU-vs-GPU measurement surface, decision
diagnostics, availability-guarded test coverage, and README updates. Phase 5
(extending offload to other analyses) depends on these numbers.

## Proposed changes

### 1. CLI measurement surface

- `samples/NivaraIncident/NivaraIncident.Cli/Program.cs`:
  - Add `bench-gpu` mode and `BenchGpu(...)` method.
  - `bench-gpu` acquires a `SilkRuntime` inside try/catch; on failure prints
    "no OpenCL GPU available" and returns 0.
  - Measures CPU-forced vs GPU-forced grouped aggregation medians (warmup + N
    iterations, GC between runs), prints ratio, transfer bytes, device name, rows,
    and the gate's estimate vs measured.
  - Extend `--benchmark` so that when `--gpu` is set, the grouped-aggregation
    section prints CPU-forced, GPU-forced, and gated medians plus the decision.
  - Accept `--gpu` globally; non-grouped analyses ignore it with a one-line
    "gpu: no applicable kernel for this analysis yet" note.

### 2. Tests (availability-guarded, no timing assertions)

- `tests/Nivara.Tests/Incident/GpuIncidentTests.cs`: extend the parity check to
  1k / 10k / 100k-record datasets (exact counts, tolerance 1e-4 on averages).
- `tests/Nivara.Tests/Incident/GpuProfitabilityTests.cs`: add monotonicity checks
  (EstimateCpuMs scales with rows, EstimateGpuMs scales with bytes, Decide flips
  to profitable as CPU estimate grows).
- `tests/Nivara.Tests/Incident/GpuOffloadDecisionTests.cs` (new): forced run records
  `Offloaded=true` with non-null CpuMs/GpuMs/TransferBytes; gated run records a
  reason containing "profitable" or "not profitable".

### 3. Documentation

- `samples/NivaraIncident/README.md`: add `bench-gpu`, `--gpu`, `--gpu-force` to
  the options table and quick start; add a "GPU offload (Phase 4 pilot)" subsection
  with commands (bash fences only), a placeholder results table, the profitability
  rule, and a note that only Grouped Aggregation is offloaded.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test ... --filter "FullyQualifiedName~Gpu"` (availability-guarded)
- Human-run: `bench-gpu` and `--benchmark --gpu` at 10k/100k/1M records on AC power.

## Planned commits

1. docs: plan Phase 4 in TODO.md
2. feat(incident): add bench-gpu mode and CPU-vs-GPU benchmark split
3. test(incident): availability-guarded parity and decision coverage
4. docs(incident): document --gpu, --gpu-force, bench-gpu and measurement protocol

## Blast radius

- CLI argument parsing and `Analyze`/benchmark output (sample only, no library API).
- Test files under `tests/Nivara.Tests/Incident`.
- README documentation.
- No changes to `src/` kernels or storage.

## GitHub issues log

- (none yet)
