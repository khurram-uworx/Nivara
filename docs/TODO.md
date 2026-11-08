# Plan: NivaraIncident Silk (--silk) conversion

## Problem

`samples/NivaraIncident` exposes GPU offload as `--gpu`/`--gpu-force` and a `bench-gpu` subcommand. Every other sample in the repo uses `--silk` for the Silk/OpenCL backend, with the convention "use OpenCL, fail loudly if no device". NivaraIncident's `--gpu` is additionally gated by `GpuProfitability`, which can silently keep work on CPU and swallow OpenCL failures into CPU fallback. That is inconsistent and defeats the fail-loud contract.

## Proposed changes

### 1. CLI surface (`samples/NivaraIncident/Program.cs`)
- Replace `--gpu`/`--gpu-force` with `--silk`.
- Drop `gpu`/`gpuForce` variables; use `silk`.
- Rename `bench-gpu` -> `bench-silk`.
- Update `Analyze(..., silk)` and `BenchSilk(...)`.
- Benchmark branch: CPU aggregation when `!silk`, Silk aggregation when `silk`.
- Non-benchmark branch prints Silk notice when `silk`, calls `Analysis.AnalyzeGroupedAggregation(dsPath, sc, silk)`.
- `BenchSilk` constructs `SilkRuntime` without swallowing exceptions.
- Update `PrintUsage()`.

### 2. Analysis API (`samples/Nivara.Samples/Incident/Analysis.cs`)
- `AnalyzeGroupedAggregation(string, IncidentScenario, bool useSilk = false)`.
- `!useSilk`: CPU aggregation only, decision offloaded=false.
- `useSilk`: construct `SilkRuntime` (no try/catch), run `SilkAnalytics.GroupedHashReduce`, offloaded=true. OpenCL init failures propagate.
- Remove the profitability gate and CPU fallback from the Silk path.
- Keep `LastGpuOffloadDecision` property/type names as-is to minimise churn (optional rename later).

### 3. Profitability code left in place but de-wired
- Rename `GpuProfitability` -> `SilkProfitability` (file + class), logic untouched.
- Do not reference it from the Silk path.
- Document as experimental/non-wired in README.

### 4. Docs
- `samples/NivaraIncident/README.md`: replace `--gpu`/`--gpu-force`/`bench-gpu` with `--silk`/`bench-silk`; add "Silk profitability (experimental, not wired up)" subsection with disclaimer.
- `docs/SILK.md` §13d wiring line update to `--silk`.

### 5. Dead-ish references
- None expected; verify build.

## Blast radius
- Sample-only code; no core library or tests reference these flags.
- Public(ish) sample API signatures change (`AnalyzeGroupedAggregation`); only sample uses them.
- `GpuProfitability` rename requires a new file name but no logic change.

## Verification
- `dotnet build samples/NivaraIncident/NivaraIncident.csproj -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Inspect `git diff` for drift/unintended files.

## GitHub issues log
- None captured.