# Plan: Phase A — kernel parity tests for grouped-hash-reduction Silk path (#1025)

## Problem

`--silk` for NivaraIncident is streamlined, but correctness evidence lives mostly in
end-to-end output. The kernel-level contract for `SilkAnalytics.GroupedHashReduce`
vs `Analysis.AggregateOnCpu` should be pinned by tests with explicit max-diff and
offload verdict assertions.

## Proposed changes

### 1. Tests (`tests/Nivara.Tests/Incident/GpuIncidentTests.cs`)
- Keep/extend GPU-category parity tests.
- Assert per-service totals are exact, and `ErrorRate`/`AvgDurationMs` max diffs
  are within the defined tolerance.
- Assert `LastGpuOffloadDecision.Offloaded == true` on forced Silk runs.
- Keep OpenCL-unavailable devices on `Assert.Ignore` (no CPU fallback in test path).

### 2. Additional focused test class (optional)
- Add `SilkGroupedReductionParityTests.cs` for compact max-diff diagnostics to
  avoid changing the larger benchmark-style test names.

### 3. No CLI changes
- Sample remains `--silk` / `--benchmark` only.
- Timing evidence continues to come from `analyze --benchmark` / `--benchmark --silk`.

### 4. Documentation
- Ensure README notes that parity tests are GPU-category and skipped on OpenCL-less machines.

## Blast radius

- Test files under `tests/Nivara.Tests/Incident`.
- No core library / sample CLI changes.
- Sample behavior unchanged.

## Verification

- `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj -c Release`
- `dotnet test ... -c Release --filter "FullyQualifiedName~Nivara.Tests.Incident"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## GitHub issues log

- [ ] #1025 — Phase A kernel parity tests (this branch)
- [ ] #1026 — Phase B rolling-mean kernel
- [ ] #1027 — Phase C additional kernels
- [ ] #1028 — Phase D wiring>