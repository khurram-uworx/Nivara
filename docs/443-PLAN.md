# Plan for issue #443 — GPU offload pilot for NivaraIncident analytics

Status: Phases 0–3 implemented. Phases 4–5 (measurement/reporting and further kernels) are pending.

## Summary

Phase 2 of the GPU long-term plan (tracking issue #134): pilot GPU offload for the
NivaraIncident columnar analytics, delivered through the promoted GPU backend. The
policy is "offload where it makes sense" — a profitability heuristic decides per
kernel whether the GPU path beats the measured CPU path.

## Current state vs. the issue text

Issue #443 was filed against an older roadmap state. It assumes no GPU code exists
and names `Nivara.Extensions/Gpu` with an ILGPU 1.5.3 `GpuRuntime`. The repository
has since moved on:

- Production GPU backend is now **Silk (OpenCL)** under `src/Nivara.Extensions/Silk/`
  (`SilkRuntime`, `SilkKernels`, runners), already promoted and tested.
- ILGPU still lives in `samples/Nivara.Samples/Gpu/` and is used for model inference
  (BERT / ModernBERT / Laya), not analytics.
- There are **no analytics GPU kernels** in either backend today (no hash-reduction,
  rolling window, multi-column filter, or radix-sort / hash-join kernels).

Decision: build the analytics kernels on the already-promoted Silk/OpenCL backend rather
than re-promoting ILGPU for this issue. Revisit ILGPU only if a kernel needs
`ILGPU.Algorithms` (for example `XMath`).

## Candidate kernels (from the issue)

- Grouped hash-reduction (many groups x many rows)
- Large-span rolling windows
- Multi-column filter vectors
- GPU radix sort / hash-join keys

## Pilot scope (smallest honest slice)

- Pilot analysis: `AnalyzeGroupedAggregation` (manual dictionary loop over filtered
  rows). It has the cleanest CPU reference and avoids the harder sorting / ranking
  surface of Regional Partitioning.
- First GPU kernel: a grouped hash-reduction (key = service, accumulators = count,
  errors, sum of duration), with a CPU fallback.
- Defer rolling windows, multi-column filter vectors, and radix sort / hash-join to
  later phases, gated on measured profitability.

## Measurement rules

- AC power only.
- Warm up both routes, interleave A/B rounds, alternate which side is measured first.
- Assert on the median ratio with a bound well above the observed noise floor.
- No best-of-N taken separately per route.

## Phases

### Phase 0 — CLI surface and plumbing (no GPU yet)

- Add `--gpu` flag parsing to `samples/NivaraIncident/NivaraIncident.Cli/Program.cs`.
- Thread a `useGpu` parameter through `Analyze(...)` and the benchmark loop.
- Add a `GpuOffloadDecision` record (offloaded?, reason, cpuMs, gpuMs, transferBytes,
  kernel) and print it in the CLI summary, mirroring `ExecutionDiagnostics`.

### Phase 1 — GPU grouped hash-reduction kernel (Silk)

- New OpenCL C kernel in `src/Nivara.Extensions/Silk/` (for example a new
  `SilkAnalyticsKernels`): `GroupedHashReduce`, taking input arrays (service ids as
  int, status codes, durations) and producing per-group accumulators.
- Host-side driver: upload columns, dispatch, download the accumulator table, map group
  ids back to service strings.
- Availability-guarded construction: if no OpenCL GPU is present, `--gpu` reports "no
  GPU; CPU path" and exits 0. It must never crash the CLI.
- Parity test versus the manual CPU loop: exact integer counts; tolerance on floating
  sums (ideally exact where summation order is controlled).

### Phase 2 — Profitability gate

- Implement `GpuProfitabilityHeuristic` with a cost model: estimated GPU time =
  kernel launch + host-to-device + device-to-host, compared against measured CPU time
  for the same kernel phase.
- Policy: offload only when `estimatedGpuMs < cpuMs * margin` (margin around 0.9 to
  avoid marginal wins). Surface the decision and the numbers in diagnostics.
- Add a forced mode (for measurement runs) so the GPU path is exercised even when the
  heuristic says no, enabling honest curves. Default `--gpu` stays profitability-gated.

### Phase 3 — Integration into the pilot analysis

- `Analysis.AnalyzeGroupedAggregation(..., useGpu)` (or a sibling `...Gpu` method)
  that runs the filter on CPU (I/O stays on CPU, per the issue) and then either the
  CPU manual reduction or the GPU hash-reduction.
- Keep the CPU path as the byte-for-byte reference. The GPU path must produce the same
  aggregation values within tolerance.

### Phase 4 — Measurement and documentation

- CLI timing: report CPU vs GPU median for the pilot analysis, transfer bytes, and the
  decision.
- Add a `bench-gpu` mode (or extend `--benchmark`) with GPU-vs-CPU timing curves,
  availability-guarded.
- Update `samples/NivaraIncident/README.md` with `--gpu` usage and a results table
  (AC power, machine line, date).
- Availability-guarded `Nivara.Tests` coverage where a GPU is present.

### Phase 5 — Extend only where profitable (later)

- Rolling windows on GPU (only if measured profitable; shared-DRAM iGPU likely loses on
  bandwidth-bound windows).
- Multi-column filter vectors.
- Radix sort / hash-join keys (hardest; needs careful memory management).

## Implementation status (2026-10-19)

- Phase 0 — done: `--gpu`/`--gpu-force` plumbing and the `GpuOffloadDecision` record.
- Phase 1 — done: `SilkAnalytics.GroupedHashReduce` on the `grouped_hash_reduce_partial`
  OpenCL kernel, availability-guarded.
- Phase 2 — done: `GpuProfitability` calibrates the CPU cost from a bounded sample and
  compares it against a launch + transfer + compute model with a `0.9` margin.
  `--gpu-force` bypasses the gate for measurement runs. The `EstimatedGpuMsPerRow`
  constant is a first-pass placeholder, not a measured throughput.
- Phase 3 — done: `Analysis.AnalyzeGroupedAggregation(..., useGpu, gpuForce)` keeps I/O on
  the CPU and switches only the reduction; the decision is published through
  `Analysis.LastGpuOffloadDecision` and printed by the CLI.
- Phase 4/5 — pending: no GPU-vs-CPU timing curve or results table yet, and the
  cost model has not been calibrated against a measured GPU run.

## Files likely touched

- `samples/NivaraIncident/NivaraIncident.Cli/Program.cs` — `--gpu` parsing, routing,
  diagnostics printing.
- `samples/Nivara.Samples/Incident/Analysis.cs` — `useGpu` parameter on grouped
  aggregation.
- `src/Nivara.Extensions/Silk/` — new analytics kernel and host driver (new file),
  reusing `SilkRuntime` as-is.
- `tests/Nivara.Tests/` — availability-guarded parity test for grouped hash-reduction.
- `samples/NivaraIncident/README.md` — usage and results table.
- Possibly `docs/` — a short note on the GPU analytics offload policy.

## Key risks and decisions needed

1. Backend choice: Silk (OpenCL, already promoted) vs ILGPU (as the issue literally
   says). Recommendation: Silk, to avoid a second promoted backend.
2. Pilot analysis: Grouped Aggregation (recommended) vs Regional Partitioning.
   Regional Partitioning needs GPU ranking / sort, which is much larger.
3. Profitability margin and thresholds: decide the margin (around 0.9) and the minimum
   row count before GPU is even considered.
4. Scope expectation: this is a multi-session engineering effort (new OpenCL kernels,
   host drivers, tests, measurement). Confirm whether to start with Phase 0 + Phase 1
   only, or plan the whole thing and wait for a go-ahead to implement.

## What this plan does not do

No code changes, no commits beyond this document, and no test edits are part of this
plan itself. Implementation requires an explicit go-ahead and a switch to an
implementation agent.