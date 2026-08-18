# Phase 4 — Performance Assessment

**Status:** FOLLOWS 3a
**Scope:** Benchmark harness + core gap measurement using the Phase 3a CLI and dataset
**Depends on:** Phase 3a complete (CLI, generator, analyses, streaming all working)
**Parent plan:** `samples/Incident-PLAN.md`
**Related:** `docs/PHASE3A.md`, `tests/Nivara.PerformanceTests/` (existing harness)

---

## Goal

Produce a benchmark report with **real numbers** (not illustrative) that answers:

1. How fast is the full incident analysis pipeline end-to-end?
2. Does streaming actually stay chunked for window-heavy queries, or does it fall back?
3. What percentage of kernels run vectorized (SIMD/fused) vs scalar?
4. What is the AutoDiff SIMD impact from Phase 2 (Pow, RMSNorm-grad)?

Results go into `samples/NivaraIncident/README.md` Performance section. Any core limitation
found is escalated as a GitHub issue with evidence.

---

## Existing infrastructure

The repo already has the pieces needed:

| What | Where | Purpose |
|------|-------|---------|
| `DiagnosticsTracker` | `src/Nivara/Diagnostics/OperationDiagnostics.cs` | Per-kernel-type tracking (`KernelType.Scalar` / `.Vectorized`), allocation bytes, elapsed time |
| `OperationSummary` | same file | Aggregate: `VectorizationRate`, per-type counts, per-op allocation totals |
| `ExecutionDiagnostics` | `src/Nivara/Diagnostics/ExecutionDiagnostics.cs` | Query-level: `RowsRead`, `RowsReturned`, `MaterializedColumns`, `PeakMemoryUsage`, `GenerateReport()` |
| `KernelSelector` | `src/Nivara/KernelSelector.cs` | `DetermineKernelType(length, isVectorizable)` — Scalar unless vectorizable + SIMD + length ≥ threshold |
| `PerformanceTests` harness | `tests/Nivara.PerformanceTests/` | Custom stopwatch harness, 21 scenarios, multi-process median, regression gate |
| CLI diagnostics | `NivaraIncident.Cli` (from 3a) | `--stream` flag, `LastExecutionDiagnostics` output |

**Key API for kernel tracking:**
```csharp
DiagnosticsTracker.IsEnabled = true;
// ... run operations ...
var summary = DiagnosticsTracker.GetSummary();
// summary.VectorizationRate  (double, 0–100)
// summary.VectorizedOperations / summary.TotalOperations
```

---

## Sub-steps

### 4.1 — End-to-end analysis benchmark

**What:** Measure the full pipeline — generate dataset, load, run all 4 analyses (A–D),
capture timing + memory + row counts.

**How:**
- Add a `--bench` mode to the CLI (or a separate `BenchmarkRunner` class in
  `Nivara.Samples/Incident/`).
- For each scenario (A–D), run the analysis twice: eager and streaming.
- Capture from `QueryFrame.LastExecutionDiagnostics`:
  - `RowsRead`, `RowsReturned`, `MaterializedColumns`
  - `TotalExecutionTime` (elapsed)
  - `PeakMemoryUsage` (GC.GetTotalMemory high-water mark)
- Run 5 iterations per scenario, report median.
- Report format:

```
BENCHMARK: End-to-end Analysis

Scenario A (Database degradation)
  Eager:   12.4M rows → 31,842 rows  |  1.83s  |  412 MB peak  |  7 materialized cols
  Stream:  12.4M rows → 31,842 rows  |  2.01s  |  187 MB peak  |  3 materialized cols

Scenario B (Bad deployment)
  ...
```

**Files:**
- `samples/Nivara.Samples/Incident/BenchmarkRunner.cs` (new) — or extend CLI with `--bench`.
- `samples/NivaraIncident.Cli/Program.cs` (add `bench` mode).

**Validation:** numbers are real (same dataset, same machine, reproducible). Median over 5 runs.

---

### 4.2 — Streaming vs eager: memory curve + gap 5 measurement

**What:** Does `AsStream` stay chunked for window-heavy queries, or does it fall back to
single-frame materialization? This directly measures **gap 5** (window semantics across
chunk boundaries).

**How:**
- Take the most window-heavy analysis (likely scenario A or C — rolling windows + rank +
  shift).
- Run it with `--stream` at multiple chunk sizes: 10K, 50K, 100K, 500K, 1M rows.
- For each chunk size, measure:
  - Number of chunks yielded by `AsStream` (1 = fell back to single frame; >1 = truly chunked)
  - Peak memory
  - Total elapsed
  - Whether any non-streamable operation triggered the fallback
- Also run the same query via `CollectAsync` (eager) for comparison.
- Log the `StreamingExecutionStrategy` behavior — did it detect non-streamable ops?
- Report:

```
BENCHMARK: Streaming vs Eager (Scenario A, window-heavy query)

Chunk size   Chunks   Peak memory   Elapsed    Fallback?
----------   ------   -----------   -------    ---------
10,000       1        890 MB        3.21s      YES (RollingWindow)
50,000       1        890 MB        3.18s      YES (RollingWindow)
100,000      1        890 MB        3.15s      YES (RollingWindow)
500,000      1        890 MB        3.12s      YES (RollingWindow)
Eager        N/A      890 MB        3.10s      N/A
```

Or, if streaming works:

```
Chunk size   Chunks   Peak memory   Elapsed    Fallback?
----------   ------   -----------   -------    ---------
100,000      124      187 MB        4.82s      NO
500,000      25       340 MB        3.91s      NO
```

**Decision point:** If window-heavy queries always fall back (gap 5), file a core issue
with this evidence titled "Cross-chunk window computation for streaming replay" referencing
the measured data. If they sometimes succeed, document the conditions.

**Files:**
- Extend `BenchmarkRunner.cs` or add streaming benchmark methods.
- May need to instrument `StreamingExecutionStrategy` to log fallback reason (check if
  `NonStreamableOperations` list is already exposed).

**Validation:** numbers are real. The fallback/no-fallback determination is binary and
unambiguous.

---

### 4.3 — Kernel-selection visibility: vectorized vs scalar %

**What:** What percentage of kernel invocations during a full analysis run are vectorized
(SIMD/fused) vs scalar?

**How:**
- Enable `DiagnosticsTracker.IsEnabled = true` before each analysis.
- Run all 4 scenarios (A–D) with the full dataset.
- After each run, call `DiagnosticsTracker.GetSummary()` to get:
  - `VectorizationRate` (percent)
  - `VectorizedOperations` / `TotalOperations` breakdown
  - Per-operation-type kernel distribution
  - `TotalAllocatedBytes` / `AverageAllocatedBytes`
- Also pull `ExecutionDiagnostics.GenerateReport()` which includes the kernel selection
  breakdown section.
- Report:

```
BENCHMARK: Kernel Selection Visibility

Scenario A:
  Total kernel ops:     847
  Vectorized:           612 (72.3%)
  Scalar:               235 (27.7%)
  With nulls:            41 ( 4.8%)
  Total allocated:   12.4 MB

  Top vectorized: ElementwiseAddition(312), FusedExpression(189), RollingMean(67)
  Top scalar:     RankKernel(89), QuantileSelect(72), StringCompare(48)
```

**Files:**
- Extend `BenchmarkRunner.cs` with kernel tracking section.
- Clear `DiagnosticsTracker` between scenarios to avoid cross-contamination.

**Validation:** `VectorizationRate` is a real number, not hardcoded. The breakdown should
make sense (rank/quantile are expected to be scalar; arithmetic/window ops should be mostly
vectorized).

---

### 4.4 — AutoDiff SIMD impact (Pow + RMSNorm-grad)

**What:** Before/after measurements for the Phase 2 SIMD improvements (items 2.2 and 2.4).

**How:**
- This is a standalone training microbenchmark, independent of the incident sample.
- Use `tests/Nivara.PerformanceTests/` or create a focused benchmark in
  `tests/Nivara.Tests/AutoDiff/`.
- **Pow benchmark:** create a `ReverseGradTensor<float>` of size N (1M, 10M), run
  `Pow(exponent=2.5)` forward + backward, measure elapsed and allocated bytes.
  Compare with the old `Math.Pow` scalar path (if still reachable) or with a reference
  measurement from before Phase 2.
- **RMSNorm-grad benchmark:** create a gradient tensor, run `RMSNorm` backward, measure.
  The Phase 2.4 change replaced element loops with `TensorPrimitives.Multiply`/`MultiplyAdd`.
- Use the existing perf harness pattern: warmup + iterations + multi-process median.
- Report:

```
BENCHMARK: AutoDiff SIMD Impact (Phase 2)

Pow (1M elements, forward + backward):
  Before (scalar Math.Pow):  4.82 ms  |  12.4 MB allocated
  After (TensorPrimitives):  0.91 ms  |   4.0 MB allocated  |  5.3× faster

RMSNorm grad (1M elements):
  Before (element loop):     2.14 ms  |  8.0 MB allocated
  After (TensorPrimitives):  0.38 ms  |  4.0 MB allocated  |  5.6× faster
```

If "before" numbers aren't available (the scalar paths were deleted in Phase 2), measure
against a hand-rolled scalar baseline in the test itself (a simple `for` loop doing the
same math).

**Files:**
- `tests/Nivara.PerformanceTests/AutoDiffBenchmarks.cs` (new) — or extend `Program.cs`.
- May need a scalar reference implementation for comparison.

**Validation:** numbers are real. Speedup factor is computed, not asserted.

---

### 4.5 — Assemble report + escalate gaps

**What:** Collect all benchmark results, write the README Performance section, file any
core issues.

**How:**
- Compile results from 4.1–4.4 into a single report.
- Update `samples/NivaraIncident/README.md`:
  - Add "Performance" section with real numbers.
  - Include machine specs (CPU, RAM, OS) for reproducibility context.
  - Document dataset size used (row count, file size on disk).
- File GitHub issues for any core limitations found:
  - Gap 5 fallback (from 4.2) → issue with streaming fallback evidence.
  - Memory budget exceeded (from 4.1/4.2) → issue with peak memory measurements.
  - Any unexpected scalar fallbacks (from 4.3) → issue with kernel selection evidence.
- Do NOT file issues for expected behavior (rank kernel being scalar is expected, not a gap).

**Validation:**
- README Performance section contains real numbers with machine context.
- Any filed issues have concrete evidence (not hypotheticals).
- No regressions in existing test suite.

---

## Definition of done (Phase 4)

- [x] `--bench` mode (or benchmark runner) produces reproducible numbers.
- [x] End-to-end timing + memory for all 4 scenarios (eager + streaming).
- [x] Streaming vs eager comparison with fallback determination (gap 5 measured).
- [x] Kernel selection visibility report with real vectorization rate.
- [x] AutoDiff SIMD before/after for Pow and RMSNorm-grad.
- [x] `samples/NivaraIncident/README.md` updated with Performance section (real numbers).
- [x] Core gaps (if any) escalated as GitHub issues with evidence.
- [ ] All existing tests still pass; no regressions. *(awaiting final test run)*

---

## Execution notes

- **Ask before running `dotnet test`** (repo rule).
- The perf harness in `tests/Nivara.PerformanceTests/` uses multi-process median (`--runs N`)
  to avoid JIT tiering skew — use the same pattern for 4.4.
- `DiagnosticsTracker.IsEnabled` is off by default; must be set to `true` before analysis
  runs, and `ClearRecordedOperations()` between scenarios to avoid cross-contamination.
- Machine specs matter — record CPU model, core count, RAM, OS in the report header.
- Keep benchmark code separate from production code (in test project or benchmark runner,
  not in the sample's Analysis.cs).
