# Phase 4 Completion — Remaining Tasks

Branch: `khurram/phase4` (tasks 2 + 7 already committed)

## Tasks (in order)

### 1. Fix instance-telemetry generator (DatasetGenerator.cs)
Bug: all instance rows get `baseTime.Ticks`, filter eliminates everything.
Fix: per-minute snapshots across the 30-minute timeline, per-row incident check, QueueDepth rises during incident.

### 2. Extend CLI --benchmark (Program.cs)
- All 4 scenarios A-D (loop or `--scenario all`)
- 5 iterations, report median
- Capture RowsRead, RowsReturned, MaterializedColumns, PeakMemoryUsage from diagnostics
- Own GC.GetTotalMemory sampler for peak memory

### 3. Streaming-vs-eager sweep (new --bench-stream mode)
- Chunk sizes 10K, 50K, 100K, 500K over 1M dataset
- Count frames yielded, peak memory, total elapsed
- Compare against filter-only prefix (no window ops)

### 4. Kernel vectorization report (fold into --benchmark or --bench-kernels)
- DiagnosticsTracker.IsEnabled = true, capture VectorizationRate
- Report unexpected scalar fallbacks

### 5. AutoDiff SIMD microbenchmarks
- Pow and RMSNorm backward at 1M elements
- Hand-rolled scalar baseline for speedup ratio

### 6. Verify gap inventory, file remaining issues
- Re-verify each gap row against current main
- File issues for still-open items with measured evidence

### 7. Update docs
- README: real numbers, fix stale notes, update gap table
- PHASE4.md: tick checkboxes
- Incident-PLAN.md: mark Phase 4 COMPLETE

## GitHub issues log

- [ ] #305 — GroupBy(keys, aggregations) silently drops aggregations
- [ ] #306 — Post-aggregation ranking not expressible in QueryFrame DSL
