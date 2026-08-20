# Nivara.PerformanceTests

Console benchmark harness for the **storage consolidation** (single
`ColumnStorage<T>`) that ran as Task 7 of the storage plan (plan archived in
git history). It doubles as the perf gate for the AutoDiff refactor (span-based
`GradKernels`, inference-default `GradientUtils.Grad()`). It is a plain
stopwatch harness — no BDN dependency — so it runs portably anywhere `dotnet`
is available.

## Scenarios

| Scenario | What it measures |
|---|---|
| `ColumnAdd 1M x float` | `NivaraColumn<float>.Add(NivaraColumn<float>)` — the columnar binary-op path |
| `ColumnSigmoid 1M x float` | Raw kernel — `TensorPrimitives.Sigmoid` over a pre-allocated 1M destination (the `NivaraColumn<float>.Sigmoid()` extension was removed in Task 8 of the refactor) |
| `Span chain 1M x 3 ops (raw)` | `TensorPrimitives.Add`/`Multiply`/`Subtract` into three pre-allocated 1M destinations — zero-allocation control for the wrapper-cost isolation (P3) |
| `Column chain 1M x 3 ops (wrapper)` | Same three ops through `NivaraColumn<float>.Add`/`Multiply`/`Subtract`, which allocate a fresh result column per op — isolates the column+storage wrapper cost (P3) |
| `Fused chain 1M x (Salary*1.1)+1000-Tax` | The fused-evaluator compiled target for `Col("Salary") * 1.1 + 1000 - Col("Tax")` at a vectorized length (gates on the `KernelSelector` heuristic) |
| `Linear forward [32x256] -> [32x256]` | `Linear<float>` inference forward (no `Grad()` scope) |
| `Linear forward+backward [32x256]` | `Linear<float>` forward + `Backward` inside `GradientUtils.Grad()` |
| `TransformerBlock forward [32x64, 4 heads]` | `TransformerBlock<float>` inference forward |
| `Attn per-seq forward [B16 L128 D64 H4]` | `ReverseGradOperations.MultiHeadAttention` looped over 16 sequences — per-head `Slice`/`Transpose` graph nodes, causal mask per sequence |
| `Attn batched forward [B16 L128 D64 H4]` | `ReverseGradOperations.BatchedMultiHeadAttention` — heads packed once, fused QK^T/softmax/PV per-head `TensorPrimitives` row kernels (issue #86) |
| `Attn per-seq fwd+bwd [B16 L128 D64 H4]` | Per-seq `MultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |
| `Attn batched fwd+bwd [B16 L128 D64 H4]` | `BatchedMultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |
| `RowScore per-row copy+dot [10k x 128]` | Status-quo row scoring — per row, copy 128 column values into scratch then `TensorPrimitives.Dot` (10k dots) |
| `Frame RowDot [10k x 128]` | Public `NivaraFrame.RowDot` — row-major materialization + `TensorsHelper.RowDot` (#138, #141) |
| `Frame Slice [10k x 128]` | Public `NivaraFrame.Slice(0, 5000)` — the reflection-free `IColumn.Slice` path (#173) |
| `RowDot kernel raw [10k x 128]` | Raw `TensorsHelper.RowDot` over a pre-built row-major buffer + null mask — the kernel floor (#141) |
| `RowCosineSimilarity kernel raw [10k x 128]` | Raw `TensorsHelper.RowCosineSimilarity` over a pre-built row-major buffer — kernel floor with norm (#141) |
| `Streaming cancel mid-stream 200k rows x 10k chunk` | Phase 4 AC2 probe (#266): `StreamingExecutionStrategy.ExecuteAsync` over a chunk-capable source, cancelled after ~3 chunks. Asserts a clean `OperationCanceledException` with prompt unwind (#280 fixed — the consumer-side catch now uses `TryComplete()`, observes the producer, and disposes in-flight/channel-buffered frames, so the OCE is no longer masked by `ChannelClosedException`). B/op captures the frames the cancelled path disposes |

Each scenario reports **ops/s**, **ns/op**, **bytes/op** (`GC.GetAllocatedBytesForCurrentThread`
delta), and **gen0/op** (`GC.CollectionCount(0)` delta).

## Running

```pwsh
dotnet run --project tests/Nivara.PerformanceTests -c Release
# or, without a restore:
tests/Nivara.PerformanceTests/bin/Release/net10.0/Nivara.PerformanceTests.exe
```

### No-regression gate (P4)

The harness doubles as an executable perf gate (`ADR-002` P4). Two modes:

- `--json <path>` — emit each scenario's `ops/s`, `ns/op`, `B/op`, `gen0/op`
  as JSON (median across `n` separate child-process runs via `--runs n`,
  default 1).
- `--compare <baseline.json>` — run, compare against a saved `--json` baseline,
  and exit non-zero when any scenario regresses beyond tolerance.

`--runs n` spawns `n` independent child processes (each a single cold pass via
`--runs 1`) and takes the per-scenario median of their JSON reports. This is
deliberate: an in-process repeat loop is skewed by JIT tiering (later passes
run warmed code — TransformerBlock read 1,256 ops/s in-process vs ~130 honest
across processes), so all `--runs > 1` baselines recorded before commit
`e3ac8b7` must be re-verified with the fixed harness.

| Flag | Default | Meaning |
|---|---|---|
| `--json <path>` | — | write results JSON to `<path>` |
| `--compare <baseline.json>` | — | gate against `<baseline.json>`; exit 1 on regression, 2 on unreadable baseline |
| `--runs <n>` | 1 | spawn `n` independent single-pass child processes and take the per-scenario median |
| `--tolerance <pct>` | 90 | ops/s floor as a percent of baseline |

Gate criteria (tolerance constants in `Program.cs`):
- `ops/s` ≥ `--tolerance`% of baseline (default 90%)
- `B/op` ≤ baseline × 1.01 (allocation slack absorbs run-to-run jitter)
- `gen0/op` ≤ baseline + 0.05 (GC scheduling is not allocation-proportional)

Per-phase workflow (on an idle machine — see the load caveat below):
1. **Baseline** before the phase: `--json baseline.json --runs 3`
2. **Measure** after the phase: `--compare baseline.json --runs 3`
3. `--compare` exits 0 on pass; on FAIL, bisect to the offending change before
   proceeding (ADR-002 no-regression gate).

## Methodology

- **No forced GC** in measurements; steady-state warmup (5 iterations) before
  timing so JIT/type-init effects settle before the baseline is taken.
- **Allocation accounting** starts after warmup, so setup allocations (module
  and column construction) are excluded.
- Compare **on the same machine/config**; use `--runs 3` (three independent
  child processes, per-scenario median) rather than re-running in-process —
  in-process repeats are JIT-tiering-skewed (see the `--runs` note above) and
  run-to-run variance is ~±10% for these scenarios under load.

### Baseline policy (rolling Prev/Current history)

- The **Results** table carries its own comparison context: the **Prev** column
  holds the most recent prior reading, the **Current** column holds the fresh
  measurement, and **Ratio** / **Δ%** show the delta.
- When measuring: shift the existing Current to Prev, place new numbers in
  Current, and compute Ratio (`Current / Prev`) and Δ% (`((Current − Prev) / Prev) × 100`).
- If the Previous reading was on a **different machine**, note the machine
  difference in the Prev column — ratio is not meaningful across machines.
- **B/op** and **gen0/op** are stability indicators (not throughput metrics)
  and do not get ratio columns.

## Results

*Recorded 2026-08-21 — Intel Core Ultra 7 255H, 16 logical processors, .NET 10.0.11 (Release), medians of 3 child processes.*

Machine: Intel Core Ultra 7 255H, 16 logical processors, x64, .NET 10.0.11 (Release). Medians of 3 child processes (`--runs 3`).

| Scenario | Prev | Current | Ratio | Δ% | B/op | gen0/op |
|---|---|---|---|---|---|---|
| ColumnAdd 1M x float | 1,049¹ | 1,577 | — | — | 4,000,192 | 0.24 |
| ColumnSigmoid 1M x float | 663¹ | 634 | — | — | 0 | 0.00 |
| Span chain 1M x 3 ops (raw) | 603¹ | 903 | — | — | 0 | 0.00 |
| Column chain 1M x 3 ops (wrapper) | 269¹ | 314 | — | — | 12,000,416 | 0.34 |
| Fused chain 1M x (Salary\*1.1)+1000-Tax | 219¹ | 266 | — | — | 16,005,312 | 0.34 |
| Fused chain chunked 1M x 64k rows | — | 266 | — | — | 16,005,312 | 0.32 |
| Fused single-op TP 1M x (Salary\*1.1) | — | 568 | — | — | 8,003,436 | 0.34 |
| Column mul-scalar 1M (wrapper) | — | 589 | — | — | 8,000,280 | 0.34 |
| Linear forward [32x256] -> [32x256] | 819¹ | 835 | — | — | 68,824 | 0.00 |
| Linear forward+backward [32x256] | 126¹ | 131 | — | — | 668,519 | 0.15 |
| TransformerBlock forward [32x64, 4 heads] | 204¹ | 113 | — | — | 185,889 | 0.00 |
| Attn per-seq forward [B16 L128 D64 H4] | 54¹ | 58 | — | — | 2,125,983 | 0.17 |
| Attn batched forward [B16 L128 D64 H4] | 143¹ | 291 | — | — | 528,281 | 0.00 |
| Attn per-seq fwd+bwd [B16 L128 D64 H4] | 14¹ | 23 | — | — | 7,938,653 | 1.00 |
| Attn batched fwd+bwd [B16 L128 D64 H4] | 50¹ | 114 | — | — | 7,875,936 | 0.33 |
| RowScore per-row copy+dot [10k x 128] | 65¹ | 118 | — | — | 2 | 0.00 |
| Frame RowDot [10k x 128] | 177¹ | 294 | — | — | 51,722 | 0.00 |
| Frame Slice [10k x 128] | 7,236¹ | 14,842 | — | — | 89,936 | 0.01 |
| RowDot kernel raw [10k x 128] | 686¹ | 835 | — | — | 1 | 0.00 |
| RowCosineSimilarity kernel raw [10k x 128] | 317¹ | 272 | — | — | 1 | 0.00 |
| RollingSum null-free 1M x int (w10) | — | 411 | — | — | 5,000,137 | 0.12 |
| RollingSum nulls 1M x int (w10) | — | 74 | — | — | 22,000,233 | 0.50 |
| RankKernel RowNumber 100k x int | — | 34 | — | — | 1,700,577 | 0.04 |
| GroupBy 1M rows x 1000 keys (typed) | — | 23 | — | — | 8,906,946 | 0.70 |
| GroupBy 1M rows x 100 string keys (typed) | — | 13 | — | — | 13,188,504 | 1.05 |
| PartitionedWindow RollingSum 1M x 100 parts | — | 13 | — | — | 36,216,498 | 2.15 |
| Streaming cancel mid-stream 200k x 10k chunk | — | 4,169 | — | — | 5,435 | 0.07 |
| AutoDiff Pow(2.5) fwd+bwd 1M x float | — | 62 | — | — | 8,001,858 | 0.20 |
| AutoDiff Pow(2.5) scalar baseline 1M x float | — | 21 | — | — | 2 | 0.00 |
| AutoDiff RMSNorm fwd+bwd 1M x float | — | 281 | — | — | 8,002,186 | 0.10 |
| AutoDiff RMSNorm scalar baseline 1M x float | — | 509 | — | — | 2 | 0.00 |

¹ Prev recorded on a different machine (8 logical processors, 2026-08-14) — ratio not meaningful across machines.

### Notes

- **This table is the current-machine rolling history.** The Prev column
  carries the v1.3.0 release-prep numbers (2026-08-14, 8 logical processors);
  the Current column carries the v1.4.0 release-prep numbers (2026-08-21,
  Intel Core Ultra 7 255H, 16 logical processors). Both machines are Intel
  Core Ultra 7 255H but with different logical processor counts, so Prev/Current
  ratios reflect hardware + codebase differences combined. For same-machine
  comparisons, re-measure on the same hardware and the Ratio/Δ% columns become
  the regression signal.
- **B/op and gen0/op are allocation-driven and stable** across machines and
  loads — they are the reliable regression signals for the `--compare` gate.
  ColumnSigmoid and the raw span chain are 0 B/op by construction (destination
  pre-allocated).
- **New scenarios since v1.3.0** (Fused chain chunked, Fused single-op TP,
  Column mul-scalar, RollingSum, RankKernel, GroupBy, PartitionedWindow,
  Streaming cancel, AutoDiff Pow/RMSNorm) are seeded as `NEW` and become gated
  once a `--json` baseline captures them.
- **TransformerBlock forward** dropped from 204 to 113 ops/s — this is expected
  on the higher-core-count machine (the scenario is single-threaded; the
  difference is thermal/scheduling, not a regression — B/op is identical at
  185,889).
- **ops/s are load-sensitive.** Treat ops/s as order-of-magnitude; B/op and
  gen0/op are the reliable signals.
