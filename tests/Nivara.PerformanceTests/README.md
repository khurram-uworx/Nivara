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
| `Streaming cancel mid-stream 200k rows x 10k chunk` | Phase 4 AC2 probe (#266): `StreamingExecutionStrategy.ExecuteAsync` over a chunk-capable source, cancelled after ~3 chunks. Asserts a clean `OperationCanceledException` with prompt unwind. **Currently failing on issue #280** (consumer-side cancellation throws `QueryExecutionException` wrapping `ChannelClosedException` instead of OCE); goes green when #280 is fixed. B/op once green will also capture the in-flight/channel-buffered chunk frames that are never disposed on cancellation |

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

### Baseline policy (do not re-litigate)

- The **Results** table below is the canonical baseline (**point A**) for
  future A/B comparisons — last recorded 2026-08-14 on the v1.3.0 release-prep
  HEAD (`release/130`) with the fixed `--runs` harness.
- **Never rebuild past commits to re-derive point A.** The previous A/B tables
  (built in git worktrees at `549c6cc` and recorded 2026-08-04/05) are
  superseded; their findings still hold (the ColumnAdd branch-removal win, the
  AutoDiff allocation reductions, and the batched-attention fusion are the
  reason this harness exists), but the numbers are no longer the comparison
  point and the 2026-08-05 table is from a different machine.
- When measuring a change: record new results on the same machine, compare
  against the **Results** table, and replace it with the new numbers. That
  keeps point A rolling forward and avoids hunting through git history.

## Results

**Baseline (point A)** recorded **2026-08-14** on the v1.3.0 release-prep HEAD
(`release/130`). **Medians of 3 independent child-process runs** (`--runs 3` on
the fixed harness). This replaces the 2026-08-06 point A, which was recorded on
a different machine (16 logical processors) and is **not comparable** to this
machine's numbers (e.g. ColumnAdd ~620 there vs ~1,049 here) — compare only
within this table from here on. The harness also gained two scenarios since the
2026-08-06 baseline (`Fused chain`, `Frame Slice`), both documented in the
scenario table above.

Machine: 8 logical processors, x64, .NET 10.0.11 (Release). Medians of 3 child processes.

| Scenario | ops/s | B/op | gen0/op |
|---|---|---|---|
| ColumnAdd 1M x float | 1,049 | 4,000,192 | 0.24 |
| ColumnSigmoid 1M x float | 663 | 0 | 0.00 |
| Span chain 1M x 3 ops (raw) | 603 | 0 | 0.00 |
| Column chain 1M x 3 ops (wrapper) | 269 | 12,000,416 | 0.34 |
| Fused chain 1M x (Salary*1.1)+1000-Tax | 219 | 16,003,330 | 0.34 |
| Linear forward [32x256] -> [32x256] | 819 | 68,536 | 0.02 |
| Linear forward+backward [32x256] | 126 | 668,240 | 0.20 |
| TransformerBlock forward [32x64, 4 heads] | 204 | 185,889 | 0.03 |
| Attn per-seq forward [B16 L128 D64 H4] | 54 | 2,125,991 | 0.50 |
| Attn batched forward [B16 L128 D64 H4] | 143 | 528,259 | 0.00 |
| Attn per-seq fwd+bwd [B16 L128 D64 H4] | 14 | 7,939,247 | 1.50 |
| Attn batched fwd+bwd [B16 L128 D64 H4] | 50 | 7,876,385 | 0.50 |
| RowScore per-row copy+dot [10k x 128] | 65 | 2 | 0.00 |
| Frame RowDot [10k x 128] | 177 | 51,722 | 0.00 |
| Frame Slice [10k x 128] | 7,236 | 89,936 | 0.01 |
| RowDot kernel raw [10k x 128] | 686 | 1 | 0.00 |
| RowCosineSimilarity kernel raw [10k x 128] | 317 | 1 | 0.00 |

### Notes

- **All 17 rows were recorded 2026-08-14 on the same machine (8 logical
  processors).** The 2026-08-07 row-scoring rows (previously on an 8-processor
  machine while the rest were on 16) are now part of the same-machine table.
  `Frame RowDot`'s B/op dropped from 452,639 to 51,722 since the 2026-08-06
  point A — result construction no longer materializes a boxed default index
  (the `NivaraSeries<T>` virtual-index change, #231) — and gen0/op fell from
  0.10 to 0.00; the raw `RowDot`/`RowCosineSimilarity` kernels stay ~1 B/op,
  and the frame API beats the per-row status quo ~2.7× on this machine (177 vs
  65 ops/s). New scenario rows are seeded as `NEW` in the gate and become gated
  once a `--json` baseline captures them.

- **This table is the current-machine point A.** The 2026-08-06 point A was
  recorded on a different machine (16 logical processors); its shape still holds
  (ColumnSigmoid and the raw span chain are allocation-free, batched attention
  outruns per-seq, Linear forward+backward allocation is the ArrayPool-backed
  `61ff968` number), but its ops/s are not comparable to this machine. Do not
  mix numbers across machines.
- **ops/s remain load-sensitive on this machine.** The table above was recorded
  with the machine busy-ish (idle CPU ~4–9%): ColumnAdd ~1,049 here. Under
  heavier load the same harness reads ~2× lower, so treat ops/s as
  order-of-magnitude until measured under comparable load; **B/op and gen0/op
  are allocation-driven and stable** and are the reliable regression signals
  for the `--compare` gate.
- **ColumnSigmoid is not comparable to the 2026-08-03 point A.** Task 8
  stripped the `NivaraColumn<float>.Sigmoid()` extension, so the scenario now
  measures the raw kernel with the destination array allocated up front —
  B/op is 0 and gen0/op is 0 by construction. The old ~8 MB/op / 0.66 gen0
  signal (issue #109) is obsolete; the conclusion from that issue (gen0/op is
  GC-scheduling-sensitive, not allocation-proportional) still holds.
- **The two `1M x 3 ops` rows are the ADR-002 P3 wrapper-cost isolation** (added
  2026-08-06). Both run the identical `Add`/`Multiply`/`Subtract` chain over 1M
  floats; the raw row reuses pre-allocated destinations (0 B/op by construction)
  while the wrapper row goes through `NivaraColumn`, allocating a fresh result
  column (~4 MB) per op. The wrapper path therefore carries the ~2 objects/op +
  result-column allocation the ADR-002 decision called out as the remaining
  per-op cost. Interpretation: the wrapper's incremental cost over raw spans is
  the immutable result-column allocation itself (each op must produce a new
  column), not an overhead that Option B's raw `Tensor<T>` backing would remove
  — closing F7 with evidence on hand (no ADR-002 amendment filed).
- **The `Fused chain` and `Frame Slice` rows were added after the 2026-08-06
  point A** (fused-evaluator and #173-slice regression gates). They are gated
  from the next baseline capture forward.
