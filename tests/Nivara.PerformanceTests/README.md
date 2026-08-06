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
| `Linear forward [32x256] -> [32x256]` | `Linear<float>` inference forward (no `Grad()` scope) |
| `Linear forward+backward [32x256]` | `Linear<float>` forward + `Backward` inside `GradientUtils.Grad()` |
| `TransformerBlock forward [32x64, 4 heads]` | `TransformerBlock<float>` inference forward |
| `Attn per-seq forward [B16 L128 D64 H4]` | `ReverseGradOperations.MultiHeadAttention` looped over 16 sequences — per-head `Slice`/`Transpose` graph nodes, causal mask per sequence |
| `Attn batched forward [B16 L128 D64 H4]` | `ReverseGradOperations.BatchedMultiHeadAttention` — heads packed once, fused QK^T/softmax/PV per-head `TensorPrimitives` row kernels (issue #86) |
| `Attn per-seq fwd+bwd [B16 L128 D64 H4]` | Per-seq `MultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |
| `Attn batched fwd+bwd [B16 L128 D64 H4]` | `BatchedMultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |

Each scenario reports **ops/s**, **ns/op**, **bytes/op** (`GC.GetAllocatedBytesForCurrentThread`
delta), and **gen0/op** (`GC.CollectionCount(0)` delta).

## Running

```pwsh
dotnet run --project tests/Nivara.PerformanceTests -c Release
# or, without a restore:
tests/Nivara.PerformanceTests/bin/Release/net10.0/Nivara.PerformanceTests.exe
```

## Methodology

- **No forced GC** in measurements; steady-state warmup (5 iterations) before
  timing so JIT/type-init effects settle before the baseline is taken.
- **Allocation accounting** starts after warmup, so setup allocations (module
  and column construction) are excluded.
- Compare **on the same machine/config**; run each build 3 times and take the
  median — run-to-run variance is ~±10% for these scenarios.

### Baseline policy (do not re-litigate)

- The **Results** table below is the canonical baseline (**point A**) for
  future A/B comparisons — last recorded 2026-08-04 against the then-current
  HEAD.
- **Never rebuild past commits to re-derive point A.** The previous A/B table
  (built in a git worktree at `549c6cc`) is superseded; its findings still
  hold (the ColumnAdd branch-removal win and the AutoDiff allocation
  reductions are the reason this harness exists), but the numbers are no
  longer the comparison point.
- When measuring a change: record new results on the same machine, compare
  against the **Results** table, and replace it with the new numbers. That
  keeps point A rolling forward and avoids hunting through git history.

## Results

**Baseline (point A)** recorded **2026-08-05** on the current HEAD (post-v1.2.0
release prep). Supersedes the 2026-08-04 point A that was recorded after the
AutoDiff refactor; the numbers below are the canonical reference going forward.
The four batched-attention scenarios (fused `MultiHeadAttention` /
`BatchedMultiHeadAttention` kernels, issue #86) were already in the harness but
are now documented here for the first time.

Machine: 16 logical processors, x64, .NET 10.0.9 (Release). Medians of 6 runs.

| Scenario | ops/s | B/op | gen0/op |
|---|---|---|---|
| ColumnAdd 1M x float | 1,672 | 4,000,537 | 0.33 |
| ColumnSigmoid 1M x float | 599 | 0 | 0.00 |
| Span chain 1M x 3 ops (raw) | 409 | 0 | 0.00 |
| Column chain 1M x 3 ops (wrapper) | 193 | 12,001,888 | 0.48 |
| Linear forward [32x256] | 369 | 69,920 | 0.00 |
| Linear forward+backward [32x256] | 171 | 1,492,187 | 0.35 |
| TransformerBlock forward [32x64, 4 heads] | 135 | 196,159 | 0.07 |
| Attn per-seq forward [B16 L128 D64 H4] | 90 | 2,137,638 | 0.17 |
| Attn batched forward [B16 L128 D64 H4] | 405 | 528,686 | 0.00 |
| Attn per-seq fwd+bwd [B16 L128 D64 H4] | 31 | 7,963,817 | 1.17 |
| Attn batched fwd+bwd [B16 L128 D64 H4] | 137 | 7,878,336 | 0.33 |

### Notes

- **Linear forward is the one outlier vs the 2026-08-04 point A** (369 vs 427
  ops/s, −14%): it also showed the widest run-to-run spread of any scenario
  (245–442 ops/s across the six runs), so this is machine-load noise rather than
  a code regression. Every other core scenario is within ±7% of point A
  (ColumnAdd 1,672 vs 1,755, ColumnSigmoid 599 vs 642, Linear forward+backward
  171 vs 166, TransformerBlock 135 vs 137).
- **ColumnSigmoid is not comparable to the 2026-08-03 point A.** Task 8
  stripped the `NivaraColumn<float>.Sigmoid()` extension, so the scenario now
  measures the raw kernel with the destination array allocated up front —
  B/op is 0 and gen0/op is 0 by construction. The old ~8 MB/op / 0.66 gen0
  signal (issue #109) is obsolete; the conclusion from that issue (gen0/op is
  GC-scheduling-sensitive, not allocation-proportional) still holds.
- **ColumnAdd still dominates the table.** The single-`ColumnStorage<T>`
  layout removed the pre-consolidation pooled-copy branch from
  `applyElementwiseBinary` — that win is the reason this harness exists.
  Throughput holds at ~1,672 ops/s (vs 1,755 at the 2026-08-04 point A) with the
  same 4 MB/op allocation profile, within the documented ±10% variance.
- **Batched attention is the fast path for transformers.** `BatchedMultiHeadAttention`
  forward runs ~4.5× per-op faster than looping `MultiHeadAttention` per
  sequence (405 vs 90 ops/s) with ~4× the allocation (528,686 vs 2,137,638
  B/op): heads are packed once per forward and QK^T/softmax/PV run as a single
  per-head pass with no per-head `Slice`/`Transpose` graph nodes. The fused
  forward+backward row (137 ops/s, 7,878,336 B/op) is the number that matters
  for `TransformerBlock` training pipelines.
- **Linear forward+backward** allocates ~295 KB/op less than the 2026-08-04
  point A (1,492,187 vs 1,787,036 B/op) at parity throughput (171 vs 166 ops/s):
  the drop is the rank-2 MatMul backward transpose buffers now rented from
  `ArrayPool<T>.Shared` (commit `61ff968`, merged 2026-08-05 via PR #121), on
  top of the refactor's span-kernel and zero-copy column-wrap work. gen0/op is
  down accordingly (0.35 vs 0.45).
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
- **2026-08-06 caveat:** these new rows (and any ops/s column this session) were
  recorded under elevated machine load — columnar throughput ran ~2× below the
  canonical point A (e.g. ColumnAdd 815 vs 1,672 ops/s) even though no
  production code changed since P1c. B/op and gen0/op are allocation-driven and
  stable across runs; treat the new rows' ops/s as order-of-magnitude only until
  re-measured on an idle machine.
