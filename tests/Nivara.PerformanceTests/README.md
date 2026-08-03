# Nivara.PerformanceTests

Console benchmark harness for the **storage consolidation** (single
`ColumnStorage<T>`) that ran as Task 7 of the storage plan (plan archived in
git history). It is a plain stopwatch harness — no BDN dependency — so it runs
portably anywhere `dotnet` is available.

## Scenarios

| Scenario | What it measures |
|---|---|
| `ColumnAdd 1M x float` | `NivaraColumn<float>.Add(NivaraColumn<float>)` — the columnar binary-op path |
| `ColumnSigmoid 1M x float` | `NivaraColumn<float>.Sigmoid()` extension |
| `Linear forward [32x256] -> [32x256]` | `Linear<float>` inference forward (no `Grad()` scope) |
| `Linear forward+backward [32x256]` | `Linear<float>` forward + `Backward` inside `GradientUtils.Grad()` |
| `TransformerBlock forward [32x64, 4 heads]` | `TransformerBlock<float>` inference forward |

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
  future A/B comparisons — recorded 2026-08-03 against the then-current HEAD.
- **Never rebuild past commits to re-derive point A.** The previous A/B table
  (built in a git worktree at `549c6cc`) is superseded; its findings still
  hold (the ColumnAdd branch-removal win and the AutoDiff allocation
  reductions are the reason this harness exists), but the numbers are no
  longer the comparison point.
- When measuring a change: record new results on the same machine, compare
  against the **Results** table, and replace it with the new numbers. That
  keeps point A rolling forward and avoids hunting through git history.

## Results

**Baseline (point A)** recorded **2026-08-03** on the then-current HEAD
(post-storage-consolidation single `ColumnStorage<T>`). Supersedes the earlier
pre-consolidation A/B table (see "Baseline policy" above).

Machine: 16 logical processors, x64, .NET 10.0.x (Release). Medians of 6 runs.

| Scenario | ops/s | B/op | gen0/op |
|---|---|---|---|
| ColumnAdd 1M x float | 872 | 4,000,537 | 0.33 |
| ColumnSigmoid 1M x float | 182 | 8,000,840 | 0.66 |
| Linear forward [32x256] | 441 | 69,247 | 0.00 |
| Linear forward+backward [32x256] | 94 | 1,787,949 | 0.50 |
| TransformerBlock forward [32x64, 4 heads] | 148 | 191,486 | 0.02 |

### Notes

- **ColumnSigmoid gen0/op is high relative to its allocation profile.** The
  ~8 MB/op is dominated by a single ~4 MB result array (>85 KB, so LOH, not
  gen0); gen0/op counts how often the *small-object* gen0 budget is hit in the
  measured window. B/op and throughput are stable across runs; treat gen0/op as
  a GC-scheduling-sensitive signal, not an allocation-volume metric. Tracked as
  issue #109 (conclusion: no regression, metric is not allocation-proportional).
- **ColumnAdd dominates the table** because the single-`ColumnStorage<T>`
  layout removed the pre-consolidation pooled-copy branch from
  `applyElementwiseBinary` — that win is the reason this harness exists.
