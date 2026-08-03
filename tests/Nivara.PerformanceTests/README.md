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

**Baseline (point A)** recorded **2026-08-04** on the then-current HEAD (after
the AutoDiff refactor: `GradKernels` span kernels, inference-default
`GradientUtils.Grad()`, optimizer zero-copy column wraps, and the Task 8
`NivaraTensorExtensions` strip). Supersedes the 2026-08-03 point A that was
recorded after storage consolidation; the numbers below are the canonical
reference going forward.

Machine: 16 logical processors, x64, .NET 10.0.9 (Release). Medians of 6 runs.

| Scenario | ops/s | B/op | gen0/op |
|---|---|---|---|
| ColumnAdd 1M x float | 1,755 | 4,000,537 | 0.33 |
| ColumnSigmoid 1M x float | 642 | 0 | 0.00 |
| Linear forward [32x256] | 427 | 69,944 | 0.00 |
| Linear forward+backward [32x256] | 166 | 1,787,036 | 0.45 |
| TransformerBlock forward [32x64, 4 heads] | 137 | 196,118 | 0.03 |

### Notes

- **ColumnSigmoid is not comparable to the 2026-08-03 point A.** Task 8
  stripped the `NivaraColumn<float>.Sigmoid()` extension, so the scenario now
  measures the raw kernel with the destination array allocated up front —
  B/op is 0 and gen0/op is 0 by construction. The old ~8 MB/op / 0.66 gen0
  signal (issue #109) is obsolete; the conclusion from that issue (gen0/op is
  GC-scheduling-sensitive, not allocation-proportional) still holds.
- **ColumnAdd still dominates the table.** The single-`ColumnStorage<T>`
  layout removed the pre-consolidation pooled-copy branch from
  `applyElementwiseBinary` — that win is the reason this harness exists.
  Throughput reads ~1,755 ops/s now (vs 872 at the 2026-08-03 point A) with the
  same 4 MB/op allocation profile.
- **Linear forward+backward** allocates ~0.9 KB less per op than at point A
  (1,787,036 vs 1,787,949 B/op) and runs at ~1.8× the ops/s, consistent with
  the refactor's span-kernel and zero-copy column-wrap work. **TransformerBlock**
  is within the documented ±10% run-to-run variance of point A.
