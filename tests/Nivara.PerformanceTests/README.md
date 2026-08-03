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

## Results

**AFTER** = post-consolidation HEAD (single `ColumnStorage<T>`).
**BEFORE** = pre-consolidation two-storage layout (`TensorStorage<T>` /
`MemoryStorage<T>` split), same harness built in a git worktree at commit
`549c6cc` with identical `Program.cs`.

Machine: 16 logical processors, x64, .NET 10.0.9 (Release). Medians of 3 runs.

| Scenario | BEFORE ops/s | AFTER ops/s | Δ | BEFORE B/op | AFTER B/op | Δ B/op |
|---|---|---|---|---|---|---|
| ColumnAdd 1M x float | 67 | 669 | **+10.0x** | 4,000,771 | 4,000,586 | ~0% |
| ColumnSigmoid 1M x float | 181 | 164 | −9% (noise) | 8,000,579 | 8,000,795 | ~0% |
| Linear forward [32x256] | 359 | 388 | +8% (noise) | 103,763 | 69,169 | −33% |
| Linear forward+backward [32x256] | 77 | 89 | +16% | 2,904,207 | 1,787,573 | −38% |
| TransformerBlock forward [32x64, 4 heads] | 118 | 118 | ~0% | 330,043 | 191,545 | **−42%** |

### Findings

- **Columnar-op branch removal (ColumnAdd +10x).** Pre-consolidation
  `applyElementwiseBinary` had a "tensor-backed or mixed storage" branch that
  materialized *both* operands into `ArrayPool` buffers element-by-element on
  every op before running the kernel. `Create` routed `float` columns to
  `TensorStorage<T>`, so `Add` always hit that path. The single
  `ColumnStorage<T>` layout removes the branch entirely — both operands are
  already plain arrays and the direct-span kernel runs with no per-op
  materialization. This is exactly the branch-removal win the plan predicted.
- **Entry zero-copy.** `FromColumn`/tensor-from-column wrappers were already
  wrapper-only pre-consolidation; the flattened-copy elimination lands in
  `FromArray`/`FromMatrix` (`CreateFromOwnedArray`) and in `AsTensor()` no
  longer producing a per-column `Tensor<T>` + flattened cache. That is reflected
  in the AutoDiff allocation reduction below rather than a dedicated
  micro-benchmark.
- **AutoDiff indirect gains (allocations −33%/−38%/−42% on Linear forward /
  forward+backward / TransformerBlock).** With no per-column `Tensor<T>` wrapper
  + flattened cache, and the pooled-copy branches gone from the ops AutoDiff
  calls, the graph ops allocate markedly less. Throughput for these
  SIMD-dominated paths is within run-to-run noise; the structural win is
  allocation pressure.
- **ColumnSigmoid flat (within noise).** The `Sigmoid` extension runs the same
  `TensorsHelper.Sigmoid` span kernel pre/post; it never hit the removed
  pooled-copy branch, so throughput is unchanged. gen0/op rose (0.35 → 0.63)
  with identical B/op, consistent with unchanged allocation volume on a
  different GC segment layout — tracked as issue #109.
