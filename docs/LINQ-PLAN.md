# LINQ / Data Pipeline Operators for AutoDiff Integration

## Purpose

This document breaks a reviewed data-pipeline feature area into concrete, assignable tasks for coding agents. The goal is not API completeness — it is a focused set of columnar query operators that solve real friction points in the Nivara AutoDiff training pipeline.

## Motivation

The original pain point identified was `TensorDataset.BuildTensor()`: a manual nested loop that reimplements row extraction the query pipeline could handle. During implementation of Tasks 1-4, a deeper analysis revealed that **the query pipeline is the right tool for one-time setup and preprocessing, but the `ArrayPool`-based direct approach is the right tool for the training hot loop** (per-batch extraction within `GetBatch`). The plan has been updated to reflect this distinction.

## Guiding principles

1. **Add operators because they solve a concrete data-movement problem in AutoDiff**, not because LINQ has them.
2. **Allocation-awareness**: column-returning methods (`Select`, `Zip`, `Transform`) allocate new `T[]` + `bool[]` backing arrays plus storage objects. Scalar-returning methods (`Sum`, `Mean`, `Min`, `Max`) are allocation-free via `TensorPrimitives`. Use column-returning methods for one-time/preprocessing paths; use direct `ArrayPool` paths for the training hot loop.
3. **Hot path vs cold path separation**: `GetBatch` / `BuildTensor` is called thousands of times per training run and must minimize GC pressure. `Split`, `Normalize`, `Distinct`, and test setup code run once or a few times — they can freely use the LINQ-style operators.

## Status

### ✅ Completed — Tasks 1–4

| Task | What | Key files |
|------|------|-----------|
| 1 | `SelectRowsOperation` + `Skip`/`Take`/`Slice` on `QueryFrame` — row-index-based selection and paging | `SelectRowsOperation.cs`, `QueryFrame.cs`, `OperationType.cs` |
| 2 | `NivaraColumn<T>.Select()`, `Zip()`, `Sum()`, `Mean()`, `Min()`, `Max()` — column-level transforms and scalar reductions | `NivaraColumn.cs`, `NivaraTensorExtensions.cs` |
| 3 | `NivaraFrame.Create<T>()` + `ToFrame()` extension — eliminates `(IColumn)` cast boilerplate | `NivaraFrame.cs`, `NivaraFrameExtensions.cs` |
| 4 | `Distinct()` / `Distinct(params string[])` on `QueryFrame` — hash-based row dedup | `DistinctOperation.cs`, `QueryFrame.cs` |
| — | Shared `ColumnFilterHelper.CreateFilteredColumn` — deduplicates row-slicing logic across Filter/Slice/SelectRows | `ColumnFilterHelper.cs` |

1623 tests passing. See `ANCHORED-SUMMARY.md` for full implementation details.

---

### ⏳ Remaining — Tasks 5–8

5. `TensorDataset.BuildTensor()` — allocation analysis complete, see below for revised scope
6. `DataParallelTrainer` gradient accumulation — approach refined per allocation analysis
7. Test cleanup (remove `(IColumn)` casts, use declarative style)
8. `Split()` + `Normalize()` data prep helpers

---

## Performance Notes

### Allocation profile of column-returning methods

| Method | Allocates new column | Uses ArrayPool | Intermediate arrays | TensorPrimitives |
|--------|---------------------|----------------|---------------------|------------------|
| `Select` / `Transform` | Yes | No | `TResult[L]`, `bool[L]` + reflection nullable array | No |
| `Zip` | Yes | Yes (>=1024) | Rented `TResult[L]`, rented `bool[L]` + `.ToArray()` copies | No |
| `Sum` / `Mean` / `Min` / `Max` | No (scalar) | No | None | Yes |

Key takeaways:
- **Scalar reductions are allocation-free** — always prefer `Sum()`, `Mean()`, `Min()`, `Max()` in hot paths.
- **`Select` always allocates** (no `ArrayPool`, no `TensorPrimitives` path) — use for preprocessing, avoid in hot loops.
- **`Zip` uses `ArrayPool` for L >= 1024** but still allocates intermediate columns — acceptable for per-epoch ops, avoid for per-batch chaining.

### Hot path rule

Any method called inside `GetBatch()` or a tighter loop per training iteration **must not allocate intermediate `NivaraColumn<T>` objects via the query pipeline**. Use `ArrayPool<T>.Shared.Rent` + direct indexer access instead.

---

## Task 5: TensorDataset.BuildTensor() — allocation-aware scope

**Decision**: Keep the current `ArrayPool`-based hot path. Do not introduce `SelectRows` inside `GetBatch()`.

The `BuildTensor()` method currently uses `ArrayPool<T>.Shared.Rent` + direct indexer access — this is correct for the per-batch hot path. Introducing `SelectRows` would add ~10× intermediate column allocations per batch (one per column in the batch frame), each allocating `T[L]` + `bool[L]` + `NivaraColumn<T>` + storage. For a 10-column batch × 1000 iterations, that's 10,000 extra column objects hitting the GC.

**Scope**:
- Leave `BuildTensor` / `GetBatch` untouched
- Ensure no allocation regression is introduced by any Task 5-8 changes
- If upstream refactoring requires touching `TensorDataset.cs`, preserve the `ArrayPool` pattern

**Files involved**: `src/Nivara/AutoDiff/Training/TensorDataset.cs` — no changes needed.

---

## Task 6: DataParallelTrainer gradient accumulation — refined scope

**Decision**: Use `Zip` for `SumAndApplyGradients` (per-epoch, not per-batch). Avoid `Select` in `ComputeGradientNorm` — use direct loop or `TensorPrimitives.SumOfSquares` instead.

These methods run once per epoch, so allocation impact is ~batchCount× smaller than `BuildTensor`. `Zip`'s `ArrayPool` path is acceptable. `Select` in `ComputeGradientNorm` would allocate an intermediate squared-values column — replace with a direct sum-of-squares loop.

**Scope**:
- `CloneGradients`: snapshot `tensor.Grad` (already `NivaraColumn<T>`) directly — no copy needed
- `SumAndApplyGradients`: use `Zip((a, b) => a + b)` to sum chunk gradients
- `ComputeGradientNorm`: use a direct `for` loop over `tensor.Grad` elements computing `sum += x * x`, or `TensorPrimitives.SumOfMagnitudes` on `Tensor<T>` if available — avoid allocating `Select(x => x * x)`

**Files involved**: `src/Nivara/AutoDiff/Training/DataParallelTrainer.cs`

---

## Task 7: Test cleanup

Replace `(IColumn)` casts with `Create<T>()` / `ToFrame()`. Replace verbose `for`-loop test setup with `Select`/`Zip` where it improves readability. No assertion changes.

**Files involved**: `TrainingTests.cs`, `DataParallelTests.cs`, `SerializationTests.cs`, `NnTests.cs`, `CrossFrameworkParityTests.cs`, `GradOperationsTests.cs`, `LinqQueryTests.cs`

---

## Task 8: Split + Normalize helpers

Add to `NivaraFrameExtensions`:
- `Split(double trainRatio, int? seed = null) → (NivaraFrame train, NivaraFrame test)` — Fisher-Yates shuffle, uses `SelectRows` / `Skip`/`Take`
- `Normalize(params string[] columnNames) → NivaraFrame` — z-score via `Select`, skips nulls
- `Standardize(params string[] columnNames) → NivaraFrame` — alias

**Files involved**: `src/Nivara/NivaraFrameExtensions.cs`, `tests/Nivara.Tests/NivaraFrameExtensionsTests.cs` (or existing test file)


