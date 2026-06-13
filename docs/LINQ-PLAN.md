# LINQ / Data Pipeline Operators for AutoDiff Integration

## Purpose

This document breaks a reviewed data-pipeline feature area into concrete, assignable tasks for coding agents. The goal is not API completeness — it is a focused set of columnar query operators that solve real friction points in the Nivara AutoDiff training pipeline.

## Motivation

The single biggest pain point is `TensorDataset.BuildTensor()` (`src/Nivara/AutoDiff/Training/TensorDataset.cs:49-104`): a manual nested loop over rows and columns that reimplements row extraction the query pipeline could handle. Every other gap — duplicated test helpers, manual gradient accumulation, boilerplate `(IColumn)` casts — stems from the same root: **the query layer lacks the operators the training pipeline needs**.

## Guiding principle

Add operators because they solve a concrete data-movement problem in AutoDiff, not because LINQ has them.

## Execution Order

1. Task 1: `SelectRowsOperation` + `Skip`/`Take` on `QueryFrame` (prerequisite for everything else)
2. Task 2: `NivaraColumn<T>.Select()` + `Zip()` + terminal reductions (column-level transforms)
3. Task 3: `NivaraFrame.Create<T>()` overloads (remove `(IColumn)` cast boilerplate)
4. Task 4: `DistinctOperation` on `QueryFrame` (data cleaning)
5. Task 5: Rewrite `TensorDataset.BuildTensor()` using new operators
6. Task 6: Rewrite `DataParallelTrainer` gradient accumulation using column ops
7. Task 7: Update tests to use new declarative style
8. Task 8: Add `Split()` + `Normalize()` data prep helpers

## Coordination Notes

- Tasks 1 and 2 are the foundation — everything else depends on them.
- Tasks 3 and 4 depend on nothing and can run in parallel with each other after Tasks 1-2.
- Tasks 5-6 consume the output of Tasks 1-2 and should be implemented together.
- Task 7 is pure test cleanup and depends on Tasks 2-4 being stable.
- Task 8 depends on Task 1 (`Take`/`Skip` for `Split`; `Select` for `Normalize`).
- Files that may create merge conflicts: `NivaraFrame.cs`, `NivaraColumn.cs` — keep tasks editing these sequential or coordinate carefully.

---

## Task 1: SelectRowsOperation + Skip/Take on QueryFrame

### Priority

High

### Goal

Add row-index-based selection and paging to the query pipeline, enabling frame-level batch extraction without manual loops.

### Why this exists

`TensorDataset.BuildTensor()` currently uses a nested `for` loop over row indices and columns (lines 65-81 in `TensorDataset.cs`) to extract batch data from columns. With `SelectRows(indices)`, this becomes a query-pipeline operation that reuses existing `createFilteredColumn` logic already in `NivaraFrame` (lines 66-131). `Skip`/`Take` provide the paging primitives needed for train/test splitting.

### Scope

- Add `SelectRowsOperation : IQueryOperation` class that accepts `int[]` row indices
  - Implement `Execute()` by calling the existing `createFilteredColumn` per column (or its typed equivalent)
  - Implement `TransformSchema()` — schema is unchanged
- Add `QueryFrame.SelectRows(IEnumerable<int> indices)` fluent method
- Add `QueryFrame.Skip(int count)` — wraps `SliceOperation` with offset
- Add `QueryFrame.Take(int count)` — wraps `SliceOperation` with length
- Add `QueryFrame.Slice(int skip, int take)` — convenience for combined paging
- Register `OperationType.SelectRows` constant in `OperationType.cs`
- Add `SelectRowsOperation` support to `ExecutionStrategyBase` validator if needed (it's a pure column-slice, no special strategy needed)
- Write unit tests for:
  - `SelectRows` with contiguous, sparse, and reversed indices
  - `SelectRows` with null propagation (nulls in selected rows are preserved)
  - `Skip` / `Take` / `Slice` boundary cases (empty result, all rows, negative values → throw)
  - Chaining: `Skip(n).Take(m)` produces correct slice
  - `SelectRows` in parallel execution strategy (since it's row-indexed, parallel should work)

### Acceptance criteria

- `SelectRows` correctly extracts rows by index from all columns, preserving types and null masks
- `Skip(n)` / `Take(n)` produce same results as equivalent `SliceOperation`
- All new methods return `QueryFrame` for fluent chaining
- Parallel execution round-trips produce identical results to serial
- 8+ passing unit tests covering normal, edge, and null cases

### Files likely involved

- `src/Nivara/Query/QueryFrame.cs` — add `SelectRows()`, `Skip()`, `Take()`, `Slice()` methods
- `src/Nivara/Query/OperationType.cs` — add `SelectRows` constant
- `src/Nivara/Operations/` — new `SelectRowsOperation.cs` file
- `src/Nivara/Execution/ExecutionStrategyBase.cs` — ensure new op type is handled
- `tests/Nivara.Tests/Query/QueryFrameTests.cs` — add tests

---

## Task 2: NivaraColumn\<T\>.Select + Zip + Terminal Reductions

### Priority

High

### Goal

Expose element-wise transform (`Select`), pairwise combine (`Zip`), and scalar reductions (`Sum`, `Mean`, `Min`, `Max`) directly on `NivaraColumn<T>`.

### Why this exists

- `Transform<TResult>(Func<T, TResult>)` already exists at `NivaraColumn.cs:2724` but is not discoverable as `Select`. The LINQ name matters for user expectations.
- `Zip` is needed by `DataParallelTrainer` gradient accumulation (sum gradient column pairs) and cross-framework test comparisons.
- Terminal reductions (`Sum`, `Mean`, `Min`, `Max`) are used by `ComputeGradientNorm()` and `SumAndApplyGradients()` in `DataParallelTrainer`, which currently use manual `for` loops or raw `TensorPrimitives`.

### Scope

- Add `public NivaraColumn<TResult> Select<TResult>(Func<T, TResult> transform)` — delegate to `Transform` (or make `Transform` the implementation and add `Select` as an alias)
- Add `NivaraColumn<TResult> Zip<T2, TResult>(NivaraColumn<T2> other, Func<T, T2, TResult> combine)`:
  - Validates lengths match
  - Null propagation: null in either column → null in result
  - Uses `ArrayPool` for large columns (>1024 elements) per AGENTS.md rules
  - Returns `NivaraColumn<TResult>` with proper null mask
- Add `T Sum()` — skips nulls, uses `TensorPrimitives.Sum` for `float`/`double`, scalar loop for other `INumber<T>` types
- Add `double Mean()` — `Sum() / count_non_null`
- Add `T Min()` / `T Max()` — null-skipping, uses `TensorPrimitives.Min`/`Max` for `float`/`double`
- All operations preserve null semantics (SQL-style: nulls are skipped in aggregation, propagated in Zip)
- Write unit tests for:
  - `Select` with identity transform → equal column
  - `Select` with nulls → nulls preserved in output positions
  - `Zip` with matching lengths, mismatched lengths → throw
  - `Zip` null propagation (null left, null right, both null)
  - `Sum` / `Mean` / `Min` / `Max` with mixed null/non-null data
  - `Sum` on empty column → exception or zero (document chosen behavior)
  - Vectorized path vs scalar fallback consistency check

### Acceptance criteria

- `Select` matches `Transform` output bit-for-bit for the same input
- `Zip` correctly propagates nulls with mask-OR semantics
- `Sum` returns correct result using vectorized path for `float`/`double`, scalar for other types
- All methods throw `ArgumentException` on length mismatch (Zip) or empty column (aggregations)
- 10+ passing tests covering normal, null, and edge cases

### Files likely involved

- `src/Nivara/NivaraColumn.cs` — add `Select`, `Zip`, `Sum`, `Mean`, `Min`, `Max` methods (or add as extension methods in a new partial / extensions file)
- `src/Nivara/Interfaces.cs` — check if `IColumn<T>` needs new members (likely not — keep on concrete class)
- `tests/Nivara.Tests/NivaraColumnTests.cs` — add tests

---

## Task 3: NivaraFrame.Create\<T\>() Overloads

### Priority

Medium

### Goal

Eliminate the `(IColumn)` cast required when passing `NivaraColumn<T>` to `NivaraFrame.Create()`.

### Why this exists

Every frame creation in the codebase — and ~20+ instances in tests — requires an explicit `(IColumn)` cast:
```csharp
NivaraFrame.Create(("f1", (IColumn)NivaraColumn<float>.Create(data)));
```
This is noise that increases friction and reduces readability.

### Scope

- Add generic static method: `NivaraFrame.Create<T>(string name, NivaraColumn<T> column)` — returns a single-column frame
- Add generic static method: `NivaraFrame.Create<T1, T2>(...)` — two-column convenience (maybe overkill, use `params` pattern instead)
- Better approach: Add `NivaraFrame.Create(params NivaraColumnInstance[] columns)` where `NivaraColumnInstance` is a small wrapper that pairs a string name with an `IColumn`, with implicit conversion from `NivaraColumn<T>`:

  ```csharp
  public readonly record struct NamedColumn(string Name, IColumn Column)
  {
      public static implicit operator NamedColumn<T>((string, NivaraColumn<T>) tuple)
          => new(tuple.Item1, tuple.Item2);
  }
  ```
  But this gets complex. Simpler: just add a `NivaraFrame.CreateTuple<T>(string name, NivaraColumn<T> column)` factory overload. Actually, the simplest fix:

  ```csharp
  public static NivaraFrame Create<T>(string name, NivaraColumn<T> column)
      => new(new[] { (name, (IColumn)column) });
  ```

  And a `Create` accepting `IEnumerable<(string Name, IColumn Column)>` already exists via the dictionary overload.

  **Final decision**: add `Create<T>(string name, NivaraColumn<T> column)` and `Create(params (string Name, IColumn Column)[] namedColumns)` already exists — since C# `params` doesn't support generic type inference per-element, the best addition is:

  ```csharp
  public static NivaraFrame Create<T>(string name, NivaraColumn<T> column)
  ```

  For multi-column, the existing `params (string Name, IColumn Column)[]` remains, but we can add an extension:

  ```csharp
  public static NivaraFrame ToFrame<T>(this NivaraColumn<T> column, string name)
      => NivaraFrame.Create(name, column);
  ```

  This is idiomatic: `myColumn.ToFrame("name")`.

### Scope (refined)

- Add `public static NivaraFrame Create<T>(string name, NivaraColumn<T> column)` static method
- Add `public NivaraFrame ToFrame(this NivaraColumn<T> column, string name)` extension method
- Update existing test code that uses `(IColumn)` cast to use the new overload (Task 7 covers this, but do a quick scan now)
- Update any internal callers (e.g. in `NivaraFrameExtensions`, `TensorDataset`) to use the new overload

### Acceptance criteria

- `NivaraFrame.Create("x", intColumn)` compiles without cast
- `intColumn.ToFrame("x")` produces equivalent frame
- `(IColumn)` cast can be removed from at least 10 existing call sites (tests + internal)
- Backward compatible: existing `(IColumn)` callers continue to compile

### Files likely involved

- `src/Nivara/NivaraFrame.cs` — add `Create<T>()`
- `src/Nivara/NivaraFrameExtensions.cs` — add `ToFrame` extension
- `tests/Nivara.Tests/LinqQueryTests.cs` — remove casts
- `tests/Nivara.Tests/AutoDiff/TrainingTests.cs` — remove casts
- `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs` — remove casts

---

## Task 4: DistinctOperation on QueryFrame

### Priority

Medium

### Goal

Add a `Distinct()` query operator that removes duplicate rows via hash-based comparison.

### Why this exists

Data deduplication is a common preprocessing step before training. The existing `GroupKey` class in `GroupByOperation.cs` already has the hashing infrastructure needed — `Distinct` is essentially `GroupBy` on all columns followed by key extraction.

### Scope

- Add `DistinctOperation : IQueryOperation`:
  - Reuses `GroupKey` for composite row hashing
  - Compares all columns (or specified columns) for equality
  - For each unique composite key, keeps the first occurrence
  - Implements `TransformSchema()` — schema is unchanged
- Add `QueryFrame.Distinct()` method
- Add `QueryFrame.Distinct(params string[] columnNames)` — distinct by specific columns only
- Register `OperationType.Distinct` constant
- Handle nulls: null == null for distinct purposes (consistent with `GroupKey`)
- Write unit tests for:
  - `Distinct` removes duplicate rows
  - `Distinct` preserves first occurrence order
  - `Distinct` with null values (null rows are deduplicated correctly)
  - `Distinct(columnNames)` deduplicates by subset of columns
  - Chaining: `Where().Distinct().Select()`

### Acceptance criteria

- `Distinct()` removes all duplicate rows from a frame
- `Distinct("key")` removes rows where the key column values match, keeping first
- Null handling is consistent with `GroupKey` semantics
- Works correctly with parallel execution strategy
- 6+ passing tests

### Files likely involved

- `src/Nivara/Query/QueryFrame.cs` — add `Distinct()` methods
- `src/Nivara/Query/OperationType.cs` — add `Distinct` constant
- `src/Nivara/Operations/` — new `DistinctOperation.cs` file
- `src/Nivara/Operations/GroupByOperation.cs` — may reuse `GroupKey` (already public)
- `tests/Nivara.Tests/Query/QueryFrameTests.cs` — add tests

---

## Task 5: Rewrite TensorDataset.BuildTensor() Using New Operators

### Priority

High

### Goal

Replace the manual nested loop in `TensorDataset.BuildTensor()` with the query pipeline's `SelectRows` + column-level operations.

### Why this exists

The current `BuildTensor()` (lines 49-104 in `TensorDataset.cs`) manually:
1. Gets typed columns from the frame
2. Rents an `ArrayPool` buffer
3. Does a double `for` loop over rows and columns
4. Checks `IsNull` per element
5. Constructs a `NivaraColumn<T>` from the flat buffer
6. Wraps it in `ReverseGradTensor<T>` and reshapes

With `SelectRows`, most of this becomes: extract a mini-frame, then convert columns to tensors.

### Scope

- Replace `BuildTensor()` body:
  ```csharp
  ReverseGradTensor<T> BuildTensor(string[] columnNames, ReadOnlySpan<int> indices, bool requiresGrad)
  {
      var batchFrame = _frame.AsQueryFrame()
          .SelectRows(indices.ToArray())
          .Collect();

      var columns = columnNames.Select(name => batchFrame.GetColumn<T>(name)).ToArray();
      var numCols = columns.Length;
      var batchSize = indices.Length;

      // Flatten row-major from individual columns
      var data = new T[batchSize * numCols];
      bool? hasNulls = null;
      var nullMask = new bool[batchSize * numCols];

      for (int j = 0; j < numCols; j++)
      {
          var col = columns[j];
          var hasColNulls = col.HasNulls;
          for (int i = 0; i < batchSize; i++)
          {
              data[i * numCols + j] = col[i];
              if (hasColNulls && col.IsNull(i))
              {
                  nullMask[i * numCols + j] = true;
                  hasNulls = true;
              }
          }
      }

      var column = hasNulls == true
          ? NivaraColumn<T>.CreateFromSpans(data.AsSpan(), nullMask.AsSpan())
          : NivaraColumn<T>.Create(data.AsSpan());

      var tensor = new ReverseGradTensor<T>(column, requiresGrad);
      tensor.Reshape(batchSize, numCols);
      return tensor;
  }
  ```
  This still has a loop but is cleaner. A future optimization could use `NivaraColumn<T>` flatten-to-tensor directly.

  **Alternative — use `Select` + column ops instead of manual array fill**:
  If we had a column-level `FlattenToRowMajor(columns[])` helper, this would be even cleaner. But the immediate improvement is using `SelectRows` to let the query pipeline handle index-based extraction, then keeping the column-to-tensor conversion focused on its one job.

- Ensure `SelectRows` integration doesn't regress performance:
  - Benchmark: measure current `BuildTensor` vs new version for batch sizes 1, 32, 128, 1024
  - The `SelectRows` path adds a `QueryPlan` + operation dispatch overhead. For small batches this is negligible. For large batches the `createFilteredColumn` logic does essentially the same work as the manual loop — but it's now centralized and reusable.

### Acceptance criteria

- `TensorDataset.GetBatch()` produces bit-identical tensors for the same indices (including null masks)
- Performance is within 10% of the original implementation for batch sizes 32+
- Null propagation is identical: same positions are null in the output
- All existing `TrainingTests` and `DataParallelTests` continue to pass

### Files likely involved

- `src/Nivara/AutoDiff/Training/TensorDataset.cs` — rewrite `BuildTensor()`
- `tests/Nivara.Tests/AutoDiff/TrainingTests.cs` — verify no regression
- `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs` — verify no regression

---

## Task 6: Rewrite DataParallelTrainer Gradient Accumulation Using Column Ops

### Priority

Medium

### Goal

Replace manual `TensorPrimitives.Add` and `for` loops in `DataParallelTrainer` with `NivaraColumn<T>.Sum()` / `Zip()` / column arithmetic.

### Why this exists

`SumAndApplyGradients()` (lines 201-224 in `DataParallelTrainer.cs`) manually sums gradient arrays from parallel chunks using `TensorPrimitives.Add` on raw `T[]`. `ComputeGradientNorm()` (lines 226-244) manually loops over elements computing sum-of-squares. Both can use the new column-level ops.

### Scope

- Refactor `SumAndApplyGradients()`:
  - Before: extracts `T[]` from each chunk's gradient, sums via `TensorPrimitives.Add`, creates new column
  - After: use column-level `Zip` to sum gradients, then assign the summed column back:
    ```csharp
    var summedColumn = allGradients[0][name]; // NivaraColumn<T>
    for (int c = 1; c < chunkCount; c++)
        summedColumn = summedColumn.Zip(allGradients[c][name], (a, b) => a + b);
    tensor.Grad = summedColumn;
    ```
  - This requires `CloneGradients()` to return `NivaraColumn<T>` instead of `T[]`

- Refactor `CloneGradients()`:
  - Currently copies `.Grad` data into `T[]` via `CopyTo(gradData.AsSpan(), T.Zero)`
  - Change to copy into `NivaraColumn<T>`:
    ```csharp
    var gradData = new T[length];
    tensor.Grad.CopyTo(gradData.AsSpan(), T.Zero);
    snapshot[name] = NivaraColumn<T>.Create(gradData);
    ```
  - Or simpler: `NivaraColumn<T>.Create(tensor.Grad.Select(x => x).ToArray())` but that re-allocates. Best: add a `NivaraColumn<T>.Clone(IColumn)` or use `tensor.Grad` directly since `Grad` is already a `NivaraColumn<T>`.

  Actually, the simplest approach: **just store the `NivaraColumn<T>` reference from `tensor.Grad` directly** in the snapshot, since we're reading it immediately in the sequential post-processing loop. The snapshot needs to capture the gradient values *at that point in time*, since subsequent backward passes may mutate `.Grad`.

- Refactor `ComputeGradientNorm()`:
  ```csharp
  double ComputeGradientNorm()
  {
      double sumSq = 0;
      foreach (var (_, tensor) in _model.Parameters())
      {
          if (tensor.Grad == null) continue;
          var gradCol = tensor.Grad; // NivaraColumn<T>
          // Use Select + Sum: sum of squares
          var sqCol = gradCol.Select(x => x * x);
          sumSq += double.CreateChecked(sqCol.Sum());
      }
      return Math.Sqrt(sumSq);
  }
  ```

### Acceptance criteria

- `DataParallelTrainer` produces identical results (same loss curve, same final model) before and after refactoring
- `CloneGradients()` correctly captures gradient snapshots (subsequent backward passes don't affect stored snapshots)
- `ComputeGradientNorm()` returns the same value within floating-point tolerance
- All `DataParallelTests` pass

### Files likely involved

- `src/Nivara/AutoDiff/Training/DataParallelTrainer.cs` — refactor `CloneGradients`, `SumAndApplyGradients`, `ComputeGradientNorm`
- `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs` — verify convergence equivalence

---

## Task 7: Update Tests to Use New Declarative Style

### Priority

Low (cleanup)

### Goal

Update test code to use the new operators (`Select`, `Zip`, `ToFrame`, `Create<T>`) instead of manual loops and casts, making tests more readable and serving as living documentation.

### Why this exists

The current test suite has:
- Duplicated `CreateTestFrame` helper in `TrainingTests.cs` and `DataParallelTests.cs` that uses `for` loops
- ~20+ `(IColumn)` cast instances
- Manual `FlattenBatches` helper that could use `Select`
- Manual element-wise assertion loops that could use `Zip` + `All`

### Scope

- Replace the duplicated `CreateTestFrame` in both training test files with `Select`-based declarative construction:
  ```csharp
  // Before:
  var f1 = new float[numRows]; for (...) { f1[i] = i; label[i] = 2*i+1; }
  
  // After:
  var f1 = NivaraColumn<float>.Create(Enumerable.Range(0, numRows).Select(i => (float)i).ToArray());
  var label = f1.Select(x => 2f * x + 1f);
  return NivaraFrame.Create("f1", f1).WithColumn("label", label);
  ```
- Replace `(IColumn)` casts in test files with `Create<T>()` or `ToFrame()`
- Replace `FlattenBatches` helper with column-level `Select` + iteration if practical, or keep as-is if it adds complexity
- Update manual element-wise assertion loops to use `Zip` + `All` where it improves readability
- Do NOT change the test assertions themselves — only the data setup and comparison mechanics
- Do NOT create a shared test helper that couples the two training test files — the declarative style should stand on its own

### Acceptance criteria

- All existing tests continue to pass without assertion changes
- No remaining `(IColumn)` casts in test files (at least 15+ removed)
- `CreateTestFrame` duplication is eliminated or significantly reduced
- Test setup code is visibly more declarative (less `for` loops, more column operations)

### Files likely involved

- `tests/Nivara.Tests/LinqQueryTests.cs` — remove casts
- `tests/Nivara.Tests/AutoDiff/TrainingTests.cs` — remove casts + replace `CreateTestFrame`
- `tests/Nivara.Tests/AutoDiff/DataParallelTests.cs` — remove casts + replace `CreateTestFrame`
- `tests/Nivara.Tests/AutoDiff/NivaraIntegrationTests.cs` — remove casts
- `tests/Nivara.Tests/AutoDiff/SerializationTests.cs` — remove casts
- `tests/Nivara.Tests/AutoDiff/NnTests.cs` — remove manual loops where possible
- `tests/Nivara.Tests/AutoDiff/CrossFrameworkParityTests.cs` — use `Zip` for comparison loops
- `tests/Nivara.Tests/AutoDiff/GradOperationsTests.cs` — remove casts

---

## Task 8: Add Split + Normalize Data Prep Helpers

### Priority

Medium

### Goal

Add `NivaraFrame.Split(double trainRatio)` for train/test splitting and `NivaraFrame.Normalize(columnNames)` for z-score normalization, built on top of the new operators.

### Why this exists

Train/test splitting and data normalization are the two most common preprocessing steps before any training run. Currently there is no built-in support — users must manually implement index splitting and column normalization. With `Skip`/`Take` (Task 1) and `Select` (Task 2), both become straightforward.

### Scope

- Add `NivaraFrame.Split(double trainRatio, int? seed = null) → (NivaraFrame train, NivaraFrame test)`:
  - Shuffles row indices using Fisher-Yates with optional seed
  - Splits at `(int)(rowCount * trainRatio)`
  - Uses `SelectRows` internally (or `Skip`/`Take`)
  - Returns a tuple of two frames
  - Throws if `trainRatio` is not in (0, 1)

- Add `NivaraFrame.Normalize(params string[] columnNames) → NivaraFrame`:
  - Computes z-score `(x - mean) / std` for each specified column
  - Uses `Select` internally: `col.Select(x => (x - mean) / std)`
  - Replaces the specified columns in a new frame
  - Returns frame unchanged if no column names specified (normalize all numeric columns)
  - Skips columns that are not numeric or not found (with optional warning)

- Add `NivaraFrame.Standardize(params string[] columnNames) → NivaraFrame`:
  - Alias for `Normalize` (z-score is the standard normalization)

- Write unit tests for:
  - `Split` with ratio 0.8 → train has ~80%, test has ~20%
  - `Split` with seed → deterministic outcome
  - `Split` with ratio 0 → empty train, full test
  - `Split` with ratio 1 → full train, empty test
  - `Normalize` produces zero-mean, unit-variance columns (within tolerance)
  - `Normalize` with null values (nulls remain null, normalization is based on non-null stats)
  - `Normalize` with non-numeric columns (left unchanged)

### Acceptance criteria

- `Split` returns frames whose row counts sum to the original row count
- `Split` is deterministic when seed is provided
- `Normalize` produces columns with mean ≈ 0 and std ≈ 1 for float/double columns
- Null propagation: nulls in input remain null in output after normalization
- All methods preserve the frame schema (column names and types, except normalization changes values)
- 8+ passing tests

### Files likely involved

- `src/Nivara/NivaraFrameExtensions.cs` — add `Split()`, `Normalize()`, `Standardize()` methods
- `tests/Nivara.Tests/NivaraFrameExtensionsTests.cs` — add tests (or existing frame test file)

---

## Suggested Agent Handout Batches

### Batch A: foundation (start first, must be sequential)

- Task 1: `SelectRowsOperation` + `Skip`/`Take` on `QueryFrame`
- Task 2: `NivaraColumn<T>.Select` + `Zip` + terminal reductions

### Batch B: dependent implementation (after Batch A)

- Task 3: `NivaraFrame.Create<T>()` overloads (independent of A, can run parallel with A if desired)
- Task 4: `DistinctOperation` (independent of A, can run parallel with A if desired)
- Task 5: Rewrite `TensorDataset.BuildTensor()` (depends on Task 1)
- Task 6: Rewrite `DataParallelTrainer` gradient accumulation (depends on Task 2)

### Batch C: polish and helper APIs

- Task 7: Update tests to use new declarative style (after Tasks 2-4 stable)
- Task 8: Add `Split` + `Normalize` helpers (depends on Tasks 1-2)

---

## Final Checklist

- [x] every task has a clear owner-sized scope
- [x] every task has acceptance criteria
- [x] decision-gate tasks are clearly marked (Task 1 — `SelectRowsOperation` API shape)
- [x] likely files are listed to reduce agent search time
- [x] execution order reflects real dependencies
