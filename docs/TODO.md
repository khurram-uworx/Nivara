# Plan: Row-wise tensor scoring — row-slice kernels (#141), frame API (#138), benchmarks (#142)

## Problem

Feature-vector workflows need row-wise scoring over embedding/sample/item rows: rows are
vectors, and users score each row against a query vector. The old frame APIs
(`Dot`, `CosineSimilarity`, `RowNorms`) were removed in the AutoDiff refactor (Task 10)
because they had no production callers; `TensorPrimitives` is the sanctioned replacement.
Scoped in `docs/TENSORS.md` → "Scoped tensor ambitions" (issues #138, #141, #142).

- **#141** — internal row-slice `TensorPrimitives` kernels over the row-major buffer
  (`RowDot` / `RowCosineSimilarity` / `RowNorms`). Prerequisite for the frame API.
- **#138** — public `NivaraFrame.RowDot<T>` / `RowCosineSimilarity<T>(query, labels?)`.
- **#142** — benchmark coverage comparing per-row status quo vs row-major materialization
  vs row-slice kernels (the evidence the removed APIs never had).

The reference to `docs/plan/FUTURE.md` in the issues is stale — FUTURE.md is gone; these are
tracked as GitHub issues. `docs/TENSORS.md` is the current canonical doc.

## Decisions (confirmed with human)

- **Order:** #141 (kernels) → #138 (frame API) → #142 (benchmarks) → docs.
- **Orientation:** rows are vectors; `query` length == frame `ColumnCount` (one element per
  dimension); one score per row → `NivaraSeries<T>` of length `RowCount`.
- **Null semantics (mask-first, SQL-like):** a null in a row makes only that row's score null;
  a null in the query makes all scores null; placeholder values at null positions are NOT valid
  output — the result always carries a null mask.
- **Constraints (MS docs):** `TensorPrimitives.Dot<T>` needs `INumber<T>`;
  `CosineSimilarity<T>` / `Norm<T>` need `IRootFunctions<T>`. Kernels are already BCL — no
  swap-target annotation needed (they ARE the target).
- **No public `RowNorms` on the frame** (#141 keeps the norm kernel internal-only).
- **No BLAS matmul in core; no column-as-tensor-axis redesign** — interop conveniences only.
- **Kernel placement:** `TensorsHelper` (the "single file to check when upgrading .NET").

## Changes

### 1. Kernels — `src/Nivara/Tensors/TensorsHelper.cs` (#141)

Add three internal row-slice kernels (no per-row copy — each row is a contiguous
`rowMajor.Slice(r*cols, cols)` into the materialized buffer):

- `RowDot<T>(rowMajor, rowMajorNullMask, query, queryNullMask, output, outputMask, rows, cols)
  where T : struct, INumber<T>`
- `RowCosineSimilarity<T>(...) where T : struct, IRootFunctions<T>`
- `RowNorms<T>(rowMajor, rowMajorNullMask, output, outputMask, rows, cols)
  where T : struct, IRootFunctions<T>` (no query — row-null → that row's norm null)

Mask/argument logic (shared private helper):
- Validate lengths: `output.Length >= rows`, `outputMask.Length >= rows`,
  `rowMajor.Length >= rows*cols`, `query.Length == cols` (query kernels only).
- `queryHasNull` (any true in `queryNullMask`) → `outputMask.Fill(true)`, `output.Clear()`,
  return — mask is authoritative, placeholders invalid.
- No `rowMajorNullMask` → compute all rows, `outputMask.Clear()` (fast path).
- Else per row: `rowHasNull = any(true)` over the row's mask slice; score =
  `rowHasNull ? default : TensorPrimitives.X(row slice, query)`; `outputMask[r] = rowHasNull`.

Tests — `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`:
- Match scalar per-row fallback (`TensorPrimitives.Dot`/`CosineSimilarity`/`Norm` per row).
- Row-null masks only that row; query-null masks all rows; no-nulls clears the mask;
  empty mask (`Length == 0`) = no nulls.
- Argument validation (short output/mask, query-length mismatch).

### 2. Frame API — `src/Nivara/NivaraFrame.cs` (#138)

- Internal `MaterializeRowMajor<T>(Span<T> data, Span<bool> mask)` — mirrors
  `CopyToRowMajor`'s pooled-temp pattern, using `NivaraColumn<T>.CopyTo(dest, fill, maskDest)`
  per column then scattering row-major (reuses the existing column mask copy path).
- Public API:
  ```csharp
  public NivaraSeries<T> RowDot<T>(NivaraSeries<T> query, IColumn? labels = null)
      where T : unmanaged, INumber<T>
  public NivaraSeries<T> RowCosineSimilarity<T>(NivaraSeries<T> query, IColumn? labels = null)
      where T : unmanaged, IRootFunctions<T>
  ```
- Validation: disposed guard; `ArgumentNullException` on query; `query.Length == ColumnCount`;
  provided `labels.Length == RowCount`. `RowCount == 0` → empty series early return.
  `ColumnCount == 0` → `RowCosineSimilarity` throws (empty-vector parity with removed API);
  `RowDot` returns empty (dot over zero-length is 0).
- Rent row-major buffers when `rows*cols >= 1024`; query span/mask via
  `query.Values.TryGetSpan` / `TryGetNullMask`.
- Build result via `ColumnStorageFactory.CreateFromOwnedArray(scores, mask)` +
  `new NivaraColumn<T>(storage)`; labels default to positional index, otherwise materialize
  `IColumn` → `object[]` via `GetValue`. Return `NivaraSeries<T>`.
- Wrap in `DiagnosticsTracker.MeasureOperation("FrameRowDot"/"FrameRowCosineSimilarity", ...)`
  per the existing `ToTensors` pattern.

Tests — `tests/Nivara.Tests/NivaraFrameTests.cs`:
- Correctness vs per-row `TensorPrimitives` scalar for both methods.
- Row-null → only that row's score `IsNull`; query-null → all scores `IsNull` (SQL-like).
- Query length mismatch throws; labels length mismatch throws; labels column → labeled series;
  no labels → positional index.
- Empty frame returns empty series; disposed frame throws; mixed-type frame throws
  (`ColumnTypeMismatchException` via `GetColumn<T>`).

### 3. Benchmarks — `tests/Nivara.PerformanceTests/Program.cs` (#142)

Add row-wise scoring scenarios to `RegisterScenarios()` (reuse `Run(...)` harness):
- **Status quo (per-row):** loop rows, slice each row's data + call
  `TensorPrimitives.Dot`/`CosineSimilarity` per row.
- **Row-major materialization:** `CopyToRowMajor` into a pooled buffer, then score.
- **Row-slice kernels:** `TensorsHelper.RowDot`/`RowCosineSimilarity` over the materialized
  buffer with pre-allocated output.
- Allocation pressure for repeated row-wise scoring (B/op + gen0/op columns already tracked).
- Realistic row counts (e.g. 10k rows × 128 dims, matching the embedding-row use case).
- Update `tests/Nivara.PerformanceTests/README.md` scenario table.

### 4. Docs & verification

- `docs/TENSORS.md` — update "Scoped tensor ambitions": mark #141 kernels + #138 frame API
  delivered; note `RowDot`/`RowCosineSimilarity` re-added as row-wise interop conveniences
  (not a column-axis change, no public `RowNorms`).
- `CHANGELOG.md` entry.
- `dotnet build Nivara.slnx`; ask before running `dotnet test`.

## Planned commits

1. `docs: plan row-wise tensor scoring (#138/#141/#142) in TODO.md`
2. `feat: add row-slice TensorPrimitives kernels to TensorsHelper (#141)` + kernel tests
3. `feat: add NivaraFrame.RowDot/RowCosineSimilarity row scoring (#138)` + frame tests
4. `test: add row-wise scoring benchmark scenarios (#142)`
5. `docs: mark row-wise tensor scoring delivered (#138/#141/#142)`

## Follow-ups / out of scope

- #142 benchmark *results* table update is the evidence step (record on same machine per
  the harness README baseline policy).
- No public `RowNorms` on the frame (kernel-only).
- BFloat16 kernels remain deferred to the net11 migration (unchanged).
