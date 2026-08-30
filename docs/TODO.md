# TODO plan — Issue #349: eliminate per-element boxing for nullable-element columns in ColumnFilterHelper

Branch: `khurram/349` (off `main`)

## Problem

`NivaraFrame.FilterByMask` (and `ColumnFilterHelper` reorder / concatenate / scatter kernels)
copy selected elements per row. For a **nullable-element** column (`NivaraColumn<int?>`, generic
parameter `T = int?`, `ElementType == typeof(int?)`):
- `CreateFilteredColumn` computes `unwrapNullable(column.ElementType)` → `int` (strips the `?`).
- `MakeGenericMethod(typeof(int))` → `createFilteredColumnTyped<int>`.
- `column is NivaraColumn<int>` → **false** (a `NivaraColumn<int?>` is not a `NivaraColumn<int>`).
- Falls into the **boxed fallback** `column.GetValue(indices[i])` → one `object` boxing per element.

Same boxed fallback in `reorderColumnTyped`, `concatenateColumnsTyped`, `scatterPartsTyped`; and the
unwrapped element type is likewise dropped in `createEmptyColumnTyped` / `createNullColumnTyped`
(producing `NivaraColumn<int>` where `int?` was expected).

Measured (issue): ~24 B/row extra allocation vs mask-based columns (~237 KB / 10 000 rows).
The `RunRowWhereScenarios` performance harness row already gates the residual in `FilterByMask`
as #349.

## Fix

Stop unwrapping nullable in `ColumnFilterHelper`; use the column's actual `ElementType`
(`typeof(int?)` for `NivaraColumn<int?>`). Instantiating the typed kernels with `T = int?` makes
the existing fast path `column is NivaraColumn<T>` match and eliminates boxing.

Why `T = int?` works in the existing kernels (grounded via Microsoft Learn):
- The kernels are **unconstrained** (`createFilteredColumnTyped<T>`, no `struct`/`class` constraint),
  so `MakeGenericMethod(typeof(int?))` is valid — `NivaraColumn<int?>` is a normal closed generic.
- `typeof(int?).IsValueType` → true (nullable value types are structs).
- `column is NivaraColumn<int?>` → true for the source column's exact type.
- `typed[index]` returns `int?`; nulls tracked via the existing `bool[]` mask (`IsNull`).
- `NivaraColumn<int?>.CreateFromOwnedArray` / `CreateFromSpans` / `Create` all work for a nullable
  `T` (nullable value types are supported column element types across the codebase).

### Edits — `src/Nivara/Helpers/ColumnFilterHelper.cs`

1. `CreateFilteredColumn`: `unwrapNullable(column.ElementType)` → `column.ElementType`
2. `ReorderColumn`: `unwrapNullable(column.ElementType)` → `column.ElementType`
3. `CreateEmptyColumn`: `unwrapNullable(elementType)` → `elementType`
4. `ConcatenateColumns`:
   - `unwrapNullable(columns[0].ElementType)` → `columns[0].ElementType`
   - mismatch check `unwrapNullable(column.ElementType) != elementType` → `column.ElementType != elementType`
5. `CreateNullColumn`: `unwrapNullable(elementType)` → `elementType`
6. `ScatterPartsColumn`:
   - `unwrapNullable(parts[0].ElementType)` → `parts[0].ElementType`
   - mismatch check `unwrapNullable(column.ElementType) != elementType` → `column.ElementType != elementType`
7. Remove now-unused private helper `unwrapNullable` and `using System.Reflection` stays (still
   needed for `MethodInfo` / `getMethod`).

### Behavior / blast radius

- Non-nullable `NivaraColumn<T>`: `ElementType == typeof(T)` already, unchanged. Columns built from
  `NivaraColumn.CreateFromNullable(...)` are `NivaraColumn<T>` (element type already unwrapped) —
  unaffected.
- Nullable-element columns now round-trip their element type (`int?` → `int?`) instead of being
  converted to `NivaraColumn<int>` with a manual mask. This is the documented "preserve source
  element type" contract.
- `ConcatenateColumns` / `ScatterPartsColumn` mismatch validation now compares actual element types:
  mixing `NivaraColumn<int>` + `NivaraColumn<int?>` throws. No legitimate caller mixes these
  (concat/scatter join parts of one source column type), so no real-world breakage.

Downstream callers of the six helpers (all already pass `column.ElementType`, so they inherit the
round-trip preservation):
- `NivaraFrame.FilterByMask` / `Where`/`Distinct`/`SelectRows`/`Slice`/`Sort` (`NivaraFrame.cs`,
  `NivaraFrameExtensions.cs`, `FilterOperation.cs`, `DistinctOperation.cs`, `SelectRowsOperation.cs`,
  `SliceOperation.cs`, `SortOperation.cs`)
- `StreamingWindowProcessor`, `PartitionedWindowStreamer`, `PartitionedWindowEngine` (window/streaming)
- `ConcatenationOperation`, `ParallelExecutionHelper` / `ParallelExecutionStrategy` (parallel/streaming)
- `ColumnFilterHelperTests`, `NivaraFrameFilteringSlicingTests` etc.

## Blast radius assessment

- **Files changed:** `src/Nivara/Helpers/ColumnFilterHelper.cs` (core), plus tests.
- **Files that depend on the changed helpers:** the callers listed above (all behavior-neutral except
  the intended nullable round-trip fix and the concat/scatter type-mismatch tightening).
- **Tests covering them:** `ColumnFilterHelperTests`, `NivaraFrameFilteringSlicingTests`,
  `SortingIntegrationTests`, `WindowFunctionsTests`, `ConcatenationOperationTests`,
  `ParallelExecutionHelperTests`, and the perf harness `RunRowWhereScenarios` (`Nivara.PerformanceTests`).

## Verification

- `dotnet build Nivara.slnx` (ask before running).
- Run `ColumnFilterHelperTests` + lint-relevant existing tests (ask before `dotnet test`).
- Optional: perf harness `RunRowWhereScenarios` — the boxing residual should drop once a new
  baseline row is recorded.

## Planned commits

1. `fix: route nullable-element columns through typed ColumnFilterHelper kernels (issue #349)` —
   core fix in `ColumnFilterHelper.cs` (commit after build).
2. `test: cover nullable-element round-trip in ColumnFilterHelper (issue #349)` — new unit tests +
   frame-level `FilterByMask`/`Where` round-trip test.
3. If review surfaces a gap: additional fix/test commit on the same branch.

## GitHub issues log

- [ ] (none yet — capture any deferred work discovered here during execution)
