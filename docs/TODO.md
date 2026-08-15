# Plan: #251 window allocation reduction + #252 window/expression test coverage

Branch: `khurram/issues` (off `main`). Executing via the iterative-work workflow: one logical
change per commit, `dotnet test` runs only with explicit human confirmation, no pushes.

## Problem statements

- **#251 (performance, medium)**: Window kernels allocate heavily on hot paths. LINQ `OrderBy`
  per partition in `RankKernel` and `PartitionedWindowEngine`; 4–6 full-array allocations per
  rolling-kernel call (`buildEffective` effective/valid copies, widened/normal prefix arrays,
  `deque = new int[length]`, plus result/mask); `PartitionedWindowEngine` does full-column passes
  per call (reorder → per-partition compute → concatenate → inverse scatter → reorder);
  `GroupByOperation.CreateGroupsInternal` allocates a boxed `object?[]` + `GroupKey` per row.
- **#252 (test coverage, medium)**: No streaming-vs-eager chunk-equivalence test (mask-aware); no
  all-null inputs, window-larger-than-data, or Shift boundary (0/±length) cases; no property tests
  for rolling mean/min/max (only sum + cumulative have them); no property tests for partitioned
  (spec'd) rolling; no Polars fixtures for rolling ops; the fused-expression chunked helpers compare
  only visible values+masks, never masked-position backing values.

## Reference corrections

- The Polars **window** fixture generator lives at `samples/NivaraIncident/Python/gen_reference.py`
  (+ `requirements.txt`, `polars>=1.0.0`), moved from `samples/NivaraWindow/`. It emits
  `samples/data/polars-window/manifest.json`, consumed by
  `tests/Nivara.Tests/Query/PolarsWindowCrossValidationTests.cs`. All fixture work below uses this
  path. `samples/NivaraTorch/gen_reference.py` (PyTorch) is unrelated.
- #251's roadmap risk note is at `docs/plan/POLARS-ROADMAP.md:113` ("pooled scratch").

## Design decisions (user-confirmed)

1. **PartitionedWindowEngine**: index-map scatter refactor — eliminate concatenation + inverse +
   final reorder; keep the single source gather (SIMD-friendly contiguous slices). Deeper
   index-aware kernels that read via indirection are rejected (they force scalar per-element access,
   losing `TensorPrimitives` fast paths) — recorded as a follow-up issue if still relevant.
2. **Group keys**: Tier B — full typed multi-column grouping, breaking changes allowed. Replace the
   per-row boxed `object?[]` + `GroupKey` with typed readers over `NivaraColumn<T>`, per-row typed
   composite hashes, and a `GroupKey` built once per distinct group.
3. **Polars rolling fixtures**: both — extend the Python generator AND add C# property tests.

## Issue #251 changes

### 1. Rolling/cumulative kernels — null-free fast path + pooled scratch
`src/Nivara/Tensors/WindowFunctions.cs`

- Add fast paths when `!column.HasNulls && column.TryGetSpan(out var span)` for `RollingSum`,
  `RollingMean`, `RollingMin`, `RollingMax`, and `cumulativeScan`. They skip `buildEffective`
  (drops `effective` + `valid`) and, for rolling, skip the `prefixCount` array entirely
  (`windowCount = min(i + 1, windowSize)` — every position is valid).
- Pool the prefix buffer (`ArrayPool<T>.Shared` / `ArrayPool<long>.Shared`) and the deque
  (`ArrayPool<int>.Shared`) when `length > 1024`; return in `finally`. `result`/`resultMask` are
  outputs and stay.
- New private helpers: `buildPrefixFromSpan<T>`, `buildWidenedPrefixFromSpan<T>` (checked long
  accumulation, preserving the #248 contract — **no** sliding-window accumulator), and
  `rollingExtremeFromSpan<T>`.
- Numerics: null-free sum/mean keep the widened checked `long` prefix for the int family and `T`
  prefix otherwise — bit-for-bit identical results to today.
- Slow (null-bearing) path unchanged.
- Acceptance: null-free rolling sum drops from 6 arrays to 2 outputs + pooled scratch; existing
  `WindowFunctionsTests` stay green.

### 2. RankKernel — shared scratch + stable-equivalent in-place sort
`src/Nivara/Tensors/RankKernel.cs` (:69-138)

- One pooled `int[]` scratch (rowCount, `ArrayPool<int>.Shared` when `> 1024`) replaces per-partition
  `OrderBy(...).ToArray()`, `new List<int>`, `valid.ToArray()`, and the single-partition
  `Enumerable.Range`.
- Per partition: copy indices into scratch, `Array.Sort(scratch, 0, count, tieBreakComparer)`.
- New internal `RankTieBreakComparer : IComparer<int>` delegates to `MultiColumnComparer`
  (SortOperation.cs:283, already an `IComparer<int>`); on a 0-comparison breaks ties by row index.
  `MultiColumnComparer` returns 0 only on full key ties and partitions are in ascending row order,
  so this reproduces `OrderBy`'s stable order exactly — preserves RowNumber tie numbering and the
  #254 null-key ordering.
- `hasNullKey` gating unchanged (mask rows with null order keys for rank/dense_rank/percent_rank).
- Acceptance: no per-partition LINQ allocations; `RankFunctionsTests` + `PolarsWindowCrossValidationTests`
  stay green.

### 3. PartitionedWindowEngine — index-map scatter refactor
`src/Nivara/Tensors/PartitionedWindowEngine.cs` (:49-88), new `src/Nivara/Tensors/PartitionScatterKernel.cs`

- Pool `sortedAll` (`ArrayPool<int>.Shared`, >1024).
- Keep the single `ReorderColumn(sourceColumn, sortedAll)` source gather (:71) — contiguous
  per-partition slices keep the rolling kernels SIMD-fast.
- Delete `ConcatenateColumns` (:81), the `inverse` map (:83-85), and the final
  `ReorderColumn(sortedResult, inverse)` (:87). Replace with a typed scatter: for each partition,
  write the computed part's value+null at sorted position `j` into `final[sortedAll[cursor+j]]`,
  then build the result column from owned `T[]` + `bool[]` (`CreateFromSpans`).
- `PartitionScatterKernel.Scatter(IColumn[] computedParts, int[] sortedAll, int[] offsets, int[] lengths)`:
  type-switch on `computedParts[0].ElementType` → generic scatter over `NivaraColumn<T>` (indexer +
  `IsNull`, no boxing).
- **Executed as `ColumnFilterHelper.ScatterPartsColumn` / `scatterPartsTyped<T>`** (not a new
  `PartitionScatterKernel.cs` file): single-pass `result[positions[i]] = value[i]` with type
  consistency checks. The typed path also gained a **box-free fast path** (see #251-5 commit 2):
  `NivaraColumn<T>` casts + indexer/`IsNull` + `CreateFromOwnedArray`, dropping ~16MB of per-row
  `GetValue` boxing per 1M-row reorder/scatter.
- Result: full-column passes 3 → 1 (only the source gather); ~2 result-sized buffers + inverse
  eliminated per call.

### 4. Tier B — typed multi-column grouping (breaking changes allowed)
`src/Nivara/Operations/GroupByOperation.cs`, `src/Nivara/Operations/DistinctOperation.cs`,
new `src/Nivara/Operations/GroupKeyReaders.cs`

- **`GroupKeyReaders.cs`**:
  - `internal interface IGroupKeyReader { Type ElementType { get; } int GetHashCode(int row);
    bool Equals(int rowA, IGroupKeyReader other, int rowB); object? GetValue(int row); }`
  - `internal sealed class GroupKeyReader<T> : IGroupKeyReader` over `NivaraColumn<T>` via
    `EqualityComparer<T>.Default` + null-mask (null==null, null-hash 0).
  - Static factory `Create(IColumn)` — type-switch over the element-type switch domain; boxed
    fallback reader (reads `GetValue`, `object.Equals`) for exotic columns.
  - `internal static class TypedGroupHash` with `ComputeRowHashes(IReadOnlyList<IGroupKeyReader>,
    int rowCount, Span<int>)` — typed composite hash, no boxing.
- **Rework `GroupKey`** (GroupByOperation.cs:79-139): typed key = `IReadOnlyList<IGroupKeyReader>` +
  `rowIndex` + cached typed hashCode; `Equals`/`GetHashCode` box-free; `GetValue(int)`/`Values`
  lazily boxed (O(groups)); **remove** the `params object?[]` ctor, add static
  `GroupKey.FromValues(IReadOnlyList<object?>)` for non-hot callers (tests/aggregation).
- **Rework `CreateGroupsInternal`** (:340-370): readers → pooled `int[]` hashes →
  `Dictionary<int, List<int>> hashBuckets` (hash → representative rows) + `Dictionary<GroupKey,
  List<int>> groups` + rep→key map; per row: typed hash, scan bucket reps (typed equality), join or
  open group (GroupKey built once per group). Eliminates per-row `object?[]` (:353) and per-row
  `GroupKey`. `offset` handling preserved (stored as `row + offset`).
- **Rework `DistinctOperation.Execute`** (:39-51): typed readers over selected columns, pooled
  hashes, `Dictionary<int, List<int>>` collision buckets, unique row indices, filtered columns. No
  per-row `GroupKey`.
- **Parallel merge** (`ParallelExecutionStrategy.cs:195-209,404-415`, `MergeGroupByDictionaries`
  in ParallelExecutionHelper.cs:155-170): contracts unchanged; typed `GroupKey` equality is
  value-based across chunk column instances, so cross-chunk merge works as-is.
- **Test updates** (breaking change): `GroupedDataTests` (all `new GroupKey(...)` sites →
  `FromValues`), `ParallelExecutionHelperTests` (:191, :207, :223), `AggregationFunctionTests`
  (~18 `new GroupKey("A")` sites → `FromValues`). New tests: multi-type key grouping
  (int+string+null), equality across distinct column instances, Distinct typed keys,
  hash-collision disambiguation.

### 5. Measurement (#251 acceptance) — DONE, two commits
- **Commit 2 (production fixes)** — `src/Nivara/Operations/SortOperation.cs`,
  `src/Nivara/Helpers/ColumnFilterHelper.cs`:
  - `MultiColumnComparer.Compare` was doing `foreach` over an `IReadOnlyList<SortKey>` — **boxes a
    new enumerator per comparison** (32B/call × O(n log n) during `Array.Sort`). Replaced with an
    index loop. Measured effect: RankKernel RowNumber 100k **47MB → 2.5MB/op** (also benefits the
    Sort operation and PartitionedWindowEngine's per-partition sorts).
  - `reorderColumnTyped<T>` / `scatterPartsTyped<T>` boxed every element via `GetValue` (~16MB per
    1M rows) and double-copied through `CreateFromSpans → Create`. Added a typed fast path
    (value types only): `NivaraColumn<T>` cast, indexer + `IsNull`, `CreateFromOwnedArray` when
    null-free. Measured effect: PartitionedWindow 1M × 100 parts **96MB → 40MB/op**.
- **Commit 1 (scenarios + budgets)** — `tests/Nivara.PerformanceTests/Program.cs` +
  new `tests/Nivara.Tests/Tensors/WindowAllocationTests.cs`:
  - Scenarios: `RollingSum null-free 1M int (w10)`, `RollingSum nulls 1M int (w10)`,
    `RankKernel RowNumber 100k`, `GroupBy 1M × 1000 keys (typed)`,
    `GroupBy 1M × 100 string keys (typed)`, `PartitionedWindow RollingSum 1M × 100 parts`.
  - Regression budgets via `GC.GetAllocatedBytesForCurrentThread` (pattern:
    `tests/Nivara.Tests/AutoDiff/PerfTests.cs`), wide margins from harness measurements:
    null-free < null-masked; null-free < 16MB; RankKernel < 8MB (guards the boxing fix);
    GroupBy typed < 25MB; PartitionedWindow < 70MB (guards the box-free reorder/scatter).
- **Deferred (#259)**: `createFilteredColumnTyped<T>` (ColumnFilterHelper.cs:106) and
  `concatenateColumnsTyped<T>` (:184) still box every element via `column.GetValue`. Same
  box-free typed fast path applies; out of #251's window/rank/group-by scope.

## Issue #252 changes

### A. Streaming-vs-eager chunk equivalence
`tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

- Property test: chunked source **with nulls** + real `Filter`/`Select` under `Streaming` vs `Lazy`,
  asserting value **and** mask equality across chunk sizes. Extend `StubChunkedQuerySource` with a
  nullable column; add a mask-aware frame-equality helper.
- Window-bearing `Select` property test: Streaming falls back to Lazy with mask+value identity and
  zero chunks read (extends the existing single-case fallback tests).

### B. All-null / window>length / Shift boundaries
`tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`, `tests/Nivara.Tests/Query/WindowOperationTests.cs`

- All-null columns: RollingSum/Min/Max/Mean (all masked; `nullHandler` fills), cumulative, `Shift`.
- `windowSize > length` → all null; `windowSize == length` → first output at last row.
- `Shift`/`Lead` at periods `0`, `±length`, and with in-range nulls.

### C. Rolling property tests
`tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`

- Mirror `RollingSum_Int_RandomArrays_MatchesNaive` for `RollingMean`/`RollingMin`/`RollingMax` over
  random null-bearing arrays, looping `window` × `minPeriods`.

### D. Partitioned rolling property tests
new `tests/Nivara.Tests/Query/PartitionedWindowPropertyTests.cs`

- Randomized partitions + order keys; `RollingSum/Min/Max/Mean`, `CumulativeSum`, `Shift` via spec
  against a naive per-partition sorted reference.

### E. Polars rolling fixtures
`samples/NivaraIncident/Python/gen_reference.py`, `samples/data/polars-window/manifest.json`,
new `tests/Nivara.Tests/Query/RollingWindowCrossValidationTests.cs`

- Extend the generator with rolling cases (sum/mean/min/max, partitions, nulls, explicit
  `min_periods=1` matching Nivara's ignore-nulls gating) appended as `rolling_*` cases.
- New test mirrors `PolarsWindowCrossValidationTests` (reads committed JSON; no Python at runtime).

### F. Expression chunked masked-backing values
`tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`

- `AssertChunkedMatchesWhole` skips masked positions; extend to assert masked-position **backing
  values** equal between chunked and whole via typed `NivaraColumn<T>` access.
- The other listed expression gaps (literal-signature collisions, heterogeneous null-bearing
  division, literal-only plans) are already covered by the #246/#249/#250 batch — add cross-reference
  notes only.

## Verification

1. `dotnet build Nivara.slnx` after each step.
2. `dotnet test` only with explicit human confirmation; the full suite is the #252 gate.
3. `tests/Nivara.PerformanceTests` with `--compare` (baseline vs post-change) for #251
   no-regression.
4. Regenerate Polars fixtures: `python samples/NivaraIncident/Python/gen_reference.py` (needs
   `pip install -r requirements.txt`) — with human confirmation; commit the manifest.

## Blast radius

- **WindowFunctions.cs**: internal; public API unchanged. Callers: `NivaraFrameExtensions` rolling
  methods, `WindowOperations` rolling/cumulative/shift ops. Tests: `WindowFunctionsTests`,
  `WindowOperationTests`, `RollingAllocationTests` (new), perf harness.
- **RankKernel.cs**: internal. Callers: `RankOperation` (lazy), rank window frame extensions.
  Tests: `RankFunctionsTests`, `RankOperationTests`, `PolarsWindowCrossValidationTests`.
- **PartitionedWindowEngine.cs / PartitionScatterKernel.cs**: internal. Callers:
  `WindowFrameExtensions` (:274), `WindowOperations` (:157). Tests: `WindowOperationTests`,
  `WindowFunctionsTests` spec paths, new partitioned property tests.
- **GroupKeyReaders.cs / GroupKey / GroupedData / CreateGroupsInternal / DistinctOperation**:
  internal types with a public-ish internal surface used by `GroupedDataTests`,
  `ParallelExecutionHelperTests`, `AggregationFunctionTests`, `ParallelExecutionStrategy`,
  `DistinctOperation`. Breaking change to the `GroupKey` ctor — call sites migrated to
  `FromValues`. GroupedData/Dictionary<GroupKey,List<int>> surface preserved.
- **Perf harness**: additive only.

## Planned commits

1. `docs: plan #251 window allocation + #252 window/expression test coverage` — ✓ `4986045`
2. `perf: null-free fast path + pooled scratch in rolling/cumulative window kernels (#251)` — ✓ `8ad71f7`
3. `refactor: typed multi-column grouping in group-by/distinct (#251)` — ✓ `76be81f`
4. `perf: pooled scratch + stable in-place sort in RankKernel (#251)` — ✓ `d81ecd3`
5. `perf: index-map scatter in PartitionedWindowEngine (#251)` — ✓ `13763d9`
6. `perf: kill per-comparison enumerator boxing + box-free reorder/scatter fast paths (#251)` — ✓ `8eb80c5`
7. `test: rolling allocation budgets + perf scenarios (#251)` — ✓ `04898b6`
8. `test: streaming-vs-eager chunk equivalence with masks (#252)` — ✓ done
9. `test: all-null, window>length, and shift boundary cases (#252)` — ✓ done
10. `test: rolling mean/min/max property tests (#252)` — ✓ done
11. `test: partitioned rolling property tests (#252)` — ✓ done
12. `test: Polars rolling fixtures + cross-validation (#252)` — ⏳ next
13. `test: chunked masked-position backing-value assertions (#252)`
14. `docs: remove TODO.md — plan executed` (only after full review)

## GitHub issues log

- **#259** — `ColumnFilterHelper` filter/concat kernels still box per row via `GetValue`
  (`createFilteredColumnTyped` :106, `concatenateColumnsTyped` :184); apply the same box-free typed
  fast path as the #251-5 reorder/scatter kernels.

> Reminder: as each task executes, if you find deferred work or a concern (known limitations,
> follow-ups, refactors) outside the current plan, create a GitHub issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number here. Don't rely on memory
> or wait until the plan finishes.
