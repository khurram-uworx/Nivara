# TODO — Issue #156: Rank family window functions

Tracker: https://github.com/khurram-uworx/Nivara/issues/156
Branch: `khurram/156` (off `main`)

## Problem

PR #147 delivered the Phase 3 core window set (#135: rolling/cumulative/shift)
but not the ranked remainder. No `Over`/`Rank`/`DenseRank`/`PercentRank`/
`RowNumber` exists in `src/Nivara`. This plan delivers the rank family over
partition + order-by keys, following the delivered 3-layer pattern (column
kernel → `IQueryOperation` → `QueryFrame`/`NivaraFrame` API).

## Requirements

- **Explicit (#156):** `row_number` / `rank` / `dense_rank` / `percent_rank`
  over partition + order-by keys; `Over` partition/order plumbing as a
  first-class operation (`OperationType` addition).
- **Implicit:** deliver on all three layers (column primitives, eager
  `NivaraFrame`, lazy `QueryFrame`); explicit null-mask semantics (ADR-001
  boundary, no NaN); register non-parallelizable/non-streamable; docs
  (`docs/LINQ.md`, `CHANGELOG.md`, `docs/plan/POLARS-ROADMAP.md`); tests.
- **Reference semantics (roadmap §Phase 3 spec = Polars SQL):**
  - `RANK`: gaps (1,1,3)
  - `DENSE_RANK`: no gaps (1,1,2)
  - `ROW_NUMBER`: unique per partition (1,2,3)
  - `PERCENT_RANK`: (rank−1)/(partitionSize−1)
  - [Polars SQL window docs](https://docs.pola.rs/api/python/stable/reference/sql/functions/window.html)

## Locked decisions (human-confirmed)

- **Scope:** keep scoped to the rank family; #159 (expression composition)
  remains separate.
- **API:** flat window-style methods on `QueryFrame` + `NivaraFrameExtensions`;
  `Over(..)`/`WindowSpec` builder → follow-up issue.
- **Nulls:** null order key → null rank output; excluded from numbering and
  from the percent_rank denominator.
- **Result types:** `long` for row_number/rank/dense_rank (matches
  `CumulativeCount` → `long`, WindowFunctions.cs:69), `double` for percent_rank.
- **Order keys:** rank/dense_rank/percent_rank require ≥1 order key
  (SQL/Polars requirement); `RowNumber` allows empty orderBy (global
  sequential).

## Design

### Kernel

New `src/Nivara/Tensors/RankKernel.cs` (internal static). Input:
`IReadOnlyDictionary<string,IColumn>` source columns, partition columns,
order `SortKey[]`; output `NivaraColumn<long>` (row_number/rank/dense_rank)
or `NivaraColumn<double>` (percent_rank).

1. Partition rows by key columns (reuse `GroupByOperation.CreateGroupsInternal`
   / `GroupKey`, GroupByOperation.cs:78-120).
2. Per partition, stable-sort indices by order keys via `MultiColumnComparer`
   + `Enumerable.OrderBy` (stable; `Array.Sort` is introsort/unstable, so
   ties must not rely on sort order).
3. One pass over the sorted partition emits ranks; ties detected by
   `MultiColumnComparer.Compare == 0`.

### Operation

`RankOperation : IQueryOperation` (in `src/Nivara/Operations/WindowOperations.cs`
or new `RankOperations.cs`), holding `PartitionBy`, `OrderBy (SortKey[])`,
`ResultColumn`, `RankKind`. `TransformSchema` reuses comparable-type
validation (extract `SortOperation.IsComparableType` from `private` →
`internal`). New `OperationType.Rank = "Rank"`. Delivered window ops produce
no runtime plan nodes (matches existing pattern — operation + `OperationType`
only).

### Strategy registration

- `ParallelExecutionStrategy.isParallelizable`: `Rank => false`
  (ParallelExecutionStrategy.cs:11-30)
- `StreamingExecutionStrategy.NonStreamableOperations` += `OperationType.Rank`
  (StreamingExecutionStrategy.cs:8)

### Public API (flat, per decision)

- `QueryFrame.Rank(string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)`
- `QueryFrame.DenseRank(...)`, `QueryFrame.PercentRank(...)` same shape
- `QueryFrame.RowNumber(string resultColumn, string[]? partitionBy = null, IReadOnlyList<SortKey>? orderBy = null)`
- Same 4 on `NivaraFrame` in `WindowFrameExtensions.cs`.

## Blast radius

- **Files (new):** `src/Nivara/Tensors/RankKernel.cs`,
  `tests/Nivara.Tests/Tensors/RankFunctionsTests.cs`,
  `tests/Nivara.Tests/Tensors/RankFunctionsFrameTests.cs`,
  `tests/Nivara.Tests/Query/RankOperationTests.cs`
- **Files (modified):** `src/Nivara/Operations/WindowOperations.cs` (or new
  `RankOperations.cs`), `src/Nivara/Query/OperationType.cs`,
  `src/Nivara/Query/QueryFrame.cs`, `src/Nivara/WindowFrameExtensions.cs`,
  `src/Nivara/Execution/ParallelExecutionStrategy.cs`,
  `src/Nivara/Execution/StreamingExecutionStrategy.cs`,
  `src/Nivara/Operations/SortOperation.cs` (visibility only), docs
  (`docs/LINQ.md`, `CHANGELOG.md`, `docs/plan/POLARS-ROADMAP.md`).
- **Symbols:** new `RankOperation`, `RankKernel`, `OperationType.Rank`,
  4 × `QueryFrame` + 4 × `NivaraFrameExtensions` methods. Reused:
  `SortKey`, `MultiColumnComparer` (SortingHelper.cs:66), `SortOperation.IsComparableType`,
  `GroupByOperation.CreateGroupsInternal`.
- **Downstream callers:** `QueryFrame.Collect`/`CollectAsync` →
  `ExecutionEngine`/`QueryExecutor` (additive; no public API breaks).
- **Tests (regression guards, untouched):** `WindowOperationTests.cs`,
  `WindowFunctionsTests.cs`, `QueryNodeTests.cs`.

## Planned commits (one logical unit each)

1. `docs: plan issue-156 rank window functions in TODO.md`
2. `feat: add rank family window kernels (row_number/rank/dense_rank/percent_rank)` + tests
3. `feat: add RankOperation + OperationType.Rank + strategy registration`
4. `feat: add QueryFrame rank window API`
5. `feat: add NivaraFrame eager rank window API`
6. `docs: document rank window functions, update roadmap + changelog`
7. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx` after each step (fast, no confirmation needed).
- Ask the human before `dotnet test` / long-running verification.
- Targeted when tests are run: `RankFunctionsTests`, `RankOperationTests`,
  then `WindowOperationTests`/`WindowFunctionsTests`/`QueryNodeTests` as
  regression guards.

## Grounding

- **Code:** `WindowOperationBase`/`RollingOperation`/`CumulativeOperation`/
  `ShiftOperation` → WindowOperations.cs:11-255 · `MultiColumnComparer` →
  SortingHelper.cs:66 / SortOperation.cs:283-394 · `SortKey`/`SortDirection`/
  `NullOrdering` → SortOperation.cs:11-87 · `GroupKey`/`CreateGroupsInternal` →
  GroupByOperation.cs:78-120 · `isParallelizable` → ParallelExecutionStrategy.cs:11-30
  · `NonStreamableOperations` → StreamingExecutionStrategy.cs:8 ·
  `OperationType` → OperationType.cs · `CumulativeCount → long` →
  WindowFunctions.cs:69-94 · `QueryFrame` window API → QueryFrame.cs:563-720 ·
  eager extensions → WindowFrameExtensions.cs.
- **Docs:** [Array.Sort (introsort, unstable)](https://learn.microsoft.com/dotnet/api/system.array.sort)
  · [Comparer\<T\>.Default](https://learn.microsoft.com/dotnet/api/system.collections.generic.comparer-1.default)
  · [Enumerable.OrderBy (stable)](https://learn.microsoft.com/dotnet/api/system.linq.enumerable.orderby?view=net-10.0)
  · [Polars SQL window semantics](https://docs.pola.rs/api/python/stable/reference/sql/functions/window.html)
  · [POLARS-ROADMAP §Phase 3](docs/plan/POLARS-ROADMAP.md).

## GitHub issues log

- [ ] Follow-up to file — `Over(..)`/`WindowSpec` builder API (per human G2
      decision; file via `gh issue create` when execution confirms the need).

## Reminder

As each task executes, if you find deferred work or a concern (known
limitations, follow-ups, refactors) that is outside the current plan, create a
GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and
record its number in the GitHub issues log above — don't rely on memory or
wait until the end of the plan, as compaction during execution can lose
important items.
