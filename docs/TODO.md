# TODO — #162 Over(...)/WindowSpec builder API for window functions

Branch: `khurram/162` (off `main`). GitHub repo: `khurram-uworx/Nivara`.

## Problem

Issue #156 delivered the rank-family window functions (`RowNumber`/`Rank`/`DenseRank`/`PercentRank`)
at three layers: column primitives (`RankKernel`), eager `NivaraFrame` extensions
(`WindowFrameExtensions.cs`), and the lazy `QueryFrame` pipeline (`RankOperation` +
`OperationType.Rank`). The API is flat per-method:

```csharp
QueryFrame Rank(string resultColumn, IReadOnlyList<SortKey> orderBy, params string[] partitionBy)
```

Issue #162 asks for an `Over(..)` / `WindowSpec` builder that captures a reusable
partition-by + order-by specification and passes it to every window method:

```csharp
var spec = frame.Over().PartitionBy("dept").OrderBy(new SortKey("score", SortDirection.Descending));
frame.AsQueryFrame().Rank("rnk", spec);
frame.AsQueryFrame().RowNumber("rn", spec);
```

Rolling/cumulative/shift currently have **no** partition/order support (single source column
only), so the spec is applied to them with full SQL-style partition + order semantics.

## Decisions (confirmed with the human)

- **Partitioned rolling/cumulative/shift**: full SQL-style implementation — partition rows,
  sort within each partition, compute the window per partition, scatter results back to
  original row order. Reuses `GroupByOperation.CreateGroupsInternal` +
  `MultiColumnComparer` + `ColumnFilterHelper.ReorderColumn`/`ConcatenateColumns`.
- **Named columns only**: `WindowSpec` carries string partition names + `SortKey` order keys.
  No expression-key variants (would require publicizing `SortExpressionKey`). The existing
  expression-based `Rank`/`DenseRank`/`PercentRank`/`RowNumber` overloads are untouched.

## Proposed changes

### 1. `WindowSpec` type — `src/Nivara/Operations/WindowSpec.cs` (new)

Public, immutable, fluent. `SortKey`/`SortDirection` live in `Nivara.Operations`, so the spec
lives there too.

```csharp
public sealed class WindowSpec
{
    public IReadOnlyList<string> PartitionBy { get; }   // empty = single partition
    public IReadOnlyList<SortKey> OrderBy { get; }      // empty = row order
    public bool IsEmpty { get; }                        // no partition, no order keys

    public WindowSpec PartitionBy(params string[] partitionBy);
    public WindowSpec OrderBy(params SortKey[] orderBy);
    public WindowSpec OrderBy(params string[] columns);          // ascending convenience
    public WindowSpec OrderBy(string column, SortDirection direction, NullOrdering nullOrdering = NullsLast);
}
```

- Fluent methods return a **new** spec (immutable; chain order is irrelevant).
- Validation: null/whitespace column names throw `ArgumentException`.

### 2. `Over()` entry points

- `NivaraFrameExtensions.Over(this NivaraFrame frame)` → `new WindowSpec()` (public, in
  `WindowFrameExtensions.cs`).
- `QueryFrame.Over()` instance method → `new WindowSpec()` (parity with the lazy layer).
- `new WindowSpec()` works standalone; the spec is frame-independent.

### 3. Shared partitioned-window engine — `src/Nivara/Tensors/PartitionedWindowEngine.cs` (new)

```csharp
internal static IColumn Compute(
    IReadOnlyDictionary<string, IColumn> columns,
    IColumn sourceColumn,
    WindowSpec spec,
    Func<IColumn, IColumn> partitionCompute)
```

1. Validate partition columns exist; order columns exist and are comparable
   (`SortOperation.IsComparableType`).
2. Build `sortedAll` = concatenation of each partition's rows, stable-sorted within partition
   via `MultiColumnComparer` (honors `SortKey.NullOrdering`).
3. `ColumnFilterHelper.ReorderColumn(sourceColumn, sortedAll)` → partitions contiguous/sorted.
4. Slice per partition (`NivaraColumn<T>.Slice`), run `partitionCompute`, combine via
   `ColumnFilterHelper.ConcatenateColumns`.
5. Scatter back: `ColumnFilterHelper.ReorderColumn(sortedResult, sortedAll)` (a permutation).

### 4. Eager `NivaraFrame` API — `WindowFrameExtensions.cs`

New overloads (spec required-positional to keep existing overloads unambiguous):

- `Rank`/`DenseRank`/`PercentRank(string resultColumn, WindowSpec spec)`
- `RowNumber(string resultColumn, WindowSpec spec)`
- `RollingSum/Mean/Min/Max(string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null)`
- `CumulativeSum/Max/Min/Product(string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null)`
- `CumulativeCount(string source, string resultColumn, WindowSpec spec)`
- `Shift`/`Lead(string source, string resultColumn, int periods, WindowSpec spec, object? fillValue = null)`

Rank family reuses `addRankColumn` + `RankKernel.Compute`. Rolling/cumulative/shift: when
`spec.IsEmpty` fast-path to the existing `addWindowColumn` (no behavior change); otherwise route
through `PartitionedWindowEngine.Compute` with `CalculateRolling`/`CalculateCumulative`/
`CalculateCumulativeCount`/`CalculateShift` delegates.

### 5. Lazy `QueryFrame`/operations API

- `WindowOperationBase` (`WindowOperations.cs`): optional `WindowSpec? Spec`; `TransformSchema`
  validates spec partition/order columns; `Execute` passes the full `input` through when a
  partitioned spec is present.
- `RollingOperation`/`CumulativeOperation`/`ShiftOperation`: new ctors accepting `WindowSpec`;
  partitioned compute via the shared engine.
- `RankOperation` (`RankOperations.cs`): ctor `(string resultColumn, RankKind kind, WindowSpec spec)`
  → forwards to the existing named-column ctor.
- `QueryFrame.cs`: `Over()` + mirror of the §4 overloads.

### 6. Semantics

- Partitioned rolling/cumulative/shift order null-order-key rows per `SortKey.NullOrdering`
  (default NULLS LAST) and **include** them in the window (SQL-faithful; `sortedAll` is a
  permutation). Diverges from the rank family (which nulls those rows) — intentional, rank
  behavior is unchanged.
- `spec.IsEmpty` → identical to today's behavior on every method.
- Rank family keeps its order-key-required validation (throws for empty `OrderBy` on
  Rank/DenseRank/PercentRank; RowNumber allows it).

### 7. Tests

- New `tests/Nivara.Tests/Query/WindowSpecTests.cs`: builder behavior, immutability, chain order,
  validation.
- Eager (`RankFunctionsFrameTests.cs`, `WindowFunctionsFrameTests.cs`): spec overloads match
  method-argument results; partitioned rolling/cumulative/shift vs known values + naive reference.
- Lazy (`RankOperationTests.cs`, `WindowOperationTests.cs`): `QueryFrame` spec overloads,
  partitioned behavior, `TransformSchema` validation.
- Parity: eager vs lazy produce identical results for the same spec.

### 8. Docs & changelog

- Update `docs/LINQ.md` window section with `Over()`/`WindowSpec`.
- **Create `docs/WINDOWS.md`** (fixes the broken reference at `docs/LINQ.md:197`) documenting the
  eager window surface incl. the builder.
- `CHANGELOG.md` entry.

## Blast radius

- New files: `WindowSpec.cs`, `PartitionedWindowEngine.cs`, `docs/WINDOWS.md`, test file.
- Modified: `WindowFrameExtensions.cs`, `QueryFrame.cs`, `WindowOperations.cs`, `RankOperations.cs`,
  `docs/LINQ.md`, `CHANGELOG.md`.
- No existing public signatures change (all new overloads are additive). `QueryFrame` is internal;
  its callers are the typed LINQ layer (`NivaraQuery<T>`), internal tests (InternalsVisibleTo), and
  `NivaraFrame.AsQueryFrame()`.
- Existing tests that cover the affected symbols: `RankOperationTests.cs`, `RankFunctionsTests.cs`,
  `RankFunctionsFrameTests.cs`, `WindowOperationTests.cs`, `WindowFunctionsTests.cs`,
  `WindowFunctionsFrameTests.cs`, `WindowExpressionTests.cs`, `WindowExpressionEvaluationTests.cs`,
  `WindowExpressionOperationTests.cs`, `WindowPlanVisitorTests.cs`. They stay green — no contracts change.

## Verification

- `dotnet build Nivara.slnx` after each step.
- `dotnet test` (requires human confirmation) on the window/rank fixtures plus the full suite at the end.

## Planned commits

1. `docs: plan #162 in TODO.md`
2. `feat: add WindowSpec builder and Over() entry points`
3. `feat: add partitioned window engine kernel`
4. `feat: add eager NivaraFrame WindowSpec overloads`
5. `feat: add lazy QueryFrame WindowSpec overloads`
6. `test: cover WindowSpec and partitioned windows`
7. `docs: document WindowSpec and eager window surface`
8. `docs: remove TODO.md — #162 plan executed`

## GitHub issues log

- [ ] none yet — created issues get recorded here with `gh issue create --repo khurram-uworx/Nivara`
