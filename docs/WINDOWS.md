# Window Functions

Nivara exposes window functions on `NivaraFrame` (eager) and `QueryFrame` (lazy). Both layers
share the same kernels, so results agree between eager and lazy paths.

Every window function returns a new `NivaraFrame` / `QueryFrame` with the result appended as a
new column; the input column is untouched.

## The `WindowSpec` builder

`Over()` starts a `WindowSpec`, which is the recommended way to describe partitioning and
ordering. The builder is immutable — each call returns a new spec.

```csharp
using Nivara.Operations; // WindowSpec, SortKey, SortDirection, NullOrdering

var spec = frame.Over()
    .PartitionBy("dept")              // one or more column names
    .OrderBy("salary");               // ascending, NULLS LAST by default
```

`OrderBy` has three overloads:

```csharp
spec.OrderBy(new SortKey("salary", SortDirection.Descending));
spec.OrderBy("salary", SortDirection.Descending, NullOrdering.NullsFirst);
spec.OrderBy("salary");                       // ascending, NULLS LAST
```

- Partition keys are optional; with no partition keys the whole frame is one partition.
- Order keys are optional for rolling / cumulative / shift / lead and `RowNumber`, and required
  for `Rank` / `DenseRank` / `PercentRank`.
- A spec with no keys at all (`new WindowSpec()`) is valid and matches the un-partitioned,
  row-order behavior of the plain overloads.
- Partition keys and order keys are validated up front: missing columns and non-comparable
  order-key columns throw `ArgumentException` (eager) or a wrapped
  `SchemaValidationException` (lazy `Collect()`).

## Semantics

A window function with a spec runs in four phases:

1. **Partition** — rows are grouped by the `PartitionBy` keys (stability preserved).
2. **Sort** — each partition is ordered by the `OrderKeys` (SQL `NULLS LAST` default).
3. **Compute** — the window kernel runs over the sorted partition (per-partition reset).
4. **Scatter** — results are mapped back to original row positions.

Order-key nulls participate as ordinary rows — with `NULLS LAST` the null-key rows sort to the
end of their partition and are not dropped from the result.

## Eager `NivaraFrame` surface

All eager members are extension methods on `NivaraFrame`. Every function has both a plain
overload and a `WindowSpec` overload; the plain overloads are equivalent to passing
`new WindowSpec()`.

### Rolling

| Member | Result type |
|--------|-------------|
| `RollingSum(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |
| `RollingMean(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | `double` |
| `RollingMin(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |
| `RollingMax(source, result, windowSize, spec?, minPeriods?, nullHandler?)` | source type |

The leading `windowSize - 1` rows of each partition yield null (or `minPeriods` if set).

### Cumulative

| Member | Result type |
|--------|-------------|
| `CumulativeSum(source, result, spec?, nullHandler?)` | source type |
| `CumulativeMax(source, result, spec?, nullHandler?)` | source type |
| `CumulativeMin(source, result, spec?, nullHandler?)` | source type |
| `CumulativeProduct(source, result, spec?, nullHandler?)` | source type |
| `CumulativeCount(source, result, spec?)` | `long` |

### Shift / Lead

`Shift(source, result, periods, spec?, fillValue?)` and `Lead(...)` move values within each
partition. Rows that would cross a partition boundary (or the frame edge) become null unless
`fillValue` is supplied.

### Rank family

| Member | Result type |
|--------|-------------|
| `RowNumber(result, spec?)` | `long` |
| `Rank(result, spec)` | `long` |
| `DenseRank(result, spec)` | `long` |
| `PercentRank(result, spec)` | `double` |

The rank family reuses the existing `RankKernel`, so a null order-key produces a null rank
result (unchanged semantics). `RowNumber` accepts a spec without order keys (partition order
within a partition is insertion order); the other rank functions require order keys.

## Lazy `QueryFrame` surface

`QueryFrame` mirrors the eager surface: `Over()`, plus the same overloads
(`RollingSum`, `CumulativeCount`, `Shift`, `Rank`, ...). Because the pipeline is lazy,
partition/order-key validation happens at `Collect()` time (thrown as
`QueryExecutionException` wrapping a `SchemaValidationException`), or immediately when
`Schema` is read.

## Example

```csharp
using var frame = new NivaraFrame(("dept", deptColumn), ("t", timeColumn), ("v", valueColumn));

var spec = frame.Over().PartitionBy("dept").OrderBy("t");

var eager = frame
    .RollingSum("v", "roll2", 2, spec)
    .CumulativeSum("v", "cum", spec)
    .Shift("v", "lag1", 1, spec)
    .Rank("rnk", spec);

var lazy = frame.AsQueryFrame()
    .RollingSum("v", "roll2", 2, spec)
    .CumulativeSum("v", "cum", spec)
    .Shift("v", "lag1", 1, spec)
    .Rank("rnk", spec)
    .Collect();
```

See also:

- `src/Nivara/Operations/WindowSpec.cs` — the builder.
- `src/Nivara/Tensors/PartitionedWindowEngine.cs` — the shared partition/sort/scatter kernel.
- `src/Nivara/WindowFrameExtensions.cs` — eager overloads.
- `src/Nivara/Query/QueryFrame.cs` — lazy overloads.
- `docs/LINQ.md` — window functions in the expression DSL (`ColumnExpressions`).
