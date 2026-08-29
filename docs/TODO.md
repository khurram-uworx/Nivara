# Plan: remove per-element `object?` boxing from group-by aggregations

## Problem

An exhaustive boxing audit of the columnar engine (after Phases 1–4 of `docs/plan/POLARS-ROADMAP.md`)
found that the **query expression hot path** (Filter/Select/OrderBy/Sort), **window functions**,
**group-by key hashing** (`GroupKeyReader<T>`), and **sort comparers** (`SortOperation.compareTyped<T>`)
are all boxing-free (typed `T` paths). The one genuine stray per-element boxing is the **group-by
aggregation family**: `Sum`, `Min`, `Max`, `Mean`, `StdDev`, `Variance` each build a `List<object>` via
`column.GetValue(index)` per element (and `Quantile`/`Median` were already modernized to the boxing-free
`QuantileKernel.TypedQuantile<T>`).

This is outside the strict Phase 1–4 acceptance criteria (which govern the query expression hot path), but
it is the only remaining per-element boxing anywhere in the engine and the user wants it gone before
signing off Phases 1–4.

## Approach

Mirror `QuantileKernel`'s already-shipped pattern: a typed extraction that reads via the generic
`IColumn<T>` indexer (returns `T`, no boxing) and skips nulls, feeding the existing box-free kernels
(`TensorPrimitives`, `MomentsKernel.Variance/StdDev(ReadOnlySpan<double>)`).

Add one shared helper in `AggregationFunction.cs`:

```csharp
static T[] ExtractValidTyped<T>(IColumn column, IReadOnlyList<int> groupIndices)
    where T : struct
{
    var typed = (IColumn<T>)column;
    int count = 0;
    foreach (var idx in groupIndices) if (!column.IsNull(idx)) count++;
    var values = new T[count];
    int pos = 0;
    foreach (var idx in groupIndices) if (!column.IsNull(idx)) values[pos++] = typed[idx];
    return values;
}
```

### Changes per aggregation (results identical to today, only allocation changes)

- **SumAggregation** — `SumVectorized<TSource,TResult>(List<object>)` →
  `SumVectorized<TSource,TResult>(IColumn, IReadOnlyList<int>)` using `ExtractValidTyped<TSource>` +
  `TResult.CreateChecked`. `SumVectorizedBool<TResult>` and `SumScalarDecimal` gain the same typed
  signature. Drop `ExtractValidValues` + `GetZeroValue` (empty group → `TensorPrimitives.Sum` of an
  empty span = promoted zero, identical to `GetZeroValue`).
- **MinAggregation / MaxAggregation** — inline `List<object>` + `MinVectorized<T>(List<object>)` →
  `MinVectorized<T>(IColumn, IReadOnlyList<int>)` (`TensorPrimitives.Min/Max`, no boxing) for the
  `INumber<T>` types; add `nint`/`nuint`/`Int128`/`UInt128`/`Half`/`BFloat16` to the typed switch
  (all `INumber<T>`, identical comparison results); add `MinScalar<T>(IColumn, ...)` using
  `Comparer<T>.Default` for `char`/`bool`; keep a **boxed** fallback only for genuinely non-numeric
  types (`string`, `DateTime`, custom structs) — preserving today's behavior for those.
- **MeanAggregation** — replace its own `List<object>` count with a non-boxed `column.IsNull(idx)`
  loop; reuse `SumAggregation.Apply` (now box-free internally) for the widened sum, then
  `ToDouble(sum) / count`.
- **StdDevAggregation / VarianceAggregation** — add `MomentsKernel.ComputeFromColumn(IColumn,
  IReadOnlyList<int>, int ddof, bool variance)` (17-type switch → `double[]` via `double.CreateChecked`,
  then `MomentsKernel.Variance/StdDev(ReadOnlySpan<double>)`). Remove the now-unused
  `ComputeStdDevFromBoxed` / `ComputeVarianceFromBoxed` from `MomentsKernel`. Remove the duplicate
  `ExtractValidValues` copies in `StdDev`/`Variance`.

## Blast radius

- `src/Nivara/Operations/AggregationFunction.cs` — `SumAggregation`, `MinAggregation`, `MaxAggregation`,
  `MeanAggregation`, `StdDevAggregation`, `VarianceAggregation` (all private helper signatures change;
  no public API change — `Apply(IColumn, IReadOnlyList<int>)` signature unchanged).
- `src/Nivara/Helpers/MomentsKernel.cs` — add `ComputeFromColumn`; remove `ComputeStdDevFromBoxed` /
  `ComputeVarianceFromBoxed` (only referenced by the two aggregations above; verify no other refs).
- `QuantileKernel.ComputeFromBoxed` + `ToDouble` stay (still used by `QuantileKernel.ComputeFromColumn`
  fallback for unknown element types).
- Tests: `tests/Nivara.Tests/Operations/*Aggregation*Tests.cs` (existing suite is the consistency guardrail).

## Verification

- `dotnet build src/Nivara/Nivara.csproj` after each commit (compile check).
- `dotnet test` on the aggregation test suite before final sign-off (requires human confirmation per
  AGENTS.md — ask before running).
- Behavior is unchanged: results are bit-identical; only the per-element `List<object>` allocation is
  removed. Existing aggregation property/equivalence tests must stay green.

## Planned commit list

1. `docs: plan aggregation boxing removal in TODO.md`
2. `docs: fix roadmap doc-map drift for renamed NivaraLinqExtensions → NivaraQuery`
3. `refactor(aggregation): make Sum box-free via typed extraction`
4. `refactor(aggregation): make Min/Max box-free for numeric/char/bool`
5. `refactor(aggregation): make Mean box-free (reuse typed Sum)`
6. `refactor(aggregation): make StdDev/Variance box-free via MomentsKernel.ComputeFromColumn; drop dead boxed helpers`

## GitHub issues log

- [x] #343 — StreamingWindowProcessor.AddConstant boxes every element on the streaming lookahead-window (Lead / negative Shift) correction branch. Genuine stray per-element boxing on a window path, outside the aggregation-family scope; tracked for a follow-up typed dispatch.
- Min/Max keep a per-element boxed fallback **by design** for genuinely non-numeric element types (string/DateTime/custom structs); numeric/char/bool are box-free. Not filed as an issue — intentional.
