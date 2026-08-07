# TODO — Window functions: rolling, cumulative, shift/lag (#135)

Feature branch: `khurram/135`

## Problem

IDEA.md §13 Phase 2 lists window functions as a roadmap item. None exist
(`docs/plan/POLARS-ROADMAP.md` Phase 3). No `Window`/`Rolling`/`Cumulative`/
`Shift`/`Lag` symbols anywhere in `src/Nivara`. Requires integration with the
lazy query pipeline (`OperationType` + `QueryNode` + strategies) and explicit
null-mask semantics (ADR-001, no NaN semantics).

## Scope

Delivered at three layers with a **single, consistent per-aggregate method
shape** (no public enums, no core/convenience split):

1. Rolling window aggregates: sum / mean / min / max
2. Cumulative ops: sum / min / max / product / count
3. Shift / lag / lead
4. Lazy `QueryFrame` + eager `NivaraFrame` surfaces
5. Null-mask preservation (ignore-nulls + `minPeriods` rolling semantics, with
   optional `Func` null-replacement handler)

Out of scope (tracked separately): `Over(...)` partitioning, `Rank`/`DenseRank`
(POLARS-ROADMAP Phase 3 remainder), grouped windows.

## Design decisions (locked)

- **API shape**: per-aggregate methods, identical names/arg order/semantics on
  all layers. Typed on `NivaraColumn<T>`; runtime-dispatch on `NivaraFrame` and
  `QueryFrame` (identical signatures, no forced type args, matching the untyped
  `Filter`/`Sort`/`Standardize` precedent).
- **nullHandler**: parameterless `Func<T>?` (column) / `Func<object?>?`
  (frames), called once per null position. Default `null` = Polars-compatible
  ignore-nulls semantics. When set, nulls are replaced by the handler output and
  counted as valid.
- **RollingMean result type**: `NivaraColumn<double>` on all layers (matches
  existing `Mean() → double`).
- **Cumulative**: skip nulls; null positions stay null, value carries forward
  (Polars `skip_nulls=true`). `CumulativeCount` = non-null running count,
  returns `long`, type-agnostic.
- **Shift/Lead**: `output[i] = input[i ∓ periods]`; boundaries → null or
  `fillValue`. `Lead = Shift(-periods)`.
- **minPeriods** ∈ [1, windowSize], default = windowSize.
- **Result column naming**: `resultColumn` param; appended, throw
  `ArgumentException` on name collision (uniform with `NivaraFrame.WithColumn`).
- **Query layer**: `OperationType.Rolling` / `.Cumulative` / `.Shift` category
  constants; ops carry an internal aggregate kind. Whole-column ops are
  **non-parallelizable** and **non-streamable**.

## Canonical signatures

Column layer (`WindowFunctions` static class, namespace `Nivara.Tensors`,
extensions on `NivaraColumn<T>`):

```csharp
// T : struct, INumber<T>
NivaraColumn<T>     RollingSum(int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
NivaraColumn<double> RollingMean(int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
NivaraColumn<T>     RollingMin(int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
NivaraColumn<T>     RollingMax(int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
NivaraColumn<T>     CumulativeSum(Func<T>? nullHandler = null)
NivaraColumn<T>     CumulativeMax(Func<T>? nullHandler = null)
NivaraColumn<T>     CumulativeMin(Func<T>? nullHandler = null)
NivaraColumn<T>     CumulativeProduct(Func<T>? nullHandler = null)
NivaraColumn<long>  CumulativeCount()                    // type-agnostic
NivaraColumn<T>     Shift(int periods, T? fillValue = null)   // type-agnostic
NivaraColumn<T>     Lead(int periods, T? fillValue = null)    // type-agnostic
```

NivaraFrame (extensions in `NivaraFrameExtensions.cs`) and QueryFrame (instance
members in `QueryFrame.cs`) — identical shape:

```csharp
RollingSum(source, resultColumn, windowSize, minPeriods?, nullHandler?)
RollingMean(source, resultColumn, windowSize, minPeriods?, nullHandler?)
RollingMin(source, resultColumn, windowSize, minPeriods?, nullHandler?)
RollingMax(source, resultColumn, windowSize, minPeriods?, nullHandler?)
CumulativeSum(source, resultColumn, nullHandler?)
CumulativeMax / CumulativeMin / CumulativeProduct(source, resultColumn, nullHandler?)
CumulativeCount(source, resultColumn)
Shift(source, resultColumn, periods, fillValue?)
Lead(source, resultColumn, periods, fillValue?)
```

## Kernels (O(n))

- Cumulative: single sequential scan skipping nulls.
- Rolling sum/mean: prefix-sum + prefix-count arrays (window delta), gated by
  `minPeriods`. Note: no `TensorPrimitives.CumulativeSum` in 10.0.10 (verified),
  so scalar scans; generic `INumber<T>` arithmetic like `NivaraTensorExtensions`.
- Rolling min/max: monotonic deque over valid (non-null) indices in window.
- Shift: span copy with boundary fill.
- Results built via `NivaraColumn<T>.CreateFromSpans(values, nullMask)`.

## Planned commits

1. `docs: plan window functions (#135) in TODO.md`
2. `feat: add column window primitives (rolling/cumulative/shift)`
   — `src/Nivara/Tensors/WindowFunctions.cs`
3. `test: cover column window functions null-mask semantics`
   — `tests/Nivara.Tests/Tensors/WindowFunctionsTests.cs`
4. `feat: add eager NivaraFrame window extensions`
   — `src/Nivara/NivaraFrameExtensions.cs`
5. `test: cover NivaraFrame window extensions`
   — `tests/Nivara.Tests/Tensors/WindowFunctionsFrameTests.cs`
6. `feat: add window operations to the lazy query pipeline`
   — `OperationType`, `RollingOperation`/`CumulativeOperation`/`ShiftOperation`,
   `QueryFrame` members, strategy wiring, `QueryNode` + visitors
7. `test: cover window operations in the query pipeline`
   — `tests/Nivara.Tests/Query/WindowOperationTests.cs`, `QueryNodeTests`
8. `docs: document window functions and mark roadmap delivered`
   — `docs/LINQ.md`, `CHANGELOG.md`, `docs/plan/POLARS-ROADMAP.md`
9. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx` before each commit (ask before `dotnet test`).
- `dotnet test` for the full suite on the go-ahead; targeted tests for window
  files after Layers A/B/C.
- Property-style NUnit tests for null-mask propagation (AGENTS.md pattern).

## Follow-ups

- SIMD cumulative scan / block-scan when a BCL `CumulativeSum` lands (net11).
- Grouped/partitioned (`Over`) windows + `Rank`/`DenseRank` — POLARS-ROADMAP
  Phase 3 remainder, separate issue.
