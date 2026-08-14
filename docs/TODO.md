# TODO — issue #231: NivaraSeries index, aggregates, indexer

Branch: `khurram/231`

## Problem

Issue #231 (`https://github.com/khurram-uworx/Nivara/issues/231`) lists three findings:

1. `NivaraSeries<T>.Index` is a `NivaraColumn<object>` holding boxed ints per position for the default index — wasteful allocation/boxing on the common path.
2. `NivaraSeries<T>` has no instance `Sum`/`Min`/`Max` — only `Average()`. The old instance members were removed in commit `11c8269` (AutoDiff refactor, Task 9); issue #231 reverses that.
3. `this[int]` (position) vs `this[object]` (label) indexer ambiguity — a boxed `int` routes to the label indexer.

## Decisions (human-confirmed)

1. **Index**: virtual positional default index — no boxed `NivaraColumn<object>` for default series; `Index` property stays `NivaraColumn<object>` but is lazily materialized only when accessed. Custom labels keep `NivaraColumn<object>`.
2. **Indexer**: remove `this[object]`; add `this[string]`. `series["a"]` keeps working; int/DateTime/other labels go through explicit `GetByLabel`.
3. **TopKDescending**: keep `(string? Label, T Score)[]`; map labels with `label?.ToString()` instead of `label is string s ? s : null`.

## Proposed changes

### 1. `src/Nivara/NivaraSeries.cs`

- Delete `createDefaultIndex` (line ~46); move body into private `materializeDefaultIndex(int length)` used only by `Index` property.
- Field `readonly NivaraColumn<object> index` → `readonly NivaraColumn<object>? index` (null = default positional). Add `NivaraColumn<object>? materializedDefaultIndex` cache.
- `Index` property (line ~298): `index ?? (materializedDefaultIndex ??= materializeDefaultIndex(Length))`.
- `findLabelPosition`: fast path — `label is int position && (uint)position < (uint)Length ? position : -1`; else existing loop.
- `getAlignedPairs`: virtual-aware — label = `index?[pos] ?? pos`, null-skip only for custom index; both sides.
- `GetLabel`: `index != null ? index[position] : position`.
- `Slice`: `index == null` → `new NivaraSeries<T>(slicedValues)`; else slice custom column.
- `Align`/`AlignBoth`: build `object[]` index only when `index != null`; default-series results stay virtual (`Create(values)`); empty-result branches → `Create(Array.Empty<T>())`.
- `Add`/`Multiply` empty branches → `Create(Array.Empty<T>())`.
- `Dispose`: also dispose `materializedDefaultIndex` when non-null.
- Add instance `Sum()`/`Min()`/`Max()` (after `Average()`, line ~653). Guard mirrors `Average()`: empty → throw; non-numeric (not in `TypeCompatibilityValidator.GetNumericTypes()`) → throw. Semantics match column reductions (`NivaraTensorExtensions`): Sum all-null → `default(T)`; Min/Max all-null → throw; empty → throw. Extract `List<T>? getValidValues()` helper from `averageVectorized`. Non-null path uses `values.AsSpan()` (zero-copy).
- Remove `this[object]` (line ~322); add `public T this[string label]` → `GetByLabel(label)`.
- `TopKDescending` (both overloads + heap helper): `label is string s ? s : null` → `label?.ToString()`.

### 2. `src/Nivara/Helpers/NumericKernelDispatcher.cs`

- Add `Min`, `Max` to `Operation` enum; add `public static T Min<T>(ReadOnlySpan<T>)` / `Max<T>(...)` entry points via `getArithmetic` with min/max messages; add the two `Operation` arms in `createArithmetic` → `new Func<ReadOnlySpan<U>, U>(NumericTensorKernels<U>.Min/Max)`.

### 3. Cleanup (demonstrates the unboxing win)

- `src/Nivara.Extensions/MLNet/TensorConversions.cs` (`FromBatchTensors`, ~line 152): replace boxed `Enumerable.Range(...).Cast<object>()` index with `NivaraSeries<T>.Create(values)` (virtual index).
- `samples/Nivara.SampleApp/AggregateExample.cs`: `series.Values.Sum()/Min()/Max()` → `series.Sum()/Min()/Max()`.

### 4. Tests

- `tests/Nivara.Tests/NivaraSeriesTests.cs`: `series3[(object)42]`→`GetByLabel(42)`, `series3[DateTime.Today]`→`GetByLabel(DateTime.Today)` (lines ~57-58); `series[(object)1]`→`GetByLabel(1)`, `series[(object)"string"]`→`series["string"]`, `series[(object)DateTime.Today]`→`GetByLabel(DateTime.Today)` (lines ~150-152). Add indexer-disambiguation regression test (`series[1]` = position; `series["1"]` on default-index series → `KeyNotFoundException`).
- `tests/Nivara.Tests/NivaraSeriesAggregateTests.cs`: update header doc ("NivaraSeries keeps Average; Sum/Min/Max live on NivaraColumn<T>"); add series-level Sum/Min/Max tests (ints/floats/doubles, null handling, empty/all-null throws, extended domain `Half`/`nint`/`nuint`/`Int128`/`UInt128`, non-numeric `string` throws, disposed throws).
- `tests/Nivara.Tests/Tensors/TensorInteropTests.cs`: `TopKDescending_WithPositionalIndex_ReturnsNullLabels` → expect `"0"`/`"1"` (stringified positions), rename accordingly.
- `tests/Nivara.Tests/MixedTypeIntegrationTests.cs`: no changes needed (uses `GetByLabel`/position indexer).

### 5. `CHANGELOG.md`

- `[Unreleased]` Added: instance `NivaraSeries<T>.Sum()/Min()/Max()`; virtual positional default index (`Index` lazily materialized). Breaking: `this[object]` label indexer removed in favor of `this[string]` + `GetByLabel`. Changed: `TopKDescending` stringifies non-string labels.

## Blast radius

- **`NivaraSeries.cs`**: core change. Public API: `Sum()`/`Min()`/`Max()` added; `this[object]` removed, `this[string]` added. `Index`/`GetLabel`/`GetByLabel`/`Slice`/`Align`/`Add`/`Multiply`/`TopKDescending` internal behavior. Only repo caller of `.Index` is `NivaraSeriesTests.cs:205`; no production code uses `.Index`/`GetByLabel`/`GetLabel`/label indexer.
- **`NumericKernelDispatcher.cs`**: additive (new `Min`/`Max`). Existing `Sum`/`DivideByCount` untouched.
- **`TensorConversions.cs`** (Extensions): `FromBatchTensors` label construction only; ML.NET interop tests cover.
- **`AggregateExample.cs`** (sample): `series.Values.X()` → `series.X()`.
- **Tests**: `NivaraSeriesTests.cs`, `NivaraSeriesAggregateTests.cs`, `TensorInteropTests.cs` (TopK). `MixedTypeIntegrationTests.cs` uses custom `NivaraColumn<object>` index ctor + `GetByLabel` — must stay green.
- Downstream consumers of the boxed label indexer: only tests/samples (verified via grep). String indexer callers keep working via new `this[string]`.

## Verification

- `dotnet build Nivara.slnx` after each code step (0 warnings).
- `dotnet test` — ask human before running.
- Grep guards: no `createDefaultIndex`, no `this[object]` on series, no `.Values.Sum()` in sample, only lazy materializer boxes ints.

## Planned commit list

1. `docs: plan issue #231 in TODO.md`
2. `feat: virtual positional default index for NivaraSeries (#231)` — NivaraSeries.cs index model
3. `feat: add NivaraSeries Sum/Min/Max instance aggregates (#231)` — NivaraSeries.cs + NumericKernelDispatcher.cs
4. `refactor: replace series object indexer with string indexer (#231)` — NivaraSeries.cs + test updates
5. `fix: stringify TopKDescending labels instead of nulling (#231)` — NivaraSeries.cs + TensorInteropTests.cs
6. `refactor: use virtual series index in TensorConversions and sample (#231)`
7. `test: add series-level Sum/Min/Max and indexer disambiguation tests (#231)`
8. `docs: changelog for #231 series improvements`
9. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] none yet — add entries here as issues are created during execution
