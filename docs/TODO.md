# TODO — Normalize/Standardize full `INumber<T>` surface (issue #144)

**Branch:** `khurram/143` · **Tracks:** khurram-uworx/Nivara issue #144 (open) · **Plan source:** `docs/143-PLAN.md` · **Grounded via:** Microsoft Learn (`TensorPrimitives`, generic math)

## Problem

`NivaraFrameExtensions.Normalize` / `Standardize` (delivered in #143) only z-score `float`/`double`
columns. `IsNumericColumn` hardcodes `typeof(float) || typeof(double)` and `NormalizeColumn` is a
two-branch switch, so int/long/decimal/byte/short columns are silently skipped on auto-select and
throw `NotSupportedException` when named explicitly. Issue #144 asks for the full numeric surface.

## Decisions (grounded; resolves the plan's open questions)

- **Support predicate:** interface-based — schema type implements `INumber<>` (subsumes
  `IFloatingPointIeee754<>`; confirmed `IFloatingPointIeee754<T> : INumber<T>` on MS Learn).
  No `typeof(float)/typeof(double)` checks.
- **Scope:** all `INumber<T>` primitives. Per MS Learn, `INumber<>` is also implemented by `char`,
  `BigInteger`, `Int128`, `UInt128`, `NFloat` — these are **excluded** via an explicit blocklist
  (z-scoring a `char` column is surprising; `BigInteger` is not a storage-friendly scalar).
  `nint`/`nuint` stay included (plan proposal).
- **Integer/decimal output:** promote to `NivaraColumn<double>` (z-scores are fractional).
  float/double/Half behavior unchanged (backwards compatible).
- **Kernel split:** SIMD TensorPrimitives for `IFloatingPointIeee754<T>`; generic `INumber<T>`
  path converts to `double` then runs the same TensorPrimitives chain.
- **Integer path via `TensorPrimitives.ConvertChecked<T,double>`** (confirmed net-10 API,
  `TFrom/TTo : INumberBase<>`, computes `TTo.CreateChecked`). This replaces the plan's scalar
  `double.CreateChecked` loop + LINQ/PLINQ threshold entirely:
  - single vectorized conversion, then `Average<double>` + `StdDev<double>` + `Subtract`/`Divide`;
  - drops the fragile `sumSq/count − mean²` + clamp (BCL `StdDev<double>` is numerically stable);
  - identical population-σ semantics to the float path → exact cross-family parity.
- **`Half`:** stays on the SIMD path, output `Half` (confirmed Half implements
  `IFloatingPointIeee754` + `IRootFunctions`). Also fixes a latent bug — the generic core already
  supported `Half` but the dispatcher rejected it.
- **Error message:** `"Normalization for column '{name}' of type {type} is not supported. Only INumber<T> columns can be normalized."`

## Changes

### 1. Kernels — `src/Nivara/Tensors/TensorsHelper.cs`

New section with the repo's BCL-swap annotations:

1. `public static bool TryNormalizeInPlace<T>(Span<T> values) where T : struct, IFloatingPointIeee754<T>`
   - `Average<T>` + `StdDev<T>` (10.0.10 names; net-11 renames `Mean`/`StdDev`), then in-place
     `Subtract` + `Divide`.
   - Returns `false` when `stdDev == T.Zero` (caller keeps the column).
   - `StdDev<T>` requires `IRootFunctions<T>` — satisfied by `IFloatingPointIeee754<T>`.

2. `public static void NormalizeToDouble<T>(ReadOnlySpan<T> values, Span<double> destination) where T : struct, INumber<T>`
   - `TensorPrimitives.ConvertChecked<T, double>(values, destination)` (vectorized `CreateChecked`).
   - `mean = Average<double>`, `stdDev = StdDev<double>`; zero-variance → caller handles (returns
     unchanged) so this kernel writes the normalized values only.
   - `Subtract` + `Divide` on the `double` destination.
   - `INumber<T>` does NOT satisfy `IRootFunctions<T>` → `StdDev<T>` unavailable on the raw type,
     hence the convert-to-double step. Integer arithmetic would also truncate.

### 2. Dispatch — `src/Nivara/NivaraFrameExtensions.cs` (data-prep region)

- `IsNumericColumn`: `IsSupportedNumericType(type)` = implements `INumber<>` AND not in
  exclusion set (`char`, `BigInteger`, `Int128`, `UInt128`).
- Two generic cores returning `IColumn`:
  - `NormalizeFloatCore<T>(NivaraColumn<T>) where T : struct, IFloatingPointIeee754<T>` → `NivaraColumn<T>`:
    null-free `TryGetSpan` → `TryNormalizeInPlace`; else pack → normalize packed → scatter →
    `CreateFromSpans`.
  - `NormalizeIntegerCore<T>(NivaraColumn<T>) where T : struct, INumber<T>` → `NivaraColumn<double>`:
    same null path with `NormalizeToDouble`.
- `NormalizeColumn` dispatches via `ConcurrentDictionary<Type, Func<NivaraFrame, string, IColumn>>`:
  first use does the interface check + `MakeGenericMethod` + `Expression.Lambda(...).Compile()` once;
  zero reflection after. Unsupported → `NotSupportedException`.

### 3. Tests — `tests/Nivara.Tests/NivaraDataPrepTests.cs`

- Per-type mean=0 / variance=1 for `int`, `long`, `short`, `byte`, `uint`, `ushort`, `sbyte`,
  `decimal` (generic) and `float`, `double` (SIMD).
- Integer/decimal output column type is `double` (via `frame.Schema.GetColumnType`).
- Null-mask preservation per type; zero-variance unchanged (both families).
- Mixed `int` + `double` auto-select normalizes both.
- Unsupported (`bool`, `string`, `DateTime`, `char`) still throws `NotSupportedException`.
- Cross-family parity: same data as `float` vs `int` → equivalent z-scores.
- **Invert existing contract:** `Standardize_DefaultsToAllSupportedNumericColumns` (int now
  normalized to double), `Normalize_ExplicitUnsupportedColumn_ThrowsNotSupported` (int now
  supported), `Normalize_AutoSelect_SkipsUnsupportedNumericColumns` (int now selected).

### 4. Docs

- `CHANGELOG.md` `[Unreleased]`: full `INumber<T>` surface, promote-to-double, interface dispatch.
- `docs/TENSORS.md`: extended numeric surface note.
- `docs/143-PLAN.md`: mark status → implemented; update branch reference.

## Verification

- `dotnet build Nivara.slnx` after each code change.
- Full `dotnet test` — only on explicit human approval.

## Commit plan

1. `docs: plan Normalize/Standardize full INumber surface (#144) in TODO.md` (includes `docs/143-PLAN.md`)
2. `feat: add TryNormalizeInPlace and NormalizeToDouble kernels to TensorsHelper`
3. `feat: interface-based Normalize/Standardize dispatch for all INumber columns`
4. `test: cover per-type Normalize/Standardize and invert int-skip expectations`
5. `docs: document full numeric surface for Normalize/Standardize (#144)`
6. `docs: remove TODO.md — plan executed` then offer push + PR

## Follow-ups

- `nint`/`nuint` columns: covered by the generic path via `INumber<>`; confirm `GetColumn<nint>`
  works end-to-end in the test suite.
- `NFloat`: satisfies `INumber<>`; not explicitly tested (storage support is value-type based).
