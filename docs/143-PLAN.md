# 143-PLAN — interface-based Normalize/Standardize kernels

**Status:** draft for team review · **Tracks:** khurram-uworx/Nivara issues #143 (delivered), #144 (this work) · **Scope:** `src/Nivara` data-prep (`NivaraFrameExtensions`) + `src/Nivara/Tensors/TensorsHelper.cs` · **Branch (proposed):** `khurram/144`

This plan is the second act of issue #143. #143 (add `Standardize` z-score alias + promote data-prep to core) is delivered and merged. This plan captures design feedback on how the promoted code dispatches on column type, and extends it to the full generic-math numeric surface. It is written for discussion before any implementation starts.

---

## 0. Why

The promoted data-prep code hardcodes type checks:

- `src/Nivara/NivaraFrameExtensions.cs:1313-1320` — `IsNumericColumn` is `columnType == typeof(float) || columnType == typeof(double)`.
- `src/Nivara/NivaraFrameExtensions.cs:1325-1336` — `NormalizeColumn` is an explicit `typeof(float)` / `typeof(double)` switch.
- `src/Nivara/NivaraFrameExtensions.cs:1343-1393` — `NormalizeCore<T>` is constrained to `IFloatingPointIeee754<T>` and uses TensorPrimitives.

Problem statement (from design feedback): instead of `typeof` checks, the support predicate should be a generic-math interface check (`INumber<T>` / `IFloatingPointIeee754<T>` — the interfaces TensorPrimitives and generic math are built on). AutoDiff staying float-only (`TypeValidator<T> : IFloatingPointIeee754<T>`) is fine, but core data-prep should accept integer and decimal columns too, with two kernel families so both correctness and performance have a home.

This work resolves issue #144 ("Normalize/Standardize: support int, long, decimal, byte, short columns").

### Current behavior worth preserving

- Nulls are skipped from statistics and preserved in the result via the null mask.
- Zero-variance columns are returned unchanged.
- Explicitly requesting an unsupported column throws `NotSupportedException`.
- No-arg / empty-arg calls auto-select all supported numeric columns (fixed in #143: `columns is null || columns.Length == 0`).

---

## 1. Design decisions (agreed)

| Decision | Choice |
| --- | --- |
| Support predicate | Interface-based: schema type implements `INumber<>` (subsumes `IFloatingPointIeee754<>`). No `typeof(float)/typeof(double)` checks. |
| Scope of types | All `INumber<T>` primitives: `int`, `long`, `short`, `byte`, `uint`, `ushort`, `sbyte`, `nint`, `nuint`, `decimal`, `Half`, `float`, `double`. |
| Integer/decimal output type | Promote to `double` (`NivaraColumn<double>`). Z-scores are fractional; float/double behavior unchanged (backwards compatible). |
| Dispatch mechanism | Interface check + per-type cached generic dispatch (see §2.2). |
| Kernel split | Two families: TensorPrimitives (SIMD) for `IFloatingPointIeee754<T>`; math/LINQ/PLINQ in `double` for everything else. |

### Open for team discussion

- **PLINQ threshold.** Generic kernel uses `AsParallel()` / `Parallel.For` only above a length threshold (proposed 4096); LINQ below. Agree on the constant and whether it belongs next to the existing vectorization heuristic (`KernelSelector`, `Length >= vectorSize * 4`).
- **`Half` output.** `Half` is IEEE and passes through the SIMD path unchanged (output `Half`). Confirmed OK to support, or should `Half` also promote to `double` for precision? (Proposed: keep as-is — `Half` is already IEEE-754 and TensorPrimitives handles it; this matches `IFloatingPointIeee754<T>` semantics.)
- **`nint`/`nuint`.** Runtime size-dependent; storage already supports value types. Include, or exclude for predictability? (Proposed: include — they satisfy `INumber<>`; auto-select covers them.)
- **Error message.** `NotSupportedException` message should name the interface requirement (e.g. "only INumber<T> columns can be normalized"), not a closed type list. Confirm wording.

---

## 2. Implementation plan

### 2.1 Span kernels — `src/Nivara/Tensors/TensorsHelper.cs`

TensorsHelper is the repo's designated central kernel file ("the single file to check when upgrading to a new .NET version"), so the two kernels live there with BCL-replacement notes, matching the existing pattern.

1. `public static bool TryNormalizeInPlace<T>(Span<T> values) where T : struct, IFloatingPointIeee754<T>`
   - `TensorPrimitives.Average<T>` + `TensorPrimitives.StdDev<T>` (10.0.10 API names; net-11 renames `Mean`/`StdDev`).
   - `TensorPrimitives.Subtract` + `Divide` in place.
   - Returns `false` when `stdDev == T.Zero` (caller keeps the column); caller only replaces on `true`.
   - Note: `StdDev<T>` requires `IRootFunctions<T>` — satisfied by `IFloatingPointIeee754<T>`.

2. `public static void NormalizeToDouble<T>(ReadOnlySpan<T> values, Span<double> destination) where T : struct, INumber<T>`
   - Convert each element with `double.CreateChecked(v)` (generic-math conversion, valid because `INumber<T> : INumberBase<T>`).
   - Population stddev in double: `mean = sum/count`, `variance = max(0, sumSq/count - mean²)`, `stdDev = sqrt(variance)` (clamp tiny-negative from rounding).
   - `z = (double.CreateChecked(v) - mean) / stdDev`.
   - PLINQ (`AsParallel().Sum`) and `Parallel.For` transform only above the threshold from §1; LINQ below.
   - Note: TensorPrimitives is mathematically unavailable here — `INumber<T>` does not satisfy the `IRootFunctions<T>` constraint that `StdDev` requires, and integer arithmetic would truncate.

### 2.2 Dispatch — `src/Nivara/NivaraFrameExtensions.cs`

1. `IsNumericColumn` becomes an interface predicate: `type.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(INumber<>))`.
2. Two generic cores, both returning `IColumn`:
   - `NormalizeFloatCore<T>(NivaraColumn<T>) where T : struct, IFloatingPointIeee754<T>` → `NivaraColumn<T>` (SIMD path, null-pack + `TryNormalizeInPlace`).
   - `NormalizeIntegerCore<T>(NivaraColumn<T>) where T : struct, INumber<T>` → `NivaraColumn<double>` (generic path, null-pack + `NormalizeToDouble`).
   - Both preserve the null path: pack non-null values → stats → transform → scatter → `NivaraColumn.CreateFromSpans(values, nullMask)`.
3. `NormalizeColumn` dispatches through `ConcurrentDictionary<Type, Func<NivaraFrame, string, IColumn>>`:
   - First use per type: interface check selects the kernel family → `MakeGenericMethod` once → `Expression.Lambda(...).Compile()` a delegate wrapping `frame.GetColumn<T>(name)` + the core call.
   - Zero reflection per call after first use; no `Span` crosses `Invoke` (delegates take/return columns).
   - Unsupported type → `NotSupportedException`.

### 2.3 Tests — `tests/Nivara.Tests/NivaraDataPrepTests.cs`

NUnit 4, no `[TestCase]`, explicit `[Test]` per type:

- Per-type mean=0 / variance=1 for `int`, `long`, `short`, `byte`, `uint`, `ushort`, `sbyte`, `nint`, `decimal` (generic path) and `float`, `double`, `Half` (SIMD path).
- Integer/decimal output column is `double` (assert via `frame.Schema.GetColumnType`).
- Null-mask preservation per type.
- Zero-variance returns column unchanged (both families).
- Auto-select over a mixed `int` + `double` frame selects both.
- Unsupported (`bool`, `string`, `DateTime`) still throws `NotSupportedException`.
- Cross-family parity: identical data as `float` vs `int` promoted to `double` produce equivalent z-scores within tolerance.

### 2.4 Docs

- `CHANGELOG.md` (`[Unreleased]`): Normalize/Standardize accept any `INumber<T>` column; integer/decimal output promotes to `double`; interface-based kernel dispatch.
- `docs/TENSORS.md`, `CONTRIBUTING.md`: reflect extended numeric surface.
- This file `docs/143-PLAN.md`: after the work ships, mark status → implemented (or convert to ADR if the team wants a recorded decision).

---

## 3. Execution order (after team sign-off)

1. Branch `khurram/144` from `main`.
2. `docs/TODO.md` plan → commit.
3. Kernels in TensorsHelper → build `src/Nivara`.
4. Dispatch rework in NivaraFrameExtensions → build.
5. Tests → build both projects → full `dotnet test` (run only on approval per repo guidance).
6. Docs/CHANGELOG updates.
7. Push `khurram/144`, open PR, close #144 — all human-approved.

---

## 4. References

- Issue #143 — https://github.com/khurram-uworx/Nivara/issues/143 (closed)
- Issue #144 — https://github.com/khurram-uworx/Nivara/issues/144 (this work)
- PR #145 — https://github.com/khurram-uworx/Nivara/pull/145 (merged delivery of #143)
- `src/Nivara/NivaraFrameExtensions.cs` data-prep region (lines ~1258-1396)
- `src/Nivara/Tensors/TensorsHelper.cs` (central kernel pattern)
- `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` (`IFloatingPointIeee754<T>` — AutoDiff stays float-only)
