# Plan: Standardize alias + promote data-prep to core (#143)

## Problem

Issue #143 asks for a `Standardize` alias for the z-score `Normalize` frame data-prep helper.
`Normalize(this NivaraFrame, params string[]?)` currently lives in
`src/Nivara.Extensions/MLNet/MLNetExtensions.cs` alongside `TrainTestSplit`. Three quality gaps exist:

1. The `Standardize` alias is missing (the issue's literal ask).
2. `NormalizeColumn` reads `column[i]` at null positions (returns `default(T)`) and rebuilds via
   `NivaraColumn<T>.Create(...)`, which drops the null mask — the issue's promised "skipping null
   values" semantics do NOT hold today.
3. `IsNumericColumn` claims int/long/decimal/byte/short are numeric, but `NormalizeColumn` only
   implements float/double — so `Normalize()` (auto-select) crashes with `NotSupportedException`
   on frames containing int columns.

Also: the issue references `docs/plan/FUTURE.md`, which does not exist in the repo (stale reference).

## Decisions (confirmed with human)

- **Scope:** alias + null-skip fix. float/double only (no int/long normalization in this pass).
- **Placement:** promote the data-prep surface to core `src/Nivara/NivaraFrameExtensions.cs`
  (namespace `Nivara`), remove it from `MLNetExtensions`. Core stays dependency-free — the moved
  code uses only LINQ + Nivara primitives + `System.Numerics.Tensors` (already a core dependency).
- **TensorPrimitives (10.0.10) opportunistically:** `Average<T>` / `StdDev<T>` (population) for
  statistics, `Subtract` / `Divide` for the transform.
- **Deviation from the earlier plan note:** `IsNumericColumn` is narrowed to float/double.
  Explicitly naming an unsupported column now throws a clear `NotSupportedException`; auto-select
  skips non-float/double columns instead of crashing.

## Changes

### 1. Core — `src/Nivara/NivaraFrameExtensions.cs`

- Add `using System.Numerics;` and `using System.Numerics.Tensors;`.
- Public `Standardize(this NivaraFrame, params string[]? columns)` — delegates to `Normalize`.
- Public `Normalize(this NivaraFrame, params string[]? columns)` — moved from MLNet.
- Private `IsNumericColumn(NivaraFrame, string)` — float/double only.
- Private `NormalizeColumn(NivaraFrame, string)` — dispatches to generic `NormalizeCore<T>`.
- Private `NormalizeCore<T>(NivaraColumn<T>) where T : struct, IFloatingPointIeee754<T>`:
  - **No nulls** (`TryGetSpan`): `Average`/`StdDev` on the zero-copy span; if `stdDev == T.Zero`
    return the column unchanged; else `Subtract`/`Divide` into a new array and `Create(...)`.
  - **Nulls:** pack non-null values, `Average`/`StdDev` on the packed span, transform the packed
    span in place via `Subtract`/`Divide`, scatter back, rebuild with `NivaraColumn<T>.CreateFromSpans`
    preserving the null mask. All-null column → skipped stats, all-null output.

### 2. Extensions — `src/Nivara.Extensions/MLNet/MLNetExtensions.cs`

- Remove `Normalize`, `IsNumericColumn`, `NormalizeColumn` (moved to core).
- Keep `ConvertToFloat` (still used by `CreateFeatureMatrix`).

### 3. Tests — `tests/Nivara.Tests/NivaraDataPrepTests.cs` (new fixture)

- Move `Normalization_ProducesZeroMeanUnitVariance` from `MLNet/MLNetIntegrationTests.cs`.
- `Standardize_IsAliasForNormalize`
- `Standardize_SkipsNulls_PreservesNullMask`
- `Standardize_DefaultsToAllNumericColumns` (float normalized; string + int untouched)
- `Standardize_ZeroVariance_LeavesValuesUnchanged`
- `Standardize_NullFrame_Throws`
- `Normalize_ExplicitUnsupportedColumn_ThrowsNotSupported`
- `Normalize_AutoSelect_SkipsIntColumns` (no crash on frames with int columns)

### 4. Docs

- `docs/TENSORS.md` — drop #143 from the scoped-ambitions list (lines ~222, ~245).
- `CONTRIBUTING.md` — note data-prep (`Normalize`/`Standardize`) lives in core
  `NivaraFrameExtensions.cs`.
- `CHANGELOG.md` — add an `## [Unreleased]` entry describing the promotion + alias + null-skip fix.

## Verification

- `dotnet build Nivara.slnx`
- Run new + MLNet tests (ask human before `dotnet test`).

## Commits (planned)

1. `docs: plan Standardize alias (#143) in TODO.md`
2. `feat: add Normalize/Standardize data-prep to core NivaraFrameExtensions`
3. `refactor: remove Normalize from MLNetExtensions (moved to core)`
4. `test: add NivaraDataPrepTests + move Normalization test`
5. `docs: update TENSORS.md/CONTRIBUTING.md/CHANGELOG.md`
6. `docs: remove TODO.md — plan executed`

## Follow-ups (filed as issues)

- Type broadening: int/long/decimal/byte/short z-score normalization parity with `IsNumericColumn`
  (float/double only today).
