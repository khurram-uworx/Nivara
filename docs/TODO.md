# Plan: Issue #1063 — Metrics nullable-element inputs

## Problem

`NivaraMetrics` accepts `IColumn` values that realise as `NivaraColumn<T>`.
A caller can instead build a `NivaraColumn<int?>` / `NivaraColumn<double?>`
where null positions live inside the values rather than in a separate mask.
Today such columns reach:

- `ClassificationMetrics.ensureComparableLabels`, whose
  `IsComparable` is evaluated against the unwrapped `ElementType` (`int`),
  so the eventual failure is a misleading cast/generic error or the
  “must be comparable” message; and
- `RegressionMetrics.IsNumeric`, which keys on exact types and rejects
  `Nullable`1` with “is not numeric”.

Both exceptions are `ArgumentException`, but neither names the ergonomic
fix: `NivaraColumnFactory.CreateFromNullable<T>(...)`, which yields a
mask-backed `NivaraColumn<T>`.

## Proposed changes

1. Add `ensureNonNullableElementType(IColumn, string)` in `NivaraMetrics` and
   call it for `predicted` and `actual` at the start of `validatePair`.
   Detection checks both the runtime generic type argument (catches
   `NivaraColumn<int?>`) and `column.ElementType` (catches any IColumn whose
   `ElementType` is `Nullable<T>`). Throw `ArgumentException` naming the parameter
   and `CreateFromNullable`.
2. Documentation sentence in `docs/METRICS.md` (“The three operations”) naming
   the supported shape.
3. Tests in `tests/Nivara.Tests/Metrics/` asserting the message contains
   `CreateFromNullable` for both classification and regression.
4. Review `DocsCoveragePins` only if a doc file/citation count changes
   (expected: none, because only a sentence is added).

## Blast radius

- Files: `src/Nivara/Metrics/NivaraMetrics.cs`,
  `docs/METRICS.md`,
  `tests/Nivara.Tests/Metrics/*`.
- Symbols: `NivaraMetrics.validatePair`,
  private helper `ensureNonNullableElementType`,
  public `ConfusionMatrixFrame` / `PerClassMetricsFrame` / `RegressionSummaryFrame`.
- Existing tests: all `tests/Nivara.Tests/Metrics/*` (must stay green).
- Docs gate: `DocsCoveragePins` unchanged unless counts move.

## Planned commits

1. `docs: plan issue 1063 in TODO.md`
2. `fix(metrics): reject nullable-element inputs with CreateFromNullable guidance`
3. `docs(metrics): state supported column shape`
4. `test(metrics): assert CreateFromNullable guidance for nullable inputs`
5. (G2) any review-driven fixup commits
6. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx`
- targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Metrics"`
- docs gate (if `docs/` touched): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## GitHub issues log

(none yet)
