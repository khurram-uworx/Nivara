# Plan: Issue #1061 — Metrics: reject non-finite regression input

## Problem
`docs/METRICS.md` (null-rule section) promises "a metric is never quietly NaN". Regression
(`src/Nivara/Metrics/RegressionMetrics.cs::summarize`) excludes null-mask positions only;
NaN/±∞ in a non-null cell flows into `TensorPrimitives` and makes every output NaN while
`Count` still counts the pair. Classification treats NaN labels as an opaque class, which
is deterministic under .NET's IComparable contract but undocumented.

## Proposed changes
1. Regression path rejects non-finite values up front:
   - fast path (`NivaraColumn<double>` + `TryGetSpan` both): vectorised scan
     `TensorPrimitives.IsFiniteAll<double>` then loop to name the offending row/column.
   - conversion path: `double.IsFinite(double.CreateChecked(...))` check per row.
   - throw `ArgumentException` naming row and column (`predicted`/`actual`).
2. Classification unchanged in code; documented as opaque.
3. `docs/METRICS.md`: updated sentence + two table rows; Regression section note.
4. Tests as in issue acceptance.

## Verification
- `dotnet build Nivara.slnx`
- targeted Metrics suite
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- snippet gate across Docs namespace

## Planned commits
1. docs: plan #1061 in TODO.md
2. fix: reject non-finite values in RegressionSummaryFrame
3. docs: clarify non-finite and NaN-label behaviour in METRICS.md
4. test: pin non-finite regression rejection and NaN classification label

## Blast radius
- Files: `src/Nivara/Metrics/RegressionMetrics.cs`, `docs/METRICS.md`,
  `tests/Nivara.Tests/Metrics/RegressionSummaryFrameTests.cs`,
  `tests/Nivara.Tests/Metrics/ConfusionMatrixFrameTests.cs` (or PerClass).
- Downstream: none in `src/`; metrics is only consumed by tests/samples.
- Diagnostics unchanged.

## Grounding
- ADR-001: AutoDiff non-null boundary (fail early).
- Microsoft Learn / .NET: `TensorPrimitives.IsFiniteAll<T>` (net-11.0-pp);
  `double.CompareTo` orders NaN before all other values.

## GitHub issues log
- (none created during this work so far)
