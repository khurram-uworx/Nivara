# Plan: issue #1066 — cross-check Nivara.Metrics against ML.NET evaluation

## Problem

`docs/METRICS.md` asserts a boundary (Nivara owns frame-producing metrics, ML.NET owns
evaluation) that is only prose. No test verifies the two agree. Need one deterministic,
asymmetric three-class fixture comparing Nivara's confusion-matrix cells and
`MicroAccuracy` with ML.NET's `MulticlassClassificationMetrics`.

## Proposed changes

1. Add `tests/Nivara.Tests/MLNet/MLNetMetricsCrossCheckTests.cs`:
   - Deterministic 3-class, ~192-row fixture with deliberately asymmetric confusion:
     `[[60,10,4],[3,45,8],[1,6,55]]` (rows = predicted, cols = actual).
   - `NivaraColumn<int>` predicted/actual.
   - Nivara: `NivaraMetrics.ConfusionMatrixFrame`.
   - ML.NET: build `IDataView` directly with `Label` (float actual),
     `PredictedLabel` (float predicted), and `[VectorType(3)] Score` one-hot;
     call `ctx.MulticlassClassification.Evaluate(dv, "Label", "Score", "PredictedLabel")`.
   - Assert every cell count exact:
     `Nivara Actual:{a}[p] == (long)ml.ConfusionMatrix.Counts[a][p]`.
   - Assert `ml.MicroAccuracy == sum(Correct)/sum(Support)` within `1e-12`.
2. Add a short "Cross-check against ML.NET" note in `docs/METRICS.md`
   referencing the test method.

## Blast radius

- New test file only; no production code.
- ML.NET Evaluate lands in unmarked test (first-use Evaluate typically <2s).
- `docs/METRICS.md` prose edit only; no citation/coverage pin changes.
- Depends on branch `khurram/1064` (metrics accumulators) for base.

## Planned commits

1. `docs: plan 1066 cross-check in TODO.md`
2. `test(metrics): cross-check confusion matrix and micro-accuracy against ML.NET`
3. `docs(metrics): reference ML.NET cross-check test`
4. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx` (Release)
- targeted: `--filter "FullyQualifiedName~MLNetMetricsCrossCheckTests"`
- Docs gate: `--filter "FullyQualifiedName~Nivara.Tests.Docs"` after doc edit
- Review `DocsCoveragePins` unchanged (no new citations)

## GitHub issues log

- (none yet)
