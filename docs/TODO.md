# Plan: frame-scoped metrics module (#868)

## Problem

Issue #868 asks for a minimal, frame-producing metrics module under `src/Nivara/Metrics/`:
a confusion-matrix frame, a per-class accuracy frame and a regression summary frame,
computed over predicted/target columns. It is explicitly **not** an evaluation service:
pipeline/feature-level evaluation stays ML.NET's, and `docs/TENSORS.md` forbids a new math
library. Nothing in `src/` implements any of this today; the only metrics are ad-hoc
`correct / total` counters inside samples.

## Decisions cleared by the human before execution

1. Public surface: static `Nivara.Metrics.NivaraMetrics` methods taking `IColumn` pairs.
2. Confusion matrix carries `Correct` (diagonal) and `Support` (predicted-row total) columns.
3. Regression summary is one wide row; residuals are out of scope for this issue.
4. Undefined metrics are null; zero valid pairs throws; constant-target R² is 1.0 when
   perfect and 0.0 otherwise.
5. Classification counting is a single typed hash pass recorded as `KernelType.Scalar`;
   the issue's "vectorized" criterion is satisfied by regression and documented as such.

## Proposed changes

### New: `src/Nivara/Metrics/NivaraMetrics.cs`

```csharp
namespace Nivara.Metrics;

public static class NivaraMetrics
{
    public static NivaraFrame ConfusionMatrixFrame(IColumn predicted, IColumn actual);
    public static NivaraFrame PerClassAccuracyFrame(IColumn predicted, IColumn actual);
    public static NivaraFrame RegressionSummaryFrame(IColumn predicted, IColumn actual);
}
```

Shared validation: null arguments, equal lengths, and `IColumn` realising as
`NivaraColumn<T>` (the only in-repo implementation, and what `Slice` returns). Classification
additionally requires identical `ElementType` and a comparable label type. Regression accepts
two different numeric element types. Each call records one `OperationMeasurement` with valid and
excluded pair counts in `notes`.

### New: `src/Nivara/Metrics/ClassificationMetrics.cs`

`ClassificationKernel<T>` entered through a cached `MakeGenericMethod` dispatch keyed on
`ElementType`, reusing `GroupKeyReaderFactory`/`IGroupKeyReader` so hashing and equality do not
box per row (the `string` reader already reuses `TryGetStringHashCodes`).

- Valid pair = `!predicted.IsNull(i) && !actual.IsNull(i)`.
- Class universe comes from valid pairs only, so per-class denominators agree with the matrix.
- Order: `StringComparer.Ordinal` for `string`, `Comparer<T>.Default` otherwise.
- Matrix columns: `Predicted` (`T`), `Actual:{label}` (`long`) per class, `Correct` (`long`),
  `Support` (`long`). The `Actual:` prefix keeps labels named `Correct`/`Support`/`Predicted`
  from colliding; labels that collide case-insensitively throw.
- Per-class columns: `Class` (`T`), `Tp`, `Fp`, `Fn` (`long`), `Precision`, `Recall`, `F1`
  (nullable `double`). `tp = diagonal`, `fp = support - tp`, `fn = actualTotal - tp`.
  `Precision` is null when the class is never predicted, `Recall` when never actual, `F1`
  when either is null or both are zero.

### New: `src/Nivara/Metrics/RegressionMetrics.cs`

`RegressionKernel<T>` converts each column to `double[]` once (span path when the column is
already `double` and null-free), compacts valid pairs when nulls exist, then runs the whole
calculation through `TensorPrimitives` over a pooled residual buffer:

- `Subtract` -> `SumOfMagnitudes` for MAE, `SumOfSquares` for MSE, `Sqrt` for RMSE.
- `Average` -> `Subtract` -> `SumOfSquares` for the R² denominator, reusing the residual buffer.
- Accumulate in `double` deliberately: `float` accumulation over large n would not match
  hand-computed references.

### New tests: `tests/Nivara.Tests/Metrics/`

- `ConfusionMatrixFrameTests` — hand-computed 2-class and 90/10 imbalance; ordinal order;
  labels named `Correct`/`Predicted`/`Support`; case-collision throw; strings.
- `PerClassAccuracyFrameTests` — hand-computed 3-class; null `Precision`/`Recall`/`F1`;
  tp/support consistency with the matrix.
- `RegressionSummaryFrameTests` — perfect prediction; hand-computed case; constant target
  perfect and imperfect; single valid pair; mixed numeric types; zero valid pairs throws.
- `MetricsNullHandlingTests` — property-style: cells + `Correct` + excluded == total;
  per-class consistency; regression `Count == total - excluded`.
- `MetricsDiagnosticsTests` — `KernelType.Scalar` for classification, `Vectorized` for
  regression, with notes; enable/clear/disable around each test because the tracker is global.

Exactness: dyadic fixtures assert equality; tolerance only for `Rmse`/`R2`.

### Docs and gates

- `docs/METRICS.md` — positioning (frame-scoped vs ML.NET), signatures, null-exclusion rule,
  undefined-metric table, vectorization honesty table, O(k²) confusion-matrix warning, and a
  prose `TrainingLoop` note (`OnEpochEnd` is already `protected virtual`, so no source change).
  One gated csharp block; the `TrainingLoop` example is an inline code span to keep the
  excluded/ungated pins still.
- `tests/Nivara.Tests/Docs/DocSnippetCompiler.cs` — add `using Nivara.Metrics;` to `BaseUsings`.
- `tests/Nivara.Tests/Docs/DocumentationSnippetTests.cs` — add `docs/METRICS.md` to
  `GatedDocuments`; gated 111 -> 112, all 259 -> 260; ungated and excluded unchanged.
- `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` — `ExpectedDocumentCount` 56 -> 57,
  `ExpectedCitedDocumentCount` 17 -> 18, `ExpectedCitationCount` / `ExpectedCodeCitationCount`
  + n symbol citations, `ExpectedDocumentReferenceCount` + m doc links. Each pin's arithmetic
  comment is extended rather than the bare number bumped.
- `README.md`, `AGENTS.md`, `docs/plan/AISTACK-ROADMAP.md` (Phase 4 status, including
  correcting the stale claim that `TrainingLoop` needs a hook), `docs/TENSORS.md` (one
  paragraph drawing the Nivara-vs-ML.NET evaluation boundary).

## Semantics (normative)

| Case | Result |
| --- | --- |
| Null in either column at row i | Pair excluded from every denominator |
| Classification label only in excluded rows | Not a class |
| `tp + fp == 0` | `Precision` null |
| `tp + fn == 0` | `Recall` null |
| `Precision`/`Recall` null, or both 0 | `F1` null |
| Zero valid pairs (any function) | `ArgumentException` |
| Constant target, `mse == 0` | `R2 = 1.0` |
| Constant target, `mse != 0` | `R2 = 0.0` |
| Labels unequal only by case | `ArgumentException` |
| Regression element type not numeric | `ArgumentException` |
| Predicted/actual length mismatch | `ArgumentException` |

## Blast radius

- **New code only**: three files under `src/Nivara/Metrics/`; nothing existing changes
  behaviour. `NivaraFrame`, `NivaraColumn<T>`, `IColumn`, `GroupKeyReaderFactory`,
  `DiagnosticsTracker` and `TensorPrimitives` are used as-is.
- **Diagnostics**: new operation names (`Metrics.ConfusionMatrix`, `Metrics.PerClassAccuracy`,
  `Metrics.RegressionSummary`) appear in the global tracker when enabled. No existing
  operation name changes.
- **Docs gates**: one new reader-facing document, one new gated block, new citations and
  references; the `DocsCoveragePins` constants move. `SlowTestCategorisationTests` is
  unaffected because no new test measures wall clock or allocations.
- **No change** to `TrainingLoop`, AutoDiff, IO, or any sample.

## Verification

```
dotnet build Nivara.slnx
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Metrics"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
```

The Docs namespace run is required because `docs/` changes. No full suite, no benchmarks.

## Planned commits

1. `docs: plan frame-scoped metrics (#868) in TODO.md`
2. `feat(metrics): add the frame-scoped metrics module` — one commit, not two: the public
   `NivaraMetrics` surface names all three entry points, so splitting classification from
   regression would leave a commit that does not compile.
3. `test(metrics): pin classification, regression, null handling and diagnostics`
4. `docs(metrics): add METRICS.md and the ML.NET evaluation boundary`
5. `docs: register the metrics module in README, AGENTS and roadmap`
6. `docs: update documentation coverage pins for METRICS.md`
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- None yet. Any deferred work or concern found during execution is filed immediately and
  recorded here.
