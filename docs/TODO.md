# Plan: Close #233 (ML.NET surface) + #235 (hygiene) in one PR

Branch: `khurram/issues` (off `main`). Single PR, closes both low-severity
code-review hygiene issues (REVIEW findings #12 and #14). Verified current
state of every touched line against `main` (2026-08-15); no production callers
exist for any API being changed.

## Problem

**#233 — ML.NET surface**
- Public sample POCOs (`TwoColumnData`, `TwoColumnFeatureData`, `ThreeColumnData`,
  `GenericData`) leak test scaffolding into the library namespace
  (`src/Nivara.Extensions/MLNet/MLNetInterop.cs:387-409`).
- `ModelIntegration.TrainAndEvaluate` returns `(ITransformer Model, object Metrics)`
  — untyped metrics (`ModelIntegration.cs:90`).
- `TensorConversions` has Array-based `ReshapeToArray` / `FlattenFromTensor(Array)`
  duplicating core `TensorInteropExtensions` typed `ReshapeToTensor` /
  `FlattenFromTensor(Tensor<T>)` (`TensorConversions.cs:252`, `:290`).
- Argument-order trap: static `MLNetInterop.ToNivaraFrame(IDataView, MLContext)`
  vs extension `MLNetExtensions.ToNivaraFrame(this MLContext, IDataView)`.

**#235 — Hygiene**
- `public` members on `internal` classes: `ColumnStorageFactory` (5 statics),
  `TensorsHelper` (10 statics), `RankKernel.Compute`.
- `TrainingResult.PrintSummary()` / `DataParallelTrainingResult.PrintSummary()`
  write to `Console` directly.
- `JsonOptions.SchemaInferenceRecords` vs `CsvOptions.SchemaInferenceRows` naming mismatch.

## Grounding

- ML.NET `LoadFromEnumerable<TRow>` requires a user-defined row type; there is no
  public dynamic-schema path. Therefore the four POCOs must remain somewhere in the
  assembly (internal) for `ToDataView` — they cannot be deleted entirely. Confirmed
  via microsoft-learn docs (`DataOperationsCatalog.LoadFromEnumerable`).
- ML.NET evaluation metrics are task-specific types (`BinaryClassificationMetrics`,
  `MulticlassClassificationMetrics`, `RegressionMetrics`) deriving from
  `ModelMetricsBase` with no useful shared members; a task-kind wrapper record is the
  clean strong-typing approach (microsoft-learn "Evaluate your ML.NET model with metrics").

## Proposed changes

### Part 1 — #233 (`src/Nivara.Extensions/MLNet/`)

1. **Scope sample POCOs to `internal`** (`MLNetInterop.cs:387-409`).
   Tests keep access via Extensions→`Nivara.Tests` `InternalsVisibleTo`.
2. **Strongly-type `TrainAndEvaluate` metrics** (`ModelIntegration.cs`):
   - `public enum ModelTaskKind { BinaryClassification, MulticlassClassification, Regression }`
   - `public sealed record ModelEvaluationResult(ModelTaskKind Kind, BinaryClassificationMetrics? Binary, MulticlassClassificationMetrics? Multiclass, RegressionMetrics? Regression)` — exactly one non-null per detected pipeline.
   - `TrainAndEvaluate` returns `(ITransformer Model, ModelEvaluationResult Metrics)`.
   - `EvaluateModel` returns `ModelEvaluationResult`.
3. **Remove Array-based conversions** (`TensorConversions.cs`): delete `ReshapeToArray`
   and `FlattenFromTensor(Array)`; update `MLNetIntegrationTests.cs` reshape/flatten tests
   to core `TensorInteropExtensions.ReshapeToTensor` / `FlattenFromTensor(Tensor<T>)`.
4. **Remove flipped `ToNivaraFrame`**: delete public static
   `MLNetInterop.ToNivaraFrame(IDataView, MLContext)`; make `ConvertFromDataView`
   `internal`; re-point `MLNetExtensions` (`ToNivaraFrame`/`Transform`/`Predict`) to it;
   update the 8 test call sites to `mlContext.ToNivaraFrame(dataView)`.

### Part 2 — #235

5. **`public` → `internal` on internal-class members**:
   - `ColumnStorageFactory.cs`: `Create<T>(ReadOnlySpan<T>)`, `Create<T>(ReadOnlySpan<T>, ReadOnlyMemory<bool>?)`, `Create<T>(ReadOnlySpan<T?>)`, `IsVectorizable<T>()`, `IsVectorizable(Type)`.
   - `TensorsHelper.cs`: all 10 public statics (`Transpose` ×2, `Multiply` ×3, `MultiplyCore`, `RowDot`, `RowCosineSimilarity`, `TryNormalizeInPlace`, `TryNormalizeToDouble`).
   - `RankKernel.cs`: `Compute`. **Keep `RankKind` public** (used by public rank API).
   Semantically a no-op (classes already internal; all callers are assembly friends).
6. **`PrintSummary(TextWriter? writer = null)`** in `TrainingLoop.cs` and
   `DataParallelResult.cs`, writing to `writer ?? Console.Out`; extend the two
   `PrintSummary_DoesNotThrow` tests to assert output via a `StringWriter`.
7. **Rename `CsvOptions.SchemaInferenceRows` → `SchemaInferenceRecords`**
   (property, ctors, `With` param, internal reader `CsvDataSource.cs:388`);
   update `IoOptionsTests.cs`.

### Part 3 — wrap-up

8. CHANGELOG entry (breaking notes: `TrainAndEvaluate` return type,
   `SchemaInferenceRows` rename, removal of static `ToNivaraFrame` +
   `ReshapeToArray`/`FlattenFromTensor(Array)`).
9. Verify: `dotnet build Nivara.slnx` (ask before `dotnet test`).

## Blast radius

- **#233**: only `src/Nivara.Extensions/MLNet/*` + `MLNet*Tests.cs`.
  `TrainAndEvaluate`/`ModelIntegration` have no production callers; `ReshapeToArray`
  /`FlattenFromTensor(Array)` only referenced by tests; static `ToNivaraFrame` only
  used by Extensions internals + tests. Samples don't touch the ML.NET surface.
- **#235**: core storage/tensor/training internals — scoping is cosmetic (all callers
  are InternalsVisibleTo friends: `Nivara.Tests`, `Nivara.Extensions`,
  `Nivara.PerformanceTests`). `PrintSummary` optional param keeps all 3 samples and 2
  tests compiling. CSV rename touches Extensions public API (breaking, low severity)
  + `IoOptionsTests`.

## Planned commit list

1. `docs: plan #233 + #235 cleanup in TODO.md`
2. `refactor: scope ML.NET sample POCOs to internal (#233)`
3. `refactor: strongly-type TrainAndEvaluate metrics via ModelEvaluationResult (#233)`
4. `refactor: drop Array-based TensorConversions reshape/flatten overloads (#233)`
5. `refactor: remove argument-flipped MLNetInterop.ToNivaraFrame static (#233)`
6. `refactor: scope internal-class public members to internal (#235)`
7. `refactor: route TrainingResult/DataParallelTrainingResult PrintSummary through TextWriter (#235)`
8. `refactor: unify SchemaInferenceRecords naming on CsvOptions (#235)`
9. `docs: changelog for #233 + #235 surface cleanup`
10. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build Nivara.slnx` before each commit; full `dotnet test` (asked, not
  auto-run) at the end — watch MLNet, Storage, Tensors, IO, AutoDiff suites.

## GitHub issues log

- (empty) — file issues discovered during execution here via
  `gh issue create --repo khurram-uworx/Nivara`.
