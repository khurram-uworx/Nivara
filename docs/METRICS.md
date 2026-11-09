# Metrics: frames in, frames out

Nivara owns the **table-level** metric. `Nivara.Metrics` turns a predicted column and an
actual column into a `NivaraFrame` you can filter, join, export or print. It does not own
*pipeline* evaluation: cross-validation, feature importance, automatic model selection and
metric logging belong to ML.NET's first-party evaluation surface, and Nivara does not
re-implement them.

That boundary is the same one recorded in [docs/TENSORS.md](TENSORS.md): the platform owns
the math and the ML workflow, Nivara owns the data layer. This module adds no numeric
library; it reuses the column kernels and `TensorPrimitives`.

## The three operations

| Method | Returns | Rows |
| --- | --- | --- |
| `NivaraMetrics.ConfusionMatrixFrame(IColumn predicted, IColumn actual)` | `Predicted`, one `Actual:{label}` per class, `Correct`, `Support` | one per class |
| `NivaraMetrics.PerClassMetricsFrame(IColumn predicted, IColumn actual)` | `Class`, `Tp`, `Fp`, `Fn`, `Precision`, `Recall`, `F1` | one per class |
| `NivaraMetrics.RegressionSummaryFrame(IColumn predicted, IColumn actual)` | `Count`, `Mae`, `Mse`, `Rmse`, `R2` | exactly one |

Classification requires both columns to have the same `ElementType` and a comparable label
type. Regression accepts two different numeric element types and converts both to `double`.
Every method requires an `IColumn` that realises as `NivaraColumn<T>` - the only in-repo
implementation, and what `Slice` returns - because the kernels work from the column's stored
values and null mask rather than from the `IColumn` surface. The element type `T` must itself
be non-nullable (`NivaraColumn<int>`, `NivaraColumn<double>`, and so on); null positions are
carried in the separate null mask produced by `NivaraColumnFactory.CreateFromNullable<T>(...)`.
A `NivaraColumn<T?>` whose nulls live inside its values is rejected with `ArgumentException`
naming the parameter and `CreateFromNullable`.

## The null rule

A row is a **valid pair** only when neither column is null there. A null in *either* column
excludes the row from every numerator, every denominator and every count, and the excluded
rows are not counted in any class. A label that appears only in excluded rows is not a
class. The rule follows the storage layer's null-mask contract: storage stays nullable and
each consumer decides what a null means, so AutoDiff throws at its domain boundary
(`docs/adr/001-autodiff-nonnullable-domain.md`) while metrics excludes the pair.

Zero valid pairs throws `ArgumentException`. Regression also rejects a non-finite value
(`NaN`, `+∞` or `-∞`) in a non-null cell up front, throwing `ArgumentException` that names
the offending row and column, so a summary frame is never quietly `NaN`. Classification
keeps labels opaque: a `double.NaN` label hashes equal to itself and becomes the class
`Actual:NaN`, and the default comparer orders it before every other label, which is the
standard .NET ordering.

An undefined *per-class* metric is null instead of zero, because zero is a real value and
would be read as a failed class:

| Situation | Result |
| --- | --- |
| Class never predicted (`Tp + Fp == 0`) | `Precision` null |
| Class never actual (`Tp + Fn == 0`) | `Recall` null |
| `Precision` or `Recall` null, or both are `0` | `F1` null |
| Constant target, every residual `0` | `R2 = 1.0` |
| Constant target, any non-zero residual | `R2 = 0.0` |
| Non-finite regression input (`NaN`, `+∞`, `-∞`) in a non-null cell | `ArgumentException` naming the row and column |
| `double.NaN` classification label | class named `Actual:NaN`, ordered before every other label |

## Reading the confusion matrix

Each row is a **predicted** class. Each `Actual:{label}` column counts rows whose actual
class is that label, so the cell `(predicted = p, actual = a)` is one count. `Correct` is
the diagonal cell for that row and `Support` is that row's total — the number of rows
*actually predicted* as that class. `Support` deliberately means predicted support here, so
`Tp + Fp` reproduces it; the actual-class total is `Tp + Fn` and is available in the
per-class frame.

The frame is one row and one column per class, so it grows as **O(k²)** in the class count.
That is fine for tens or hundreds of classes and wrong for tens of thousands; aggregate a
high-cardinality label first (`GroupBy`, bucket the tail) rather than materialising a column
per distinct value.

Labels are written into column names as `Actual:{label}`. The prefix keeps a label called
`Correct`, `Support` or `Predicted` from colliding with the reserved columns. Two labels
whose text differs only by case produce the same column name, because frame column names are
case-insensitive, and that is an `ArgumentException` rather than a silent merge. Class order
is deterministic: ordinal for `string`, default comparer ordering otherwise.

## Precision, recall and F1

`Tp` is the diagonal cell, `Fp` is the rest of the predicted row, and `Fn` is the rest of the
actual column. `Precision = Tp / (Tp + Fp)`, `Recall = Tp / (Tp + Fn)`, and
`F1 = 2PR / (P + R)`, each null under the table above. All three are `double?`, so an
undefined value survives the null mask rather than collapsing to `0`.

Per-class accuracy is `Recall` (`Tp / (Tp + Fn)`): a class's share of its actual support that
was predicted correctly. There is no separate `Accuracy` column in the per-class frame, because
that would duplicate `Recall`. Classifier (global) accuracy is a scalar over the whole
population and is available from the confusion matrix frame as
`Sum(Correct) / Sum(Support)`; it is not a per-class quantity.

## Regression

`Mae = mean |predicted − actual|`, `Mse = mean (predicted − actual)²`, `Rmse = √Mse`, and
`R2 = 1 − SS_res / SS_tot` with `SS_tot = Σ(actual − mean(actual))²`. Because `R2` uses
sums of squares, a prediction that is genuinely worse than the mean is negative — that is
the definition, not a bug, and it is reported as-is.

Every input value must be finite: a `NaN`, `+∞` or `-∞` in a non-null cell raises
`ArgumentException` instead of flowing into `Mae`, `Mse`, `Rmse` or `R2`.

Accumulation is in `double` regardless of the input element type: summing thousands of
`float` residuals loses precision against a hand-computed reference, and the frame has one
row, so the extra width costs nothing. A `double` column pair with no nulls is read in place
through `TryGetSpan`; any other pairing is converted once into pooled buffers, and null
exclusion happens during that conversion. `Count` is the number of valid pairs, so it also
tells you how much of the input the row was computed from.

## What the diagnostics record

Each call records one operation when `Nivara.Diagnostics.DiagnosticsTracker` is enabled:

| Operation | Kernel recorded | Why |
| --- | --- | --- |
| `Metrics.ConfusionMatrix` | `Scalar` | one typed hash pass over the valid pairs; there is no SIMD histogram primitive to hand this to |
| `Metrics.PerClassMetrics` | `Scalar` | same pass, same reason |
| `Metrics.RegressionSummary` | `KernelSelector.DetermineKernelType<double>(validPairs)` | the calculation really is `TensorPrimitives` over `double` |

Classification is **not** vectorized and the diagnostics say so. Class counting is a hash
lookup per row; a claim of SIMD here would be a claim about a kernel that does not exist.
The notes field carries `"{validPairs} of {total} pairs used"`, and `HadNulls` says whether
the exclusion came from a mask.

## Using it

```csharp
var predicted = NivaraColumn<string>.Create(new[] { "spam", "ham", "spam", "spam" });
var actual = NivaraColumn<string>.Create(new[] { "spam", "spam", "spam", "ham" });

NivaraFrame matrix = NivaraMetrics.ConfusionMatrixFrame(predicted, actual);
NivaraFrame perClass = NivaraMetrics.PerClassMetricsFrame(predicted, actual);

Console.WriteLine(string.Join(", ", matrix.ColumnNames));
Console.WriteLine(string.Join(", ", perClass.ColumnNames));
```

The frames are ordinary frames, so the same result can be filtered, joined to a label map,
written to CSV or inspected with `GetColumn<T>`.

## Watching a training loop

No hook is needed. `TrainingLoop.OnEpochEnd` is already `protected virtual`
(`src/Nivara/AutoDiff/Training/TrainingLoop.cs::OnEpochEnd`), so a subclass overrides it and
calls these methods on predictions it collects itself; `NivaraMetrics` holds no state and is
safe to call per epoch. Loss curves stay a training concern and are not part of this module.

## Cross-check against ML.NET

Nivara's confusion matrix (predicted classes in rows, actual classes in columns) and its
`MicroAccuracy = Sum(Correct) / Sum(Support)` agree with ML.NET's
`MulticlassClassificationMetrics` on a deterministic, deliberately asymmetric
three-class fixture:
`tests/Nivara.Tests/MLNet/MLNetMetricsCrossCheckTests.cs::ConfusionMatrix_MatchesMLNetEvaluation_Exactly`.
">

- [docs/TENSORS.md](TENSORS.md) — the Nivara-versus-platform positioning this module follows
- [docs/AUTODIFF.md](AUTODIFF.md) — the training stack that produces the predictions
- [docs/EMBEDDINGS.md](EMBEDDINGS.md) — the other frame-first AI interop convention
