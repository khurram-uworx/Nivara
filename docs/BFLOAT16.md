# BFloat16 in Nivara

`System.Numerics.BFloat16` is a brain-floating-point format: an 8-bit exponent
(wide dynamic range, the same as `float`) with a 7-bit mantissa (low
precision). On **.NET 11** it implements `IBinaryFloatingPointIeee754<BFloat16>`,
so it natively satisfies the `IFloatingPointIeee754<T>` constraint that Nivara's
AutoDiff and numeric kernels are built around, and the BCL `TensorPrimitives`
exposes `BFloat16` arithmetic overloads.

Nivara supports `BFloat16` in **two layers**:

1. **The AutoDiff domain** — admitted in issue **#137** (merged to `main` after
   `v1.4.0`). `BFloat16` is a first-class gradient type: it flows through every
   op, optimizer, and module.
2. **The column / query-analytics layer** — wired in **POLARS-ROADMAP Phase 2**
   (branch `khurram/bfloat16`). `BFloat16` is now a first-class numeric column
   type: vectorized arithmetic, window functions, sorting, aggregation, and
   fused query expressions.

This document covers both. Related references:
[`AUTODIFF.md`](AUTODIFF.md),
[`TENSORS.md`](TENSORS.md) (type-support note),
[`INTEGERS.md`](INTEGERS.md),
[`plan/POLARS-ROADMAP.md`](plan/POLARS-ROADMAP.md).

---

## Why BFloat16

- **Memory**: half the storage of `float` — attractive for large columnar
  datasets and model activations/gradients.
- **Range**: the same exponent range as `float32`, so it survives the wide
  dynamic range of gradients and normalized activations without under/overflow
  (where `Half` can be marginal).
- **.NET 11**: the type is built in, and `TensorPrimitives` provides the SIMD
  arithmetic kernels — so Nivara's generic `TensorPrimitives`-based paths apply
  with no scalar fallback.

---

## 1. AutoDiff domain (issue #137)

### What works

`BFloat16` is admitted into `TypeValidator`'s supported set, so it is treated
exactly like `float`/`double`/`Half` across the autograd engine:

- **All operations** — element-wise, `MatMul` (runs through the BCL
  `TensorPrimitives.Dot` row dot-product; the old hand-rolled `Vector<T>` SIMD
  branch that threw `NotSupportedException` for `BFloat16` was removed),
  reductions, normalization, activations, attention, convolutions, VAE/Transformer
  modules.
- **Optimizers** — `SGD<BFloat16>`, `Adam<BFloat16>`, `AdamW<BFloat16>` with
  their `TensorPrimitives`-based state buffers.
- **Modules** — `Linear<BFloat16>`, `Sequential<BFloat16>`, `Embedding`,
  `Conv1d/2d`, `BatchNorm`, `LayerNorm`, `TransformerBlock`, `VAE`, etc., since
  they are all generic over `T : struct, IFloatingPointIeee754<T>`.
- **Transformer token-ID correctness** — `Embedding<T>` (and `BertEncoder<T>`,
  `MiniLMDistilled<T>`, `DistilBertForSequenceClassification<T>`) take token IDs
  as **exact `int[]`** via `Forward(int[] tokenIds, ...)` overloads. BFloat16 (and
  `Half`) cannot represent vocabulary indices (~30k) exactly — only integers up to
  256 — so passing token IDs as a `T` tensor before the embedding lookup corrupts
  them (e.g. `30522 → 30512`) and produces garbage output (~7 logit diff vs the
  F32 reference). Keeping the indices as `int` (independent of the compute dtype)
  makes BFloat16/Half transformer inference correct; the existing
  `ReverseGradTensor<T>` overloads remain for F32/F64. End-to-end,
  `DistilBertForSequenceClassification<BFloat16>` matches the F32 HuggingFace
  reference at **8/8 argmax** with a **~0.33 max logit diff**.
- **Frame → tensor batch** — `ToReverseGradTensorsAuto` now converts `BFloat16`
  frame columns (it previously skipped them).
- **Model serialization** — state dicts persist `BFloat16` weights via
  base64-encoded binaries.

### SafeTensors

`SafeTensorsLoader` still performs **BF16 → F32 widening as the default** for
`float`/`double` pipelines (lossless for inference), while `Read<BFloat16>`
performs **F32 → BF16 truncation** (genuine 7-bit-mantissa weights) so callers
can run inference in BFloat16. `ConvertBF16<BFloat16>` is available for native
`BFloat16` reads (BF16 → F32 → BF16 is lossless).

### Example — train a `BFloat16` linear model

```csharp
using System.Numerics;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;

// BFloat16 is a supported AutoDiff type
TypeValidator.IsSupportedType(typeof(BFloat16));   // true
NivaraAutoGradExtensions.IsAutoGradSupported<BFloat16>(); // true

// Trainable weights in BFloat16
var w = new ReverseGradTensor<BFloat16>(
    NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)0.5f, (BFloat16)0.5f }),
    requiresGrad: true);

var x = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)1.0f, (BFloat16)2.0f });
var y = ReverseGradTensor<BFloat16>.FromArray(new BFloat16[] { (BFloat16)3.0f, (BFloat16)5.0f });

using (GradientUtils.Grad())
{
    var pred = ReverseGradOperations.Multiply(x, w);            // [0.5, 1.0]
    var diff = pred - y;                                        // [-2.5, -4.0]
    var loss = ReverseGradOperations.Mean(ReverseGradOperations.Multiply(diff, diff));
    loss.Backward();                                            // fills w.Grad
}
// w.Grad now holds BFloat16 gradients
```

The `BFloat16Tests` suite verifies forward/backward parity with `float`
references, `Linear<BFloat16>` training under `SGD`/`Adam`, and the
inference-default graph guard.

---

## 2. Column / query-analytics layer (POLARS-ROADMAP Phase 2)

> Delivered on branch `khurram/bfloat16` (this branch). Mirrors the existing
> `Half` path one-for-one.

### Typed columns

```csharp
var col  = NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)1.5f, (BFloat16)2.5f, (BFloat16)3.5f });
var ncol = NivaraColumn.CreateFromNullable(new BFloat16?[] { (BFloat16)1.5f, null, (BFloat16)3.5f });
```

`BFloat16` is recognized as a numeric type everywhere the type system dispatches:
`NumericKernelDispatcher.arithmeticDomain`, `NumericPromoter`
(`BFloat16` promotes to `float`/`double` like `Half` — `BFloat16 + int` →
`double`, since there is no implicit `BFloat16`↔integral conversion),
`TypeCompatibilityValidator.GetNumericTypes`, and `TypeExtensions.IsNumericType`.

### Vectorized arithmetic (null-mask preserved)

```csharp
var scaled = col.Multiply((BFloat16)2.0f);   // [3.0, 5.0, 7.0] via generic TensorPrimitives SIMD path
var ratio  = col.Divide((BFloat16)2.0f);      // [0.75, 1.25, 1.75]
var added  = col.Add(otherColumn);            // column-on-column
```

Arithmetic runs through the same generic `TensorPrimitives` SIMD path as
`Half`; **null masks are preserved** (a null input position yields a null
output, like every other numeric type).

### Window functions

```csharp
var frame  = NivaraFrame.Create(("v", NivaraColumn<BFloat16>.Create(new BFloat16[] { (BFloat16)1, (BFloat16)2, (BFloat16)3 })));
var rolled = frame.RollingSum("v", "sum", 3);   // typed BFloat16 output column
var cum    = frame.CumulativeSum("v", "cum");
```

`Rolling*` / `Cumulative*` / `Shift` / `Lead` all accept `NivaraColumn<BFloat16>`
via `Over()`/`WindowSpec`.

### Sorting

```csharp
var sorted = frame.OrderBy("v");             // multi-column sort + comparers support BFloat16
```

### Aggregation (precision-promoted to `double`)

`Sum`, `Mean`, `Quantile`, and `Median` over a `BFloat16` column all produce
`double` (the same precision-preserving promotion `Half` uses):

```csharp
var allRows = Enumerable.Range(0, col.Length).ToList();
AggregationFunctions.Sum().Apply(col, allRows);          // double
AggregationFunctions.Mean().Apply(col, allRows);         // double
AggregationFunctions.Quantile(0.5).Apply(col, allRows);  // double (median)
AggregationFunctions.Median().Apply(col, allRows);       // double
```

### Fused query expressions

```csharp
using Nivara.Expressions;

var input = new Dictionary<string, IColumn> { ["A"] = col };
var fused = new FusedExpressionEvaluator();

// Same-type expression runs through the fused evaluator
var sameType = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Col("A"), input);
// sameType.ElementType == typeof(BFloat16)

// Mixed BFloat16 + int promotes to double (safe superset, like a C# error pair)
var promoted = fused.Evaluate(ColumnExpressions.Col("A") + 1, input);
// promoted is NivaraColumn<double>
```

### NivaraSeries

`NivaraSeries` gains a `BFloat16` conversion arm so quantile/aggregation over
series works as well.

> **Comparisons** (`>`, `<`, `==`) on `BFloat16` columns fall back to
> `Comparer<T>.Default` (scalar, not SIMD) — identical to `Half`, which is also
> absent from the comparison fast-path domain. They are correct, just not
> vectorized.

---

## What you could not do before

- **AutoDiff (pre-#137):** every `BFloat16` autograd operation threw
  `NotSupportedException`. The hand-rolled `Vector<T>` matmul SIMD branch
  rejected `BFloat16`, and `TypeValidator` excluded it. Now it is admitted at
  runtime and exercises the BCL `TensorPrimitives.Dot` path.
- **Column / query (pre-Phase 2):** `NivaraColumn<BFloat16>` arithmetic threw
  `NotSupportedException` (it was absent from `NumericKernelDispatcher.arithmeticDomain`);
  `ExpressionTypeInferer` excluded `BFloat16` from the fused evaluator; and
  aggregation, quantile, window functions, and sorting had no `BFloat16` arm.
  All of those are now wired (mirroring `Half`).

---

## Precision & limitations

- **Low precision (like `Half`).** Aggregation promotes to `double` to avoid
  loss; keep this in mind for cumulative sums over many rows.
- **Comparisons are scalar**, not SIMD — same as `Half`.
- **Frame convenience ops with their own type switches** (e.g. the data-prep
  `Normalize` / `Standardize` helpers) may still not support `BFloat16`. This is
  consistent with how `Half` is treated and is a known follow-up, not a
  regression.
- **Non-nullable at the AutoDiff boundary (ADR-001).** Resolve nulls
  (`FillNull` / `DropNulls`) before converting a `BFloat16` column to a gradient
  tensor.

---

## Provenance

- **AutoDiff (`#137`)** — commits `eb50279` … `595704f` (merged to `main` after
  `v1.4.0` via PR #339). `CHANGELOG.md` "Unreleased" records the admission;
  `TENSORS.md` carries the type-support note.
- **Column / query (Phase 2)** — branch `khurram/bfloat16`; see
  `POLARS-ROADMAP.md` Phase 2 "Scope (remaining)". The Step-0 probe confirmed
  the active net11 BCL `TensorPrimitives` exposes `BFloat16` arithmetic
  overloads, so the generic SIMD path applies with no scalar fallback.
