# Half (F16) in Nivara

`System.Numerics.Half` is the IEEE-754 half-precision (binary16) floating-point
format: a 5-bit exponent with a 10-bit mantissa (plus implicit bit), giving
roughly 3 decimal digits of precision and a range of about ±65504. On **.NET 11**
it implements `IBinaryFloatingPointIeee754<Half>`, so it natively satisfies the
`IFloatingPointIeee754<T>` constraint that Nivara's AutoDiff and numeric kernels
are built around, and the BCL `TensorPrimitives` exposes `Half` arithmetic
overloads (e.g. `TensorPrimitives.Add<Half>`).

> **Vectorization note — read this first.** `Half` (like `BFloat16`) is
> **first-class and correct** in Nivara, but **not SIMD-accelerated** on .NET 11.
> The same ambiguity as with BFloat16 applies:
>
> - **Type support (TRUE).** `TensorPrimitives` exposes *generic* overloads such
>   as `Add<T>`, `Multiply<T>`, `Dot<T>`, `Sum<T>` constrained only to
>   `IAdditionOperators<T,T,T>` + `IAdditiveIdentity<T,T>`. `Half` satisfies those
>   interfaces, so `TensorPrimitives.Add<Half>` **compiles and returns correct
>   results** — no `NotSupportedException`.
> - **SIMD acceleration (FALSE).** Microsoft documents that TensorPrimitives is
>   *"hardware-accelerated for the element types that `Vector<T>` supports
>   (`Vector<T>.IsSupported`)."* Because `Vector<Half>.IsSupported` is `false` at
>   every width (64/128/256), the BCL cannot build a `Vector<Half>` and silently
>   drops to its **scalar** fallback loop. The operation is correct — just not
>   vectorized.
>
> So "Half is supported in Nivara" means: *the same generic code paths run
> end-to-end with correct results*, **not** that the kernel is vectorized.
> BFloat16 is in the identical situation; the two differ mainly in range/precision
> (see [Why Half](#why-half)) and in the cost of the manual widen-compute-narrow
> recovery path (Half needs a real cross-format conversion, not a bit shift).

| `TensorPrimitives` aspect | `Half` / `BFloat16` | `float` / `double` |
| --- | --- | --- |
| Generic overloads compile & run (`Add<T>`, `Dot<T>`, …) | Yes | Yes |
| Result correctness | Correct | Correct |
| SIMD / `Vector<T>` fast path | No — scalar BCL fallback | Yes |
| Relative throughput | ≈ hand-written scalar loop | much faster |

> **Hints for .NET `Numerics` engineers.**
> - `TensorPrimitives` is an *encapsulated* dispatcher: per element type `T` it
>   picks the SIMD kernel when `Vector<T>.IsSupported` is `true`, otherwise a
>   scalar loop. **You don't choose — the runtime does, per `T`.**
> - `Half` implements the generic-math interfaces, so every **generic**
>   `TensorPrimitives` overload compiles and returns correct results. Type-level
>   support ≠ acceleration.
> - SIMD in .NET is `Vector<T>`-gated. `Vector<Half>` reports `IsSupported == false`
>   on .NET 11 at every width. No `Vector<T>` support ⇒ no SIMD kernel in
>   `TensorPrimitives`, however "native" the type is.
> - The BCL does **not** auto-widen narrow floats to `float` to gain SIMD.
>   `TensorPrimitives.Add<Half>` operates on `Half` *as* `Half` and falls back to
>   scalar.
> - To actually vectorize Half you must do it yourself: convert the 16-bit values
>   to `float` lanes, run the genuinely-SIMD `TensorPrimitives<float>`, then
>   narrow back. Nivara ships this as `NarrowFloatKernels`/`WidenPrimitives`
>   (portable per-element conversion, since .NET 11 exposes no F16C batch
>   intrinsic).
> - Watch `Vector<Half>` in future .NET: if it ever flips to supported, the BCL
>   scalar fallback disappears with **no Nivara code change**.

Nivara supports `Half` in **two layers**:

1. **The column / query-analytics layer** — `Half` is a numeric column type:
   element-wise arithmetic (scalar by default), window functions, sorting,
   aggregation, and fused query expressions.
2. **The AutoDiff domain** — `Half` is a first-class gradient type: it flows
   through every op, optimizer, and module.

This document covers both. Related references:
[`AUTODIFF.md`](AUTODIFF.md), [`TENSORS.md`](TENSORS.md) (type-support note),
[`BFLOAT16.md`](BFLOAT16.md), [`INTEGERS.md`](INTEGERS.md),
[`SAFETENSORS.md`](SAFETENSORS.md).

---

## Why Half

- **Memory**: half the storage of `float` — attractive for large columnar
  datasets and model weights/activations/gradients.
- **Precision**: 10-bit mantissa (~3 decimal digits). Adequate for many inference
  workloads, but noticeably coarser than BF16's relationship to F32 and far
  coarser than float.
- **Range**: only ±65504. Unlike BF16 (which inherits float's 8-bit exponent),
  `Half` can overflow where gradients or normalized activations have a wide
  dynamic range. This is the main reason BF16 is often preferred for training
  even though both halve memory.
- **.NET 11**: the type is built in and `TensorPrimitives` exposes `Half`
  overloads, so Nivara's generic code paths apply with **no `NotSupportedException`**.
  They are **not** SIMD-accelerated: the BCL executes a scalar loop for `Half`
  arithmetic and reductions (see the *Vectorization note* above).

For concrete end-to-end numbers (weight memory F32 vs FP16 and
accuracy-vs-reference), see the *Narrow-precision inference* section of
`samples/NivaraInference/README.md`. Both FP16 (`Half`) and BF16 halve weight
memory (2 B/param vs `float`'s 4): the sample measures ~91→~45.5 MB (MiniLM) and
~255→~128 MB (DistilBERT / SST-2).

---

## Column / query-analytics layer

### Typed columns

```csharp
var col  = NivaraColumn<Half>.Create(new Half[] { (Half)1.5f, (Half)2.5f, (Half)3.5f });
var ncol = NivaraColumnFactory.CreateFromNullable(new Half?[] { (Half)1.5f, null, (Half)3.5f });
```

`Half` is recognized as a numeric type wherever the type system dispatches:
`NumericKernelDispatcher`'s arithmetic domain, `NumericPromoter`, `TypeExtensions.IsNumericType`,
and `TypeCompatibilityValidator.GetNumericTypes`. `Half` (like `BFloat16`) promotes to
`float`/`double` in mixed arithmetic — `Half + int` → `double`, since there is
no implicit `Half`↔integral conversion.

### Element-wise arithmetic (null-mask preserved)

```csharp
var scaled = col.Multiply((Half)2.0f);   // via the generic TensorPrimitives path (scalar for Half)
var ratio  = col.Divide((Half)2.0f);      // [0.75, 1.25, 1.75]
var added  = col.Add(otherColumn);        // column-on-column
```

Null masks propagate (`resultNullMask = leftNullMask OR rightNullMask`). There is
no separate specialized fast path beyond the generic `TensorPrimitives<Half>` loop;
`Vector<Half>` is unsupported on .NET 11, so dispatch is scalar. Element-wise kernels auto-widen for large buffers (on by default; opt out via `NivaraPrimitives.DisableWidenSimd`) — see
[Precision & limitations](#precision--limitations) — Half's path is more
expensive than BF16's because of the cross-format conversion.

### Window / cumulative / aggregation

`Rolling*` / `Cumulative*` / `Shift` / `Lead` all accept `NivaraColumn<Half>`.
`Sum`, `Mean`, `Quantile`, and `Median` over a `Half` column produce `double`
(the same precision-preserving promotion Half uses everywhere else; see
[`TENSORS.md`](TENSORS.md) data-prep note).

### Sorting and comparison

Multi-column sort, single-column comparers, and `OrderBy` accept `Half` columns.
Comparisons (`>`, `<`, `==`) follow SQL-like null-mask semantics (null positions
carry a null mask and `false` in the boolean output).

### Fused expressions

`ColumnExpression` fused evaluation supports `Half` columns. Mixed `Half + int`
promotes to `double`.

### Promotion

| Pair | Result |
| --- | --- |
| `Half + Half` | `Half` |
| `Half + int` | `double` |
| `Half + float` | `float` (widest floating operand wins) |
| `Half + double` | `double` |

---

## AutoDiff domain

`Half` is admitted through the `IFloatingPointIeee754<T>` constraint, so it is a
first-class gradient type across the AutoDiff stack:

- `ReverseGradTensor<Half>`, `ForwardGradTensor<Half>`.
- `TypeValidator` lists `typeof(Half)` as supported.
- Ops: all element-wise, matmul/linear, convs, attention, pooling, normalization,
  embeddings/lookup.
- Optimizers: `SGD<Half>`, `Adam<Half>`, `AdamW<Half>` (SIMD kernels are
  generic/scalar for Half; matmul row-dot falls back to scalar unless
  widen is on by default).
- Modules: `Linear<Half>`, `Sequential<Half>`, `Embedding<Half>`,
  `Conv1d`/`Conv2d`/`ConvTranspose2d`, `BatchNorm`, `LayerNorm`, `TransformerBlock`,
  etc. Note `Embedding` takes `int[]` token ids internally — Half (like all
  narrow types) can only exactly represent integers up to 2048, so avoid storing
  token ids as `Half` tensors (see [`INTEGERS.md`](INTEGERS.md)).
- Losses: MSE, L1, BCE, cross-entropy; cross-entropy targets are `int[]`.
- Initializers: Xavier/Kaiming/normal/uniform overloads for `Half`.
- Serialization: state dicts persist `Half` weights; SafeTensors round-trips.

AutoDiff is a non-nullable domain (ADR-001): convert nullable columns with
`FillNull`/`DropNulls` before crossing into `ReverseGradTensor<Half>`.

`NivaraAutoGradExtensions.ToReverseGradTensorsAuto` converts `Half` columns.

---

## Math helpers (`TensorsHelper`, `ModuleHelpers`)

- `MatMul`/`Dot` for `Half` runs through `TensorPrimitives.Dot<Half>` (`Dot<Half>`
  itself is the BCL scalar path, but Nivara's `TensorsHelper.MultiplyCore<T>` lifts
  those row dots to widen-compute-narrow above the length gate by default; opt
  out via `NivaraPrimitives.DisableWidenSimd`).
- Normalization/standardization: `Half` uses the in-place SIMD path via
  `TensorsHelper.TryNormalizeInPlace` (generic, scalar fallback for Half where
  vectorized).
- Data-prep `Normalize`/`Standardize` accept `Half` columns directly (no
  conversion to double), unlike integer/decimal columns.

---

## SafeTensors / sample I/O

The sample `SafeTensorsLoader` (and the `Nivara.Extensions` loader) support
on-disk `F16` (`Half`) tensors:

- `Read<Half>(path)` returns `Half[]` directly (no widening).
- Reading an F16 checkpoint as `float[]` widens losslessly.
- Reading an F32 checkpoint as `Half[]` narrows via `T.CreateChecked`.
- Any other dtype raises `NotSupportedException` with guidance.

See [`SAFETENSORS.md`](SAFETENSORS.md).

---

## Sample narrow-precision runs (inference)

`samples/NivaraInference` supports `--precision fp16`/`f16`/`half` where a Half
runner is implemented:

```bash
dotnet run --project samples/NivaraInference -c Release -- minilm --precision fp16
dotnet run --project samples/NivaraInference -c Release -- distilbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision fp16
dotnet run --project samples/NivaraInference -c Release -- modernbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- laya --precision fp16
dotnet run --project samples/NivaraInference -c Release -- smollm --precision fp16
```

- Vision models (MobileNetV2/ResNet-18) have no fp16 path.
- Qwen does not support fp16 (Half's range/precision is insufficient for Qwen2.5-0.5B; use `f32` or `bf16`).
- Silk/OpenCL (`--silk`) supports `--precision bf16` (widened on device); `--precision fp16` is rejected — bf16 is preferred (no measured speedup at current shapes, 0.97–1.07× vs f32). Use bf16 instead.

---

## Precision & limitations

- **Low precision (10-bit mantissa)**: Half holds ~3 decimal digits. Aggregation
  promotes to `double` to avoid accumulated error; widening F16→F32 for compute
  is lossless (Half's value set is a subset of float's).
- **Range**: ±65504 max. Prefer BF16 (float exponent) when gradients or
  activations span a wide dynamic range.
- **SIMD / Vector Lane Support (empirical, .NET 11)**: `Vector<Half>.IsSupported`
  is `false` at every width. Matmul (`TensorsHelper.MultiplyCore<T>`) routes only
  `float`/`double` to SIMD row-dot kernels; by default `Half` uses the widen path
  above its length gate (and the scalar `MultiplyRowScalar` path when the gate
  is missed or widen is disabled — that scalar path is ~26× slower in the disabled
  case, verified via
  `NivaraInference` benchmark). **Auto-widened SIMD path (default on):**
  `src/Nivara/Primitives/` provides `WidenPrimitives` + `NarrowFloatKernels`
  that convert `ushort` lanes to `float`, run in `Vector128<float>`, and narrow
  back, for element-wise `Add/Sub/Mul/Div` (`NumericTensorKernels`) and matmul
  row-dot (`TensorsHelper.MultiplyCore`, lifts AutoDiff). Enabled by default;
  opt out via the hidden AppContext switch `Nivara.Primitives.DisableWidenSimd`
  (or `NivaraPrimitives.DisableWidenSimd`). `ShouldWiden<T>` also
  applies a length gate (`Vector<byte>.Count * 4`) +
  `Vector.IsHardwareAccelerated` check (per-op, no temp `float` buffer —
  fused per-vector). Half's conversion is portable per-element
  (no F16C batch intrinsic in .NET 11), so its widen gain is smaller than
  BF16's (~10× is BF16's measured). Correctness unchanged by the toggle
  (Half/BF16 parity gates).

---

## Related

- [`BFLOAT16.md`](BFLOAT16.md) — BFloat16 matrix (compare range/precision).
- [`AUTODIFF.md`](AUTODIFF.md) — AutoDiff operations and domain.
- [`TENSORS.md`](TENSORS.md) — type-support note for tensors.
- [`SAFETENSORS.md`](SAFETENSORS.md) — dtype handling.