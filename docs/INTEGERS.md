# Integer Types (I32, I64) in Nivara AutoDiff

## Status: Research & Documentation

This document covers how integer-typed tensors (I32, I64) are handled when
loading model weights from SafeTensors files, and what options exist for
improving or bypassing the current widening-to-float approach.

---

## 1. Core Constraint

AutoDiff's generic constraint is:

```csharp
where T : struct, IFloatingPointIeee754<T>
```

This excludes all non-floating-point types at compile time:

| Type   | Satisfies `IFloatingPointIeee754`? | Can be `ReverseGradTensor`? |
|--------|------------------------------------|-----------------------------|
| float  | Yes                                | Yes                         |
| double | Yes                                | Yes                         |
| Half   | Yes                                | Yes                         |
| BFloat16 | No (NET 10). Planned for NET 11  | No (until NET 11)           |
| int    | No                                 | No                          |
| long   | No                                 | No                          |
| short  | No                                 | No                          |
| byte   | No                                 | No                          |

**Rationale**: AutoDiff relies on `TensorPrimitives` (which requires
floating-point for most operations like `Exp`, `Log`, `Sigmoid`) and on
differentiability (integer types are not differentiable). The constraint is
correct and should not be relaxed.

---

## 2. Where Integer Tensors Appear in Practice

### 2a. Model Weights (SafeTensors)

Some SafeTensors files contain integer weight tensors:

- **I32**: uncommon. Occasionally used for biases or older model formats
  (e.g., some BERT variants store `token_type_embeddings` as I32).
- **I64**: used for large embedding tables (vocab >2^31) or for positional
  encoding tables. Rare in typical transformer models.

In practice, these integer weights represent real-valued quantities that were
stored as integers for file-size or format reasons (e.g., ONNX exported with
integer quantization). They must be widened to a float type for computation.

### 2b. Token IDs (Model Input)

`Embedding<T>.Forward()` takes `ReverseGradTensor<T>` as input, then
extracts `int[]` internally via `int.CreateChecked(input.Data[i])`.

```csharp
// Embedding.cs:45-47
var tokenIds = new int[totalTokens];
for (int i = 0; i < totalTokens; i++)
    tokenIds[i] = int.CreateChecked(input.Data[i]);
```

This means token IDs are currently stored as `float[]` / `Half[]` / `double[]`
in the input tensor and then narrowed to `int[]` inside the Embedding layer.
The float→int conversion is **lossless for float** (all int32 values up to 2^24
are exactly representable), but **lossy for Half** (Half can only exactly
represent integers up to 2048).

### 2c. Class Labels (Loss Functions)

`CrossEntropyLoss<T>.Forward()` accepts `int[]` targets directly — no tensor
wrapper.

```csharp
// CrossEntropyLoss.cs:21
public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, int[] targets)
```

Labels bypass the tensor system entirely. No change needed.

---

## 3. Current Behavior in SafeTensorsLoader

`samples/Nivara.Samples/SafeTensorsLoader.cs` currently widens ALL dtypes to
`float[]`:

| Source Dtype | Conversion                 | Precision                     |
|-------------|----------------------------|-------------------------------|
| F32         | Zero-copy reinterpret      | Exact                         |
| F16         | `(float)src[i]`            | Exact (Half→float is lossless) |
| BF16        | Bit-shift + Unsafe.As      | Exact (BF16→float preserves all 7 mantissa bits) |
| I32         | Implicit widening `int→float` | Exact for `|v| ≤ 2^24`. Approximate for larger values (nearest even) |
| I64         | Implicit widening `long→float` | Exact for `|v| ≤ 2^24`. Approximate for larger values (rounds to ~7 significant digits) |

### Precision Analysis: I32 → float

| I32 range              | float representation           |
|------------------------|--------------------------------|
| `[-2^24, 2^24]`       | Exact                          |
| Outside `[-2^24, 2^24]` | Rounds to nearest even float |

Since model weight values are typically small (bias values in [-10, 10],
weight values in [-5, 5] after normalization), this is lossless for all
practical weight tensors.

### Precision Analysis: I64 → float

| I64 range              | float representation           |
|------------------------|--------------------------------|
| `[-2^24, 2^24]`       | Exact                          |
| `[2^24, 2^25)`        | Multiples of 2                 |
| `[2^25, 2^26)`        | Multiples of 4                 |
| ...                    | ...                            |
| `[2^51, 2^52)`        | Multiples of 2^28 (~268M)      |

For large embedding tables stored as I64, precision loss can be significant.
However, such tables are rare and typically represent token counts or
cumulative statistics, not learnable weight values.

---

## 4. Current Approach (Phase A Implementation)

The generic `SafeTensorsLoader.Read<T>()` method widens I32/I64 via
`T.CreateChecked(src[i])`:

```csharp
// Generic widening for integer types
static T[] ConvertI32<T>(ReadOnlySpan<byte> bytes)
    where T : struct, IFloatingPointIeee754<T>
{
    var src = MemoryMarshal.Cast<byte, int>(bytes);
    var result = new T[src.Length];
    for (int i = 0; i < src.Length; i++)
        result[i] = T.CreateChecked(src[i]);
    return result;
}
```

### Precision by Target Type

| Target | I32 exact              | I64 exact               |
|--------|------------------------|-------------------------|
| float  | Up to ±2^24 (16,777,216) | Up to ±2^24 (16,777,216) |
| double | All exact              | Up to ±2^53 (~9e15)    |
| Half   | Up to ±2048            | Up to ±2048             |

The target type is chosen by the model. Typical usage:

- `Read<float>()` for existing float models
- `Read<Half>()` for memory-optimized inference
- `Read<double>()` for high-precision training

---

## 5. Alternatives Considered

### 5a. Separate Integer Tensor for Index/ID Data

**Idea**: Create a non-differentiable `IndexTensor` type (not backed by
`ReverseGradTensor`) used exclusively for embedding lookups and label data.

**Pros**:
- Avoids wasteful float→int conversion in `Embedding.Forward`
- Halves memory for token ID storage
- Type-safe: can't accidentally differentiate through indices

**Cons**:
- New type and API surface
- Requires `Embedding.Forward(int[])` overload
- Doesn't solve the weight-tensor widening problem (weights are still float)

**Verdict**: Worth considering for a future release. Not a blocker for
Phase A-C.

### 5b. BFloat16 Instead of Half for I64 Weight Tensors

**Idea**: Wait for .NET 11 `BFloat16` support, which maintains larger
dynamic range than `Half` for integer widening.

**Pros**:
- BFloat16 has 8-bit exponent (same as float32), so can represent much larger
  values than Half (5-bit exponent)
- Lower precision loss for large I64→BF16 than I64→Half

**Cons**:
- Not available in .NET 10
- No `IFloatingPointIeee754<BFloat16>` (planned for NET 11)
- Half is sufficient for current model weight ranges

**Verdict**: Track via GitHub issue (#76). Do not block on this.

### 5c. Quantization-Aware Loader

**Idea**: Keep integer weights as compressed integers in memory and
dequantize on-the-fly during `Linear.Forward` or `Embedding.Forward`.

**Pros**:
- Maximum memory efficiency (I8, I16 weight storage)
- True low-precision deployment

**Cons**:
- Very large engineering effort
- Requires custom quantized linear/embedding kernels
- Post-training quantization is Not Implemented

**Verdict**: Out of scope for current work. Document for future.

### 5d. BCL Tensor<int> / Tensor<long> Storage (No AutoDiff)

**Idea**: Use `System.Numerics.Tensors.Tensor<int>` directly for non-differentiable
integer data within `NivaraColumn<int>` (already supported via `ColumnStorage<int>`).

**Pros**:
- Zero-copy storage for integer columns in DataFrames
- Already works (`ColumnStorage<int>` supports int, long, etc.)

**Cons**:
- Can't be used with AutoDiff
- No gradient tracking (by design)
- Requires separate code paths for integer data

**Verdict**: Already the case — integer data in DataFrames uses
`ColumnStorage<int>`. The issue is only about
AutoDiff tensor conversion.

### 5e. Revisit: Remove Embedding.Forward T Cast Overhead

**Idea**: Add `Embedding.Forward(int[] indices)` and `Embedding.Forward(NivaraColumn<int> indices)`
overloads that bypass the float→int extraction.

```csharp
public ReverseGradTensor<T> Forward(int[] indices)
{
    var embedding = ReverseGradOperations.Gather(weight.Tensor, indices);
    return embedding;
}
```

**Pros**:
- No wasteful float→int CreateChecked conversion
- Clearer API for integer-indexed lookups
- Works with any T (float/Half/double)

**Cons**:
- Additional API surface
- Callers must convert token IDs from `ReverseGradTensor<T>` → `int[]` manually
  (or use `NivaraColumn<int>` path)
- CrossEntropyLoss already takes int[] directly (pattern exists)

**Verdict**: Worth implementing — low effort, high value.

---

## 6. Key Files

| File | Purpose |
|------|---------|
| `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` | Defines supported types |
| `src/Nivara/AutoDiff/Nn/Embedding.cs` | Embedding layer with float→int extraction |
| `src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs` | Loss with int[] target overload |
| `samples/Nivara.Samples/SafeTensorsLoader.cs` | Weight loader with dtype→float conversion |

---

## 7. Open Questions

1. **Should `Embedding<T>.Forward(ReverseGradTensor<T>)` warn when T=Half and
   token IDs > 2048?** Silent precision loss is dangerous.

2. **Should we add `Embedding<T>.Forward(int[])`?** Low effort, matches
   `CrossEntropyLoss` pattern. See 5e.

3. **Should I32/I64 widening warn or throw on precision loss?** E.g., if
   `T=Half` and an I32 value is 100000, the result is `Half.PositiveInfinity`.

4. **Should `SafeTensorsLoader.Read<T>()` reject integer dtypes when T=Half
   to avoid silent overflow?** Or just let `T.CreateChecked` throw?

5. **Does .NET 11 BFloat16 change anything?** BFloat16 has 8-bit exponent,
   so it can represent up to ~3.4e38 (same as float32), but only 7 mantissa
   bits. This would make I64→BFloat16 widening much safer for large values.
