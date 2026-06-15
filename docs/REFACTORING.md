# Autograd / Autodiff Refactoring — Discussion Plan

## Status

This document is a **proposal for a future architectural refactor**, not the
current implementation contract. It should be discussed with the team before
coding starts.

Since the original draft, the smaller AutoDiff completion plan has been
implemented:

- Inference is now the default. Reverse-mode graph construction only happens
  inside `using (GradientUtils.Grad())`.
- Built-in training APIs enter `GradientUtils.Grad()` internally.
- `Module<T>.StateDict()` and `Module<T>.LoadStateDict(...)` are implemented.
- `ModelSerializer.StateDictToJson(...)` and
  `ModelSerializer.JsonToStateDict<T>(...)` are implemented.
- `docs/AUTODIFF-PLAN.md` has been removed because its actionable items are
  complete.
- Full test suite status after those changes: `1740` passing tests.

Therefore this refactor should preserve the user-facing direction:
**predict by default, train explicitly**. Do not introduce `NoGrad` as the
primary public API.

## Observations

These observations were gathered after the original plan and should be used
when revising the implementation approach:

- `TensorPrimitives` APIs are span-based. Inputs are generally
  `ReadOnlySpan<T>` and outputs are `Span<T>`.
- `TensorPrimitives` does not operate directly on `IReadOnlyMemory<T>` or
  `ReadOnlyMemory<T>`. Memory remains useful for ownership or transport, but
  callers should use `.Span` before invoking `TensorPrimitives`.
- Tensor-backed paths should first try to get a contiguous
  `ReadOnlySpan<T>` or `Span<T>` from `Tensor<T>`, `ReadOnlyTensorSpan<T>`, or
  `TensorSpan<T>`. If that is unavailable, flatten or copy into a temporary
  span-backed buffer before calling `TensorPrimitives`.
- Generic `TensorPrimitives` overloads are governed by generic math
  constraints per operation, not only by a hardcoded SIMD-supported type list.
- SIMD support is an implementation and performance detail. The API contract
  is the relevant generic math interface support for each operation.
- Operation constraints vary by method:
  - `Add`, `Subtract`, and `Multiply` use operator and identity constraints.
  - `Exp` and `Sigmoid` use `IExponentialFunctions<T>`.
  - `Log` uses `ILogarithmicFunctions<T>`.
  - `Tanh` uses hyperbolic-function constraints or specific overloads,
    depending on the package/API version.
  - `Norm` and `CosineSimilarity` use floating/root-style constraints.
- The current repo targets `net10.0` and references
  `System.Numerics.Tensors` `10.0.9`.
- Any future `GradKernels` design should use `ReadOnlySpan<T>` input and
  `Span<T>` output signatures, not `IReadOnlyMemory<T>`.
- The refactor should distinguish "type satisfies the API constraints" from
  "type is SIMD-accelerated."
- AutoDiff may not need a full raw-`Tensor<T>` rewrite to use
  `TensorPrimitives` well. The current common path already creates
  `NivaraColumn<T>` values through `ColumnStorageFactory`, which selects
  `TensorStorage<T>` for supported unmanaged numeric types, and many AutoDiff
  paths already call `TryGetSpan(...)` before using `TensorPrimitives`.
- The stronger requirement is that AutoDiff declare its boundary explicitly:
  tensors entering AutoDiff should be dense, non-null, numeric, and
  span-capable/tensor-backed for the operations being performed.
- Once that boundary is enforced, AutoDiff does not need to carry nullable
  column semantics through the graph. Null cleaning should happen before entry
  (`DropNulls`, `FillNull`, or explicit rejection), and gradients should remain
  dense non-null numeric values.
- A smaller revision may be enough: keep `NivaraColumn<T>` as the AutoDiff data
  boundary, validate `HasNulls == false` and span/tensor capability on entry,
  then implement internal kernels over `ReadOnlySpan<T>` and `Span<T>`. Only
  replace AutoDiff storage with raw `Tensor<T>` if profiling or API pressure
  proves the column wrapper is the actual problem.
- The bigger cleanup target may be the extension-method boundary rather than
  the storage type itself. Many methods in `src/Nivara/Tensors`, especially
  activation, gradient, matrix, and helper operations currently used mainly by
  AutoDiff, are written as general `NivaraColumn<T>` extensions and therefore
  carry null-aware and mixed-storage branches. See
  `src/Nivara/Tensors/NivaraTensorExtensions.cs` for definitions such as
  `Exp`, `Log`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `MatMul`,
  `Transpose`, and the gradient helpers.
- Moving AutoDiff-only tensor helpers into an AutoDiff-owned namespace/folder,
  such as `src/Nivara/AutoDiff/Tensors`, would let those methods assume the
  explicit AutoDiff boundary: dense, non-null, numeric, span-capable data. That
  would simplify implementations substantially while keeping column-level
  reductions and genuinely public tensor interop in the column/tensor layer.
- Current AutoDiff call sites include
  `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` and
  `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs`, which call these
  column extensions through expressions like `a.Data.Sigmoid()`,
  `a.Data.Tanh()`, `a.Data.MatMul(...)`, `typedGradOutput.Transpose(...)`, and
  gradient helpers such as `a.Data.ReluGradient(typedGradOutput)`.
- Under that narrower refactor, the plan should separate three categories:
  public column reductions and interop stay in `Nivara.Tensors`; obsolete
  `NivaraSeries` tensor helpers can be removed; AutoDiff-only kernels move into
  AutoDiff and become simpler span-based methods with no null-mask propagation.
- A cleaner revision may be to make `GradTensor<T>`,
  `ReverseGradTensor<T>`, and `ForwardGradTensor<T>` the explicit AutoDiff
  boundary gates. Their constructors and factories (`FromColumn`, `FromArray`,
  `FromMatrix`, `FromSeries`, and any future tensor/span factory) should
  validate the AutoDiff contract once: numeric type, dense non-null data,
  correct shape, and span/tensor capability for the needed operations.
- Those boundary gates can pluck the required representation from constructor
  inputs, using zero-copy access from tensor-backed `NivaraColumn<T>` where
  available and copying only when necessary. After construction, AutoDiff
  operations, modules, optimizers, training, and AutoDiff-owned tensor helpers
  should use `GradTensor<T>`/span-facing APIs rather than reaching back into
  general `NivaraColumn<T>` extension methods.
- This may be cleaner than the lower plan's broad raw-`Tensor<T>` rewrite:
  keep `NivaraColumn<T>` as an accepted input/output boundary, but do not make
  it the internal abstraction every AutoDiff kernel must reason about.
- After the `GradTensor<T>` boundary decision, the next plan areas to revisit
  are the dependent AutoDiff plumbing: `OpNode` and `ComputationGraph`
  gradient delegate types, `ReverseGradOperations` and `ForwardGradOperations`,
  optimizers (`SGD`, `Adam`, `AdamW`), training/data loaders, state-dict and
  model serialization, initializers, and cleanup of remaining tensor helpers.
  The important question for those sections is whether they should use raw
  `Tensor<T>` directly, or a `GradTensor<T>`/span-facing internal abstraction
  that already enforces the AutoDiff boundary.
- Future Streamix interop should influence the AutoDiff/tensor boundary.
  Streamix pipelines are `IAsyncEnumerable<T>`-based, windowed,
  cancellation-aware, error-aware, and backpressure-sensitive, so Nivara should
  expose clean frame/window-to-column/tensor conversion boundaries without
  requiring `ReadOnlySpan<T>`, `Span<T>`, or `TensorSpan<T>` to escape across
  `await`, `yield`, channel, or stream-window boundaries. Spans should be
  acquired and consumed inside synchronous kernel calls only.
- Streamix windows can materialize to `NivaraFrame`, transformations can
  produce derived columns, and prediction should enter inference through the
  same dense non-null `GradTensor<T>`/span-facing boundary. Keep Streamix
  integration out of Nivara core unless there is a strong reason otherwise;
  prefer an adapter or extension package that provides operators such as
  `ToFrame()`, `ToTensor<T>()`, and `Predict(...)` while preserving Streamix
  ordering, cancellation, error, and backpressure semantics.
- The refactor should preserve inference as the cheap/default path for
  streaming prediction. `Predict(...)` over Streamix windows should not build
  reverse-mode graphs unless explicitly inside `GradientUtils.Grad()` or a
  training API.

## Discussion points

Before implementation, align on these decisions:

1. **Scope**: keep the original full-refactor stance, or split into reviewable
   slices with the same final architecture?
2. **Column nullable model**: confirm the type-level `NivaraColumn<float>` vs.
   `NivaraColumn<float?>` storage split before changing storage contracts.
3. **AutoDiff null semantics**: confirm AutoDiff becomes a no-null pure tensor
   layer, with nullable data cleaned at the DataFrame boundary.
4. **State-dict compatibility**: decide whether the Tensor-backed serializer
   must read the current JSON/base64 state-dict format, or whether 0.x can
   break saved model files.
5. **Frame tensor methods**: confirm `Dot`, `CosineSimilarity`, `ColumnNorms`,
   and `RowNorms` move out of core into Extensions, including whether to keep
   obsolete forwarding shims for one release.
6. **Test gate**: keep `dotnet test` full-suite green at each validation point,
   with explicit `GradientUtils.Grad()` scopes for backward/training tests.

## Original session context

This document captures the architectural refactoring plan agreed during a
conversation between Khurram (maintainer) and the AI coding agent. Key
decisions and rationale:

**Original intent of AutoDiff**: The AutoDiff subsystem was created as a
correctness proof for Nivara's columnar engine — a way to verify that
null-mask propagation, shape safety, and storage abstractions work
correctly by building something real on top. Over time it grew into a
functional ML toolkit (Linear, optimizers, training loop, serialization,
forward+backward autograd) with proven parity against PyTorch (0.04%
loss-curve diff backward, 1e-5 tolerance JVP forward). But the
architectural layering drifted.

**The drum machine philosophy**: Nivara's AutoDiff is positioned as a
"drum machine with the best drum sounds" — a small, correct, tasteful set
of gradient primitives that make the 70–80% case feel complete. It is not
a PyTorch replacement and should not chase feature parity (no Conv, no
RNN, no LR schedulers, no AMP, no GPU). The current user-facing API and
exclusion rationale now live in `docs/AUTODIFF.md`; `examples/README.md`
contains the parity proofs.

**Gardening principle**: subtract → become lean → grow. Remove bad
coupling first (NivaraColumn from AutoDiff, null masks from hot paths,
obsolete code), then the remaining surface becomes cleaner and simpler.
The previously planned ergonomic additions (`GradientUtils.Grad()`,
`StateDict()`, `LoadStateDict()`, and state-dict JSON helpers) have already
landed before this refactor. The refactor must carry those semantics forward.

**Platform sugar**: Use `Tensor<T>` from System.Numerics.Tensors directly
as the backing store for AutoDiff. Use `TensorPrimitives` for SIMD-cycled
float/double kernels. No custom vectorization abstractions, no custom
intrinsics, no custom tensor hierarchy — .NET 10 owns that now. The
storage engine should consume as much platform sugar as possible.

**Nullable vs non-nullable columns**: `NivaraColumn<float>` is backed by
`Tensor<float>` with NO null mask — pure math, `HasNulls = false`. `NivaraColumn<float?>`
(`Nullable<float>`) gets a separate `NullableTensorStorage<T>` that
manages `Tensor<T> + Tensor<bool>` mask behind the scenes. The type system
drives the storage decision, not a runtime flag. Users clean their nullable
data (`DropNulls`, `FillNull`) at the boundary before entering AutoDiff.

**NivaraSeries**: Kept as a labeled-column-wrapper. All tensor math
(`Sum`, `Mean`, `AddTensor`, etc.) removed from it. Revisit after the
next usability layer if it doesn't pull its weight.

**Refactor stance to discuss**: The original plan assumed a full refactor in
one shot because Nivara is still 0.x and has no backward compatibility
obligations. That may still be the right call, but the team should explicitly
confirm scope before implementation because the current AutoDiff surface now
has passing tests, docs, examples, inference-default semantics, training
loops, state dictionaries, and model serialization.

## Motivation

The AutoDiff subsystem was originally written as a correctness proof for Nivara's
columnar engine (null-mask propagation, shape safety, etc.). Over time it grew into
a functional ML toolkit, but with architectural debt:

- `GradTensor<T>` stores data in `NivaraColumn<T>`, pulling null-mask machinery,
  series conversions, and the full columnar API into every tensor operation
- Activation and gradient functions (Sigmoid, Tanh, ReLU, MatMul, etc.) live
  on `NivaraColumn<T>` extension methods (`NivaraTensorExtensions`) when they
  are only ever called from AutoDiff
- The columnar layer pays for null-mask branches even when T is non-nullable
- Obsolete `NivaraSeries` tensor math (`AddTensor`, `DotProduct`, etc.) has no
  production callers

**Goal**: Subtract → become lean → grow. Define clean layering between the
columnar engine (null-aware, type-safe, DataFrame) and the autograd engine
(pure math, no nulls, platform primitives).

## Layering after refactoring

```
┌──────────────────────────────────────────────────┐
│             AutoDiff Layer                         │
│  ReverseGradTensor  ForwardGradTensor              │
│  GradOperations  ForwardGradOperations             │
│  GradKernels (span-based, TensorPrimitives ops)    │
│  NN Module System  Optimizers  Training            │
│  Storage: Tensor<T> from System.Numerics.Tensors   │
│  No nulls. No NivaraColumn. Pure math.             │
├──────────────────────────────────────────────────┤
│         Columnar / DataFrame Layer                  │
│  NivaraColumn<T>  NivaraColumn<T?>                 │
│  NivaraFrame  NivaraSeries (labeled wrapper)        │
│  Null semantics  Schema  I/O  Joins  GroupBy        │
│  Reductions (Sum, Mean, Min, Max) with null support │
│  Backed by Tensor<T> (non-nullable) or              │
│    Tensor<T> + bool mask (nullable)                 │
├──────────────────────────────────────────────────┤
│                Storage Layer                        │
│  TensorStorage<T>  NullableTensorStorage<T>         │
│  MemoryStorage<T>  ColumnStorageFactory             │
└──────────────────────────────────────────────────┘
```

## What moves where

| Current location | What | Moves to | Reason |
|---|---|---|---|
| `NivaraTensorExtensions` | Sigmoid, Tanh, ReLU, LeakyReLU, Exp, Log, Softmax, LogSoftmax, Abs, Clamp, Negate, Divide | `AutoDiff/GradKernels` | Only called from AutoDiff. No null handling needed. |
| `NivaraTensorExtensions` | SigmoidGradient, TanhGradient, ReluGradient, AbsGradient, ClipGradient, LeakyReluGradient, LogGradient, SoftmaxGradient, LogSoftmaxGradient | `AutoDiff/GradKernels` | Only exist for backward pass. |
| `NivaraTensorExtensions` | MatMul, Transpose | `AutoDiff/GradKernels` | NN-only operations. |
| `NivaraTensorExtensions` | Sum, Mean, Min, Max, Norm (column extensions) | **Stay on NivaraColumn** | Null-aware reductions belong on columns. |
| `NivaraTensorExtensions` | AddTensor, MultiplyTensor, SumTensor, DotProduct, Norm, TransformTensor (Series) | **Delete** | No production callers. Dead code in `NivaraSeriesIsValidTests.cs` only. |
| `NivaraFrame` | Dot, CosineSimilarity, ColumnNorms, RowNorms | **Move** to `Nivara.Extensions.Tensors` | Per TENSORS.md — column math should not pretend to be tensor axes. |
| `NivaraFrame` | MatrixMultiply | **Delete** | No callers anywhere. |
| `NivaraSeries` | Sum, Mean, tensor math methods | **Remove** from NivaraSeries (already on NivaraColumn) | NivaraSeries becomes a labeled-column-wrapper only. |
| `ColumnStorageFactory` | IsVectorizable<T>() dispatch | **Add** nullable axis | Pick TensorStorage<T> or NullableTensorStorage<T> based on T vs T?. |
| `TensorStorage<T>` | Null mask fields and branches | **Remove** for non-nullable T | T? gets its own `NullableTensorStorage<T>` class. |
| `GradTensor<T>` | NivaraColumn<T> Data | **Replace** with Tensor<T> Data | Back onto System.Numerics.Tensors. |
| `Grad/ForwardGradOperations` | 1221 + 990 lines of NivaraColumn ops | **Rewrite** to Tensor<T> + GradKernels | ~50% line reduction, no column coupling. |

## 1. Column storage: nullable vs non-nullable

### Type-level dispatch

`ColumnStorageFactory` determines the storage backend by examining T at
construction time:

```
typeof(T)           IsVectorizable?  IsNullable?      Storage backend
─────────────────────────────────────────────────────────────────────
float               ✓                 no               TensorStorage<float> (no mask)
float?              ✓                 yes              NullableTensorStorage<float> (data + mask)
double              ✓                 no               TensorStorage<double> (no mask)
double?             ✓                 yes              NullableTensorStorage<double> (data + mask)
int                 ✓                 no               TensorStorage<int> (no mask)
int?                ✓                 yes              NullableTensorStorage<int> (data + mask)
string              ✗ (ref type)     —                MemoryStorage<string> (no change)
```

Detection:

```csharp
internal static bool IsNullableType(Type type) =>
    type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>);

internal static Type GetNonNullableType(Type type) =>
    IsNullableType(type) ? type.GetGenericArguments()[0] : type;
```

### `TensorStorage<T>` — non-nullable only

Stripped of all null-mask machinery:

| Member | Before | After |
|---|---|---|
| `HasNulls` | computed from mask | `false` (constant) |
| `NullCount` | computed from mask | `0` (constant) |
| `NullMask` | `Tensor<bool>?` | empty span |
| Internal | `Tensor<T> data + Tensor<bool>? nullMask` | `Tensor<T> data` only |

All null-branch operations (mask propagation, null filtering) are removed.
The storage is a pure `Tensor<T>` wrapper.

### `NullableTensorStorage<T>` — new, nullable only

New internal class for `T?` types where `T` is unmanaged. Same semantics as
the current `TensorStorage<T>` but only instantiated for nullable types:

```csharp
internal sealed class NullableTensorStorage<T> : IColumnStorage<T?>
    where T : unmanaged
{
    private readonly Tensor<T> _data;
    private readonly Tensor<bool>? _nullMask;
    // ...
}
```

`NullableTensorStorage<T>` implements `IColumnStorage<Nullable<T>>` and
provides the same null-aware operations (mask propagation for arithmetic,
null filtering for reductions).

### `NivaraColumn<T>` — conditional behavior

`NivaraColumn<T>` is a single class that adapts its behavior based on
whether T is nullable or not:

| Member | Non-nullable T | Nullable T (i.e. U?) |
|---|---|---|
| `this[i]` | returns `T` | returns `T` (which is `Nullable<U>`) |
| `HasNulls` | `false` | computed from storage mask |
| `NullCount` | `0` | computed from storage mask |
| `DropNulls()` | returns `this` (identity) | filters, returns `NivaraColumn<U>` |
| `FillNull(T)` | no-op (identity) | replaces nulls → `NivaraColumn<U>` |
| `ToArray()` | copies `Tensor<T>` | copies nulls as `default(T)` |
| `Slice()` | slices data only | slices data + mask |

The column is constructed via `ColumnStorageFactory.Create(ReadOnlySpan<T>)`,
which selects the appropriate storage backend.

### Enter/exit AutoDiff boundary

```csharp
// Enter: columnar → tensor
// Non-nullable: zero-copy when storage is already Tensor<T>
var col = frame.GetColumn<float>("features");               // NivaraColumn<float>
var tensor = ReverseGradTensor<float>.FromColumn(col);      // Tensor<T>, no nulls

// Nullable: must clean first
var nullableCol = frame.GetColumn<float?>("features_with_nulls");
var cleaned = nullableCol.DropNulls();                      // NivaraColumn<float>
var tensor = ReverseGradTensor<float>.FromColumn(cleaned);

// Exit: tensor → columnar
var resultCol = resultTensor.ToColumn();                     // NivaraColumn<float>, no mask
frame.AddColumn("predictions", resultCol);
```

`FromColumn` validates that the column has no nulls (throws if `HasNulls`).
`ToColumn` wraps the `Tensor<T>` data in a non-nullable `NivaraColumn<T>`
with no null mask.

## 2. AutoDiff: GradTensor backs onto Tensor<T>

### Core principle

`GradTensor<T>` — and therefore `ReverseGradTensor<T>` and
`ForwardGradTensor<T>` — stores data as `Tensor<T>` from
System.Numerics.Tensors. No nulls. No NivaraColumn in the hot path.

### `GradTensor<T>` base class

| Aspect | Before | After |
|---|---|---|
| Backing store | `NivaraColumn<T>` | `Tensor<T>` (internal) |
| `Data` property (public) | `NivaraColumn<T>` | **Removed** — replaced with span access |
| `Length` | `Data.Length` | `Data.FlattenTo(stackalloc...).Length` or `_shape[0]` |
| `Shape` | `int[]` clone | `int[]` clone (unchanged) |
| `HasNulls` | delegated to `Data.HasNulls` | **Removed** — tensor T has no nulls |
| `IsNull(int)` | `Data.IsNull(i)` | **Removed** |
| `ToColumn()` | returns `Data` | wraps `Tensor<T>` in `NivaraColumn<T>` (zero-copy when possible) |
| `ToSeries()` | `new NivaraSeries<T>(Data)` | wraps via `ToColumn()` |
| `AsTensor()` | two-hop through Series | returns `Data` directly (identity) |

What replaces the public `Data` property: a method to get the underlying
span for interop:

```csharp
// For zero-copy interop with TensorPrimitives etc
public ReadOnlySpan<T> GetDataSpan() => _data.TryGetSpan(out var s) ? s : _data.FlattenTo(...);
```

But most internal code accesses the backing data through GradKernels, not
through a public property.

### `ReverseGradTensor<T>`

| Aspect | Before | After |
|---|---|---|
| `Grad` | `NivaraColumn<T>?` | `Tensor<T>?` |
| Constructor | takes `NivaraColumn<T>` | takes `Tensor<T>` (or factory methods) |
| `FromColumn` | wraps NivaraColumn | extracts Tensor<T> from column |
| `FromArray` | `NivaraColumn<T>.Create(array)` | `Tensor.Create<T>(array, [array.Length])` |
| `FromMatrix` | column wrapping | `Tensor.Create<T>(data, [rows, cols])` |
| `FromSeries` | extracts column | extracts column, then its Tensor<T> |
| `Backward()` | seeds grad as NivaraColumn | seeds grad as `Tensor.Create<T>([T.One], [1])` |

### `ForwardGradTensor<T>`

Same pattern as ReverseGradTensor. `Tangent` becomes `Tensor<T>?`.

### `OpNode`

BackwardFunction delegate signature changes:

```csharp
// Before
public Action<NivaraColumn<T>, bool> BackwardFunction { get; }

// After
public Action<Tensor<T>, bool> BackwardFunction { get; }
```

`Apply` method:

```csharp
// Before
public void Apply(NivaraColumn<T> gradOutput, bool stripGradientNulls)

// After
public void Apply(Tensor<T> gradOutput, bool stripGradientNulls)
```

### `ComputationGraph`

`Backward<T>()` method signature changes:

```csharp
// Before
public void Backward<T>(ReverseGradTensor<T> tensor,
    NivaraColumn<T>? gradient, bool stripGradientNulls = false)

// After
public void Backward<T>(ReverseGradTensor<T> tensor,
    Tensor<T>? gradient, bool stripGradientNulls = false)
```

## 3. GradKernels — the kernel workhorse

**New file**: `src/Nivara/AutoDiff/Operations/GradKernels.cs`

A single static class containing all activation, gradient, and matrix
operations. Methods take `ReadOnlySpan<T>` inputs, write to `Span<T>`
outputs. No allocations on the hot path.

**Pattern**:

```csharp
internal static class GradKernels
{
    public static void Sigmoid<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(float))
            TensorPrimitives.Sigmoid(
                MemoryMarshal.Cast<T, float>(input),
                MemoryMarshal.Cast<T, float>(output));
        else if (typeof(T) == typeof(double))
            TensorPrimitives.Sigmoid(
                MemoryMarshal.Cast<T, double>(input),
                MemoryMarshal.Cast<T, double>(output));
        else
            for (int i = 0; i < input.Length; i++)
                output[i] = T.One / (T.One + T.Exp(-input[i]));
    }

    public static void ReluGradient<T>(ReadOnlySpan<T> input,
        ReadOnlySpan<T> gradOutput, Span<T> result)
        where T : unmanaged, INumber<T> { ... }

    public static void MatMul<T>(ReadOnlySpan<T> a, int aRows, int aCols,
        ReadOnlySpan<T> b, int bRows, int bCols, Span<T> result)
        where T : unmanaged, INumber<T> { ... }

    // ~20 methods total
}
```

**Methods**:

| Method | Inputs | Output | Notes |
|---|---|---|---|
| `Sigmoid` | input | output | TensorPrimitives pass-through for float/double |
| `SigmoidGradient` | output, gradOutput | result | `result * (1 - result) * gradOutput` |
| `Tanh` | input | output | TensorPrimitives pass-through |
| `TanhGradient` | output, gradOutput | result | `(1 - output²) * gradOutput` |
| `Relu` | input | output | element-wise max(0, x) |
| `ReluGradient` | input, gradOutput | result | `(input > 0) ? gradOutput : 0` |
| `LeakyRelu` | input, slope | output | element-wise |
| `LeakyReluGradient` | input, gradOutput, slope | result | slope for x < 0 |
| `Exp` | input | output | TensorPrimitives pass-through |
| `Log` | input | output | TensorPrimitives pass-through |
| `LogGradient` | input, gradOutput | result | `gradOutput / input` |
| `Softmax` | input, classCount | output | exp/rowSum |
| `SoftmaxGradient` | softmaxOutput, gradOutput, classCount | result | Jacobian |
| `LogSoftmax` | input, classCount | output | log(softmax) |
| `LogSoftmaxGradient` | logSoftmaxOutput, gradOutput, classCount | result | Jacobian |
| `Abs` | input | output | element-wise |
| `AbsGradient` | input, gradOutput | result | sign(x) * gradOutput |
| `Clamp` | input, min, max | output | element-wise clamp |
| `ClipGradient` | input, gradOutput, min, max | result | pass-through only where in bounds |
| `Negate` | input | output | `-x` |
| `Divide` | a, b | result | `a / b` |
| `MatMul` | a, aRows, aCols, b, bRows, bCols | result | gemm-style |
| `Transpose` | input, rows, cols | result | row-major transpose |

**No null checks. No mask propagation. Pure span math.**

## 4. GradOperations / ForwardGradOperations rewrite

Both files get rewritten from:

```csharp
// Before — NivaraColumn ops with null overhead
public static ReverseGradTensor<T> Add<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b)
    where T : unmanaged, INumber<T>
{
    var data = a.Data + b.Data;         // NivaraColumn op — null check, mask merge, allocation
    // backward function ...
    return new ReverseGradTensor<T>(data, ...);
}
```

To:

```csharp
// After — Tensor<T> + GradKernels, no null branches
public static ReverseGradTensor<T> Add<T>(ReverseGradTensor<T> a, ReverseGradTensor<T> b)
    where T : unmanaged, INumber<T>
{
    var spanA = a.GetDataSpan();
    var spanB = b.GetDataSpan();
    var resultData = Tensor.CreateUninitialized<T>([a.Length]);
    var resultSpan = resultData.TryGetSpan(out var s) ? s : resultData.FlattenTo(...);
    TensorPrimitives.Add(spanA, spanB, resultSpan);

    var result = new ReverseGradTensor<T>(resultData);
    // backward function operates on Tensor<T> gradients
    return result;
}
```

**Net reduction**: ~1221 → ~600 lines for `GradOperations.cs`,
~990 → ~500 lines for `ForwardGradOperations.cs`.

All backward functions use `GradKernels` on extracted spans.

## 5. Downstream adaptation

### Optimizers (SGD, Adam, AdamW)

`tensor.Grad` changes from `NivaraColumn<T>?` to `Tensor<T>?`.

Gradient accumulation:

```csharp
// Before
var merged = NivaraColumnUtility.MergeNullMasks(existingGrad, newGrad);
tensor.Grad = merged;

// After  
TensorPrimitives.Add(existingGradSpan, newGradSpan, resultSpan);
tensor.Grad = Tensor.Create<T>(resultArray, [existingGrad.Length]);
```

No `MergeNullMasks` calls. No null-skip branches. Gradients are always
dense arrays (nulls don't exist in the gradient domain).

### Training (TensorDataset, DataLoader, TrainingLoop)

`Batch<T>` holds `ReverseGradTensor<T>` backed by `Tensor<T>`.

`TensorDataset<T>` extracts `T[]` slices from columns using `TensorPrimitives`
span operations instead of `NivaraColumn.Slice()` + null mask.

### Serialization (ModelSerializer)

Simpler format: serialize `Tensor<T>` data as base64 (no null mask per
parameter). Shape stored as JSON int array.

Preserve the existing public model:

- `Module<T>.StateDict()` returns a copied snapshot dictionary.
- `Module<T>.LoadStateDict(state, strict: false)` supports partial loading by
  default and strict full-restore validation when requested.
- `ModelSerializer.Save/Load` remain file-level wrappers over state dicts.
- `ModelSerializer.StateDictToJson` / `JsonToStateDict<T>` remain the
  in-memory JSON boundary.

The refactor may simplify the internal payload because AutoDiff tensors would
no longer carry null masks, but it should not remove the state-dict workflow.

### Initializers

Fill `T[]` arrays directly, pass to `Tensor.Create<T>()`. No NivaraColumn
intermediary.

```csharp
// Before
var tensor = NivaraColumn<T>.Create(values);   // column wrapping
var result = new ReverseGradTensor<T>(tensor);

// After
var tensor = Tensor.Create(values, [values.Length]);
var result = new ReverseGradTensor<T>(tensor);
```

### NivaraTensorExtensions (cleanup)

Remove all activation, gradient, MatMul, Transpose methods. Keep only:

- `Sum<T>(this NivaraColumn<T>)` — null-aware via type check
- `Mean<T>(this NivaraColumn<T>)` — same
- `Min<T>(this NivaraColumn<T>)` — same
- `Max<T>(this NivaraColumn<T>)` — same

Each has two paths: `TensorPrimitives` for non-nullable T, manual loop
with mask skip for nullable T (or T?).

Delete obsolete methods:
- `AddTensor`, `MultiplyTensor`, `SumTensor`, `DotProduct`, `Norm`, `TransformTensor` (Series)

### NivaraSeries

Strip tensor math. Keep:
- Labeled wrapper around `NivaraColumn<T>`
- `ToColumn()`
- LINQ helpers (`Select`, `Where`, `OrderBy`)
- Indexer with label lookup

`Sum()` / `Mean()` callers in AutoDiff (and tests) redirect to
`column.Sum()`.

## 6. NivaraFrame deprecations

Per `TENSORS.md`, these are moved to `Nivara.Extensions.Tensors`:

| Method | Current home | New home |
|---|---|---|
| `Dot(NivaraColumn<float>, NivaraColumn<float>)` | `NivaraFrame` | `Nivara.Extensions.Tensors.FrameTensorExtensions` |
| `CosineSimilarity(...)` | `NivaraFrame` | same |
| `ColumnNorms(...)` | `NivaraFrame` | same |
| `RowNorms(...)` | `NivaraFrame` | same |

Core `NivaraFrame` loses these methods. The Extensions namespace holds them
with an obsolete-forwarding shim for one release, then removed.

```csharp
// NivaraFrame.cs — removed
// Nivara.Extensions.Tensors/FrameTensorOperations.cs — added
public static class FrameTensorOperations
{
    public static NivaraColumn<float> Dot(this NivaraFrame frame,
        string colA, string colB) { ... }
}
```

## 7. File manifest

### New files

| File | Purpose |
|---|---|
| `src/Nivara/Storage/NullableTensorStorage.cs` | Nullable column storage (data + mask) |
| `src/Nivara/AutoDiff/Operations/GradKernels.cs` | All span-based kernel operations |

### Rewritten (core changes)

| File | Change summary |
|---|---|
| `src/Nivara/Storage/TensorStorage.cs` | Remove null mask, pure Tensor<T> |
| `src/Nivara/Storage/ColumnStorageFactory.cs` | Add nullable dispatch |
| `src/Nivara/NivaraColumn.cs` | Conditional null behavior per T vs T? |
| `src/Nivara/AutoDiff/GradTensor.cs` | Back onto Tensor<T> |
| `src/Nivara/AutoDiff/ReverseGradTensor.cs` | Grad as Tensor<T>? |
| `src/Nivara/AutoDiff/ForwardGradTensor.cs` | Tangent as Tensor<T>? |
| `src/Nivara/AutoDiff/OpNode.cs` | Delegate signature change |
| `src/Nivara/AutoDiff/ComputationGraph.cs` | Gradient type change |
| `src/Nivara/AutoDiff/Operations/GradOperations.cs` | 1221→~600 lines, Tensor<T> ops |
| `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs` | 990→~500 lines, Tensor<T> ops |

### Adapted (minor changes)

| File | Change |
|---|---|
| `src/Nivara/AutoDiff/Nn/Parameter.cs` | Adapt to Tensor<T> |
| `src/Nivara/AutoDiff/Nn/Initializers/*.cs` (14 files) | Direct Tensor<T> construction |
| `src/Nivara/AutoDiff/Optimizer/SGD.cs` | Tensor<T>? Grad, no MergeNullMasks |
| `src/Nivara/AutoDiff/Optimizer/Adam.cs` | Same |
| `src/Nivara/AutoDiff/Optimizer/AdamW.cs` | Same |
| `src/Nivara/AutoDiff/Training/TensorDataset.cs` | Span slices from Tensor<T> |
| `src/Nivara/AutoDiff/Training/TrainingLoop.cs` | Adapt to new Batch/Batch types |
| `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs` | No null mask per param |
| `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` | Tensor<T> factory methods |

### Stripped / cleaned

| File | Change |
|---|---|
| `src/Nivara/Tensors/NivaraTensorExtensions.cs` | Keep only Sum/Mean/Min/Max, delete rest |
| `src/Nivara/NivaraSeries.cs` | Remove tensor math methods |
| `src/Nivara/NivaraFrame.cs` | Remove Dot/CosineSimilarity/ColumnNorms/RowNorms (move to Extensions) |

### Deleted

| File | Reason |
|---|---|
| `src/Nivara/Tensors/` obsolete Series extensions (in `NivaraTensorExtensions.cs`) | No callers |

### Tests

| File | Change |
|---|---|
| `tests/Nivara.Tests/AutoDiff/GradOperationsTests.cs` | Adapt construction, keep explicit `GradientUtils.Grad()` scope for backward tests |
| `tests/Nivara.Tests/AutoDiff/ForwardGradOperationsTests.cs` | Same |
| `tests/Nivara.Tests/AutoDiff/NullHandlingTests.cs` | Keep — validates nullable→non-nullable boundary |
| `tests/Nivara.Tests/AutoDiff/BackwardPassTests.cs` | Adapt to Tensor<T>, keep explicit `GradientUtils.Grad()` scope |
| `tests/Nivara.Tests/AutoDiff/NnTests.cs` | Adapt |
| `tests/Nivara.Tests/AutoDiff/LossTests.cs` | Adapt |
| `tests/Nivara.Tests/AutoDiff/ForwardParityTests.cs` | Adapt construction |
| `tests/Nivara.Tests/AutoDiff/TypeSafetyTests.cs` | Validate FromColumn throws on nullable |
| `tests/Nivara.Tests/AutoDiff/NivaraIntegrationTests.cs` | Adapt |
| `tests/Nivara.Tests/AutoDiff/SerializationTests.cs` | Preserve StateDict/LoadStateDict and JSON state-dict coverage |
| `tests/Nivara.Tests/AutoDiff/GradientUtilsTests.cs` | Preserve inference-default tests and explicit Grad-scope tests |
| `tests/Nivara.Tests/NivaraSeriesIsValidTests.cs` | Remove obsolete tensor math tests |
| `samples/Nivara.SampleApp/ForwardParityExample.cs` | Adapt construction |
| `samples/Nivara.SampleApp/AutoDiffExample.cs` | Adapt construction |
| `samples/Nivara.SampleApp/CrossFrameworkFraudNet.cs` | Adapt construction |

### Rough line count deltas

| Before | After | Δ |
|---|---|---|
| NivaraTensorExtensions: ~1000 lines | ~200 lines | -800 |
| GradOperations: ~1221 | ~600 | -621 |
| ForwardGradOperations: ~990 | ~500 | -490 |
| TensorStorage: ~272 | ~150 | -122 |
| GradKernels: 0 | ~400 | +400 |
| NullableTensorStorage: 0 | ~150 | +150 |
| ColumnStorageFactory: ~80 | ~110 | +30 |
| **Net** | | **~-1453 lines** |

## 8. Risk and testing strategy

### Risks

1. **Tensor<T>.TryGetSpan behavior**: `Tensor<T>` may not always provide a
   contiguous span for large multi-dimensional tensors. Use `FlattenTo`
   with stack-alloc or small-array fallback. The pattern is
   `data.TryGetSpan(out var span) ? span : data.FlattenTo(scratch)`.

2. **Zero-copy FromColumn**: For non-nullable `NivaraColumn<T>` backed by
   `TensorStorage<T>`, the underlying `Tensor<T>` is already there.
   `FromColumn` can extract it via internal access. No copy.

3. **Nullable column storage performance**: `NullableTensorStorage<T>`
   stores data + mask separately. Row operations (slice, copy) must handle
   both tensors. This is correct but ~2x allocation for the nullable path.
   Acceptable — the non-nullable path is where performance matters.

4. **OpNode delegate type change**: All backward closures in GradOperations
   and ForwardGradOperations need to be updated. Each closure captures the
   inputs' Tensor data for gradient computation. This is mechanical but
   affects every operation.

### Test strategy

1. **Column storage unit tests**: Create `NivaraColumn<float>` (no mask) and
   `NivaraColumn<float?>` (with mask). Verify HasNulls, NullCount, indexer,
   DropNulls, FillNull behavior.

2. **AutoDiff boundary tests**: `FromColumn` on non-nullable succeeds,
   `FromColumn` on nullable throws, `ToColumn` returns correct type.

3. **GradKernels unit tests**: Each kernel operation tested against known
   inputs/outputs. Float and double paths both covered.

4. **Inference/training mode tests**: Preserve the current contract that
   normal forward operations do not build a graph and backward/training tests
   opt in with `GradientUtils.Grad()`.

5. **State-dict tests**: Preserve full load, partial load, strict missing-key
   validation, shape mismatch validation, and state-dict JSON round-trip.

6. **Existing parity tests (unchanged)**: ForwardCrossFrameworkParityTests,
   CrossFrameworkParityTests, GradOperationsTests, ForwardGradOperationsTests
   — adapt construction, keep assertions.

7. **Null handoff test**: `nullableCol.DropNulls()` → `FromColumn` →
   train → `ToColumn` → verify result is non-nullable.

### Recovery

If a step breaks downstream code, fix forward. The sequence is designed so
that storage changes (1-4) can be validated before AutoDiff kernel changes
(5-11). If the column tests pass, the foundation is solid.

## 9. Sequence

```
 1. NullableTensorStorage          ← new type, no deps
 2. TensorStorage clean            ← remove masks, no deps
 3. ColumnStorageFactory           ← nullable dispatch
 4. NivaraColumn<T>                ← conditional null behavior
    ─── VALIDATE: column tests pass ───
 5. GradKernels                    ← new file, spans only
 6. GradTensor rewrite             ← Tensor<T> backing
 7. ReverseGradTensor rewrite      ← Tensor<T>? Grad
 8. ForwardGradTensor rewrite      ← Tensor<T>? Tangent
 9. OpNode + ComputationGraph      ← gradient type change
10. GradOperations rewrite         ← use GradKernels + Tensor<T>
11. ForwardGradOperations rewrite  ← same
    ─── VALIDATE: Autodiff tests pass ───
12. NivaraTensorExtensions         ← strip to reductions
13. NivaraSeries cleanup           ← remove tensor math
14. Frame deprecations             ← move Dot etc to Extensions
15. Optimizer adaptation           ← Tensor<T>? Grad
16. Training adaptation            ← Tensor<T> in datasets
17. Serialization adaptation       ← Tensor<T> format, preserve state dict APIs
18. Initializer adaptation         ← direct Tensor<T>
19. Tests + samples                ← adapt everything
    ─── VALIDATE: full dotnet test ───
```
